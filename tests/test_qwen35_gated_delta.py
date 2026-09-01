from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).parents[1]
    / "nanovllm"
    / "models"
    / "qwen35_gated_delta.py"
)
SPEC = spec_from_file_location("qwen35_gated_delta_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qwen35_gated_delta = module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen35_gated_delta
SPEC.loader.exec_module(qwen35_gated_delta)

Qwen35RecurrentStatePool = qwen35_gated_delta.Qwen35RecurrentStatePool
Qwen35GatedDeltaNet = qwen35_gated_delta.Qwen35GatedDeltaNet
causal_conv1d_scan = qwen35_gated_delta.causal_conv1d_scan
causal_conv1d_prefill = qwen35_gated_delta.causal_conv1d_prefill
chunk_gated_delta_rule = qwen35_gated_delta.chunk_gated_delta_rule
effective_chunk_size = qwen35_gated_delta.effective_chunk_size
recurrent_gated_delta_rule = qwen35_gated_delta.recurrent_gated_delta_rule
recurrent_gated_delta_step = qwen35_gated_delta.recurrent_gated_delta_step


def inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 5, 3, 4, generator=generator)
    k = torch.randn(2, 5, 3, 4, generator=generator)
    v = torch.randn(2, 5, 3, 6, generator=generator)
    decay = -torch.rand(2, 5, 3, generator=generator)
    beta = torch.rand(2, 5, 3, generator=generator)
    return q, k, v, decay, beta


@pytest.mark.parametrize(
    ("sequence_length", "maximum", "expected"),
    [(1, 64, 1), (5, 64, 8), (16, 64, 16), (17, 64, 32), (65, 64, 64)],
)
def test_effective_chunk_size_avoids_excess_short_prefill_padding(
    sequence_length,
    maximum,
    expected,
):
    assert effective_chunk_size(sequence_length, maximum) == expected


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("sequence_length", [1, 2, 7, 16])
@pytest.mark.parametrize("kernel_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vectorized_causal_convolution_matches_scan(
    batch_size,
    sequence_length,
    kernel_size,
    dtype,
):
    torch.manual_seed(37)
    channels = 9
    x = torch.randn(batch_size, sequence_length, channels, dtype=dtype)
    state = torch.randn(batch_size, channels, kernel_size, dtype=dtype)
    weight = torch.randn(channels, kernel_size, dtype=dtype)
    bias = torch.randn(channels, dtype=dtype)

    expected, expected_state = causal_conv1d_scan(x, state, weight, bias)
    actual, actual_state = causal_conv1d_prefill(x, state, weight, bias)

    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(actual_state, expected_state)


def test_vectorized_causal_convolution_handles_empty_prefill():
    x = torch.empty(2, 0, 5)
    state = torch.randn(2, 5, 4)
    weight = torch.randn(5, 4)

    output, next_state = causal_conv1d_prefill(x, state, weight)

    assert output.shape == x.shape
    assert next_state is state


def test_recurrent_rule_chunked_prefill_matches_one_shot():
    q, k, v, decay, beta = inputs()
    expected, expected_state = recurrent_gated_delta_rule(q, k, v, decay, beta)

    first, state = recurrent_gated_delta_rule(
        q[:, :3], k[:, :3], v[:, :3], decay[:, :3], beta[:, :3]
    )
    second, state = recurrent_gated_delta_rule(
        q[:, 3:], k[:, 3:], v[:, 3:], decay[:, 3:], beta[:, 3:], state
    )

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


def test_single_token_decode_matches_last_prefill_token():
    q, k, v, decay, beta = inputs(1)
    prefix, state = recurrent_gated_delta_rule(
        q[:, :-1], k[:, :-1], v[:, :-1], decay[:, :-1], beta[:, :-1]
    )
    decoded, state = recurrent_gated_delta_rule(
        q[:, -1:], k[:, -1:], v[:, -1:], decay[:, -1:], beta[:, -1:], state
    )
    expected, expected_state = recurrent_gated_delta_rule(q, k, v, decay, beta)

    torch.testing.assert_close(torch.cat((prefix, decoded), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_specialized_decode_step_matches_recurrent_oracle(dtype):
    torch.manual_seed(41)
    query = torch.randn(5, 3, 4, dtype=dtype)
    key = torch.randn(5, 3, 4, dtype=dtype)
    value = torch.randn(5, 3, 6, dtype=dtype)
    decay = -torch.rand(5, 3)
    beta = torch.rand(5, 3, dtype=dtype)
    state = torch.randn(5, 3, 4, 6, dtype=dtype)

    expected, expected_state = recurrent_gated_delta_rule(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        decay.unsqueeze(1),
        beta.unsqueeze(1),
        state,
    )
    actual, actual_state = recurrent_gated_delta_step(
        query,
        key,
        value,
        decay,
        beta,
        state,
    )

    torch.testing.assert_close(actual, expected.squeeze(1))
    torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("sequence_length", [1, 5, 17, 64, 65])
def test_chunk_rule_matches_recurrent_oracle(sequence_length):
    q, k, v, decay, beta = inputs(5)
    repeats = (sequence_length + q.shape[1] - 1) // q.shape[1]
    q, k, v, decay, beta = (
        tensor.repeat(1, repeats, 1, 1)[:, :sequence_length]
        if tensor.ndim == 4
        else tensor.repeat(1, repeats, 1)[:, :sequence_length]
        for tensor in (q, k, v, decay, beta)
    )
    expected, expected_state = recurrent_gated_delta_rule(q, k, v, decay, beta)

    actual, actual_state = chunk_gated_delta_rule(
        q, k, v, decay, beta, chunk_size=16
    )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-4)


def test_causal_convolution_chunk_and_decode_are_equivalent():
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(2, 6, 7, generator=generator)
    weight = torch.randn(7, 4, generator=generator)
    initial = torch.zeros(2, 7, 4)
    expected, expected_state = causal_conv1d_scan(x, initial, weight)

    first, state = causal_conv1d_scan(x[:, :5], initial, weight)
    last, state = causal_conv1d_scan(x[:, 5:], state, weight)

    torch.testing.assert_close(torch.cat((first, last), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


def test_causal_convolution_matches_grouped_conv1d_reference():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(2, 6, 7, generator=generator)
    weight = torch.randn(7, 4, generator=generator)
    bias = torch.randn(7, generator=generator)
    initial = torch.zeros(2, 7, 4)

    actual, _ = causal_conv1d_scan(x, initial, weight, bias)
    expected = F.silu(
        F.conv1d(
            x.transpose(1, 2),
            weight.unsqueeze(1),
            bias,
            padding=3,
            groups=7,
        )[:, :, : x.shape[1]]
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_state_pool_isolates_updates_and_resets_reused_slots():
    pool = Qwen35RecurrentStatePool(2, 4, 3, 4, 6, 14, 4, device="cpu")
    slots = torch.tensor([3, 1])
    recurrent, convolution = pool.get(0, slots)
    pool.update(0, slots, recurrent + 1, convolution + 2)

    assert torch.count_nonzero(pool.recurrent[0, 3]) > 0
    assert torch.count_nonzero(pool.recurrent[0, 0]) == 0
    pool.reset(torch.tensor([3]))
    assert torch.count_nonzero(pool.recurrent[:, 3]) == 0
    assert torch.count_nonzero(pool.convolution[:, 3]) == 0
    assert torch.count_nonzero(pool.recurrent[0, 1]) > 0


def test_state_pool_can_store_recurrent_state_in_model_dtype():
    pool = Qwen35RecurrentStatePool(
        2,
        4,
        3,
        4,
        6,
        14,
        4,
        device="cpu",
        recurrent_dtype=torch.bfloat16,
        convolution_dtype=torch.bfloat16,
    )

    assert pool.recurrent.dtype == torch.bfloat16
    assert pool.convolution.dtype == torch.bfloat16
    assert pool.recurrent.element_size() == 2


def test_state_pool_converts_fp32_updates_to_compressed_storage_dtype():
    pool = Qwen35RecurrentStatePool(
        1,
        2,
        2,
        3,
        4,
        5,
        2,
        device="cpu",
        recurrent_dtype=torch.bfloat16,
        convolution_dtype=torch.bfloat16,
    )
    slots = torch.tensor([1])
    recurrent = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    convolution = torch.randn(1, 5, 2, dtype=torch.float32)

    pool.update(0, slots, recurrent, convolution)

    assert pool.recurrent.dtype == torch.bfloat16
    assert pool.convolution.dtype == torch.bfloat16
    torch.testing.assert_close(
        pool.recurrent[0, 1].float(),
        recurrent[0],
        rtol=4e-3,
        atol=4e-3,
    )
    torch.testing.assert_close(
        pool.convolution[0, 1].float(),
        convolution[0],
        rtol=4e-3,
        atol=4e-3,
    )


def tiny_config():
    return SimpleNamespace(
        hidden_size=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1e-6,
    )


def make_layer(rank=0, world_size=1, layer_config=None):
    with (
        patch.object(qwen35_gated_delta.dist, "get_world_size", return_value=world_size),
        patch.object(qwen35_gated_delta.dist, "get_rank", return_value=rank),
    ):
        return Qwen35GatedDeltaNet(layer_config or tiny_config(), layer_idx=0)


def test_qkv_and_convolution_loaders_shard_each_component_independently():
    layer = make_layer(rank=1, world_size=2)
    global_key_dim = 4
    global_value_dim = 8
    rows = 2 * global_key_dim + global_value_dim
    qkv = torch.arange(rows * 4).reshape(rows, 4).float()
    conv = torch.arange(rows * 2).reshape(rows, 1, 2).float()

    layer._load_qkv(layer.in_proj_qkv.weight, qkv)
    layer._load_conv(layer.conv1d.weight, conv)

    expected_rows = torch.cat(
        (qkv[2:4], qkv[6:8], qkv[12:16]),
        dim=0,
    )
    expected_conv = torch.cat(
        (conv[2:4], conv[6:8], conv[12:16]),
        dim=0,
    )
    torch.testing.assert_close(layer.in_proj_qkv.weight, expected_rows)
    torch.testing.assert_close(layer.conv1d.weight, expected_conv)


def test_decay_and_gated_norm_parameters_remain_fp32_under_bf16_default():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        layer = make_layer()
    finally:
        torch.set_default_dtype(original_dtype)

    assert layer.A_log.dtype == torch.float32
    assert layer.norm.weight.dtype == torch.float32
    assert layer.dt_bias.dtype == torch.bfloat16


def test_gated_delta_layer_chunked_prefill_matches_one_shot():
    torch.manual_seed(4)
    layer = make_layer()
    layer.allocate_state_cache(3, "cpu")
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    hidden_states = torch.randn(5, 4)
    current_context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        state_token_ranges=((0, 5),),
        state_slots=torch.tensor([1], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: current_context

    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        expected = layer(hidden_states)
        expected_recurrent = layer.state_pool.recurrent[:, 1].clone()
        expected_convolution = layer.state_pool.convolution[:, 1].clone()
        layer.state_pool.reset(torch.tensor([1]))
        current_context.cu_seqlens_q = torch.tensor([0, 3], dtype=torch.int32)
        current_context.state_token_ranges = ((0, 3),)
        first = layer(hidden_states[:3])
        current_context.cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
        current_context.state_token_ranges = ((0, 2),)
        current_context.state_reset_mask = torch.tensor([False])
        second = layer(hidden_states[3:])

    torch.testing.assert_close(torch.cat((first, second)), expected)
    torch.testing.assert_close(layer.state_pool.recurrent[:, 1], expected_recurrent)
    torch.testing.assert_close(layer.state_pool.convolution[:, 1], expected_convolution)


def test_tensor_parallel_layers_sum_to_single_rank_reference():
    torch.manual_seed(11)
    full = make_layer(world_size=1)
    ranks = [make_layer(rank=rank, world_size=2) for rank in range(2)]
    sources = {
        "in_proj_qkv.weight": torch.randn(16, 4),
        "in_proj_z.weight": torch.randn(8, 4),
        "in_proj_b.weight": torch.randn(4, 4),
        "in_proj_a.weight": torch.randn(4, 4),
        "conv1d.weight": torch.randn(16, 1, 2),
        "dt_bias": torch.randn(4),
        "A_log": torch.randn(4),
        "norm.weight": torch.randn(2),
        "out_proj.weight": torch.randn(4, 8),
    }

    def load(layer):
        for name, source in sources.items():
            parameter = layer.get_parameter(name)
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.data.copy_(source)
            else:
                loader(parameter, source)
        layer.allocate_state_cache(2, "cpu")

    load(full)
    for rank_layer in ranks:
        load(rank_layer)
    hidden = torch.randn(4, 4)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
        state_token_ranges=((0, 4),),
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(qwen35_gated_delta.dist, "all_reduce", return_value=None),
    ):
        expected = full(hidden)
        actual = sum(rank_layer(hidden) for rank_layer in ranks)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@torch.no_grad()
def test_official_head_layout_matches_across_tp4_and_tp8():
    torch.manual_seed(37)
    layer_config = SimpleNamespace(
        hidden_size=8,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1e-6,
    )
    sources = {
        "in_proj_qkv.weight": torch.randn(64, 8),
        "in_proj_z.weight": torch.randn(32, 8),
        "in_proj_b.weight": torch.randn(32, 8),
        "in_proj_a.weight": torch.randn(32, 8),
        "conv1d.weight": torch.randn(64, 1, 2),
        "dt_bias": torch.randn(32),
        "A_log": torch.randn(32),
        "norm.weight": torch.randn(1),
        "out_proj.weight": torch.randn(8, 32),
    }

    def load(layer):
        for name, source in sources.items():
            parameter = layer.get_parameter(name)
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.copy_(source)
            else:
                loader(parameter, source)
        layer.allocate_state_cache(1, "cpu")

    hidden = torch.randn(3, 8)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        state_token_ranges=((0, 3),),
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context
    full = make_layer(world_size=1, layer_config=layer_config)
    load(full)

    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        expected = full(hidden)
        for tp_size in (4, 8):
            ranks = [
                make_layer(
                    rank=rank,
                    world_size=tp_size,
                    layer_config=layer_config,
                )
                for rank in range(tp_size)
            ]
            for rank_layer in ranks:
                load(rank_layer)
            with patch.object(
                qwen35_gated_delta.dist,
                "all_reduce",
                return_value=None,
            ):
                actual = sum(rank_layer(hidden) for rank_layer in ranks)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_batched_decode_matches_individual_slot_updates():
    torch.manual_seed(19)
    layer = make_layer()
    layer.allocate_state_cache(4, "cpu")
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    hidden = torch.randn(3, 4)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([0, 2, 3], dtype=torch.int32),
        state_reset_mask=torch.tensor([True, True, True]),
        state_token_ranges=(),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        batched = layer(hidden)
        batched_state = layer.state_pool.recurrent.clone()
        layer.state_pool.reset(torch.tensor([0, 2, 3]))
        individual = []
        for row, slot in zip(hidden, (0, 2, 3)):
            context.state_slots = torch.tensor([slot], dtype=torch.int32)
            context.state_reset_mask = torch.tensor([True])
            individual.append(layer(row.unsqueeze(0)))

    torch.testing.assert_close(torch.cat(individual), batched)
    torch.testing.assert_close(layer.state_pool.recurrent, batched_state)


def test_equal_length_prefills_batch_without_changing_results():
    torch.manual_seed(23)
    layer = make_layer()
    layer.allocate_state_cache(4, "cpu")
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    hidden = torch.randn(6, 4)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
        state_token_ranges=((0, 3), (3, 6)),
        state_slots=torch.tensor([0, 2], dtype=torch.int32),
        state_reset_mask=torch.tensor([True, True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        batched = layer(hidden)
        batched_state = layer.state_pool.recurrent.clone()
        layer.state_pool.reset(torch.tensor([0, 2]))
        individual = []
        for start, slot in ((0, 0), (3, 2)):
            context.cu_seqlens_q = torch.tensor([0, 3], dtype=torch.int32)
            context.state_token_ranges = ((0, 3),)
            context.state_slots = torch.tensor([slot], dtype=torch.int32)
            context.state_reset_mask = torch.tensor([True])
            individual.append(layer(hidden[start : start + 3]))

    torch.testing.assert_close(torch.cat(individual), batched)
    torch.testing.assert_close(layer.state_pool.recurrent, batched_state)
