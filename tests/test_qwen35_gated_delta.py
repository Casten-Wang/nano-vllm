from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

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
recurrent_gated_delta_rule = qwen35_gated_delta.recurrent_gated_delta_rule


def inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 5, 3, 4, generator=generator)
    k = torch.randn(2, 5, 3, 4, generator=generator)
    v = torch.randn(2, 5, 3, 6, generator=generator)
    decay = -torch.rand(2, 5, 3, generator=generator)
    beta = torch.rand(2, 5, 3, generator=generator)
    return q, k, v, decay, beta


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


def make_layer(rank=0, world_size=1):
    with (
        patch.object(qwen35_gated_delta.dist, "get_world_size", return_value=world_size),
        patch.object(qwen35_gated_delta.dist, "get_rank", return_value=rank),
    ):
        return Qwen35GatedDeltaNet(tiny_config(), layer_idx=0)


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
        first = layer(hidden_states[:3])
        current_context.cu_seqlens_q = torch.tensor([0, 2], dtype=torch.int32)
        current_context.state_reset_mask = torch.tensor([False])
        second = layer(hidden_states[3:])

    torch.testing.assert_close(torch.cat((first, second)), expected)
    torch.testing.assert_close(layer.state_pool.recurrent[:, 1], expected_recurrent)
    torch.testing.assert_close(layer.state_pool.convolution[:, 1], expected_convolution)
