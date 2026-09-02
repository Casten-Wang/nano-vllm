from contextlib import nullcontext
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
gather_prefill_group = qwen35_gated_delta._gather_prefill_group
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


def test_contiguous_prefill_group_reuses_projected_storage():
    projected = torch.arange(40).view(10, 4)
    group = ((2, 5, 0), (5, 8, 1))

    batched = gather_prefill_group(projected, 3, group)

    assert batched.shape == (2, 3, 4)
    assert batched.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()
    torch.testing.assert_close(batched, projected[2:8].view(2, 3, 4))


def test_interleaved_prefill_group_keeps_copy_fallback():
    projected = torch.arange(40).view(10, 4)
    group = ((0, 2, 0), (3, 5, 2))

    batched = gather_prefill_group(projected, 2, group)

    assert batched.shape == (2, 2, 4)
    assert batched.untyped_storage().data_ptr() != projected.untyped_storage().data_ptr()
    torch.testing.assert_close(
        batched,
        torch.stack((projected[0:2], projected[3:5])),
    )


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
    assert actual_state.untyped_storage().nbytes() == (
        actual_state.numel() * actual_state.element_size()
    )


def test_vectorized_causal_convolution_handles_empty_prefill():
    x = torch.empty(2, 0, 5)
    state = torch.randn(2, 5, 4)
    weight = torch.randn(5, 4)

    output, next_state = causal_conv1d_prefill(x, state, weight)

    assert output.shape == x.shape
    assert next_state is state


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
def test_decode_convolution_can_reuse_state_storage(kernel_size):
    torch.manual_seed(43)
    x = torch.randn(3, 7)
    state = torch.randn(3, 7, kernel_size)
    weight = torch.randn(7, kernel_size)
    bias = torch.randn(7)
    expected_output, expected_state = qwen35_gated_delta.causal_conv1d_step(
        x,
        state,
        weight,
        bias,
    )
    reusable_state = state.clone()
    storage = reusable_state.data_ptr()

    with torch.inference_mode():
        actual_output, actual_state = qwen35_gated_delta.causal_conv1d_step(
            x,
            reusable_state,
            weight,
            bias,
            inplace_state=True,
        )

    assert actual_state.data_ptr() == storage
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)


def test_decode_convolution_preserves_state_when_autograd_is_enabled():
    x = torch.randn(2, 5, requires_grad=True)
    state = torch.randn(2, 5, 4)
    original = state.clone()
    weight = torch.randn(5, 4, requires_grad=True)

    _, next_state = qwen35_gated_delta.causal_conv1d_step(
        x,
        state,
        weight,
        inplace_state=True,
    )

    assert next_state.data_ptr() != state.data_ptr()
    torch.testing.assert_close(state, original)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_step_broadcasts_key_heads_without_replication(dtype):
    torch.manual_seed(43)
    query = torch.randn(2, 2, 4, dtype=dtype)
    key = torch.randn(2, 2, 4, dtype=dtype)
    value = torch.randn(2, 6, 3, dtype=dtype)
    decay = -torch.rand(2, 6)
    beta = torch.rand(2, 6, dtype=dtype)
    state = torch.randn(2, 6, 4, 3, dtype=dtype)

    expected, expected_state = recurrent_gated_delta_rule(
        query.repeat_interleave(3, dim=1).unsqueeze(1),
        key.repeat_interleave(3, dim=1).unsqueeze(1),
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


def test_decode_step_reuses_explicit_runtime_state_buffer():
    torch.manual_seed(45)
    query = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(2, 6, 3, dtype=torch.bfloat16)
    decay = -torch.rand(2, 6)
    beta = torch.rand(2, 6, dtype=torch.bfloat16)
    state = torch.randn(2, 6, 4, 3, dtype=torch.float32)
    expected, expected_state = recurrent_gated_delta_step(
        query,
        key,
        value,
        decay,
        beta,
        state.clone(),
    )
    state_storage = state.data_ptr()

    with torch.inference_mode():
        actual, actual_state = recurrent_gated_delta_step(
            query,
            key,
            value,
            decay,
            beta,
            state,
            inplace_state=True,
        )

    assert actual_state.data_ptr() == state_storage
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)


def test_decode_step_reuses_prediction_for_inference_correction():
    torch.manual_seed(451)
    query = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(2, 6, 3, dtype=torch.bfloat16)
    decay = -torch.rand(2, 6)
    beta = torch.rand(2, 6, dtype=torch.bfloat16)
    state = torch.randn(2, 6, 4, 3)
    add_storage = []
    multiply_storage = []
    original_add = torch.Tensor.add_
    original_multiply = torch.Tensor.mul_

    def record_add(tensor, other, *args, **kwargs):
        if tensor.shape == (2, 2, 3, 3):
            add_storage.append(tensor.data_ptr())
        return original_add(tensor, other, *args, **kwargs)

    def record_multiply(tensor, other, *args, **kwargs):
        if tensor.shape == (2, 2, 3, 3):
            multiply_storage.append(tensor.data_ptr())
        return original_multiply(tensor, other, *args, **kwargs)

    with (
        torch.inference_mode(),
        patch.object(torch.Tensor, "add_", record_add),
        patch.object(torch.Tensor, "mul_", record_multiply),
    ):
        recurrent_gated_delta_step(
            query,
            key,
            value,
            decay,
            beta,
            state,
            inplace_state=True,
        )

    assert add_storage == multiply_storage
    assert add_storage
    assert len(set(add_storage)) == 1


def test_decode_step_inplace_request_preserves_autograd_state():
    torch.manual_seed(46)
    query = torch.randn(1, 1, 2, requires_grad=True)
    key = torch.randn_like(query)
    value = torch.randn(1, 1, 2, requires_grad=True)
    decay = -torch.rand(1, 1)
    beta = torch.rand(1, 1)
    state = torch.randn(1, 1, 2, 2, requires_grad=True)
    original = state.detach().clone()

    output, next_state = recurrent_gated_delta_step(
        query,
        key,
        value,
        decay,
        beta,
        state,
        inplace_state=True,
    )
    (output.square().mean() + next_state.square().mean()).backward()

    assert next_state.data_ptr() != state.data_ptr()
    assert torch.equal(state, original)
    assert state.grad is not None


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


@pytest.mark.parametrize("sequence_length", [1, 17, 65])
def test_chunk_rule_broadcasts_key_heads_without_replication(sequence_length):
    torch.manual_seed(47)
    query = torch.randn(2, sequence_length, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn(2, sequence_length, 6, 3)
    decay = -torch.rand(2, sequence_length, 6)
    beta = torch.rand(2, sequence_length, 6)
    state = torch.randn(2, 6, 4, 3)

    expected, expected_state = recurrent_gated_delta_rule(
        query.repeat_interleave(3, dim=2),
        key.repeat_interleave(3, dim=2),
        value,
        decay,
        beta,
        state,
    )
    actual, actual_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        decay,
        beta,
        state,
        chunk_size=16,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-4)


def test_chunk_rule_preserves_autograd_with_separate_output_buffer():
    q, k, v, decay, beta = inputs(5)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    output, state = chunk_gated_delta_rule(
        q,
        k,
        v,
        decay,
        beta,
        chunk_size=4,
    )
    (output.square().mean() + state.square().mean()).backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_chunk_rule_inference_path_does_not_mutate_inputs():
    tensors = inputs(17)
    originals = tuple(tensor.clone() for tensor in tensors)

    with torch.no_grad():
        chunk_gated_delta_rule(*tensors, chunk_size=16)

    for actual, expected in zip(tensors, originals):
        torch.testing.assert_close(actual, expected)


def test_chunk_rule_inference_reuses_solved_values_for_correction():
    tensors = inputs(17)
    expected, expected_state = recurrent_gated_delta_rule(*tensors)
    original_subtract = torch.Tensor.sub_
    correction_shapes = []

    def track_subtract(tensor, other):
        if tensor.ndim == 5:
            correction_shapes.append(tuple(tensor.shape))
        return original_subtract(tensor, other)

    with torch.no_grad(), patch.object(torch.Tensor, "sub_", track_subtract):
        actual, actual_state = chunk_gated_delta_rule(
            *tensors,
            chunk_size=16,
        )

    assert correction_shapes
    assert set(correction_shapes) == {(2, 3, 1, 8, 6)}
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_chunk_rule_inference_reuses_recurrent_state_across_chunks(state_dtype):
    tensors = inputs(17)
    state = torch.randn(2, 3, 4, 6, dtype=state_dtype)
    state_before = state.clone()
    state_updates = []
    original_multiply = torch.Tensor.mul_

    def record_state_multiply(tensor, other, *args, **kwargs):
        if tensor.shape == (2, 3, 1, 4, 6):
            state_updates.append(tensor.data_ptr())
        return original_multiply(tensor, other, *args, **kwargs)

    with (
        torch.inference_mode(),
        patch.object(torch.Tensor, "mul_", record_state_multiply),
    ):
        _, next_state = chunk_gated_delta_rule(
            *tensors,
            initial_state=state,
            chunk_size=4,
        )

    assert len(state_updates) >= 2
    assert len(set(state_updates)) == 1
    assert next_state.data_ptr() == state_updates[0]
    assert next_state.data_ptr() != state.data_ptr()
    torch.testing.assert_close(state, state_before)


def test_grouped_chunk_rule_handles_empty_prefill():
    query = torch.empty(2, 0, 2, 4, dtype=torch.bfloat16)
    value = torch.empty(2, 0, 6, 3, dtype=torch.bfloat16)
    state = torch.randn(2, 6, 4, 3)

    output, next_state = chunk_gated_delta_rule(
        query,
        query,
        value,
        torch.empty(2, 0, 6),
        torch.empty(2, 0, 6, dtype=torch.bfloat16),
        state,
    )

    assert output.shape == value.shape
    assert output.dtype == value.dtype
    torch.testing.assert_close(next_state, state)


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


def test_state_pool_contiguous_get_returns_writable_cache_views():
    pool = Qwen35RecurrentStatePool(1, 4, 2, 3, 4, 5, 2, device="cpu")

    recurrent, convolution = pool.get_contiguous(0, 1, 2)
    recurrent.add_(1)
    convolution.add_(2)

    assert recurrent.data_ptr() == pool.recurrent[0, 1].data_ptr()
    assert convolution.data_ptr() == pool.convolution[0, 1].data_ptr()
    assert torch.count_nonzero(pool.recurrent[0, 0]) == 0
    assert torch.all(pool.recurrent[0, 1:3] == 1)
    assert torch.all(pool.convolution[0, 1:3] == 2)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (3, 2)])
def test_state_pool_rejects_invalid_contiguous_spans(start, count):
    pool = Qwen35RecurrentStatePool(1, 4, 2, 3, 4, 5, 2, device="cpu")

    with pytest.raises(ValueError, match="contiguous state span is out of bounds"):
        pool.get_contiguous(0, start, count)


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


class TrackingSlice:
    def __init__(self, tensor):
        self.tensor = tensor
        self.requests = []

    def get_shape(self):
        return self.tensor.shape

    def __getitem__(self, key):
        self.requests.append(key)
        return self.tensor[key]


def test_gated_delta_safetensors_loaders_only_read_local_tp_slices():
    layer = make_layer(rank=1, world_size=2)
    rows = 2 * layer.global_key_dim + layer.global_value_dim
    qkv = TrackingSlice(torch.arange(rows * 4).reshape(rows, 4).float())
    conv = TrackingSlice(torch.arange(rows * 2).reshape(rows, 1, 2).float())
    column = TrackingSlice(torch.arange(32).reshape(8, 4).float())
    row = TrackingSlice(torch.arange(32).reshape(4, 8).float())

    layer._load_qkv_slice(layer.in_proj_qkv.weight, qkv)
    layer._load_conv_slice(layer.conv1d.weight, conv)
    layer.in_proj_zba.packed_safetensors_loader(
        layer.in_proj_zba.weight,
        column,
        0,
    )
    layer._load_row_slice(layer.out_proj.weight, row)

    torch.testing.assert_close(
        layer.in_proj_qkv.weight,
        torch.cat((qkv.tensor[2:4], qkv.tensor[6:8], qkv.tensor[12:16])),
    )
    torch.testing.assert_close(
        layer.conv1d.weight,
        torch.cat((conv.tensor[2:4], conv.tensor[6:8], conv.tensor[12:16])),
    )
    torch.testing.assert_close(layer.in_proj_zba.weight[:4], column.tensor[4:8])
    torch.testing.assert_close(layer.out_proj.weight, row.tensor[:, 4:8])
    assert len(qkv.requests) == 3
    assert len(conv.requests) == 3
    assert len(column.requests) == 1
    assert len(row.requests) == 1


def test_gated_delta_packed_loaders_do_not_materialize_local_copy():
    layer = make_layer(rank=1, world_size=2)
    rows = 2 * layer.global_key_dim + layer.global_value_dim
    qkv_tensor = torch.arange(rows * 4).reshape(rows, 4).float()
    conv_tensor = torch.arange(rows * 2).reshape(rows, 1, 2).float()

    with patch.object(
        qwen35_gated_delta.torch,
        "cat",
        side_effect=AssertionError("loader must copy each shard directly"),
    ):
        layer._load_qkv(layer.in_proj_qkv.weight, qkv_tensor)
        layer._load_conv(layer.conv1d.weight, conv_tensor)
        qkv = TrackingSlice(qkv_tensor)
        conv = TrackingSlice(conv_tensor)
        layer._load_qkv_slice(layer.in_proj_qkv.weight, qkv)
        layer._load_conv_slice(layer.conv1d.weight, conv)

    torch.testing.assert_close(layer.in_proj_qkv.weight[:2], qkv_tensor[2:4])
    torch.testing.assert_close(layer.in_proj_qkv.weight[2:4], qkv_tensor[6:8])
    torch.testing.assert_close(layer.in_proj_qkv.weight[4:], qkv_tensor[12:16])
    torch.testing.assert_close(layer.conv1d.weight[:2], conv_tensor[2:4])
    torch.testing.assert_close(layer.conv1d.weight[2:4], conv_tensor[6:8])
    torch.testing.assert_close(layer.conv1d.weight[4:], conv_tensor[12:16])
    assert len(qkv.requests) == 3
    assert len(conv.requests) == 3


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_rmsnorm_inference_matches_reference_without_mutation(dtype):
    torch.manual_seed(39)
    norm = qwen35_gated_delta.Qwen35GatedRMSNorm(6)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    hidden = torch.randn(4, 6, dtype=dtype)
    gate = torch.randn(4, 6, dtype=dtype)
    original_hidden = hidden.clone()
    original_gate = gate.clone()
    hidden_float = hidden.float()
    expected = (
        (
            hidden_float
            * torch.rsqrt(
                hidden_float.square().mean(dim=-1, keepdim=True) + norm.eps
            )
        ).to(dtype)
        * norm.weight
        * F.silu(gate.float())
    ).to(dtype)

    with torch.inference_mode():
        actual = norm(hidden, gate)

    assert torch.equal(hidden, original_hidden)
    assert torch.equal(gate, original_gate)
    torch.testing.assert_close(actual, expected)


def test_gated_rmsnorm_autograd_preserves_backward_inputs():
    norm = qwen35_gated_delta.Qwen35GatedRMSNorm(6)
    hidden = torch.randn(4, 6, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn(4, 6, dtype=torch.bfloat16, requires_grad=True)

    norm(hidden, gate).float().square().mean().backward()

    assert hidden.grad is not None
    assert gate.grad is not None
    assert norm.weight.grad is not None


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


@pytest.mark.parametrize(
    ("grad_enabled", "shares_storage"),
    [(False, True), (True, False)],
)
def test_gated_delta_reuses_gate_projection_only_in_inference(
    grad_enabled,
    shares_storage,
):
    torch.manual_seed(7)
    layer = make_layer()
    layer.allocate_state_cache(1, "cpu")
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        state_token_ranges=((0, 3),),
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context
    storage = {}
    z_hook = layer.in_proj_zba.register_forward_hook(
        lambda _module, _inputs, output: storage.update(z=output.data_ptr())
    )
    output_hook = layer.out_proj.register_forward_pre_hook(
        lambda _module, inputs: storage.update(output=inputs[0].data_ptr())
    )
    grad_context = nullcontext() if grad_enabled else torch.inference_mode()

    try:
        with (
            patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
            grad_context,
        ):
            layer(torch.randn(3, 4, requires_grad=grad_enabled))
    finally:
        z_hook.remove()
        output_hook.remove()

    assert (storage["output"] == storage["z"]) is shares_storage


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
            packed_shard = {
                "in_proj_z.weight": 0,
                "in_proj_b.weight": 1,
                "in_proj_a.weight": 2,
            }.get(name)
            if packed_shard is not None:
                layer.in_proj_zba.weight.weight_loader(
                    layer.in_proj_zba.weight,
                    source,
                    packed_shard,
                )
                continue
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
            packed_shard = {
                "in_proj_z.weight": 0,
                "in_proj_b.weight": 1,
                "in_proj_a.weight": 2,
            }.get(name)
            if packed_shard is not None:
                layer.in_proj_zba.weight.weight_loader(
                    layer.in_proj_zba.weight,
                    source,
                    packed_shard,
                )
                continue
            parameter = layer.get_parameter(name)
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.copy_(source)
            else:
                loader(parameter, source)
        layer.allocate_state_cache(1, "cpu")

    hidden = torch.randn(3, 8)
    decode_hidden = torch.randn(1, 8)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=True,
        state_token_ranges=((0, 3),),
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context
    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        for tp_size in (4, 8):
            full = make_layer(world_size=1, layer_config=layer_config)
            load(full)
            expected = full(hidden)
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

            context.is_prefill = False
            context.state_token_ranges = ()
            context.state_reset_mask = torch.tensor([False])
            expected_decode = full(decode_hidden)
            with patch.object(
                qwen35_gated_delta.dist,
                "all_reduce",
                return_value=None,
            ):
                actual_decode = sum(
                    rank_layer(decode_hidden) for rank_layer in ranks
                )
            torch.testing.assert_close(
                actual_decode,
                expected_decode,
                rtol=3e-5,
                atol=3e-5,
            )

            context.is_prefill = True
            context.state_token_ranges = ((0, 3),)
            context.state_reset_mask = torch.tensor([True])


def test_official_state_allocations_match_tp4_and_tp8_memory_budget():
    layer_config = SimpleNamespace(
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    expected = {
        4: (15_728_640, 7_864_320, 491_520),
        8: (7_864_320, 3_932_160, 245_760),
    }
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.bfloat16)
        for tp_size, (
            fp32_recurrent_bytes,
            bf16_recurrent_bytes,
            convolution_bytes,
        ) in expected.items():
            layer = make_layer(
                rank=tp_size - 1,
                world_size=tp_size,
                layer_config=layer_config,
            )
            layer.allocate_state_cache(1, "cpu")
            assert layer.state_pool is not None
            actual_fp32_recurrent = (
                layer.state_pool.recurrent.numel()
                * layer.state_pool.recurrent.element_size()
                * 30
            )
            actual_convolution = (
                layer.state_pool.convolution.numel()
                * layer.state_pool.convolution.element_size()
                * 30
            )
            assert actual_fp32_recurrent == fp32_recurrent_bytes
            assert actual_convolution == convolution_bytes
            layer.allocate_state_cache(
                1,
                "cpu",
                recurrent_dtype=torch.bfloat16,
            )
            assert layer.state_pool is not None
            actual_bf16_recurrent = (
                layer.state_pool.recurrent.numel()
                * layer.state_pool.recurrent.element_size()
                * 30
            )
            assert actual_bf16_recurrent == bf16_recurrent_bytes
    finally:
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)


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


def test_decode_padding_scratch_slot_does_not_change_real_states():
    torch.manual_seed(29)
    layer = make_layer()
    layer.allocate_state_cache(3, "cpu")
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    real_hidden = torch.randn(2, 4)
    padded_hidden = torch.cat((real_hidden, torch.randn(2, 4)))
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        state_reset_mask=None,
        state_token_ranges=(),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with patch.dict(sys.modules, {"nanovllm.utils.context": context_module}):
        padded_output = layer(padded_hidden)
        padded_real_state = layer.state_pool.recurrent[:, :2].clone()
        layer.state_pool.reset(torch.tensor([0, 1, 2]))
        context.state_slots = torch.tensor([0, 1], dtype=torch.int32)
        expected_output = layer(real_hidden)

    torch.testing.assert_close(padded_output[:2], expected_output)
    torch.testing.assert_close(
        padded_real_state,
        layer.state_pool.recurrent[:, :2],
    )


def test_contiguous_decode_updates_state_cache_without_gather_scatter():
    torch.manual_seed(31)
    layer = make_layer()
    layer.allocate_state_cache(4, "cpu")
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    hidden = torch.randn(2, 4)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([1, 2], dtype=torch.int64),
        state_reset_mask=torch.tensor([True, True]),
        state_token_ranges=(),
        decode_state_span=(1, 2),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(
            layer.state_pool,
            "get_contiguous",
            wraps=layer.state_pool.get_contiguous,
        ) as get_contiguous,
        patch.object(layer.state_pool, "update", wraps=layer.state_pool.update) as update,
    ):
        contiguous_output = layer(hidden)

    assert get_contiguous.call_count == 1
    assert update.call_count == 0
    contiguous_state = layer.state_pool.recurrent[:, 1:3].clone()

    layer.state_pool.reset(torch.tensor([1, 2]))
    context.decode_state_span = None
    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
    ):
        indexed_output = layer(hidden)

    torch.testing.assert_close(contiguous_output, indexed_output)
    torch.testing.assert_close(
        contiguous_state,
        layer.state_pool.recurrent[:, 1:3],
    )


def test_compressed_contiguous_decode_avoids_gather_and_matches_indexed_path():
    torch.manual_seed(37)
    layer = make_layer()
    layer.allocate_state_cache(
        3,
        "cpu",
        recurrent_dtype=torch.bfloat16,
    )
    for parameter in layer.parameters():
        parameter.data.normal_(mean=0.0, std=0.2)
    hidden = torch.randn(2, 4)
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([1, 2], dtype=torch.int64),
        state_reset_mask=torch.tensor([True, True]),
        state_token_ranges=(),
        decode_state_span=(1, 2),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(layer.state_pool, "get", wraps=layer.state_pool.get) as get,
        patch.object(layer.state_pool, "update", wraps=layer.state_pool.update) as update,
        patch.object(
            layer.state_pool,
            "get_contiguous",
            wraps=layer.state_pool.get_contiguous,
        ) as get_contiguous,
    ):
        contiguous_output = layer(hidden)

    assert get.call_count == 0
    assert update.call_count == 0
    assert get_contiguous.call_count == 1
    contiguous_state = layer.state_pool.recurrent[:, 1:3].clone()

    layer.state_pool.reset(torch.tensor([1, 2]))
    context.decode_state_span = None
    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
    ):
        indexed_output = layer(hidden)

    torch.testing.assert_close(contiguous_output, indexed_output)
    torch.testing.assert_close(
        contiguous_state,
        layer.state_pool.recurrent[:, 1:3],
    )


def test_gated_delta_reuses_precomputed_reset_slots():
    layer = make_layer()
    layer.allocate_state_cache(1, "cpu")
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([0], dtype=torch.int64),
        state_reset_mask=object(),
        state_reset_slots=torch.tensor([0], dtype=torch.int64),
        state_token_ranges=(),
        state_prefill_groups=(),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(layer.state_pool, "reset", wraps=layer.state_pool.reset) as reset,
    ):
        layer(torch.randn(1, 4))

    assert reset.call_count == 1
    assert reset.call_args.args[0] is context.state_reset_slots


def test_gated_delta_reuses_beta_projection_during_inference():
    layer = make_layer()
    layer.allocate_state_cache(1, "cpu")
    projected_zba = torch.tensor(
        [[0.0] * 8 + [0.0, 1.0, -1.0, 2.0] + [0.0] * 4]
    )
    projected_beta = projected_zba[:, 8:12]
    context = SimpleNamespace(
        is_mixed=False,
        is_prefill=False,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=(),
    )
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: context
    original_decode = layer._decode_batch
    original_softplus = qwen35_gated_delta.F.softplus
    observed = {}

    def capture_beta(mixed_qkv, z, beta, log_decay, slots, state_span=None):
        observed["same_storage"] = beta.data_ptr() == projected_beta.data_ptr()
        observed["decay_reuses_softplus"] = (
            log_decay.data_ptr() == observed["softplus_output"]
        )
        return original_decode(
            mixed_qkv,
            z,
            beta,
            log_decay,
            slots,
            state_span,
        )

    def capture_softplus(value):
        result = original_softplus(value)
        observed["softplus_output"] = result.data_ptr()
        return result

    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(layer.in_proj_zba, "forward", return_value=projected_zba),
        patch.object(qwen35_gated_delta.F, "softplus", side_effect=capture_softplus),
        patch.object(layer, "_decode_batch", side_effect=capture_beta),
    ):
        layer(torch.randn(1, 4))

    assert observed["same_storage"]
    assert observed["decay_reuses_softplus"]
    torch.testing.assert_close(
        projected_beta,
        torch.sigmoid(torch.tensor([[0.0, 1.0, -1.0, 2.0]])),
    )


def test_gated_delta_precomputes_decay_rate_when_weights_load():
    layer = make_layer()
    source = torch.tensor([0.0, 1.0, 2.0, 3.0])

    layer.A_log.weight_loader(layer.A_log, source)

    assert layer._decay_rate is not None
    torch.testing.assert_close(layer._decay_rate, -source.exp())


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
