from unittest.mock import patch

import torch
import pytest

from nanovllm.models.qwen35_gptq import (
    GPTQ_PACK_FACTOR,
    Qwen35GPTQExperts,
    dequantize_gptq_int4,
    resolve_gptq_expert_parameter,
)


def pack_int4(values: torch.Tensor, axis: int) -> torch.Tensor:
    shape = list(values.shape)
    packed_size = shape[axis] // GPTQ_PACK_FACTOR
    shape[axis] = packed_size
    shape.insert(axis + 1, GPTQ_PACK_FACTOR)
    grouped = values.reshape(shape).to(torch.int32)
    shifts_shape = [1] * grouped.ndim
    shifts_shape[axis + 1] = GPTQ_PACK_FACTOR
    shifts = (
        torch.arange(GPTQ_PACK_FACTOR, dtype=torch.int32).mul(4).reshape(shifts_shape)
    )
    return torch.sum(
        torch.bitwise_left_shift(grouped, shifts),
        dim=axis + 1,
        dtype=torch.int32,
    )


def quantize_reference(weight: torch.Tensor, group_size: int):
    # Use exact integer-valued quantization so the dequantized result is exact.
    input_size = weight.shape[1]
    group_ids = torch.arange(input_size, dtype=torch.int32) // group_size
    scales = torch.ones(group_ids[-1].item() + 1, weight.shape[0])
    zeros = torch.full_like(scales, 8, dtype=torch.int32)
    quantized = weight.transpose(0, 1).to(torch.int32) + zeros[group_ids]
    qweight = pack_int4(quantized, 0)
    qzeros = pack_int4(zeros, 1)
    return qweight, qzeros, scales, group_ids


def test_gptq_dequantization_matches_known_weight():
    weight = torch.tensor(
        [
            [-3, -2, -1, 0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0, -1, -2, -3],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [-4, -3, -2, -1, 0, 1, 2, 3],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [2, 0, -2, 2, 0, -2, 2, 0],
            [3, -1, 3, -1, 3, -1, 3, -1],
            [-2, 4, -2, 4, -2, 4, -2, 4],
        ],
        dtype=torch.float32,
    )
    qweight, qzeros, scales, g_idx = quantize_reference(weight, 4)

    actual = dequantize_gptq_int4(
        qweight,
        qzeros,
        scales,
        g_idx,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(actual, weight)


def test_official_expert_name_maps_to_stacked_parameter():
    assert resolve_gptq_expert_parameter(
        "model.layers.12.mlp.experts.37.gate_proj.scales"
    ) == ("model.layers.12.mlp.experts.gate_scales", 37)
    assert (
        resolve_gptq_expert_parameter(
            "model.layers.12.mlp.shared_expert.gate_proj.scales"
        )
        is None
    )


def make_experts(rank=0, world_size=1):
    with (
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=rank),
    ):
        return Qwen35GPTQExperts(
            hidden_size=8,
            intermediate_size=8,
            num_experts=2,
            group_size=8,
        )


def load_projection(experts, projection, expert_id, weight):
    packed = quantize_reference(weight, experts.group_size)
    for component, source in zip(experts.checkpoint_components, packed):
        parameter = getattr(experts, f"{projection}_{component}")
        parameter.packed_safetensors_loader(parameter, source, expert_id)


def test_reference_experts_match_dequantized_mixture():
    experts = make_experts()
    gate = torch.eye(8)
    up = torch.eye(8) * 2
    down = torch.eye(8) * 3
    for expert_id in range(2):
        load_projection(experts, "gate", expert_id, gate)
        load_projection(experts, "up", expert_id, up)
        load_projection(experts, "down", expert_id, down)
    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    topk_ids = torch.tensor([[0, 1], [1, 0]])
    topk_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6]])

    actual = experts(hidden, topk_ids, topk_weights)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(hidden) * (hidden * 2),
        down,
    )

    torch.testing.assert_close(actual, expected)


def test_tp8_down_loader_normalizes_partial_group_indices():
    with (
        patch("torch.distributed.get_world_size", return_value=8),
        patch("torch.distributed.get_rank", return_value=3),
    ):
        experts = Qwen35GPTQExperts(
            hidden_size=8,
            intermediate_size=64,
            num_experts=1,
            group_size=128,
        )
    weight = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64) % 7 - 3
    qweight, qzeros, scales, g_idx = quantize_reference(weight, 128)

    for component, source in zip(
        experts.checkpoint_components,
        (qweight, qzeros, scales, g_idx),
    ):
        parameter = getattr(experts, f"down_{component}")
        parameter.packed_safetensors_loader(parameter, source, 0)

    assert experts.down_g_idx[0].tolist() == [0] * 8
    actual = experts._weight("down", 0, torch.float32)
    torch.testing.assert_close(actual, weight[:, 24:32])


def test_triton_backend_never_silently_runs_on_cpu():
    experts = make_experts()
    experts.backend = "triton"

    with pytest.raises(ValueError, match="requires CUDA"):
        experts._linear(torch.ones(1, 8), "gate", 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_w4a16_matches_reference_dequantization():
    from nanovllm.layers.gptq_w4a16 import gptq_w4a16_linear

    torch.manual_seed(71)
    weight = torch.randint(-7, 8, (64, 128), dtype=torch.int32).float()
    qweight, qzeros, scales, g_idx = quantize_reference(weight, 32)
    inputs = torch.randn(5, 128, dtype=torch.float16, device="cuda")

    actual = gptq_w4a16_linear(
        inputs,
        qweight.cuda(),
        qzeros.cuda(),
        scales.half().cuda(),
        g_idx.cuda(),
    )
    expected = torch.nn.functional.linear(inputs, weight.half().cuda())

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
