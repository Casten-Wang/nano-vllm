from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.models.qwen35_fp8 import (
    dequantize_fp8_block_weight,
    dequantize_fp8_block_weight_slice,
    resolve_fp8_expert_parameter,
)
from nanovllm.utils.loader import load_model

ROOT = Path(__file__).parents[1]
MOE_SPEC = spec_from_file_location(
    "qwen35_fp8_moe_under_test",
    ROOT / "nanovllm/models/qwen35_moe.py",
)
assert MOE_SPEC is not None and MOE_SPEC.loader is not None
MOE_MODULE = module_from_spec(MOE_SPEC)
MOE_SPEC.loader.exec_module(MOE_MODULE)
Qwen35Experts = MOE_MODULE.Qwen35Experts
ResidentFP8WeightBufferPool = MOE_MODULE.ResidentFP8WeightBufferPool


def test_block_fp8_dequantization_handles_partial_edge_blocks():
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    result = dequantize_fp8_block_weight(
        weight,
        scale,
        (2, 2),
        output_dtype=torch.float32,
    )

    expected = torch.tensor([[2.0, 4.0, 9.0], [8.0, 10.0, 18.0], [28.0, 32.0, 45.0]])
    torch.testing.assert_close(result, expected)


def test_block_fp8_dequantization_does_not_expand_a_full_scale_tensor(monkeypatch):
    weight = torch.ones((5, 7), dtype=torch.float8_e4m3fn)
    scale = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    allocations = []
    repeat_interleave = torch.repeat_interleave

    def record_repeat_interleave(input, repeats, *args, **kwargs):
        result = repeat_interleave(input, repeats, *args, **kwargs)
        allocations.append(result.numel())
        return result

    monkeypatch.setattr(torch, "repeat_interleave", record_repeat_interleave)
    result = dequantize_fp8_block_weight(
        weight,
        scale,
        (3, 3),
        output_dtype=torch.float32,
    )

    expected_scale = torch.tensor(
        [
            [1, 1, 1, 2, 2, 2, 3],
            [1, 1, 1, 2, 2, 2, 3],
            [1, 1, 1, 2, 2, 2, 3],
            [4, 4, 4, 5, 5, 5, 6],
            [4, 4, 4, 5, 5, 5, 6],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result, expected_scale)
    assert allocations
    assert max(allocations) <= 9
    assert max(allocations) < weight.numel()


def test_block_fp8_dequantization_broadcasts_exact_blocks_without_expansion(
    monkeypatch,
):
    weight = torch.ones((4, 6), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    def reject_repeat_interleave(*_args, **_kwargs):
        raise AssertionError("exact block grids must not expand scales")

    monkeypatch.setattr(torch, "repeat_interleave", reject_repeat_interleave)
    result = dequantize_fp8_block_weight(
        weight,
        scale,
        (2, 3),
        output_dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [2, 2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5],
            [4, 4, 4, 5, 5, 5],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result, expected)


def test_block_fp8_slice_dequantization_preserves_global_block_offsets():
    weight = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7).to(
        torch.float8_e4m3fn
    )
    scale = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    full = dequantize_fp8_block_weight(
        weight,
        scale,
        (3, 3),
        output_dtype=torch.float32,
    )

    shard = dequantize_fp8_block_weight_slice(
        weight,
        scale,
        (3, 3),
        (2, 5),
        (2, 7),
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(shard, full[2:5, 2:7])


def test_block_fp8_slice_dequantization_writes_directly_to_output():
    weight = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7).to(
        torch.float8_e4m3fn
    )
    scale = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    expected = dequantize_fp8_block_weight_slice(
        weight,
        scale,
        (3, 3),
        (2, 5),
        (2, 7),
        output_dtype=torch.float32,
    )
    output = torch.empty_like(expected)

    actual = dequantize_fp8_block_weight_slice(
        weight,
        scale,
        (3, 3),
        (2, 5),
        (2, 7),
        output_dtype=output.dtype,
        out=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_block_fp8_dequantization_rejects_incompatible_output():
    weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1)

    with pytest.raises(ValueError, match="output shape"):
        dequantize_fp8_block_weight(
            weight,
            scale,
            (2, 2),
            output_dtype=torch.float32,
            out=torch.empty(1, 4),
        )
    with pytest.raises(ValueError, match="output dtype"):
        dequantize_fp8_block_weight(
            weight,
            scale,
            (2, 2),
            output_dtype=torch.float32,
            out=torch.empty(2, 2, dtype=torch.bfloat16),
        )


def test_block_fp8_dequantization_rejects_wrong_scale_grid():
    with pytest.raises(ValueError, match="scale shape"):
        dequantize_fp8_block_weight(
            torch.ones(3, 3, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1),
            (2, 2),
            output_dtype=torch.float32,
        )


@pytest.mark.parametrize("block_size", [(0, 2), (2, 0), (-1, 2)])
def test_block_fp8_dequantization_rejects_nonpositive_blocks(block_size):
    with pytest.raises(ValueError, match="positive"):
        dequantize_fp8_block_weight(
            torch.ones(3, 3, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2),
            block_size,
            output_dtype=torch.float32,
        )


def test_fp8_expert_names_map_to_stacked_parameters():
    assert resolve_fp8_expert_parameter(
        "model.layers.2.mlp.experts.17.up_proj.weight"
    ) == ("model.layers.2.mlp.experts.gate_up_proj", (17, "up"))
    assert (
        resolve_fp8_expert_parameter("model.layers.2.self_attn.q_proj.weight") is None
    )


def test_fp8_expert_loader_owns_only_local_tp4_blocks():
    with (
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.get_rank", return_value=2),
    ):
        experts = Qwen35Experts(
            hidden_size=4,
            intermediate_size=512,
            num_experts=2,
            checkpoint_format="fp8_block",
        )
    gate = torch.arange(512 * 4, dtype=torch.float32).reshape(512, 4)
    up = gate + 10_000
    down = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512)

    experts._load_gate_up(experts.gate_up_proj, gate, (1, "gate"))
    experts._load_gate_up(experts.gate_up_proj, up, (1, "up"))
    experts._load_down(experts.down_proj, down, (1, "down"))

    torch.testing.assert_close(experts.gate_up_proj[1, :128], gate[256:384])
    torch.testing.assert_close(experts.gate_up_proj[1, 128:], up[256:384])
    torch.testing.assert_close(experts.down_proj[1], down[:, 256:384])
    assert len(experts.gate_up_proj.required_checkpoint_shards) == 4
    assert len(experts.down_proj.required_checkpoint_shards) == 2


def test_resident_fp8_experts_keep_quantized_storage_and_match_reference():
    with (
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        experts = Qwen35Experts(
            hidden_size=4,
            intermediate_size=4,
            num_experts=1,
            checkpoint_format="fp8_block",
            fp8_block_size=(2, 2),
            resident_fp8=True,
        )
    gate = torch.eye(4, dtype=torch.float8_e4m3fn)
    up = (torch.eye(4) * 2).to(torch.float8_e4m3fn)
    down = torch.eye(4, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 2)
    experts._load_gate_up_fp8_slice(
        experts.gate_up_proj, gate, scale, (0, "gate"), (2, 2)
    )
    experts._load_gate_up_fp8_slice(
        experts.gate_up_proj, up, scale, (0, "up"), (2, 2)
    )
    experts._load_down_fp8_slice(
        experts.down_proj, down, scale, (0, "down"), (2, 2)
    )
    pool = ResidentFP8WeightBufferPool()
    experts.resident_weight_buffer_pool = pool
    hidden = torch.tensor([[0.5, 1.0, -0.5, 2.0]])
    actual = experts._forward_sorted(
        hidden,
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1),
    )
    expected = torch.nn.functional.silu(hidden) * (2.0 * hidden)

    assert experts.gate_up_proj.dtype == torch.float8_e4m3fn
    assert experts.down_proj.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual, expected)
    assert experts.resident_fp8_storage_stats() == {
        "weight_bytes": (8 * 4 + 4 * 4),
        "scale_bytes": (1 * 2 * 2 * 2 + 1 * 2 * 2) * 4,
        "total_bytes": (8 * 4 + 4 * 4) + (1 * 2 * 2 * 2 + 1 * 2 * 2) * 4,
    }
    assert pool.storage_stats() == {
        "storage_bytes": 8 * 4 * hidden.element_size(),
        "allocation_count": 1,
        "reuse_count": 1,
    }


def test_resident_fp8_experts_preserve_tp8_partial_block_scales():
    with (
        patch("torch.distributed.get_world_size", return_value=8),
        patch("torch.distributed.get_rank", return_value=3),
    ):
        experts = Qwen35Experts(
            hidden_size=8,
            intermediate_size=64,
            num_experts=1,
            checkpoint_format="fp8_block",
            fp8_block_size=(128, 128),
            resident_fp8=True,
        )
    down = torch.ones(8, 64, dtype=torch.float8_e4m3fn)
    scale = torch.full((1, 1), 3.0)

    experts._load_down_fp8_slice(
        experts.down_proj,
        down,
        scale,
        (0, "down"),
        (128, 128),
    )

    assert experts.down_scale.shape == (1, 1, 1)
    torch.testing.assert_close(
        experts._resident_weight("down", 0, torch.float32),
        torch.full((8, 8), 3.0),
    )


class TinyFP8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.checkpoint_quantization_spec = SimpleNamespace(
            format="fp8_block",
            weight_block_size=(2, 2),
        )


def write_tiny_fp8_checkpoint(path: Path, *, include_scale: bool = True):
    tensors = {
        "linear.weight": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float8_e4m3fn,
        )
    }
    if include_scale:
        tensors["linear.weight_scale_inv"] = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    save_file(tensors, path / "model.safetensors")


def test_model_loader_pairs_fp8_weight_and_scale(tmp_path):
    model = TinyFP8Model()
    write_tiny_fp8_checkpoint(tmp_path)

    load_model(model, str(tmp_path))

    expected = torch.tensor([[2.0, 4.0, 9.0], [8.0, 10.0, 18.0], [28.0, 32.0, 45.0]])
    torch.testing.assert_close(model.linear.weight, expected)


def test_model_loader_rejects_fp8_weight_without_scale(tmp_path):
    model = TinyFP8Model()
    write_tiny_fp8_checkpoint(tmp_path, include_scale=False)

    with pytest.raises(RuntimeError, match="missing scale"):
        load_model(model, str(tmp_path))


class TinyFP8ExpertsModel(nn.Module):
    strict_weight_loading = True

    def __init__(
        self,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        block_size: tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        with (
            patch("torch.distributed.get_world_size", return_value=tp_size),
            patch("torch.distributed.get_rank", return_value=tp_rank),
        ):
            layer.mlp.experts = Qwen35Experts(
                hidden_size=4,
                intermediate_size=4,
                num_experts=2,
                checkpoint_format="fp8_block",
            )
        self.model.layers = nn.ModuleList([layer])
        self.checkpoint_quantization_spec = SimpleNamespace(
            format="fp8_block",
            weight_block_size=block_size,
        )

    def resolve_checkpoint_parameter(self, weight_name):
        resolved = resolve_fp8_expert_parameter(weight_name)
        if resolved is None:
            return None
        target, shard_id = resolved
        self.get_parameter(target)
        return target, shard_id


def test_model_loader_assembles_per_expert_fp8_weights(tmp_path):
    model = TinyFP8ExpertsModel()
    tensors = {}
    expected = {}
    for expert_id in range(2):
        for projection in ("gate", "up", "down"):
            name = f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"
            value = torch.full(
                (4, 4),
                expert_id * 10 + {"gate": 1, "up": 2, "down": 3}[projection],
                dtype=torch.float8_e4m3fn,
            )
            scale = torch.full((2, 2), 2.0)
            tensors[name] = value
            tensors[f"{name}_scale_inv"] = scale
            expected[(expert_id, projection)] = value.float() * 2
    save_file(tensors, tmp_path / "model.safetensors")

    load_model(model, str(tmp_path))

    experts = model.model.layers[0].mlp.experts
    for expert_id in range(2):
        torch.testing.assert_close(
            experts.gate_up_proj[expert_id, :4],
            expected[(expert_id, "gate")],
        )
        torch.testing.assert_close(
            experts.gate_up_proj[expert_id, 4:],
            expected[(expert_id, "up")],
        )
        torch.testing.assert_close(
            experts.down_proj[expert_id],
            expected[(expert_id, "down")],
        )


def test_fp8_expert_loader_dequantizes_only_non_aligned_tp_shards(tmp_path):
    model = TinyFP8ExpertsModel(tp_rank=1, tp_size=2, block_size=(3, 3))
    tensors = {}
    expected = {}
    for expert_id in range(2):
        for projection in ("gate", "up", "down"):
            name = f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"
            value = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
            value = (value + expert_id * 20).to(torch.float8_e4m3fn)
            scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            tensors[name] = value
            tensors[f"{name}_scale_inv"] = scale
            expected[(expert_id, projection)] = dequantize_fp8_block_weight(
                value,
                scale,
                (3, 3),
                output_dtype=torch.float32,
            )
    save_file(tensors, tmp_path / "model.safetensors")

    with patch(
        "nanovllm.utils.loader.dequantize_fp8_block_weight",
        side_effect=AssertionError("expert loader must not dequantize full weights"),
    ):
        load_model(model, str(tmp_path))

    experts = model.model.layers[0].mlp.experts
    for expert_id in range(2):
        torch.testing.assert_close(
            experts.gate_up_proj[expert_id, :2],
            expected[(expert_id, "gate")][2:4],
        )
        torch.testing.assert_close(
            experts.gate_up_proj[expert_id, 2:],
            expected[(expert_id, "up")][2:4],
        )
        torch.testing.assert_close(
            experts.down_proj[expert_id],
            expected[(expert_id, "down")][:, 2:4],
        )


class TinyFP8TensorParallelModel(nn.Module):
    strict_weight_loading = True

    def __init__(self):
        super().__init__()
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
        ):
            self.column = ColumnParallelLinear(5, 6)
            self.row = RowParallelLinear(6, 5)
        self.checkpoint_quantization_spec = SimpleNamespace(
            format="fp8_block",
            weight_block_size=(4, 4),
        )


def test_model_loader_uses_tp_local_fp8_dense_slices(tmp_path):
    model = TinyFP8TensorParallelModel()
    column = torch.arange(1, 31).reshape(6, 5).to(torch.float8_e4m3fn)
    row = torch.arange(1, 31).reshape(5, 6).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file(
        {
            "column.weight": column,
            "column.weight_scale_inv": scale,
            "row.weight": row,
            "row.weight_scale_inv": scale.clone(),
        },
        tmp_path / "model.safetensors",
    )
    expected_column = dequantize_fp8_block_weight(
        column,
        scale,
        (4, 4),
        output_dtype=model.column.weight.dtype,
    )[3:6]
    expected_row = dequantize_fp8_block_weight(
        row,
        scale,
        (4, 4),
        output_dtype=model.row.weight.dtype,
    )[:, 3:6]

    with patch(
        "nanovllm.utils.loader.dequantize_fp8_block_weight",
        side_effect=AssertionError("TP weights must not be fully dequantized"),
    ):
        load_model(model, str(tmp_path))

    torch.testing.assert_close(model.column.weight, expected_column)
    torch.testing.assert_close(model.row.weight, expected_row)
