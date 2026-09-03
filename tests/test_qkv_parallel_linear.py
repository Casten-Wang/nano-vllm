from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

from nanovllm.models.qwen35_fp8 import dequantize_fp8_block_weight


LINEAR_PATH = Path(__file__).parents[1] / "nanovllm" / "layers" / "linear.py"
SPEC = spec_from_file_location("linear_under_test", LINEAR_PATH)
assert SPEC is not None and SPEC.loader is not None
LINEAR = module_from_spec(SPEC)
sys.modules[SPEC.name] = LINEAR
SPEC.loader.exec_module(LINEAR)


def make_layer(rank: int, world_size: int = 4):
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=world_size),
        patch.object(LINEAR.dist, "get_rank", return_value=rank),
    ):
        return LINEAR.QKVParallelLinear(
            hidden_size=3,
            head_size=2,
            total_num_heads=16,
            total_num_kv_heads=2,
        )


class TrackingSlice:
    def __init__(self, tensor):
        self.tensor = tensor
        self.requests = []

    def get_shape(self):
        return self.tensor.shape

    def __getitem__(self, key):
        self.requests.append(key)
        return self.tensor[key]


def test_kv_heads_are_replicated_when_tp_exceeds_kv_heads():
    layers = [make_layer(rank) for rank in range(4)]

    assert all(layer.num_heads == 4 for layer in layers)
    assert all(layer.num_kv_heads == 1 for layer in layers)
    assert all(layer.num_kv_head_replicas == 2 for layer in layers)
    assert all(layer.weight.shape == (12, 3) for layer in layers)


def test_replicated_kv_weight_loader_uses_shared_source_shards():
    loaded_k = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    loaded_v = loaded_k + 100

    layers = [make_layer(rank) for rank in range(4)]
    for layer in layers:
        layer.weight_loader(layer.weight, loaded_k, "k")
        layer.weight_loader(layer.weight, loaded_v, "v")

    assert torch.equal(layers[0].weight[8:10], loaded_k[0:2])
    assert torch.equal(layers[1].weight[8:10], loaded_k[0:2])
    assert torch.equal(layers[2].weight[8:10], loaded_k[2:4])
    assert torch.equal(layers[3].weight[8:10], loaded_k[2:4])
    assert torch.equal(layers[0].weight[10:12], loaded_v[0:2])
    assert torch.equal(layers[1].weight[10:12], loaded_v[0:2])
    assert torch.equal(layers[2].weight[10:12], loaded_v[2:4])
    assert torch.equal(layers[3].weight[10:12], loaded_v[2:4])


def test_query_weight_loader_remains_rank_sharded():
    loaded_q = torch.arange(96, dtype=torch.float32).reshape(32, 3)
    layers = [make_layer(rank) for rank in range(4)]

    for layer in layers:
        layer.weight_loader(layer.weight, loaded_q, "q")

    for rank, layer in enumerate(layers):
        assert torch.equal(layer.weight[:8], loaded_q[rank * 8 : (rank + 1) * 8])


def test_query_head_size_can_differ_from_kv_head_size():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=4),
        patch.object(LINEAR.dist, "get_rank", return_value=3),
    ):
        layer = LINEAR.QKVParallelLinear(
            hidden_size=3,
            head_size=2,
            q_head_size=4,
            total_num_heads=16,
            total_num_kv_heads=2,
        )
    query = torch.arange(192, dtype=torch.float32).reshape(64, 3)
    key = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    value = key + 100

    layer.weight_loader(layer.weight, query, "q")
    layer.weight_loader(layer.weight, key, "k")
    layer.weight_loader(layer.weight, value, "v")

    assert layer.weight.shape == (20, 3)
    assert torch.equal(layer.weight[:16], query[48:64])
    assert torch.equal(layer.weight[16:18], key[2:4])
    assert torch.equal(layer.weight[18:20], value[2:4])


def test_query_head_size_must_be_positive():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=1),
        patch.object(LINEAR.dist, "get_rank", return_value=0),
        pytest.raises(ValueError, match="q_head_size must be positive"),
    ):
        LINEAR.QKVParallelLinear(3, 2, 4, q_head_size=0)


def test_standalone_kv_projection_replicates_heads_and_bias():
    loaded_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    loaded_bias = torch.arange(4, dtype=torch.float32)
    layers = []
    for rank in range(4):
        with (
            patch.object(LINEAR.dist, "get_world_size", return_value=4),
            patch.object(LINEAR.dist, "get_rank", return_value=rank),
        ):
            layer = LINEAR.KVParallelLinear(3, 2, 2, bias=True)
        layer.weight_loader(layer.weight, loaded_weight)
        layer.weight_loader(layer.bias, loaded_bias)
        layers.append(layer)

    assert torch.equal(layers[0].weight, loaded_weight[:2])
    assert torch.equal(layers[1].weight, loaded_weight[:2])
    assert torch.equal(layers[2].weight, loaded_weight[2:])
    assert torch.equal(layers[3].weight, loaded_weight[2:])
    assert torch.equal(layers[0].bias, loaded_bias[:2])
    assert torch.equal(layers[3].bias, loaded_bias[2:])


def test_lazy_kv_loader_reads_shared_source_shards_for_replica_ranks():
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    layers = []
    for rank in range(4):
        with (
            patch.object(LINEAR.dist, "get_world_size", return_value=4),
            patch.object(LINEAR.dist, "get_rank", return_value=rank),
        ):
            layers.append(LINEAR.KVParallelLinear(3, 2, 2))
    slices = [TrackingSlice(source) for _ in layers]

    for layer, loaded_slice in zip(layers, slices):
        layer.safetensors_loader(layer.weight, loaded_slice)

    assert torch.equal(layers[0].weight, source[:2])
    assert torch.equal(layers[1].weight, source[:2])
    assert torch.equal(layers[2].weight, source[2:])
    assert torch.equal(layers[3].weight, source[2:])
    assert slices[0].requests == [(slice(0, 2), slice(None))]
    assert slices[1].requests == [(slice(0, 2), slice(None))]
    assert slices[2].requests == [(slice(2, 4), slice(None))]
    assert slices[3].requests == [(slice(2, 4), slice(None))]


def test_lazy_column_and_row_loaders_read_only_local_tp_slices():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=2),
        patch.object(LINEAR.dist, "get_rank", return_value=1),
    ):
        column = LINEAR.ColumnParallelLinear(3, 8)
        row = LINEAR.RowParallelLinear(8, 3)
    column_source = TrackingSlice(torch.arange(24).reshape(8, 3).float())
    row_source = TrackingSlice(torch.arange(24).reshape(3, 8).float())

    column.safetensors_loader(column.weight, column_source)
    row.safetensors_loader(row.weight, row_source)

    assert torch.equal(column.weight, column_source.tensor[4:8])
    assert torch.equal(row.weight, row_source.tensor[:, 4:8])
    assert column_source.requests == [(slice(4, 8), slice(None))]
    assert row_source.requests == [(slice(None), slice(4, 8))]


def test_lazy_merged_column_loader_reads_each_local_packed_shard():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=2),
        patch.object(LINEAR.dist, "get_rank", return_value=1),
    ):
        layer = LINEAR.MergedColumnParallelLinear(3, [8, 4])
    gate = TrackingSlice(torch.arange(24).reshape(8, 3).float())
    up = TrackingSlice(torch.arange(12).reshape(4, 3).float() + 100)

    layer.packed_safetensors_loader(layer.weight, gate, 0)
    layer.packed_safetensors_loader(layer.weight, up, 1)

    assert torch.equal(layer.weight[:4], gate.tensor[4:8])
    assert torch.equal(layer.weight[4:], up.tensor[2:4])
    assert gate.requests == [(slice(4, 8), slice(None))]
    assert up.requests == [(slice(2, 4), slice(None))]


def test_lazy_qkv_loader_preserves_replicated_kv_source_selection():
    layer = make_layer(rank=1)
    query = TrackingSlice(torch.arange(96).reshape(32, 3).float())
    key = TrackingSlice(torch.arange(12).reshape(4, 3).float())
    value = TrackingSlice(torch.arange(12).reshape(4, 3).float() + 100)

    layer.packed_safetensors_loader(layer.weight, query, "q")
    layer.packed_safetensors_loader(layer.weight, key, "k")
    layer.packed_safetensors_loader(layer.weight, value, "v")

    assert torch.equal(layer.weight[:8], query.tensor[8:16])
    assert torch.equal(layer.weight[8:10], key.tensor[:2])
    assert torch.equal(layer.weight[10:12], value.tensor[:2])
    assert query.requests == [(slice(8, 16), slice(None))]
    assert key.requests == [(slice(0, 2), slice(None))]
    assert value.requests == [(slice(0, 2), slice(None))]


def test_fp8_column_and_row_loaders_dequantize_only_local_non_aligned_shards():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=2),
        patch.object(LINEAR.dist, "get_rank", return_value=1),
    ):
        column = LINEAR.ColumnParallelLinear(5, 6)
        row = LINEAR.RowParallelLinear(6, 5)
    column_weight = torch.arange(1, 31).reshape(6, 5).to(torch.float8_e4m3fn)
    row_weight = torch.arange(1, 31).reshape(5, 6).to(torch.float8_e4m3fn)
    column_scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    row_scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    column.fp8_safetensors_loader(
        column.weight,
        TrackingSlice(column_weight),
        TrackingSlice(column_scale),
        (4, 4),
    )
    row.fp8_safetensors_loader(
        row.weight,
        TrackingSlice(row_weight),
        TrackingSlice(row_scale),
        (4, 4),
    )

    expected_column = dequantize_fp8_block_weight(
        column_weight,
        column_scale,
        (4, 4),
        output_dtype=column.weight.dtype,
    )[3:6]
    expected_row = dequantize_fp8_block_weight(
        row_weight,
        row_scale,
        (4, 4),
        output_dtype=row.weight.dtype,
    )[:, 3:6]
    torch.testing.assert_close(column.weight, expected_column)
    torch.testing.assert_close(row.weight, expected_row)


def test_fp8_qkv_loader_preserves_replicated_kv_and_block_offsets():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=2),
        patch.object(LINEAR.dist, "get_rank", return_value=1),
    ):
        layer = LINEAR.QKVParallelLinear(
            hidden_size=5,
            head_size=2,
            total_num_heads=4,
            total_num_kv_heads=2,
        )
    query = torch.arange(1, 41).reshape(8, 5).to(torch.float8_e4m3fn)
    key = torch.arange(1, 21).reshape(4, 5).to(torch.float8_e4m3fn)
    value = (torch.arange(1, 21).reshape(4, 5) + 40).to(torch.float8_e4m3fn)
    query_scale = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
    kv_scale = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)

    layer.fp8_packed_safetensors_loader(
        layer.weight,
        TrackingSlice(query),
        TrackingSlice(query_scale),
        "q",
        (3, 3),
    )
    layer.fp8_packed_safetensors_loader(
        layer.weight,
        TrackingSlice(key),
        TrackingSlice(kv_scale),
        "k",
        (3, 3),
    )
    layer.fp8_packed_safetensors_loader(
        layer.weight,
        TrackingSlice(value),
        TrackingSlice(kv_scale),
        "v",
        (3, 3),
    )

    expected_query = dequantize_fp8_block_weight(
        query,
        query_scale,
        (3, 3),
        output_dtype=layer.weight.dtype,
    )[4:8]
    expected_key = dequantize_fp8_block_weight(
        key,
        kv_scale,
        (3, 3),
        output_dtype=layer.weight.dtype,
    )[2:4]
    expected_value = dequantize_fp8_block_weight(
        value,
        kv_scale,
        (3, 3),
        output_dtype=layer.weight.dtype,
    )[2:4]
    torch.testing.assert_close(layer.weight[:4], expected_query)
    torch.testing.assert_close(layer.weight[4:6], expected_key)
    torch.testing.assert_close(layer.weight[6:8], expected_value)
