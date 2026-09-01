from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

import torch


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


def test_qkv_safetensors_loader_reads_only_rank_source_rows():
    layer = make_layer(rank=3)
    loaded_q = TrackingSlice(torch.arange(96).reshape(32, 3).float())
    loaded_k = TrackingSlice(torch.arange(12).reshape(4, 3).float())

    layer.safetensors_loader(layer.weight, loaded_q, "q")
    layer.safetensors_loader(layer.weight, loaded_k, "k")

    assert torch.equal(layer.weight[:8], loaded_q.tensor[24:32])
    assert torch.equal(layer.weight[8:10], loaded_k.tensor[2:4])
    assert loaded_q.requests == [(slice(24, 32), slice(None))]
    assert loaded_k.requests == [(slice(2, 4), slice(None))]


def test_merged_and_row_parallel_safetensors_loaders_slice_by_rank():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=4),
        patch.object(LINEAR.dist, "get_rank", return_value=2),
    ):
        merged = LINEAR.MergedColumnParallelLinear(3, [8, 12])
        row = LINEAR.RowParallelLinear(8, 3)
    first = TrackingSlice(torch.arange(24).reshape(8, 3).float())
    second = TrackingSlice(torch.arange(36).reshape(12, 3).float())
    row_source = TrackingSlice(torch.arange(24).reshape(3, 8).float())

    merged.safetensors_loader(merged.weight, first, 0)
    merged.safetensors_loader(merged.weight, second, 1)
    row.safetensors_loader(row.weight, row_source)

    assert torch.equal(merged.weight[:2], first.tensor[4:6])
    assert torch.equal(merged.weight[2:], second.tensor[6:9])
    assert torch.equal(row.weight, row_source.tensor[:, 4:6])
    assert first.requests == [(slice(4, 6), slice(None))]
    assert second.requests == [(slice(6, 9), slice(None))]
    assert row_source.requests == [(slice(None), slice(4, 6))]


def test_column_and_replicated_kv_safetensors_loaders_slice_by_rank():
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=4),
        patch.object(LINEAR.dist, "get_rank", return_value=1),
    ):
        column = LINEAR.ColumnParallelLinear(3, 8, bias=True)
        kv = LINEAR.KVParallelLinear(3, 2, 2, bias=True)
    column_weight = TrackingSlice(torch.arange(24).reshape(8, 3).float())
    column_bias = TrackingSlice(torch.arange(8).float())
    kv_weight = TrackingSlice(torch.arange(12).reshape(4, 3).float())
    kv_bias = TrackingSlice(torch.arange(4).float())

    column.safetensors_loader(column.weight, column_weight)
    column.safetensors_loader(column.bias, column_bias)
    kv.safetensors_loader(kv.weight, kv_weight)
    kv.safetensors_loader(kv.bias, kv_bias)

    assert torch.equal(column.weight, column_weight.tensor[2:4])
    assert torch.equal(column.bias, column_bias.tensor[2:4])
    assert torch.equal(kv.weight, kv_weight.tensor[:2])
    assert torch.equal(kv.bias, kv_bias.tensor[:2])
    assert column_weight.requests == [(slice(2, 4), slice(None))]
    assert column_bias.requests == [(slice(2, 4),)]
    assert kv_weight.requests == [(slice(0, 2), slice(None))]
    assert kv_bias.requests == [(slice(0, 2),)]
