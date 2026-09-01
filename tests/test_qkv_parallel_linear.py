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
