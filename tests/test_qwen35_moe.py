from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch


ROOT = Path(__file__).parents[1]


def load_module(name: str, relative_path: str):
    spec = spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nanovllm_package = types.ModuleType("nanovllm")
layers_package = types.ModuleType("nanovllm.layers")
sys.modules.setdefault("nanovllm", nanovllm_package)
sys.modules.setdefault("nanovllm.layers", layers_package)
load_module("nanovllm.layers.activation", "nanovllm/layers/activation.py")
linear = load_module("nanovllm.layers.linear", "nanovllm/layers/linear.py")
qwen35_moe = load_module("qwen35_moe_under_test", "nanovllm/models/qwen35_moe.py")


def make_experts(rank: int = 0, world_size: int = 1):
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=world_size),
        patch.object(qwen35_moe.dist, "get_rank", return_value=rank),
    ):
        return qwen35_moe.Qwen35Experts(
            hidden_size=2,
            intermediate_size=4,
            num_experts=2,
        )


def test_router_selects_and_renormalizes_topk_experts():
    router = qwen35_moe.Qwen35TopKRouter(2, 3, 2)
    router.weight.data.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))

    weights, ids = router(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert ids.tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_expert_reference_path_matches_manual_mixture():
    experts = make_experts()
    experts.gate_up_proj.data.fill_(1.0)
    experts.down_proj.data.fill_(0.5)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0, 1], [1, 0]])
    topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])

    output = experts(hidden, topk_ids, topk_weights)

    gate_up = torch.nn.functional.linear(hidden, experts.gate_up_proj[0])
    gate, up = gate_up.chunk(2, dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up,
        experts.down_proj[0],
    )
    assert torch.allclose(output, expected)


def test_sorted_expert_dispatch_matches_naive_topk_accumulation():
    torch.manual_seed(7)
    experts = make_experts()
    experts.gate_up_proj.data.normal_()
    experts.down_proj.data.normal_()
    hidden = torch.randn(5, 2)
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    actual = experts(hidden, topk_ids, topk_weights)
    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = torch.nn.functional.linear(
                hidden[token], experts.gate_up_proj[expert]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            value = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                experts.down_proj[expert],
            )
            expected[token] += value * topk_weights[token, slot]

    torch.testing.assert_close(actual, expected)


def test_expert_weights_are_sharded_and_replicable_across_tp_ranks():
    source_gate_up = torch.arange(32, dtype=torch.float32).reshape(2, 8, 2)
    source_down = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    rank0 = make_experts(rank=0, world_size=2)
    rank1 = make_experts(rank=1, world_size=2)

    rank0._load_gate_up(rank0.gate_up_proj, source_gate_up)
    rank1._load_gate_up(rank1.gate_up_proj, source_gate_up)
    rank0._load_down(rank0.down_proj, source_down)
    rank1._load_down(rank1.down_proj, source_down)

    expected_rank0_gate_up = torch.cat(
        (source_gate_up[:, 0:2], source_gate_up[:, 4:6]), dim=1
    )
    expected_rank1_gate_up = torch.cat(
        (source_gate_up[:, 2:4], source_gate_up[:, 6:8]), dim=1
    )
    assert torch.equal(rank0.gate_up_proj, expected_rank0_gate_up)
    assert torch.equal(rank1.gate_up_proj, expected_rank1_gate_up)
    assert torch.equal(rank0.down_proj, source_down[:, :, 0:2])
    assert torch.equal(rank1.down_proj, source_down[:, :, 2:4])


class TrackingSlice:
    def __init__(self, tensor):
        self.tensor = tensor
        self.requests = []

    def get_shape(self):
        return self.tensor.shape

    def __getitem__(self, key):
        self.requests.append(key)
        return self.tensor[key]


def test_expert_safetensors_loader_reads_only_local_tp_slices():
    source_gate_up = TrackingSlice(
        torch.arange(32, dtype=torch.float32).reshape(2, 8, 2)
    )
    source_down = TrackingSlice(
        torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    )
    rank1 = make_experts(rank=1, world_size=2)

    rank1._load_gate_up_slice(rank1.gate_up_proj, source_gate_up)
    rank1._load_down_slice(rank1.down_proj, source_down)

    assert torch.equal(
        rank1.gate_up_proj,
        torch.cat(
            (source_gate_up.tensor[:, 2:4], source_gate_up.tensor[:, 6:8]),
            dim=1,
        ),
    )
    assert torch.equal(rank1.down_proj, source_down.tensor[:, :, 2:4])
    assert len(source_gate_up.requests) == 2
    assert len(source_down.requests) == 1


def test_qwen35_rmsnorm_does_not_mutate_input():
    norm = qwen35_moe.Qwen35RMSNorm(2)
    source = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    original = source.clone()

    output = norm(source)

    assert torch.equal(source, original)
    assert torch.allclose(output.pow(2).mean(dim=-1), torch.ones(1), atol=1e-5)
