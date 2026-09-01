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


def make_experts(
    rank: int = 0,
    world_size: int = 1,
    *,
    hidden_size: int = 2,
    intermediate_size: int = 4,
    num_experts: int = 2,
):
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=world_size),
        patch.object(qwen35_moe.dist, "get_rank", return_value=rank),
    ):
        return qwen35_moe.Qwen35Experts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        )


def test_router_selects_and_renormalizes_topk_experts():
    router = qwen35_moe.Qwen35TopKRouter(2, 3, 2)
    router.weight.data.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))

    weights, ids = router(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert ids.tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))


def test_router_topk_first_matches_full_softmax_reference():
    torch.manual_seed(31)
    router = qwen35_moe.Qwen35TopKRouter(16, 256, 8).to(torch.bfloat16)
    router.weight.data.normal_(mean=0.0, std=0.2)
    hidden = torch.randn(19, 16, dtype=torch.bfloat16)

    weights, ids = router(hidden)
    logits = torch.nn.functional.linear(hidden, router.weight)
    probabilities = torch.softmax(logits.float(), dim=-1)
    expected_weights, expected_ids = torch.topk(probabilities, 8, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(
        weights.float(),
        expected_weights,
        rtol=4e-3,
        atol=4e-3,
    )


def test_router_softmax_only_materializes_selected_experts():
    router = qwen35_moe.Qwen35TopKRouter(4, 256, 8)
    hidden = torch.randn(11, 4)
    softmax_shapes = []
    original_softmax = qwen35_moe.torch.softmax

    def record_softmax_shape(tensor, *args, **kwargs):
        softmax_shapes.append(tuple(tensor.shape))
        return original_softmax(tensor, *args, **kwargs)

    with patch.object(qwen35_moe.torch, "softmax", side_effect=record_softmax_shape):
        router(hidden)

    assert softmax_shapes == [(11, 8)]


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


def test_sorted_expert_dispatch_only_groups_active_experts():
    experts = make_experts()
    hidden = torch.randn(2, 2)
    topk_ids = torch.tensor([[1, 1], [1, 1]])
    topk_weights = torch.full((2, 2), 0.5)

    with patch.object(
        qwen35_moe.torch,
        "bincount",
        side_effect=AssertionError("dense expert histogram should not be built"),
    ):
        output = experts(hidden, topk_ids, topk_weights)

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


def test_tensor_parallel_expert_outputs_sum_to_single_rank_reference():
    torch.manual_seed(13)
    full = make_experts(world_size=1)
    ranks = [make_experts(rank=rank, world_size=2) for rank in range(2)]
    source_gate_up = torch.randn(2, 8, 2)
    source_down = torch.randn(2, 2, 4)
    full._load_gate_up(full.gate_up_proj, source_gate_up)
    full._load_down(full.down_proj, source_down)
    for rank in ranks:
        rank._load_gate_up(rank.gate_up_proj, source_gate_up)
        rank._load_down(rank.down_proj, source_down)
    hidden = torch.randn(7, 2)
    topk_ids = torch.randint(0, 2, (7, 2))
    topk_weights = torch.rand(7, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = full(hidden, topk_ids, topk_weights)
    with patch.object(qwen35_moe.dist, "all_reduce", return_value=None):
        actual = sum(rank(hidden, topk_ids, topk_weights) for rank in ranks)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@torch.no_grad()
def test_official_expert_count_matches_across_tp4_and_tp8():
    torch.manual_seed(41)
    kwargs = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_experts": 256,
    }
    source_gate_up = torch.randn(256, 16, 4)
    source_down = torch.randn(256, 4, 8)
    hidden = torch.randn(5, 4)
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [31, 63, 95, 127, 159, 191, 223, 255],
            [8, 16, 32, 64, 96, 128, 192, 224],
            [255, 224, 192, 128, 64, 32, 16, 8],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ]
    )
    topk_weights = torch.rand(5, 8)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    def load(experts):
        experts._load_gate_up(experts.gate_up_proj, source_gate_up)
        experts._load_down(experts.down_proj, source_down)

    full = make_experts(world_size=1, **kwargs)
    load(full)
    expected = full(hidden, topk_ids, topk_weights)

    for tp_size in (4, 8):
        ranks = [
            make_experts(rank=rank, world_size=tp_size, **kwargs)
            for rank in range(tp_size)
        ]
        for rank_experts in ranks:
            load(rank_experts)
        with patch.object(qwen35_moe.dist, "all_reduce", return_value=None):
            actual = sum(
                rank_experts(hidden, topk_ids, topk_weights)
                for rank_experts in ranks
            )
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


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
