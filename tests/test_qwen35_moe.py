from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
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
models_package = types.ModuleType("nanovllm.models")
sys.modules.setdefault("nanovllm", nanovllm_package)
sys.modules.setdefault("nanovllm.layers", layers_package)
sys.modules.setdefault("nanovllm.models", models_package)
load_module("nanovllm.layers.activation", "nanovllm/layers/activation.py")
linear = load_module("nanovllm.layers.linear", "nanovllm/layers/linear.py")
load_module("nanovllm.models.moe_dispatch", "nanovllm/models/moe_dispatch.py")
qwen35_moe = load_module("qwen35_moe_under_test", "nanovllm/models/qwen35_moe.py")


def make_experts(
    rank: int = 0,
    world_size: int = 1,
    *,
    hidden_size: int = 2,
    intermediate_size: int = 4,
    num_experts: int = 2,
    decode_backend: str = "sorted",
    decode_chunk_size: int = 8,
):
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=world_size),
        patch.object(qwen35_moe.dist, "get_rank", return_value=rank),
    ):
        return qwen35_moe.Qwen35Experts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            decode_backend=decode_backend,
            decode_chunk_size=decode_chunk_size,
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


def test_single_token_decode_skips_general_expert_grouping():
    torch.manual_seed(43)
    experts = make_experts(num_experts=4)
    experts.gate_up_proj.data.normal_()
    experts.down_proj.data.normal_()
    hidden = torch.randn(1, 2)
    topk_ids = torch.tensor([[3, 0]])
    topk_weights = torch.tensor([[0.4, 0.6]])

    expected = torch.zeros_like(hidden)
    for expert_id, route_index in ((0, 1), (3, 0)):
        gate_up = torch.nn.functional.linear(
            hidden, experts.gate_up_proj[expert_id]
        )
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up,
            experts.down_proj[expert_id],
        )
        expected.add_(expert_output * topk_weights[0, route_index])

    with patch.object(
        qwen35_moe.torch,
        "unique_consecutive",
        side_effect=AssertionError("decode should skip general grouping"),
    ):
        actual = experts(hidden, topk_ids, topk_weights)

    assert torch.equal(actual, expected)


def test_batched_single_token_decode_matches_sorted_backend():
    torch.manual_seed(47)
    sorted_experts = make_experts(num_experts=4)
    batched_experts = make_experts(
        num_experts=4,
        decode_backend="batched",
    )
    batched_experts.load_state_dict(sorted_experts.state_dict())
    hidden = torch.randn(1, 2)
    topk_ids = torch.tensor([[3, 0]])
    topk_weights = torch.tensor([[0.4, 0.6]])

    expected = sorted_experts(hidden, topk_ids, topk_weights)
    with patch.object(
        qwen35_moe.torch.Tensor,
        "cpu",
        side_effect=AssertionError("batched decode must not synchronize to CPU"),
    ):
        actual = batched_experts(hidden, topk_ids, topk_weights)

    torch.testing.assert_close(actual, expected)


def test_batched_multi_token_decode_matches_sorted_backend_in_chunks():
    torch.manual_seed(53)
    sorted_experts = make_experts(num_experts=4)
    batched_experts = make_experts(
        num_experts=4,
        decode_backend="batched",
        decode_chunk_size=2,
    )
    batched_experts.load_state_dict(sorted_experts.state_dict())
    hidden = torch.randn(5, 2)
    topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3], [2, 1], [3, 2]])
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = sorted_experts(
        hidden,
        topk_ids,
        topk_weights,
        is_decode=False,
    )
    selected_route_counts = []
    original_bmm = qwen35_moe.torch.bmm

    def record_bmm(left, right, *args, **kwargs):
        if left.shape[1] == 2 * batched_experts.local_intermediate_size:
            selected_route_counts.append(left.shape[0])
        return original_bmm(left, right, *args, **kwargs)

    with (
        patch.object(
            qwen35_moe.torch.Tensor,
            "cpu",
            side_effect=AssertionError("batched decode must not synchronize to CPU"),
        ),
        patch.object(
            qwen35_moe.torch,
            "cat",
            side_effect=AssertionError("decode chunks must write into final storage"),
        ),
        patch.object(
            qwen35_moe.torch,
            "bmm",
            side_effect=record_bmm,
        ),
    ):
        actual = batched_experts(
            hidden,
            topk_ids,
            topk_weights,
            is_decode=True,
        )

    torch.testing.assert_close(actual, expected)
    assert selected_route_counts == [4, 4, 2]


def test_batched_decode_preallocated_output_preserves_autograd():
    torch.manual_seed(59)
    experts = make_experts(
        num_experts=4,
        decode_backend="batched",
        decode_chunk_size=2,
    )
    hidden = torch.randn(5, 2, requires_grad=True)
    topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3], [2, 1], [3, 2]])
    topk_weights = torch.rand(5, 2, requires_grad=True)

    output = experts(
        hidden,
        topk_ids,
        topk_weights,
        is_decode=True,
    )
    output.square().sum().backward()

    assert hidden.grad is not None
    assert topk_weights.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None


def test_batched_backend_keeps_prefill_on_grouped_dispatch():
    experts = make_experts(decode_backend="batched")
    hidden = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    topk_weights = torch.full((3, 2), 0.5)

    with patch.object(qwen35_moe.torch, "bmm", side_effect=AssertionError):
        output = experts(
            hidden,
            topk_ids,
            topk_weights,
            is_decode=False,
        )

    assert output.shape == hidden.shape


def test_invalid_decode_backend_is_rejected():
    with pytest.raises(ValueError, match="decode_backend"):
        make_experts(decode_backend="unknown")


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


def test_sparse_moe_combines_tp_outputs_before_single_all_reduce():
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        qwen35_moe_decode_backend="batched",
        qwen35_moe_decode_chunk_size=2,
    )
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=2),
        patch.object(qwen35_moe.dist, "get_rank", return_value=0),
        patch.object(linear.dist, "get_world_size", return_value=2),
        patch.object(linear.dist, "get_rank", return_value=0),
    ):
        block = qwen35_moe.Qwen35SparseMoeBlock(config)
    hidden = torch.randn(3, 4)
    routed = torch.full_like(hidden, 2.0)
    shared = torch.full_like(hidden, 4.0)
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: SimpleNamespace(
        is_prefill=False,
        is_mixed=False,
    )
    block.shared_expert_gate.weight.data.zero_()
    topk_weights = torch.ones(3, 2)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(
            block.gate,
            "forward",
            return_value=(topk_weights, topk_ids),
        ),
        patch.object(block.experts, "forward", return_value=routed) as experts,
        patch.object(block.shared_expert, "forward", return_value=shared) as shared_expert,
        patch.object(qwen35_moe.dist, "all_reduce", return_value=None) as all_reduce,
    ):
        output = block(hidden)

    assert experts.call_count == 1
    expert_args, expert_kwargs = experts.call_args
    assert torch.equal(expert_args[0], hidden)
    assert expert_args[1] is topk_ids
    assert expert_args[2] is topk_weights
    assert expert_kwargs == {"is_decode": True, "reduce_output": False}
    assert shared_expert.call_args.kwargs == {"reduce_output": False}
    assert all_reduce.call_count == 1
    assert torch.equal(all_reduce.call_args.args[0], output)
    torch.testing.assert_close(output, routed + 0.5 * shared)


def test_mixed_batch_keeps_prefill_tokens_on_grouped_moe_dispatch():
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        qwen35_moe_decode_backend="batched",
        qwen35_moe_decode_chunk_size=2,
    )
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=1),
        patch.object(qwen35_moe.dist, "get_rank", return_value=0),
        patch.object(linear.dist, "get_world_size", return_value=1),
        patch.object(linear.dist, "get_rank", return_value=0),
    ):
        block = qwen35_moe.Qwen35SparseMoeBlock(config)
    hidden = torch.randn(7, 4)
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: SimpleNamespace(
        is_prefill=False,
        is_mixed=True,
    )
    topk_weights = torch.ones(7, 2)
    topk_ids = torch.zeros(7, 2, dtype=torch.long)

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(
            block.gate,
            "forward",
            return_value=(topk_weights, topk_ids),
        ),
        patch.object(
            block.experts,
            "forward",
            return_value=torch.zeros_like(hidden),
        ) as experts,
        patch.object(
            block.shared_expert,
            "forward",
            return_value=torch.zeros_like(hidden),
        ),
    ):
        block(hidden)

    assert experts.call_args.kwargs["is_decode"] is False


def test_combined_routed_and_shared_tp_partials_match_full_reference():
    torch.manual_seed(61)

    def make_components(rank, world_size):
        with (
            patch.object(qwen35_moe.dist, "get_world_size", return_value=world_size),
            patch.object(qwen35_moe.dist, "get_rank", return_value=rank),
            patch.object(linear.dist, "get_world_size", return_value=world_size),
            patch.object(linear.dist, "get_rank", return_value=rank),
        ):
            experts = qwen35_moe.Qwen35Experts(4, 8, 4)
            shared_expert = qwen35_moe.Qwen35SharedExpert(4, 8)
        return experts, shared_expert

    source_expert_gate_up = torch.randn(4, 16, 4)
    source_expert_down = torch.randn(4, 4, 8)
    source_shared_gate = torch.randn(8, 4)
    source_shared_up = torch.randn(8, 4)
    source_shared_down = torch.randn(4, 8)

    def load(experts, shared_expert):
        experts._load_gate_up(experts.gate_up_proj, source_expert_gate_up)
        experts._load_down(experts.down_proj, source_expert_down)
        shared_expert.gate_up_proj.weight_loader(
            shared_expert.gate_up_proj.weight,
            source_shared_gate,
            0,
        )
        shared_expert.gate_up_proj.weight_loader(
            shared_expert.gate_up_proj.weight,
            source_shared_up,
            1,
        )
        shared_expert.down_proj.weight_loader(
            shared_expert.down_proj.weight,
            source_shared_down,
        )

    full = make_components(0, 1)
    ranks = [make_components(rank, 2) for rank in range(2)]
    load(*full)
    for components in ranks:
        load(*components)

    hidden = torch.randn(5, 4)
    topk_ids = torch.randint(0, 4, (5, 2))
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    shared_gate = torch.sigmoid(torch.randn(5, 1))
    expected = (
        full[0](hidden, topk_ids, topk_weights)
        + shared_gate * full[1](hidden)
    )
    actual = sum(
        experts(
            hidden,
            topk_ids,
            topk_weights,
            reduce_output=False,
        )
        + shared_gate * shared_expert(hidden, reduce_output=False)
        for experts, shared_expert in ranks
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
