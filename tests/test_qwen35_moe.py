from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import gc
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch
import weakref

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
moe_dispatch = load_module(
    "nanovllm.models.moe_dispatch",
    "nanovllm/models/moe_dispatch.py",
)
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


def _official_router_reference(logits, top_k):
    probabilities = torch.softmax(logits, dtype=torch.float, dim=-1)
    weights, ids = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(logits.dtype), ids


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_router_topk_first_matches_official_weights_and_gradients(dtype):
    torch.manual_seed(37)
    router = qwen35_moe.Qwen35TopKRouter(8, 32, 4).to(dtype)
    router.weight.data.normal_(mean=0.0, std=0.15)
    hidden = torch.randn(7, 8, dtype=dtype, requires_grad=True)

    weights, ids = router(hidden)
    actual_loss = (weights.float() ** 2).sum()
    actual_grad = torch.autograd.grad(actual_loss, hidden)[0]

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    logits = torch.nn.functional.linear(reference_hidden, router.weight)
    expected_weights, expected_ids = _official_router_reference(logits, 4)
    expected_loss = (expected_weights.float() ** 2).sum()
    expected_grad = torch.autograd.grad(expected_loss, reference_hidden)[0]

    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=8e-3, atol=8e-3)


def test_router_extreme_logits_preserve_official_nonzero_routes():
    router = qwen35_moe.Qwen35TopKRouter(1, 4, 3)
    router.weight.data.copy_(torch.tensor([[1000.0], [600.0], [800.0], [700.0]]))
    hidden = torch.ones(1, 1)

    weights, ids = router(hidden)
    logits = torch.nn.functional.linear(hidden, router.weight)
    expected_weights, expected_ids = _official_router_reference(logits, 3)
    actual_by_expert = torch.zeros_like(logits).scatter(1, ids, weights)
    expected_by_expert = torch.zeros_like(logits).scatter(
        1, expected_ids, expected_weights
    )

    # Full softmax may choose different experts among probabilities that
    # underflowed to zero. Their contribution is zero in either formulation.
    torch.testing.assert_close(actual_by_expert, expected_by_expert)


def test_router_tied_logits_match_official_selection():
    router = qwen35_moe.Qwen35TopKRouter(1, 4, 2)
    router.weight.data.copy_(torch.tensor([[2.0], [2.0], [1.0], [0.0]]))
    hidden = torch.ones(1, 1)

    weights, ids = router(hidden)
    logits = torch.nn.functional.linear(hidden, router.weight)
    expected_weights, expected_ids = _official_router_reference(logits, 2)

    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights)


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


def test_router_inference_softmax_reuses_selected_logits():
    router = qwen35_moe.Qwen35TopKRouter(4, 8, 2)
    hidden = torch.randn(3, 4)
    original_softmax = qwen35_moe.torch.softmax
    reused = []

    def record_storage(tensor, *args, **kwargs):
        result = original_softmax(tensor, *args, **kwargs)
        reused.append(result.data_ptr() == tensor.data_ptr())
        return result

    with (
        torch.inference_mode(),
        patch.object(qwen35_moe.torch, "softmax", side_effect=record_storage),
    ):
        weights, _ = router(hidden)

    assert reused == [True]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))


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


def test_sorted_expert_dispatch_reuses_sort_permutation_for_token_indices():
    experts = make_experts()
    hidden = torch.randn(3, 2)
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 0]])
    topk_weights = torch.full((3, 2), 0.5)
    division_storage = []
    original_divide = torch.Tensor.div_

    def record_divide(tensor, *args, **kwargs):
        division_storage.append(tensor.data_ptr())
        return original_divide(tensor, *args, **kwargs)

    with (
        torch.inference_mode(),
        patch.object(torch.Tensor, "div_", record_divide),
    ):
        output = experts(hidden, topk_ids, topk_weights)

    assert output.shape == hidden.shape
    assert division_storage
    assert len(set(division_storage)) == 1


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


def test_batched_decode_does_not_allocate_discarded_wrapper_output():
    experts = make_experts(decode_backend="batched")
    hidden = torch.randn(1, 2)
    topk_ids = torch.tensor([[0, 1]])
    topk_weights = torch.tensor([[0.6, 0.4]])
    expected = torch.randn_like(hidden)

    with (
        patch.object(
            qwen35_moe,
            "batched_expert_dispatch",
            return_value=expected,
        ),
        patch.object(
            qwen35_moe.torch,
            "zeros_like",
            side_effect=AssertionError("batched output must not be preallocated"),
        ),
    ):
        actual = experts(hidden, topk_ids, topk_weights, is_decode=True)

    assert actual is expected


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
    original_matmul = qwen35_moe.torch.matmul

    def record_matmul(left, right, *args, **kwargs):
        if left.ndim == 4:
            selected_route_counts.append(left.shape[0] * left.shape[1])
        return original_matmul(left, right, *args, **kwargs)

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
            qwen35_moe.torch.Tensor,
            "expand",
            side_effect=AssertionError("decode must broadcast without copying hidden"),
        ),
        patch.object(
            qwen35_moe.torch,
            "matmul",
            side_effect=record_matmul,
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


def test_batched_decode_releases_previous_chunk_weights_before_next_chunk():
    hidden = torch.randn(2, 4)
    topk_ids = torch.tensor([[0, 1], [2, 3]])
    topk_weights = torch.full((2, 2), 0.5)
    gate_up_proj = torch.randn(4, 6, 4)
    down_proj = torch.randn(4, 4, 3)
    original_index_select = torch.Tensor.index_select
    selected_down_refs = []

    def record_index_select(tensor, dim, index):
        if tensor is gate_up_proj and selected_down_refs:
            gc.collect()
            assert selected_down_refs[-1]() is None
        result = original_index_select(tensor, dim, index)
        if tensor is down_proj:
            selected_down_refs.append(weakref.ref(result))
        return result

    with (
        torch.inference_mode(),
        patch.object(torch.Tensor, "index_select", new=record_index_select),
    ):
        output = moe_dispatch.batched_expert_dispatch(
            hidden,
            topk_ids,
            topk_weights,
            gate_up_proj,
            down_proj,
            chunk_size=1,
        )

    assert output.shape == hidden.shape
    assert selected_down_refs
    assert all(reference() is None for reference in selected_down_refs)


def test_batched_dispatch_can_fill_caller_output_buffer():
    torch.manual_seed(57)
    experts = make_experts(
        num_experts=4,
        decode_backend="batched",
        decode_chunk_size=2,
    )
    experts.gate_up_proj.data.normal_()
    experts.down_proj.data.normal_()
    hidden = torch.randn(3, 2)
    topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3]])
    topk_weights = torch.rand(3, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    expected = moe_dispatch.batched_expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        experts.gate_up_proj,
        experts.down_proj,
        2,
    )
    output = torch.empty_like(hidden)

    actual = moe_dispatch.batched_expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        experts.gate_up_proj,
        experts.down_proj,
        2,
        output=output,
    )

    assert actual is output
    torch.testing.assert_close(actual, expected)


def test_mixed_batched_backend_only_splits_decode_prefix():
    torch.manual_seed(58)
    sorted_experts = make_experts(num_experts=4)
    mixed_experts = make_experts(
        num_experts=4,
        decode_backend="batched",
        decode_chunk_size=2,
    )
    mixed_experts.load_state_dict(sorted_experts.state_dict())
    hidden = torch.randn(5, 2)
    topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3], [2, 1], [3, 2]])
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    expected = sorted_experts(hidden, topk_ids, topk_weights, is_decode=False)

    with patch.object(
        qwen35_moe,
        "batched_expert_dispatch",
        wraps=qwen35_moe.batched_expert_dispatch,
    ) as batched:
        actual = mixed_experts(
            hidden,
            topk_ids,
            topk_weights,
            is_decode=False,
            decode_token_count=2,
        )

    torch.testing.assert_close(actual, expected)
    assert batched.call_count == 1
    assert torch.equal(batched.call_args.args[0], hidden[:2])
    assert batched.call_args.kwargs["output"].data_ptr() == actual[:2].data_ptr()


def test_mixed_batched_backend_preserves_autograd():
    torch.manual_seed(60)
    sorted_experts = make_experts(num_experts=4)
    mixed_experts = make_experts(
        num_experts=4,
        decode_backend="batched",
        decode_chunk_size=2,
    )
    mixed_experts.load_state_dict(sorted_experts.state_dict())
    sorted_hidden = torch.randn(5, 2, requires_grad=True)
    mixed_hidden = sorted_hidden.detach().clone().requires_grad_()
    topk_ids = torch.tensor([[3, 0], [1, 2], [0, 3], [2, 1], [3, 2]])
    sorted_weights = torch.rand(5, 2)
    sorted_weights /= sorted_weights.sum(dim=-1, keepdim=True)
    sorted_weights.requires_grad_()
    mixed_weights = sorted_weights.detach().clone().requires_grad_()

    expected = sorted_experts(
        sorted_hidden,
        topk_ids,
        sorted_weights,
        is_decode=False,
    )
    actual = mixed_experts(
        mixed_hidden,
        topk_ids,
        mixed_weights,
        is_decode=False,
        decode_token_count=2,
    )
    expected.square().sum().backward()
    actual.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mixed_hidden.grad, sorted_hidden.grad)
    torch.testing.assert_close(mixed_weights.grad, sorted_weights.grad)
    torch.testing.assert_close(
        mixed_experts.gate_up_proj.grad,
        sorted_experts.gate_up_proj.grad,
    )
    torch.testing.assert_close(
        mixed_experts.down_proj.grad,
        sorted_experts.down_proj.grad,
    )


@pytest.mark.parametrize("decode_token_count", [-1, 6])
def test_mixed_decode_count_must_fit_batch(decode_token_count):
    experts = make_experts(decode_backend="batched")
    hidden = torch.randn(5, 2)
    topk_ids = torch.zeros(5, 2, dtype=torch.long)
    topk_weights = torch.full((5, 2), 0.5)

    with pytest.raises(ValueError, match="decode_token_count"):
        experts(
            hidden,
            topk_ids,
            topk_weights,
            is_decode=False,
            decode_token_count=decode_token_count,
        )


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


def test_batched_decode_swiglu_reuses_gate_storage_without_autograd():
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    expected = torch.nn.functional.silu(gate) * up
    gate_storage = gate.data_ptr()

    actual = moe_dispatch.silu_and_mul(gate, up)

    assert actual.data_ptr() == gate_storage
    torch.testing.assert_close(actual, expected)


def test_batched_decode_route_weighting_reuses_expert_output():
    expert_output = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    original = expert_output.clone()
    weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]])
    expected = (original.view(2, 2, 2) * weights.unsqueeze(-1)).sum(dim=1)

    actual = moe_dispatch.weighted_route_sum(expert_output, weights)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        expert_output,
        (original.view(2, 2, 2) * weights.unsqueeze(-1)).reshape_as(original),
    )


def test_batched_decode_route_sum_fills_caller_output():
    expert_output = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    original = expert_output.clone()
    weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]])
    output = torch.empty(2, 2)
    expected = (original.view(2, 2, 2) * weights.unsqueeze(-1)).sum(dim=1)

    actual = moe_dispatch.weighted_route_sum(
        expert_output,
        weights,
        output=output,
    )

    assert actual is output
    torch.testing.assert_close(actual, expected)


def test_batched_decode_route_sum_rejects_invalid_output():
    with pytest.raises(ValueError, match="weighted route sum"):
        moe_dispatch.weighted_route_sum(
            torch.randn(4, 2),
            torch.rand(2, 2),
            output=torch.empty(2, 3),
        )


def test_batched_decode_route_sum_rejects_output_with_autograd():
    with pytest.raises(ValueError, match="requires grad"):
        moe_dispatch.weighted_route_sum(
            torch.randn(4, 2, requires_grad=True),
            torch.rand(2, 2),
            output=torch.empty(2, 2),
        )


def test_sorted_route_weighting_reuses_expert_output_without_autograd():
    expert_output = torch.randn(5, 4)
    original = expert_output.clone()
    weights = torch.rand(5)
    storage = expert_output.data_ptr()

    actual = moe_dispatch.weight_expert_output(expert_output, weights)

    assert actual.data_ptr() == storage
    torch.testing.assert_close(actual, original * weights.unsqueeze(-1))


def test_sorted_route_weighting_preserves_autograd():
    expert_output = torch.randn(5, 4, requires_grad=True)
    weights = torch.rand(5, requires_grad=True)
    expected_output = expert_output.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()

    actual = moe_dispatch.weight_expert_output(expert_output, weights)
    expected = expected_output * expected_weights.unsqueeze(-1)
    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(expert_output.grad, expected_output.grad)
    torch.testing.assert_close(weights.grad, expected_weights.grad)


def test_batched_decode_route_weighting_preserves_autograd():
    expert_output = torch.randn(6, 4, requires_grad=True)
    weights = torch.rand(3, 2, requires_grad=True)
    before = expert_output.detach().clone()

    output = moe_dispatch.weighted_route_sum(expert_output, weights)
    output.sum().backward()

    torch.testing.assert_close(expert_output.detach(), before)
    assert expert_output.grad is not None
    assert weights.grad is not None


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


def test_sparse_moe_inference_reuses_tp_partial_buffers():
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=1),
        patch.object(qwen35_moe.dist, "get_rank", return_value=0),
        patch.object(linear.dist, "get_world_size", return_value=1),
        patch.object(linear.dist, "get_rank", return_value=0),
    ):
        block = qwen35_moe.Qwen35SparseMoeBlock(config)
    hidden = torch.randn(3, 4)
    routed = torch.full_like(hidden, 2.0)
    shared = torch.full_like(hidden, 4.0)
    shared_gate = torch.zeros(3, 1)
    topk_weights = torch.ones(3, 2)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: SimpleNamespace(
        is_prefill=False,
        is_mixed=False,
    )

    with (
        torch.inference_mode(),
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(block.gate, "forward", return_value=(topk_weights, topk_ids)),
        patch.object(block.experts, "forward", return_value=routed),
        patch.object(block.shared_expert, "forward", return_value=shared),
        patch.object(block.shared_expert_gate, "forward", return_value=shared_gate),
    ):
        output = block(hidden)

    assert output.data_ptr() == routed.data_ptr()
    assert torch.equal(shared_gate, torch.full_like(shared_gate, 0.5))
    assert torch.equal(shared, torch.full_like(shared, 2.0))
    assert torch.equal(output, torch.full_like(output, 4.0))


def test_sparse_moe_autograd_preserves_tp_partial_buffers():
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    with (
        patch.object(qwen35_moe.dist, "get_world_size", return_value=1),
        patch.object(qwen35_moe.dist, "get_rank", return_value=0),
        patch.object(linear.dist, "get_world_size", return_value=1),
        patch.object(linear.dist, "get_rank", return_value=0),
    ):
        block = qwen35_moe.Qwen35SparseMoeBlock(config)
    hidden = torch.randn(3, 4)
    routed = torch.full_like(hidden, 2.0, requires_grad=True)
    shared = torch.full_like(hidden, 4.0, requires_grad=True)
    shared_gate = torch.zeros(3, 1, requires_grad=True)
    routed_before = routed.detach().clone()
    shared_before = shared.detach().clone()
    topk_weights = torch.ones(3, 2)
    topk_ids = torch.zeros(3, 2, dtype=torch.long)
    context_module = types.ModuleType("nanovllm.utils.context")
    context_module.get_context = lambda: SimpleNamespace(
        is_prefill=False,
        is_mixed=False,
    )

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": context_module}),
        patch.object(block.gate, "forward", return_value=(topk_weights, topk_ids)),
        patch.object(block.experts, "forward", return_value=routed),
        patch.object(block.shared_expert, "forward", return_value=shared),
        patch.object(block.shared_expert_gate, "forward", return_value=shared_gate),
    ):
        output = block(hidden)
        output.sum().backward()

    assert torch.equal(routed, routed_before)
    assert torch.equal(shared, shared_before)
    assert routed.grad is not None
    assert shared.grad is not None
    assert shared_gate.grad is not None


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
        decode_token_count=2,
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

    assert experts.call_args.kwargs == {
        "is_decode": False,
        "decode_token_count": 2,
        "reduce_output": False,
    }


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


def test_expert_gate_up_loaders_do_not_materialize_local_packed_copy():
    source = torch.arange(32, dtype=torch.float32).reshape(2, 8, 2)
    rank1 = make_experts(rank=1, world_size=2)

    with patch.object(
        qwen35_moe.torch,
        "cat",
        side_effect=AssertionError("loader must copy each shard directly"),
    ):
        rank1._load_gate_up(rank1.gate_up_proj, source)

    assert torch.equal(rank1.gate_up_proj[:, :2], source[:, 2:4])
    assert torch.equal(rank1.gate_up_proj[:, 2:], source[:, 6:8])

    tracked = TrackingSlice(source)
    with patch.object(
        qwen35_moe.torch,
        "cat",
        side_effect=AssertionError("loader must copy each shard directly"),
    ):
        rank1._load_gate_up_slice(rank1.gate_up_proj, tracked)

    assert torch.equal(rank1.gate_up_proj[:, :2], source[:, 2:4])
    assert torch.equal(rank1.gate_up_proj[:, 2:], source[:, 6:8])
    assert len(tracked.requests) == 2


def test_qwen35_rmsnorm_does_not_mutate_input():
    norm = qwen35_moe.Qwen35RMSNorm(2)
    source = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    original = source.clone()

    output = norm(source)

    assert torch.equal(source, original)
    assert torch.allclose(output.pow(2).mean(dim=-1), torch.ones(1), atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen35_rmsnorm_inference_matches_fp32_reference_without_mutation(dtype):
    torch.manual_seed(71)
    norm = qwen35_moe.Qwen35RMSNorm(8)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    source = torch.randn(5, 8, dtype=dtype)
    original = source.clone()
    source_float = source.float()
    expected = (
        source_float
        * torch.rsqrt(source_float.square().mean(dim=-1, keepdim=True) + norm.eps)
        * norm.weight
    ).to(dtype)

    with torch.inference_mode():
        output = norm(source)

    assert torch.equal(source, original)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen35_rmsnorm_can_reuse_explicit_inference_output(dtype):
    torch.manual_seed(72)
    norm = qwen35_moe.Qwen35RMSNorm(8)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    source = torch.randn(5, 8, dtype=dtype)
    source_ptr = source.data_ptr()
    source_float = source.float()
    expected = (
        source_float
        * torch.rsqrt(source_float.square().mean(dim=-1, keepdim=True) + norm.eps)
        * norm.weight
    ).to(dtype)

    with torch.inference_mode():
        output = norm(source, inplace_output=True)

    assert output.data_ptr() == source_ptr
    torch.testing.assert_close(output, expected)


def test_qwen35_rmsnorm_autograd_preserves_backward_inputs():
    norm = qwen35_moe.Qwen35RMSNorm(8)
    source = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)

    norm(source).float().square().mean().backward()

    assert source.grad is not None
    assert norm.weight.grad is not None


def test_qwen35_rmsnorm_materializes_checkpoint_delta_once():
    norm = qwen35_moe.Qwen35RMSNorm(3)
    checkpoint_weight = torch.tensor([-0.25, 0.0, 0.5], dtype=torch.bfloat16)

    norm._load_weight(norm.weight, checkpoint_weight)

    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(
        norm.weight,
        torch.tensor([0.75, 1.0, 1.5]),
    )
