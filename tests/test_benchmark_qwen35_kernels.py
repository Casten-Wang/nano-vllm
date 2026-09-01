from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_qwen35_kernels",
    ROOT / "scripts" / "benchmark_qwen35_kernels.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_expert_dispatch_matches_naive_route_accumulation():
    torch.manual_seed(101)
    hidden = torch.randn(4, 3)
    topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1]])
    topk_weights = torch.rand(4, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(3, 4, 3)
    down = torch.randn(3, 3, 2)

    actual = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = topk_ids[token, route]
            projected = torch.nn.functional.linear(hidden[token], gate_up[expert])
            gate, up = projected.chunk(2)
            value = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                down[expert],
            )
            expected[token] += value * topk_weights[token, route]

    torch.testing.assert_close(actual, expected)
    reference = MODULE.expert_dispatch_reference(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    torch.testing.assert_close(actual, reference)
