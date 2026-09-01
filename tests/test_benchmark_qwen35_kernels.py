from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_qwen35_kernels",
    ROOT / "scripts" / "benchmark_qwen35_kernels.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_measure_preserves_every_raw_repeat():
    result = MODULE.measure(
        lambda: None,
        device=torch.device("cpu"),
        warmup=1,
        iterations=1,
        repeats=3,
    )

    assert len(result["samples_ms"]) == 3
    assert result["peak_extra_mib_samples"] == []
    assert result["median_ms"] == sorted(result["samples_ms"])[1]


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


def test_graph_safe_batched_decode_matches_current_single_token_path():
    torch.manual_seed(109)
    hidden = torch.randn(1, 4)
    topk_ids = torch.tensor([[3, 1]])
    topk_weights = torch.tensor([[0.25, 0.75]])
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    actual = MODULE.expert_dispatch_batched_decode(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )

    torch.testing.assert_close(actual, expected)


def test_graph_safe_batched_decode_matches_multi_token_path():
    torch.manual_seed(113)
    hidden = torch.randn(5, 4)
    topk_ids = torch.randint(0, 4, (5, 2))
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    actual = MODULE.expert_dispatch_batched_decode(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
        chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_expert_dispatch_sweep_preserves_requested_token_counts(monkeypatch):
    calls = []

    def fake_benchmark(args, device, dtype, token_count):
        calls.append(token_count)
        return {"tokens": token_count}

    monkeypatch.setattr(MODULE, "benchmark_expert_dispatch", fake_benchmark)
    args = SimpleNamespace(expert_token_counts=(1, 8, 32, 64, 128, 512))

    result = MODULE.benchmark_expert_dispatch_sweep(
        args,
        torch.device("cpu"),
        torch.float32,
    )

    assert calls == [1, 8, 32, 64, 128, 512]
    assert list(result) == ["1", "8", "32", "64", "128", "512"]
    assert result["128"] == {"tokens": 128}


def test_single_token_dispatch_reports_general_path_baseline():
    args = SimpleNamespace(
        moe_intermediate_size=8,
        tp_size=1,
        hidden_size=4,
        num_experts=4,
        top_k=2,
        num_hidden_layers=1,
        warmup=0,
        iterations=1,
        repeats=1,
        moe_decode_chunk_size=2,
        moe_graph_safe_min_speedup=1.05,
        moe_graph_safe_max_peak_extra_mib=64.0,
        moe_graph_safe_max_abs_error=0.05,
    )

    result = MODULE.benchmark_expert_dispatch(
        args,
        torch.device("cpu"),
        torch.float32,
        1,
    )

    assert result["single_token_decode_fast_path"]
    assert result["general_dispatch_baseline"]["median_ms"] > 0
    assert result["decode_fast_path_speedup"] > 0
    graph_safe = result["graph_safe_batched_candidate"]
    assert graph_safe["speedup_vs_current"] > 0
    assert graph_safe["estimated_selected_weight_mib"] > 0
    assert graph_safe["errors_vs_current"]["max_abs_error"] < 1e-5
    assert not graph_safe["promotion"]["promote_to_runtime"]
    assert not graph_safe["promotion"]["checks"]["cuda_measurement"]


def test_graph_safe_candidate_requires_every_promotion_gate():
    promoted = MODULE.evaluate_graph_safe_moe_candidate(
        device_type="cuda",
        speedup=1.2,
        peak_extra_mib=32.0,
        max_abs_error=0.01,
        min_speedup=1.05,
        max_peak_extra_mib=64.0,
        max_allowed_abs_error=0.05,
    )
    too_slow = MODULE.evaluate_graph_safe_moe_candidate(
        device_type="cuda",
        speedup=1.0,
        peak_extra_mib=32.0,
        max_abs_error=0.01,
        min_speedup=1.05,
        max_peak_extra_mib=64.0,
        max_allowed_abs_error=0.05,
    )

    assert promoted["promote_to_runtime"]
    assert not too_slow["promote_to_runtime"]
    assert not too_slow["checks"]["speedup"]
