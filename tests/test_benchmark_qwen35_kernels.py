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


def test_rmsnorm_benchmark_compares_workspace_reuse_to_baseline():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_rmsnorm(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["candidate_reuses_fp32_workspace"]
    assert result["avoided_fp32_copy_mib"] == 8 * 16 * 4 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0


def test_gated_rmsnorm_benchmark_compares_both_reused_workspaces():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_gated_rmsnorm(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    workspace_mib = 8 * 16 * 4 / 1024 / 1024
    assert result["candidate_reuses_fp32_workspaces"]
    assert result["reused_hidden_fp32_workspace_mib"] == workspace_mib
    assert result["reused_gate_fp32_workspace_mib"] == workspace_mib
    assert result["errors"][0]["max_abs_error"] == 0


def test_moe_output_merge_benchmark_measures_buffer_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_moe_output_merge(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    output_mib = 8 * 16 * 2 / 1024 / 1024
    assert result["reused_routed_output_mib"] == output_mib
    assert result["reused_shared_output_mib"] == output_mib
    assert result["reused_gate_mib"] == 8 * 2 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0


def test_residual_merge_benchmark_measures_branch_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_residual_merge(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["reused_branch_output_mib_per_merge"] == (
        8 * 16 * 2 / 1024 / 1024
    )
    assert result["residual_merges_per_decoder_layer"] == 2
    assert result["errors"][0]["max_abs_error"] == 0


def test_torch_kv_dequant_benchmark_measures_output_reuse():
    args = SimpleNamespace(
        prefill_tokens=256,
        key_head_dim=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_torch_kv_dequant(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        2,
    )

    assert result["selected_blocks"] == 1
    assert result["avoided_output_workspace_mib"] == (
        2 * 256 * 2 * 8 * 2 / 1024 / 1024
    )
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


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
        moe_decode_chunk_sizes=(1, 2),
        max_decode_tokens=64,
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
    assert set(result["graph_safe_chunk_sweep"]["candidates"]) == {"1", "2"}


def test_chunk_recommendation_uses_worst_decode_batch_speedup():
    def candidate(speedup, promoted=True):
        return {
            "promotion": {"promote_to_runtime": promoted},
            "speedup_vs_current": speedup,
            "peak_extra_mib": 4.0,
            "median_ms": 1.0 / speedup,
        }

    results = {
        "1": {"graph_safe_chunk_sweep": {"candidates": {
            "4": candidate(1.3),
            "8": candidate(1.2),
            "16": candidate(1.4, promoted=False),
        }}},
        "64": {"graph_safe_chunk_sweep": {"candidates": {
            "4": candidate(1.1),
            "8": candidate(1.15),
            "16": candidate(1.5),
        }}},
        "128": {"tokens": 128},
    }

    recommendation = MODULE.recommend_moe_decode_chunk_size(results, 64)

    assert recommendation["measured_decode_batches"] == [1, 64]
    assert recommendation["recommended_chunk_size"] == 8
    assert not recommendation["candidates"]["16"]["all_batches_promoted"]


def test_delta_prefill_chunk_sweep_compares_shared_input_to_chunk64():
    args = SimpleNamespace(
        delta_prefill_chunk_sizes=(4, 2, 4),
        prefill_batch=1,
        prefill_tokens=4,
        key_head_dim=2,
        value_head_dim=2,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_prefill_chunk_sweep(
        args,
        torch.device("cpu"),
        torch.float32,
        2,
        4,
    )

    assert result["baseline_chunk_size"] == 64
    assert list(result["candidates"]) == ["2", "4"]
    for chunk_size, candidate in result["candidates"].items():
        assert candidate["chunk_size"] == int(chunk_size)
        assert candidate["candidate"]["median_ms"] > 0
        assert max(
            item["max_abs_error"]
            for item in candidate["errors_vs_chunk64"]
        ) < 1e-4


def test_delta_decode_benchmark_records_state_workspace_reuse():
    args = SimpleNamespace(
        decode_batch=2,
        key_head_dim=4,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_decode(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        2,
        6,
    )

    assert result["reused_recurrent_state_mib"] == (
        2 * 6 * 4 * 3 * 4 / 1024 / 1024
    )
    assert result["avoided_full_state_intermediates"] == 2
    assert all(item["max_abs_error"] < 1e-4 for item in result["errors"])


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
