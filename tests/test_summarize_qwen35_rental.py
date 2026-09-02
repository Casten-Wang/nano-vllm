from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "summarize_qwen35_rental",
    ROOT / "scripts" / "summarize_qwen35_rental.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_attention_case(
    root,
    name,
    context_len,
    *,
    partitioned,
    max_abs_diff=0.01,
):
    results = {
        "flash_reference": {
            "status": "ok",
            "median_ms": 2.0,
        },
        "int8_v3_bt256_w8_s2": {
            "status": "ok",
            "median_ms": 1.0,
            "max_abs_diff_vs_flash_reference": max_abs_diff,
            "peak_extra_mib": 2.0,
        },
    }
    if partitioned:
        results["int8_partitioned_ps256"] = {
            "status": "ok",
            "median_ms": 0.8,
            "max_abs_diff_vs_flash_reference": 0.02,
            "peak_extra_mib": 4.0,
        }
    write(
        root / f"attention/tp4/{name}.json",
        {
            "commit": "abc",
            "git_dirty": False,
            "cuda_available": True,
            "batch_size": 4,
            "context_len": context_len,
            "num_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 256,
            "results": results,
        },
    )


def test_summary_selects_valid_performance_and_preserves_evidence(tmp_path):
    run_id = "rental-a"
    write(tmp_path / "preflight/checkpoint_mapping_audit.json", {"valid": True, "complete": True})
    write(
        tmp_path / "preflight/memory_preflight.json",
        {
            "valid": True,
            "results": {
                "tp4": {
                    "local_parameter_bytes": 16_000,
                    "max_state_bytes_per_rank": 1_000,
                    "minimum_workload_kv_bytes_per_rank": 2_000,
                    "kv_bytes_per_token_by_dtype": {
                        "auto": 10_240,
                        "int8": 5_160,
                    },
                    "required_free_bytes_per_rank": 20_000,
                    "available_budget_bytes_by_rank": [25_000] * 4,
                }
            },
        },
    )
    rows = [
        {
            "label": "sorted",
            "commit": "abc",
            "tensor_parallel_size": 4,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "qwen35_moe_decode_backend": "sorted",
            "generated_token_ids_digest": "tokens",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {
                "output_throughput_tok_s": 10,
                "avg_tpot_s": 0.2,
                "peak_torch_allocated_mib": 8,
            },
            "coefficient_of_variation": {
                "output_throughput_tok_s": 0.01,
                "avg_tpot_s": 0.02,
            },
        },
        {
            "label": "batched",
            "commit": "abc",
            "tensor_parallel_size": 4,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "qwen35_moe_decode_backend": "batched",
            "generated_token_ids_digest": "tokens",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {
                "output_throughput_tok_s": 20,
                "avg_tpot_s": 0.1,
                "peak_torch_allocated_mib": 12,
            },
            "coefficient_of_variation": {
                "output_throughput_tok_s": 0.02,
                "avg_tpot_s": 0.01,
            },
        },
    ]
    write(
        tmp_path / f"performance/{run_id}_matrix_summary.json",
        {
            "commits": ["abc"],
            "workload": {
                "checkpoint_manifest_digest": "weights",
                "max_num_seqs": 64,
            },
            "all_execution_paths_valid": True,
            "all_generation_valid": True,
            "all_output_digests_match": True,
            "runs": rows,
        },
    )
    write(
        tmp_path / f"quality/{run_id}_summary.json",
        {
            "model": "/model",
            "quality_scope": "decode",
            "cases": [{"kv_sensitive_token_rows": 2}],
            "comparisons_by_tp": {"tp4": {"baseline_decode_ppl": 3.0}},
        },
    )
    quality_dir = tmp_path / f"quality/{run_id}_qwen35_tp4"
    write(
        quality_dir / f"{quality_dir.name}.json",
        {"commit": "abc", "git_dirty": False, "checkpoint_manifest": {"digest": "weights"}},
    )
    write(quality_dir / "batch0_len128_cases.json", [{"prompt_ids": [1, 2]}])
    write(
        tmp_path / "kernels/tp4.json",
        {
            "commit": "abc",
            "git_dirty": False,
            "cuda_available": True,
            "results": {"expert_dispatch_torch": {
                batch: {"graph_safe_batched_candidate": {
                    "promotion": {"promote_to_runtime": True},
                    "median_ms": 1.0,
                    "speedup_vs_current": 1.2,
                    "peak_extra_mib": 4.0,
                    "errors_vs_current": {"max_abs_error": 0.01},
                }}
                for batch in ("1", "64")
            }},
        },
    )
    write(
        tmp_path / "cudagraph/tp4/run_1/summary.json",
        {
            "commit": "abc",
            "git_dirty": False,
            "cuda_available": True,
            "passed": True,
            "hybrid_graph_captured": True,
            "scenarios": [
                {"batch_size": 3},
                {"batch_size": 9},
                {"batch_size": 64},
            ],
        },
    )
    write_attention_case(tmp_path, "short", 4096, partitioned=False)
    write_attention_case(tmp_path, "long", 16384, partitioned=True)

    report = MODULE.summarize(tmp_path, run_id)

    assert report["valid"]
    assert report["performance"]["best_throughput"]["label"] == "batched"
    assert report["performance"]["lowest_peak_memory"]["label"] == "sorted"
    assert report["graph_safe_moe"]["all_tp_promoted"]
    assert report["hybrid_cudagraph"]["all_tp_passed"]
    assert report["graph_safe_moe"]["by_tp"]["tp4"]["promotion"][
        "selected_decode_batches"
    ] == [1, 64]
    runtime = report["graph_safe_moe"]["runtime_by_tp"]["tp4"]
    assert runtime["output_digest_matches"]
    assert runtime["throughput_speedup"] == 2.0
    assert runtime["tpot_speedup"] == 2.0
    assert runtime["peak_memory_delta_mib"] == 4
    assert runtime["promotion"]["promote_to_default"]
    attention = report["int8_attention"]["by_tp"]["tp4"]
    assert attention["short"]["best_fused"]["speedup_vs_flash_reference"] == 2.0
    assert attention["long"]["best_partitioned"]["backend"] == (
        "int8_partitioned_ps256"
    )
    memory = report["memory"]["by_tp"]["tp4"]
    assert memory["int8_kv_reduction_ratio"] == 0.49609375
    assert memory["minimum_budget_margin_bytes"] == 5_000


def test_summary_rejects_inaccurate_attention_kernel(tmp_path):
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 0.5,
                "max_abs_diff_vs_flash_reference": 0.051,
                "peak_extra_mib": 1.0,
            },
        },
        "context_len": 4096,
        "batch_size": 4,
    }

    summary = MODULE.summarize_attention_case(result, partitioned=False)

    assert not summary["fused_correctness_valid"]
    assert summary["best_fused"] is None
    assert summary["max_allowed_abs_error"] == 0.05


def test_runtime_promotion_rejects_unstable_or_regressing_candidate():
    result = MODULE.evaluate_moe_runtime_candidate(
        output_digest_matches=True,
        throughput_speedup=0.98,
        tpot_speedup=1.10,
        peak_memory_delta_mib=4.0,
        max_coefficient_of_variation=0.06,
    )

    assert not result["promote_to_default"]
    assert not result["checks"]["throughput_non_regression"]
    assert not result["checks"]["stable_repeats"]


def test_memory_summary_rejects_non_saving_int8_cache():
    report = {
        "results": {
            "tp4": {
                "kv_bytes_per_token_by_dtype": {
                    "auto": 100,
                    "int8": 100,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="does not reduce memory"):
        MODULE.summarize_memory_preflight(report)
