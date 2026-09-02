from importlib.util import module_from_spec, spec_from_file_location
from copy import deepcopy
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
        results["int8_partitioned_ps512"] = {
            "status": "ok",
            "median_ms": 0.9,
            "max_abs_diff_vs_flash_reference": 0.015,
            "peak_extra_mib": 3.0,
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


def write_cudagraph_case(root, context_name, attention_path, batch_sizes):
    write(
        root / f"cudagraph/tp4/{context_name}/run_1/summary.json",
        {
            "commit": "abc",
            "git_dirty": False,
            "cuda_available": True,
            "kv_cache_dtype": "int8",
            "passed": True,
            "hybrid_graph_captured": True,
            "scenarios": [
                {
                    "batch_size": batch_size,
                    "comparison": {
                        "expected_attention_path": attention_path,
                        "expected_eager_attention_path": True,
                        "expected_graph_attention_path": True,
                        "scratch_primed_across_bucket": True,
                    },
                }
                for batch_size in batch_sizes
            ],
        },
    )


def write_long_prefill_case(root, *, max_abs_error=0.01):
    measurement = {
        "reference": None,
        "candidate": {"median_ms": 4.0, "peak_extra_mib": 128.0},
        "speedup": None,
        "errors": [{"max_abs_error": max_abs_error}],
        "reused_fp32_output_buffer_mib": 32.0,
        "reused_fp32_pairwise_buffer_mib": 16.0,
    }
    write(
        root / "kernels_long/tp4.json",
        {
            "commit": "abc",
            "git_dirty": False,
            "cuda_available": True,
            "configuration": {
                "prefill_only": True,
                "prefill_batch": 1,
                "prefill_tokens": 8192,
                "tp_size": 4,
                "resolved_device": "cuda",
                "delta_prefill_chunk_sizes": [32, 64, 128],
            },
            "results": {
                "vectorized_prefill_convolution": measurement,
                "grouped_delta_prefill": deepcopy(measurement),
                "grouped_delta_prefill_chunk_sweep": {
                    "baseline_chunk_size": 64,
                    "candidates": {
                        str(chunk_size): {
                            **deepcopy(measurement),
                            "chunk_size": chunk_size,
                            "errors_vs_chunk64": deepcopy(
                                measurement["errors"]
                            ),
                            "candidate": {
                                "median_ms": latency,
                                "peak_extra_mib": memory,
                            },
                        }
                        for chunk_size, latency, memory in (
                            (32, 5.0, 64.0),
                            (64, 4.0, 96.0),
                            (128, 6.0, 160.0),
                        )
                    }
                },
            },
        },
    )


def write_mixed_case(root):
    for repeat, p95 in enumerate((0.019, 0.020, 0.021), start=1):
        write(
            root / f"mixed/tp4/r{repeat}.json",
            {
                "commit": "abc",
                "git_dirty": False,
                "cuda_available": True,
                "tensor_parallel_size": 4,
                "qwen35_moe_decode_backend": "batched",
                "enable_dynamic_chunked_prefill": True,
                "injected": True,
                "initial_p95_decode_gap_s": p95,
                "peak_torch_allocated_mib": 12_000.0,
                "generated_token_ids": {"digest": "mixed-tokens"},
                "execution_stats": {
                    "model_path_counts": {"mixed_eager": 3},
                },
                "execution_validation": {"valid": True},
                "generation_validation": {"valid": True},
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
                    "model_max_position_embeddings": 262_144,
                    "configured_max_model_len": 16_384,
                    "max_state_bytes_per_rank": 1_000,
                    "minimum_workload_kv_bytes_per_rank": 2_000,
                    "kv_bytes_per_token_by_dtype": {
                        "auto": 10_240,
                        "int8": 5_160,
                    },
                    "kv_capacity_by_dtype": {
                        "auto": {
                            "memory_limited_total_token_slots": 100_000,
                            "memory_limited_context_tokens_per_sequence": 1_280,
                            "effective_context_tokens_per_sequence": 1_280,
                        },
                        "int8": {
                            "memory_limited_total_token_slots": 200_000,
                            "memory_limited_context_tokens_per_sequence": 3_072,
                            "effective_context_tokens_per_sequence": 3_072,
                        },
                    },
                    "capacity_concurrent_sequences": 64,
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
    int8_baseline = deepcopy(rows[0])
    int8_baseline.update(label="sorted-int8", kv_cache_dtype="int8")
    int8_candidate = deepcopy(rows[1])
    int8_candidate.update(label="batched-int8", kv_cache_dtype="int8")
    rows.extend((int8_baseline, int8_candidate))
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
            "cases": [
                {
                    "kv_sensitive_token_rows": 2,
                    "int8_partitioned_decode_observed": True,
                }
            ],
            "comparisons_by_tp": {"tp4": {"baseline_decode_ppl": 3.0}},
            "cross_tp": {"all_passed": True, "comparisons": []},
            "quality_gates": {"all_passed": True, "thresholds": {}},
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
            "results": {
                "expert_dispatch_torch": {
                    batch: {"graph_safe_batched_candidate": {
                        "promotion": {"promote_to_runtime": True},
                        "median_ms": 1.0,
                        "speedup_vs_current": 1.2,
                        "peak_extra_mib": 4.0,
                        "errors_vs_current": {"max_abs_error": 0.01},
                    }}
                    for batch in ("1", "64")
                },
                "rmsnorm_fp32_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_fp32_copy_mib": 4.0,
                    "candidate_reuses_fp32_workspace": True,
                },
                "gated_rmsnorm_fp32_reuse": {
                    "reference": {"peak_extra_mib": 12.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_hidden_fp32_workspace_mib": 4.0,
                    "reused_gate_fp32_workspace_mib": 4.0,
                    "candidate_reuses_fp32_workspaces": True,
                },
                "moe_output_buffer_reuse": {
                    "reference": {"peak_extra_mib": 12.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_routed_output_mib": 4.0,
                    "reused_shared_output_mib": 4.0,
                    "reused_gate_mib": 0.01,
                },
                "residual_output_buffer_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_branch_output_mib_per_merge": 4.0,
                    "residual_merges_per_decoder_layer": 2,
                },
                "torch_kv_dequant_buffer_reuse": {
                    "reference": {"peak_extra_mib": 32.0},
                    "candidate": {"peak_extra_mib": 16.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_output_workspace_mib": 16.0,
                },
                "specialized_delta_decode": {
                    "reference": {"peak_extra_mib": 24.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_recurrent_state_mib": 8.0,
                    "avoided_full_state_intermediates": 2,
                },
            },
        },
    )
    write_cudagraph_case(
        tmp_path,
        "short",
        "int8_fused_decode",
        [3, 9, 64],
    )
    write_cudagraph_case(
        tmp_path,
        "long",
        "int8_partitioned_decode",
        [3],
    )
    write_attention_case(tmp_path, "short", 4096, partitioned=False)
    write_attention_case(tmp_path, "long", 16384, partitioned=True)
    write_long_prefill_case(tmp_path)
    write_mixed_case(tmp_path)

    report = MODULE.summarize(tmp_path, run_id)

    assert report["valid"]
    assert report["performance"]["best_throughput"]["label"] == "batched"
    assert report["performance"]["lowest_peak_memory"]["label"] == "sorted"
    assert report["graph_safe_moe"]["all_tp_promoted"]
    assert report["hybrid_cudagraph"]["all_tp_passed"]
    assert report["hybrid_cudagraph"]["by_tp"]["tp4"]["short"][
        "scratch_isolation_observed"
    ]
    assert report["evidence"]["long_prefill_kernel_evidence"]
    assert report["evidence"]["mixed_workload_evidence"]
    assert report["evidence"]["normalization_workspace_evidence"]
    assert report["evidence"]["buffer_reuse_evidence"]
    assert report["long_prefill"]["by_tp"]["tp4"]["valid"]
    chunk_sweep = report["long_prefill"]["by_tp"]["tp4"]["chunk_sweep"]
    assert chunk_sweep["fastest_chunk_size"] == 64
    assert chunk_sweep["lowest_memory_chunk_size"] == 32
    mixed = report["mixed_workload"]["by_tp"]["tp4"]
    assert mixed["repeat_count"] == 3
    assert mixed["median_mixed_steps"] == 3
    assert mixed["initial_p95_decode_gap_cv"] < 0.1
    assert report["mixed_workload"]["cross_tp_output_parity"]
    assert report["normalization"]["by_tp"]["tp4"]["rmsnorm"][
        "peak_extra_mib_delta"
    ] == -4.0
    assert report["normalization"]["by_tp"]["tp4"]["gated_rmsnorm"][
        "workspace"
    ]["reused_gate_fp32_workspace_mib"] == 4.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["torch_kv_dequant"][
        "workspace"
    ]["avoided_output_workspace_mib"] == 16.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["recurrent_decode"][
        "metadata"
    ]["avoided_full_state_intermediates"] == 2
    assert report["graph_safe_moe"]["by_tp"]["tp4"]["promotion"][
        "selected_decode_batches"
    ] == [1, 64]
    runtime = report["graph_safe_moe"]["runtime_by_tp"]["tp4"]["auto"]
    assert runtime["output_digest_matches"]
    assert runtime["throughput_speedup"] == 2.0
    assert runtime["tpot_speedup"] == 2.0
    assert runtime["peak_memory_delta_mib"] == 4
    assert runtime["promotion"]["promote_to_default"]
    assert report["graph_safe_moe"]["runtime_by_tp"]["tp4"]["int8"][
        "promotion"
    ]["promote_to_default"]
    attention = report["int8_attention"]["by_tp"]["tp4"]
    assert attention["short"]["best_fused"]["speedup_vs_flash_reference"] == 2.0
    assert attention["long"]["best_partitioned"]["backend"] == (
        "int8_partitioned_ps256"
    )
    assert attention["long"]["production_partitioned"]["backend"] == (
        "int8_partitioned_ps512"
    )
    memory = report["memory"]["by_tp"]["tp4"]
    assert memory["int8_kv_reduction_ratio"] == 0.49609375
    assert memory["minimum_budget_margin_bytes"] == 5_000
    assert memory["kv_capacity_by_dtype"]["int8"][
        "memory_limited_context_tokens_per_sequence"
    ] == 3_072

    mixed_path = tmp_path / "mixed/tp4/r1.json"
    mixed_result = json.loads(mixed_path.read_text())
    mixed_result["initial_p95_decode_gap_s"] = 0.2
    write(mixed_path, mixed_result)
    unstable_report = MODULE.summarize(tmp_path, run_id)
    unstable_mixed = unstable_report["mixed_workload"]["by_tp"]["tp4"]
    assert unstable_mixed["initial_p95_decode_gap_cv"] > 0.1
    assert not unstable_mixed["valid"]
    assert not unstable_report["evidence"]["mixed_workload_evidence"]

    mixed_result["initial_p95_decode_gap_s"] = 0.019
    mixed_result.pop("generated_token_ids")
    write(mixed_path, mixed_result)
    invalid_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_report["evidence"]["mixed_workload_evidence"]
    assert not invalid_report["valid"]


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


def test_long_context_requires_production_partition_size():
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 1.0,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 1.0,
            },
            "int8_partitioned_ps256": {
                "status": "ok",
                "median_ms": 0.8,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 2.0,
            },
        },
        "context_len": 16_384,
        "batch_size": 4,
    }

    summary = MODULE.summarize_attention_case(result, partitioned=True)

    assert not summary["partitioned_correctness_valid"]
    assert summary["production_partitioned"] is None


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


def test_long_prefill_rejects_inaccurate_kernel():
    result = {
        "configuration": {
            "prefill_only": True,
            "prefill_batch": 1,
            "prefill_tokens": 8192,
            "tp_size": 4,
            "resolved_device": "cuda:0",
            "delta_prefill_chunk_sizes": [32, 64],
        },
        "results": {
            name: {
                "candidate": {"median_ms": 1.0, "peak_extra_mib": 2.0},
                "errors": [{"max_abs_error": error}],
            }
            for name, error in (
                ("vectorized_prefill_convolution", 0.01),
                ("grouped_delta_prefill", 0.051),
            )
        } | {
            "grouped_delta_prefill_chunk_sweep": {
                "baseline_chunk_size": 64,
                "candidates": {
                    str(chunk_size): {
                        "chunk_size": chunk_size,
                        "candidate": {
                            "median_ms": 1.0,
                            "peak_extra_mib": 2.0,
                        },
                        "errors_vs_chunk64": [{"max_abs_error": 0.01}],
                    }
                    for chunk_size in (32, 64)
                }
            }
        },
    }

    summary = MODULE.summarize_long_prefill(result, expected_tp_size=4)

    assert not summary["valid"]
    assert not summary["cases"]["grouped_delta_prefill"]["valid"]


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


def test_buffer_reuse_summary_rejects_missing_workspace_metric():
    result = {
        "reference": {"peak_extra_mib": 2.0},
        "candidate": {"peak_extra_mib": 1.0},
        "speedup": 1.1,
        "errors": [{"max_abs_error": 0.0}],
    }

    with pytest.raises(ValueError, match="missing workspace metrics"):
        MODULE.summarize_buffer_reuse_candidate(
            result,
            ("reused_workspace_mib",),
        )


def test_buffer_reuse_summary_preserves_failed_accuracy_evidence():
    result = {
        "reference": {"peak_extra_mib": 2.0},
        "candidate": {"peak_extra_mib": 1.0},
        "speedup": 1.1,
        "errors": [{"max_abs_error": 0.051}],
        "reused_workspace_mib": 1.0,
        "expected_count": 2,
    }

    summary = MODULE.summarize_buffer_reuse_candidate(
        result,
        ("reused_workspace_mib",),
        {"expected_count": 2},
    )

    assert not summary["valid"]
    assert summary["max_abs_error"] == 0.051
    assert summary["peak_extra_mib_delta"] == -1.0
