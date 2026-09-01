from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


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


def test_summary_selects_valid_performance_and_preserves_evidence(tmp_path):
    run_id = "rental-a"
    write(tmp_path / "preflight/checkpoint_mapping_audit.json", {"valid": True, "complete": True})
    write(tmp_path / "preflight/memory_preflight.json", {"valid": True})
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
            "scenarios": [
                {"batch_size": 3},
                {"batch_size": 9},
                {"batch_size": 64},
            ],
        },
    )

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
