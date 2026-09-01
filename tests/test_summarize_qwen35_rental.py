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
            "label": "fast",
            "commit": "abc",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {"output_throughput_tok_s": 20, "peak_torch_allocated_mib": 12},
        },
        {
            "label": "small",
            "commit": "abc",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {"output_throughput_tok_s": 10, "peak_torch_allocated_mib": 8},
        },
    ]
    write(
        tmp_path / f"performance/{run_id}_matrix_summary.json",
        {
            "commits": ["abc"],
            "workload": {"checkpoint_manifest_digest": "weights"},
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
            "results": {"expert_dispatch_torch": {"1": {"graph_safe_batched_candidate": {
                "promotion": {"promote_to_runtime": True},
                "median_ms": 1.0,
                "speedup_vs_current": 1.2,
                "peak_extra_mib": 4.0,
                "errors_vs_current": {"max_abs_error": 0.01},
            }}}},
        },
    )

    report = MODULE.summarize(tmp_path, run_id)

    assert report["valid"]
    assert report["performance"]["best_throughput"]["label"] == "fast"
    assert report["performance"]["lowest_peak_memory"]["label"] == "small"
    assert report["graph_safe_moe"]["all_tp_promoted"]
