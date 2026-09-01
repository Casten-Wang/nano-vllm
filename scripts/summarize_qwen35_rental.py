"""Summarize one complete Qwen3.5 rental validation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MOE_RUNTIME_MIN_THROUGHPUT_RATIO = 0.99
MOE_RUNTIME_MIN_TPOT_SPEEDUP = 1.02
MOE_RUNTIME_MAX_PEAK_EXTRA_MIB = 64.0
MOE_RUNTIME_MAX_CV = 0.05


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required validation artifact is missing: {path}")
    return json.loads(path.read_text())


def evaluate_moe_runtime_candidate(
    *,
    output_digest_matches: bool,
    throughput_speedup: float,
    tpot_speedup: float,
    peak_memory_delta_mib: float,
    max_coefficient_of_variation: float,
) -> dict:
    checks = {
        "output_parity": output_digest_matches,
        "stable_repeats": (
            max_coefficient_of_variation <= MOE_RUNTIME_MAX_CV
        ),
        "throughput_non_regression": (
            throughput_speedup >= MOE_RUNTIME_MIN_THROUGHPUT_RATIO
        ),
        "tpot_speedup": tpot_speedup >= MOE_RUNTIME_MIN_TPOT_SPEEDUP,
        "peak_memory": (
            peak_memory_delta_mib <= MOE_RUNTIME_MAX_PEAK_EXTRA_MIB
        ),
    }
    return {
        "promote_to_default": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_throughput_ratio": MOE_RUNTIME_MIN_THROUGHPUT_RATIO,
            "min_tpot_speedup": MOE_RUNTIME_MIN_TPOT_SPEEDUP,
            "max_peak_extra_mib": MOE_RUNTIME_MAX_PEAK_EXTRA_MIB,
            "max_coefficient_of_variation": MOE_RUNTIME_MAX_CV,
        },
    }


def summarize_moe_runtime(rows: list[dict]) -> dict[str, dict]:
    baselines = {
        (
            row["tensor_parallel_size"],
            row["recurrent_state_dtype"],
            row["kv_cache_dtype"],
        ): row
        for row in rows
        if row.get("qwen35_moe_decode_backend") == "sorted"
    }
    comparisons = {}
    candidates = [
        row
        for row in rows
        if row.get("qwen35_moe_decode_backend") == "batched"
    ]
    if not candidates:
        raise ValueError("performance matrix contains no batched MoE candidate")
    for candidate in candidates:
        key = (
            candidate["tensor_parallel_size"],
            candidate["recurrent_state_dtype"],
            candidate["kv_cache_dtype"],
        )
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(
                "batched MoE candidate has no matching sorted baseline: "
                f"TP={key[0]}, state={key[1]}, KV={key[2]}"
            )
        baseline_median = baseline["median"]
        candidate_median = candidate["median"]
        stability_metrics = ("output_throughput_tok_s", "avg_tpot_s")
        max_cv = max(
            baseline["coefficient_of_variation"][metric]
            for metric in stability_metrics
        )
        max_cv = max(
            max_cv,
            *(
                candidate["coefficient_of_variation"][metric]
                for metric in stability_metrics
            ),
        )
        tp_name = f"tp{key[0]}"
        output_digest_matches = (
            baseline["generated_token_ids_digest"]
            == candidate["generated_token_ids_digest"]
        )
        throughput_speedup = (
            candidate_median["output_throughput_tok_s"]
            / baseline_median["output_throughput_tok_s"]
        )
        tpot_speedup = (
            baseline_median["avg_tpot_s"]
            / candidate_median["avg_tpot_s"]
        )
        peak_memory_delta_mib = (
            candidate_median["peak_torch_allocated_mib"]
            - baseline_median["peak_torch_allocated_mib"]
        )
        comparisons[tp_name] = {
            "configuration": {
                "recurrent_state_dtype": key[1],
                "kv_cache_dtype": key[2],
            },
            "baseline_label": baseline["label"],
            "candidate_label": candidate["label"],
            "output_digest_matches": output_digest_matches,
            "throughput_speedup": throughput_speedup,
            "tpot_speedup": tpot_speedup,
            "peak_memory_delta_mib": peak_memory_delta_mib,
            "max_coefficient_of_variation": max_cv,
            "promotion": evaluate_moe_runtime_candidate(
                output_digest_matches=output_digest_matches,
                throughput_speedup=throughput_speedup,
                tpot_speedup=tpot_speedup,
                peak_memory_delta_mib=peak_memory_delta_mib,
                max_coefficient_of_variation=max_cv,
            ),
        }
    return comparisons


def summarize(run_dir: Path, run_id: str) -> dict:
    audit = load_json(run_dir / "preflight" / "checkpoint_mapping_audit.json")
    memory = load_json(run_dir / "preflight" / "memory_preflight.json")
    performance = load_json(
        run_dir / "performance" / f"{run_id}_matrix_summary.json"
    )
    quality = load_json(run_dir / "quality" / f"{run_id}_summary.json")
    kernel_paths = sorted((run_dir / "kernels").glob("tp*.json"))
    if not kernel_paths:
        raise ValueError("no kernel benchmark artifacts were found")
    cudagraph_paths = sorted(
        (run_dir / "cudagraph").glob("tp*/run_*/summary.json")
    )
    if not cudagraph_paths:
        raise ValueError("no CUDA Graph parity artifacts were found")

    valid_runs = [
        row
        for row in performance["runs"]
        if row["repeat_output_digests_match"]
        and row["execution_paths_valid"]
        and row["generation_valid"]
    ]
    if not valid_runs:
        raise ValueError("performance matrix contains no valid configurations")
    best_throughput = max(
        valid_runs,
        key=lambda row: row["median"]["output_throughput_tok_s"],
    )
    lowest_memory = min(
        valid_runs,
        key=lambda row: row["median"]["peak_torch_allocated_mib"],
    )
    moe_runtime = summarize_moe_runtime(performance["runs"])

    kernels = {}
    configured_max_decode_batch = performance["workload"]["max_num_seqs"]
    commits = {
        row["commit"]
        for row in performance["runs"]
    }
    clean_worktrees = True
    cuda_measurements = True
    for path in kernel_paths:
        result = load_json(path)
        dispatch_results = result["results"]["expert_dispatch_torch"]
        measured_decode_batches = sorted(
            int(token_count)
            for token_count in dispatch_results
            if int(token_count) <= configured_max_decode_batch
        )
        if not measured_decode_batches or measured_decode_batches[0] != 1:
            raise ValueError(f"kernel benchmark has no batch-1 MoE result: {path}")
        selected_batches = tuple(
            dict.fromkeys((1, measured_decode_batches[-1]))
        )
        candidates_by_batch = {
            str(batch): dispatch_results[str(batch)][
                "graph_safe_batched_candidate"
            ]
            for batch in selected_batches
        }
        candidate = candidates_by_batch["1"]
        tp_name = path.stem
        kernels[tp_name] = {
            "promotion": {
                "promote_to_runtime": all(
                    item["promotion"]["promote_to_runtime"]
                    for item in candidates_by_batch.values()
                ),
                "selected_decode_batches": list(selected_batches),
            },
            "median_ms": candidate["median_ms"],
            "speedup_vs_current": candidate["speedup_vs_current"],
            "peak_extra_mib": candidate["peak_extra_mib"],
            "errors_vs_current": candidate["errors_vs_current"],
            "by_decode_batch": {
                batch: {
                    "promotion": item["promotion"],
                    "median_ms": item["median_ms"],
                    "speedup_vs_current": item["speedup_vs_current"],
                    "peak_extra_mib": item["peak_extra_mib"],
                    "errors_vs_current": item["errors_vs_current"],
                }
                for batch, item in candidates_by_batch.items()
            },
        }
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    cudagraph = {}
    for path in cudagraph_paths:
        result = load_json(path)
        tp_name = path.parents[1].name
        if tp_name in cudagraph:
            raise ValueError(f"multiple CUDA Graph summaries found for {tp_name}")
        cudagraph[tp_name] = {
            "passed": result["passed"],
            "hybrid_graph_captured": result.get(
                "hybrid_graph_captured",
                False,
            ),
            "scenario_count": len(result["scenarios"]),
            "batch_sizes": [
                scenario["batch_size"] for scenario in result["scenarios"]
            ],
        }
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    quality_case_dirs = sorted(
        path
        for path in (run_dir / "quality").glob(f"{run_id}_qwen35_*")
        if path.is_dir()
    )
    quality_case_paths = [
        case_dir / f"{case_dir.name}.json"
        for case_dir in quality_case_dirs
    ]
    if not quality_case_paths:
        raise ValueError("no quality case artifacts were found")
    checkpoint_digests = {performance["workload"]["checkpoint_manifest_digest"]}
    for path in quality_case_paths:
        result = load_json(path)
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        checkpoint_digests.add(result["checkpoint_manifest"]["digest"])

    evidence = {
        "checkpoint_mapping_valid": audit["valid"] and audit["complete"],
        "memory_preflight_valid": memory["valid"],
        "performance_paths_valid": performance["all_execution_paths_valid"],
        "performance_generation_valid": performance["all_generation_valid"],
        "performance_output_parity": performance["all_output_digests_match"],
        "moe_runtime_output_parity": all(
            item["output_digest_matches"] for item in moe_runtime.values()
        ),
        "hybrid_cudagraph_parity": (
            set(cudagraph) == set(kernels)
            and all(
                item["passed"] and item["hybrid_graph_captured"]
                for item in cudagraph.values()
            )
        ),
        "quality_reads_stored_kv": all(
            row["kv_sensitive_token_rows"] > 0 for row in quality["cases"]
        ),
        "cuda_measurements": cuda_measurements,
        "clean_worktrees": clean_worktrees,
        "single_commit": len(commits) == 1,
        "single_checkpoint": len(checkpoint_digests) == 1,
    }
    microbenchmark_promoted = all(
        item["promotion"]["promote_to_runtime"]
        for item in kernels.values()
    )
    runtime_promoted = all(
        item["promotion"]["promote_to_default"]
        for item in moe_runtime.values()
    )
    same_tp_coverage = set(kernels) == set(moe_runtime)
    return {
        "run_id": run_id,
        "model": quality["model"],
        "evidence": evidence,
        "valid": all(evidence.values()),
        "commits": sorted(commits),
        "checkpoint_digests": sorted(checkpoint_digests),
        "performance": {
            "best_throughput": best_throughput,
            "lowest_peak_memory": lowest_memory,
        },
        "quality": {
            "scope": quality["quality_scope"],
            "comparisons_by_tp": quality["comparisons_by_tp"],
        },
        "graph_safe_moe": {
            "all_tp_promoted": (
                same_tp_coverage
                and microbenchmark_promoted
                and runtime_promoted
            ),
            "same_tp_coverage": same_tp_coverage,
            "microbenchmark_all_tp_promoted": microbenchmark_promoted,
            "runtime_all_tp_promoted": runtime_promoted,
            "by_tp": kernels,
            "runtime_by_tp": moe_runtime,
        },
        "hybrid_cudagraph": {
            "all_tp_passed": all(
                item["passed"] and item["hybrid_graph_captured"]
                for item in cudagraph.values()
            ),
            "same_tp_coverage": set(cudagraph) == set(kernels),
            "by_tp": cudagraph,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir, args.run_id)
    output = args.output or args.run_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
