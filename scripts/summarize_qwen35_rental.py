"""Summarize one complete Qwen3.5 rental validation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required validation artifact is missing: {path}")
    return json.loads(path.read_text())


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

    kernels = {}
    commits = {
        row["commit"]
        for row in performance["runs"]
    }
    clean_worktrees = True
    cuda_measurements = True
    for path in kernel_paths:
        result = load_json(path)
        candidate = result["results"]["expert_dispatch_torch"]["1"][
            "graph_safe_batched_candidate"
        ]
        tp_name = path.stem
        kernels[tp_name] = {
            "promotion": candidate["promotion"],
            "median_ms": candidate["median_ms"],
            "speedup_vs_current": candidate["speedup_vs_current"],
            "peak_extra_mib": candidate["peak_extra_mib"],
            "errors_vs_current": candidate["errors_vs_current"],
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
        "quality_reads_stored_kv": all(
            row["kv_sensitive_token_rows"] > 0 for row in quality["cases"]
        ),
        "cuda_measurements": cuda_measurements,
        "clean_worktrees": clean_worktrees,
        "single_commit": len(commits) == 1,
        "single_checkpoint": len(checkpoint_digests) == 1,
    }
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
            "all_tp_promoted": all(
                item["promotion"]["promote_to_runtime"]
                for item in kernels.values()
            ),
            "by_tp": kernels,
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
