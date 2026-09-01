"""Compare reproducible nano-vLLM benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


WORKLOAD_FIELDS = (
    "num_seqs",
    "input_len",
    "output_len",
    "seed",
    "vocab_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "warmup",
)

ENVIRONMENT_FIELDS = (
    "device",
    "device_capability",
    "cuda_device_count",
    "torch_version",
    "cuda_version",
    "transformers_version",
    "triton_version",
    "flash_attn_version",
    "nvidia_smi_gpus",
)

OPTIMIZATION_FIELDS = (
    "tensor_parallel_size",
    "recurrent_state_dtype",
    "kv_cache_dtype",
    "kv_dequant_backend",
    "int8_partitioned_decode_threshold",
    "int8_partitioned_decode_partition_size",
    "sliding_window_size",
    "enable_dynamic_chunked_prefill",
    "enforce_eager",
)


def ratio(value: float, baseline: float) -> float | None:
    return value / baseline if baseline else None


def distribution(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "population_stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else None,
    }


def load_result(path: Path) -> dict:
    with path.open() as handle:
        result = json.load(handle)
    missing = [
        field
        for field in (
            *WORKLOAD_FIELDS,
            *ENVIRONMENT_FIELDS,
            "model",
            "commit",
            "git_dirty",
            "checkpoint_manifest",
            *OPTIMIZATION_FIELDS,
            "output_throughput_tok_s",
            "peak_torch_allocated_mib",
            "generated_token_ids",
            "execution_validation",
            "generation_validation",
            "metrics",
        )
        if field not in result
    ]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    checkpoint_manifest = result["checkpoint_manifest"]
    if not isinstance(checkpoint_manifest, dict) or not isinstance(
        checkpoint_manifest.get("digest"), str
    ):
        raise ValueError(f"{path} has an invalid checkpoint manifest")
    if result["git_dirty"]:
        raise ValueError(
            f"{path} was captured from a dirty worktree and is not reproducible"
        )
    if not isinstance(result["commit"], str) or result["commit"] == "unknown":
        raise ValueError(f"{path} has no reproducible Git commit identity")
    return result


def compare_results(results: list[dict], labels: list[str]) -> dict:
    if len(results) < 2 or len(results) != len(labels):
        raise ValueError("at least two results with matching labels are required")
    dirty_runs = [
        label for label, result in zip(labels, results) if result.get("git_dirty")
    ]
    if dirty_runs:
        raise ValueError(
            "benchmark runs are not reproducible because their worktrees "
            f"were dirty: {', '.join(dirty_runs)}"
        )
    baseline = results[0]
    baseline_checkpoint = baseline["checkpoint_manifest"]["digest"]
    mismatches = []
    for label, result in zip(labels[1:], results[1:]):
        for field in ("model", *WORKLOAD_FIELDS, *ENVIRONMENT_FIELDS):
            if result[field] != baseline[field]:
                mismatches.append(
                    {
                        "run": label,
                        "field": field,
                        "baseline": baseline[field],
                        "actual": result[field],
                    }
                )
        if result["checkpoint_manifest"]["digest"] != baseline_checkpoint:
            mismatches.append(
                {
                    "run": label,
                    "field": "checkpoint_manifest.digest",
                    "baseline": baseline_checkpoint,
                    "actual": result["checkpoint_manifest"]["digest"],
                }
            )
    if mismatches:
        fields = ", ".join(
            f"{item['run']}.{item['field']}" for item in mismatches
        )
        raise ValueError(f"benchmark workloads are not comparable: {fields}")

    baseline_throughput = baseline["output_throughput_tok_s"]
    baseline_ttft = baseline["metrics"]["avg_ttft_s"]
    baseline_tpot = baseline["metrics"]["avg_tpot_s"]
    baseline_memory = baseline["peak_torch_allocated_mib"]
    baseline_digest = baseline["generated_token_ids"]["digest"]
    rows = []
    for label, result in zip(labels, results):
        throughput = result["output_throughput_tok_s"]
        ttft = result["metrics"]["avg_ttft_s"]
        tpot = result["metrics"]["avg_tpot_s"]
        memory = result["peak_torch_allocated_mib"]
        rows.append(
            {
                "label": label,
                **{field: result[field] for field in OPTIMIZATION_FIELDS},
                "output_throughput_tok_s": throughput,
                "throughput_vs_baseline": ratio(throughput, baseline_throughput),
                "avg_ttft_s": ttft,
                "ttft_vs_baseline": ratio(ttft, baseline_ttft),
                "avg_tpot_s": tpot,
                "tpot_vs_baseline": ratio(tpot, baseline_tpot),
                "peak_torch_allocated_mib": memory,
                "peak_memory_vs_baseline": ratio(memory, baseline_memory),
                "output_digest_matches_baseline": (
                    result["generated_token_ids"]["digest"] == baseline_digest
                ),
                "execution_valid": result["execution_validation"]["valid"],
                "generation_valid": result["generation_validation"]["valid"],
            }
        )
    return {
        "baseline": labels[0],
        "workload": {
            "model": baseline["model"],
            "checkpoint_manifest_digest": baseline_checkpoint,
            **{field: baseline[field] for field in WORKLOAD_FIELDS},
        },
        "commits": [result["commit"] for result in results],
        "checkpoint_identity_strength": baseline["checkpoint_manifest"].get(
            "strength", "unknown"
        ),
        "environment": {
            field: baseline[field] for field in ENVIRONMENT_FIELDS
        },
        "all_output_digests_match": all(
            row["output_digest_matches_baseline"] for row in rows
        ),
        "all_execution_paths_valid": all(row["execution_valid"] for row in rows),
        "all_generation_valid": all(row["generation_valid"] for row in rows),
        "runs": rows,
    }


def summarize_repeats(results: list[dict], labels: list[str]) -> dict:
    comparison = compare_results(results, labels)
    baseline = results[0]
    mismatches = []
    for label, result in zip(labels[1:], results[1:]):
        for field in ("commit", *OPTIMIZATION_FIELDS):
            if result[field] != baseline[field]:
                mismatches.append(f"{label}.{field}")
    if mismatches:
        raise ValueError(
            "benchmark repeats do not share one implementation: "
            + ", ".join(mismatches)
        )

    metrics = {
        "output_throughput_tok_s": [
            result["output_throughput_tok_s"] for result in results
        ],
        "avg_ttft_s": [result["metrics"]["avg_ttft_s"] for result in results],
        "avg_tpot_s": [result["metrics"]["avg_tpot_s"] for result in results],
        "avg_request_latency_s": [
            result["metrics"]["avg_request_latency_s"] for result in results
        ],
        "peak_torch_allocated_mib": [
            result["peak_torch_allocated_mib"] for result in results
        ],
    }
    return {
        "commit": baseline["commit"],
        "workload": comparison["workload"],
        "environment": comparison["environment"],
        "configuration": {
            field: baseline[field] for field in OPTIMIZATION_FIELDS
        },
        "checkpoint_identity_strength": comparison[
            "checkpoint_identity_strength"
        ],
        "generated_token_ids_digest": baseline["generated_token_ids"]["digest"],
        "all_output_digests_match": comparison["all_output_digests_match"],
        "all_execution_paths_valid": comparison["all_execution_paths_valid"],
        "all_generation_valid": comparison["all_generation_valid"],
        "statistics": {
            name: distribution(values) for name, values in metrics.items()
        },
    }


def compare_repeat_summaries(summaries: list[dict], labels: list[str]) -> dict:
    if len(summaries) < 2 or len(summaries) != len(labels):
        raise ValueError("at least two summaries with matching labels are required")
    baseline = summaries[0]
    mismatches = []
    for label, summary in zip(labels[1:], summaries[1:]):
        for field in ("workload", "environment"):
            if summary.get(field) != baseline.get(field):
                mismatches.append(f"{label}.{field}")
    if mismatches:
        raise ValueError(
            "benchmark summaries are not comparable: " + ", ".join(mismatches)
        )

    metric_names = (
        "output_throughput_tok_s",
        "avg_ttft_s",
        "avg_tpot_s",
        "avg_request_latency_s",
        "peak_torch_allocated_mib",
    )
    rows = []
    for label, summary in zip(labels, summaries):
        statistics_by_name = summary["statistics"]
        rows.append(
            {
                "label": label,
                **summary["configuration"],
                "commit": summary["commit"],
                "generated_token_ids_digest": summary[
                    "generated_token_ids_digest"
                ],
                "repeat_output_digests_match": summary[
                    "all_output_digests_match"
                ],
                "execution_paths_valid": summary[
                    "all_execution_paths_valid"
                ],
                "generation_valid": summary["all_generation_valid"],
                "median": {
                    name: statistics_by_name[name]["median"]
                    for name in metric_names
                },
                "coefficient_of_variation": {
                    name: statistics_by_name[name]["coefficient_of_variation"]
                    for name in metric_names
                },
            }
        )
    baseline_median = rows[0]["median"]
    for row in rows:
        median = row["median"]
        row["vs_baseline"] = {
            "output_throughput": ratio(
                median["output_throughput_tok_s"],
                baseline_median["output_throughput_tok_s"],
            ),
            "ttft": ratio(
                median["avg_ttft_s"],
                baseline_median["avg_ttft_s"],
            ),
            "tpot": ratio(
                median["avg_tpot_s"],
                baseline_median["avg_tpot_s"],
            ),
            "request_latency": ratio(
                median["avg_request_latency_s"],
                baseline_median["avg_request_latency_s"],
            ),
            "peak_memory": ratio(
                median["peak_torch_allocated_mib"],
                baseline_median["peak_torch_allocated_mib"],
            ),
        }
    digests = {row["generated_token_ids_digest"] for row in rows}
    return {
        "baseline": labels[0],
        "workload": baseline["workload"],
        "environment": baseline["environment"],
        "checkpoint_identity_strength": baseline[
            "checkpoint_identity_strength"
        ],
        "all_output_digests_match": len(digests) == 1,
        "all_repeat_output_digests_match": all(
            row["repeat_output_digests_match"] for row in rows
        ),
        "all_execution_paths_valid": all(
            row["execution_paths_valid"] for row in rows
        ),
        "all_generation_valid": all(row["generation_valid"] for row in rows),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; result files should remain ignored.",
    )
    parser.add_argument("--require-output-parity", action="store_true")
    parser.add_argument(
        "--summarize-repeats",
        action="store_true",
        help="Require one identical implementation and summarize repeated runs.",
    )
    parser.add_argument(
        "--compare-repeat-summaries",
        action="store_true",
        help="Compare per-configuration repeat summaries from one matrix.",
    )
    args = parser.parse_args()
    labels = [path.stem for path in args.results]
    if args.summarize_repeats and args.compare_repeat_summaries:
        raise SystemExit("summary modes are mutually exclusive")
    if args.compare_repeat_summaries:
        summaries = [json.loads(path.read_text()) for path in args.results]
        comparison = compare_repeat_summaries(summaries, labels)
    else:
        results = [load_result(path) for path in args.results]
        comparison = (
            summarize_repeats(results, labels)
            if args.summarize_repeats
            else compare_results(results, labels)
        )
    rendered = json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if (
        not comparison["all_execution_paths_valid"]
        or not comparison["all_generation_valid"]
    ):
        raise SystemExit(2)
    if args.require_output_parity and not comparison["all_output_digests_match"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
