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
        "all_output_digests_match": comparison["all_output_digests_match"],
        "all_execution_paths_valid": comparison["all_execution_paths_valid"],
        "statistics": {
            name: distribution(values) for name, values in metrics.items()
        },
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
    args = parser.parse_args()
    labels = [path.stem for path in args.results]
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
    if not comparison["all_execution_paths_valid"]:
        raise SystemExit(2)
    if args.require_output_parity and not comparison["all_output_digests_match"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
