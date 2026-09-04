"""Compare reproducible nano-vLLM benchmark result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.benchmark_metadata import validate_ranked_records


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
    "cuda_devices",
    "torch_version",
    "cuda_version",
    "nccl_version",
    "transformers_version",
    "triton_version",
    "flash_attn_version",
    "nvidia_smi_gpus",
    "nvidia_smi_topology",
    "cuda_visible_devices",
    "cuda_device_order",
    "nccl_environment",
)

OPTIMIZATION_FIELDS = (
    "tensor_parallel_size",
    "recurrent_state_dtype",
    "qwen35_decode_conv_backend",
    "qwen35_moe_decode_backend",
    "qwen35_moe_decode_chunk_size",
    "sampling_chunk_size",
    "quantization_format",
    "requested_weight_quant_backend",
    "weight_quant_backend",
    "kv_cache_dtype",
    "kv_dequant_backend",
    "int8_partitioned_decode_threshold",
    "int8_partitioned_decode_partition_size",
    "sliding_window_size",
    "enable_dynamic_chunked_prefill",
    "enforce_eager",
)

STORAGE_FIELDS = (
    "model_parameter_storage",
    "model_parameter_storage_by_rank",
    "model_parameter_total_all_ranks_bytes",
    "kv_cache_storage",
    "kv_cache_storage_by_rank",
    "num_kvcache_blocks",
    "recurrent_state_storage",
    "recurrent_state_storage_by_rank",
    "recurrent_state_total_all_ranks_bytes",
    "runtime_buffer_storage",
    "runtime_buffer_storage_by_rank",
    "runtime_buffer_total_all_ranks_bytes",
)

LATENCY_METRIC_NAMES = (
    "avg_ttft_s",
    "p50_ttft_s",
    "p95_ttft_s",
    "p99_ttft_s",
    "avg_tpot_s",
    "p50_tpot_s",
    "p95_tpot_s",
    "p99_tpot_s",
    "avg_request_latency_s",
    "p50_request_latency_s",
    "p95_request_latency_s",
    "p99_request_latency_s",
)

INPUT_PREPARATION_STEP_KINDS = ("prefill", "decode", "mixed")

RANKED_STORAGE_TOTALS = (
    (
        "model_parameter_storage_by_rank",
        "total_bytes_local_rank",
        "model_parameter_total_all_ranks_bytes",
    ),
    (
        "recurrent_state_storage_by_rank",
        "total_bytes_local_rank",
        "recurrent_state_total_all_ranks_bytes",
    ),
    (
        "runtime_buffer_storage_by_rank",
        "total_bytes_local_rank",
        "runtime_buffer_total_all_ranks_bytes",
    ),
)


def ratio(value: float, baseline: float) -> float | None:
    return value / baseline if baseline else None


def validate_measurement(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        requirement = (
            "a finite positive number"
            if positive
            else "a finite non-negative number"
        )
        raise ValueError(f"{label} must be {requirement}")


def validate_ranked_evidence(result: dict, *, label: str) -> None:
    """Reject incomplete per-rank evidence and inconsistent storage totals."""

    world_size = result.get("tensor_parallel_size")
    ranked_fields = [field for field in STORAGE_FIELDS if field.endswith("_by_rank")]
    if "cuda_memory_by_rank" in result:
        ranked_fields.append("cuda_memory_by_rank")
    for field in ranked_fields:
        validate_ranked_records(
            result.get(field),
            expected_world_size=world_size,
            record_name=f"{label}.{field}",
        )
    for by_rank_field, local_total_field, aggregate_field in RANKED_STORAGE_TOTALS:
        records = result[by_rank_field]
        local_values = []
        for record in records:
            value = record.get(local_total_field)
            validate_measurement(
                value,
                label=f"{label}.{by_rank_field}.{local_total_field}",
            )
            local_values.append(value)
        aggregate = result.get(aggregate_field)
        validate_measurement(aggregate, label=f"{label}.{aggregate_field}")
        if sum(local_values) != aggregate:
            raise ValueError(
                f"{label}.{aggregate_field} does not match per-rank storage"
            )


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


def _parse_input_preparation(raw: object, *, label: str) -> dict[str, dict]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label}.input_preparation_stats must be a dictionary")
    unsupported = sorted(set(raw) - set(INPUT_PREPARATION_STEP_KINDS))
    if unsupported:
        raise ValueError(
            f"{label}.input_preparation_stats has unsupported steps: "
            + ", ".join(unsupported)
        )
    metrics = {}
    for step_kind in INPUT_PREPARATION_STEP_KINDS:
        stats = raw.get(step_kind)
        if stats is None:
            continue
        if not isinstance(stats, dict):
            raise ValueError(
                f"{label}.input_preparation_stats.{step_kind} must be a dictionary"
            )
        call_count = stats.get("call_count")
        if (
            isinstance(call_count, bool)
            or not isinstance(call_count, int)
            or call_count <= 0
        ):
            raise ValueError(
                f"{label}.input_preparation_stats.{step_kind}.call_count "
                "must be a positive integer"
            )
        total_time_s = stats.get("total_time_s")
        max_time_s = stats.get("max_time_s")
        validate_measurement(
            total_time_s,
            label=(
                f"{label}.input_preparation_stats.{step_kind}.total_time_s"
            ),
        )
        validate_measurement(
            max_time_s,
            label=f"{label}.input_preparation_stats.{step_kind}.max_time_s",
        )
        if max_time_s > total_time_s:
            raise ValueError(
                f"{label}.input_preparation_stats.{step_kind}.max_time_s "
                "cannot exceed total_time_s"
            )
        metrics[step_kind] = {
            "call_count": call_count,
            "total_time_s": total_time_s,
            "max_time_s": max_time_s,
            "average_time_s": total_time_s / call_count,
        }
    return metrics


def input_preparation_metrics(result: dict, *, label: str) -> dict[str, dict]:
    by_rank = result.get("execution_stats_by_rank")
    if by_rank is None:
        raw = result.get("execution_stats", {}).get(
            "input_preparation_stats",
            {},
        )
        return _parse_input_preparation(raw, label=label)
    if not isinstance(by_rank, list) or not by_rank:
        raise ValueError(f"{label}.execution_stats_by_rank must be a non-empty list")
    expected_ranks = set(range(result["tensor_parallel_size"]))
    rank_metrics = {}
    for item in by_rank:
        if not isinstance(item, dict):
            raise ValueError(f"{label}.execution_stats_by_rank entries must be dictionaries")
        rank = item.get("rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in expected_ranks
            or rank in rank_metrics
        ):
            raise ValueError(f"{label}.execution_stats_by_rank has invalid ranks")
        rank_metrics[rank] = _parse_input_preparation(
            item.get("input_preparation_stats", {}),
            label=f"{label}.rank{rank}",
        )
    if set(rank_metrics) != expected_ranks:
        raise ValueError(f"{label}.execution_stats_by_rank is incomplete")
    step_sets = [set(metrics) for metrics in rank_metrics.values()]
    if any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError(f"{label}.execution_stats_by_rank recorded different paths")
    aggregated = {}
    for step_kind in sorted(step_sets[0]):
        samples = [
            {"rank": rank, **metrics[step_kind]}
            for rank, metrics in sorted(rank_metrics.items())
        ]
        call_counts = {item["call_count"] for item in samples}
        if len(call_counts) != 1:
            raise ValueError(
                f"{label}.execution_stats_by_rank recorded different call counts"
            )
        slowest = max(
            samples,
            key=lambda item: (item["average_time_s"], -item["rank"]),
        )
        fastest_average = min(item["average_time_s"] for item in samples)
        aggregated[step_kind] = {
            "call_count": samples[0]["call_count"],
            "total_time_s": max(item["total_time_s"] for item in samples),
            "max_time_s": max(item["max_time_s"] for item in samples),
            "average_time_s": slowest["average_time_s"],
            "slowest_rank": slowest["rank"],
            "rank_skew": ratio(slowest["average_time_s"], fastest_average),
            "by_rank": samples,
        }
    return aggregated


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
            *STORAGE_FIELDS,
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
    for label, result in zip(labels, results):
        validate_ranked_evidence(result, label=label)
        validate_measurement(
            result.get("output_throughput_tok_s"),
            label=f"{label}.output_throughput_tok_s",
            positive=True,
        )
        validate_measurement(
            result.get("peak_torch_allocated_mib"),
            label=f"{label}.peak_torch_allocated_mib",
        )
        for name in LATENCY_METRIC_NAMES:
            validate_measurement(
                result.get("metrics", {}).get(name),
                label=f"{label}.metrics.{name}",
            )
    preparation_by_run = [
        input_preparation_metrics(result, label=label)
        for label, result in zip(labels, results)
    ]
    baseline_preparation = preparation_by_run[0]
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
    baseline_memory = baseline["peak_torch_allocated_mib"]
    baseline_digest = baseline["generated_token_ids"]["digest"]
    rows = []
    for label, result, preparation in zip(labels, results, preparation_by_run):
        throughput = result["output_throughput_tok_s"]
        memory = result["peak_torch_allocated_mib"]
        latency_metrics = {
            name: result["metrics"][name] for name in LATENCY_METRIC_NAMES
        }
        rows.append(
            {
                "label": label,
                **{field: result[field] for field in OPTIMIZATION_FIELDS},
                "output_throughput_tok_s": throughput,
                "throughput_vs_baseline": ratio(throughput, baseline_throughput),
                **latency_metrics,
                "ttft_vs_baseline": ratio(
                    latency_metrics["avg_ttft_s"],
                    baseline["metrics"]["avg_ttft_s"],
                ),
                "tpot_vs_baseline": ratio(
                    latency_metrics["avg_tpot_s"],
                    baseline["metrics"]["avg_tpot_s"],
                ),
                "latency_vs_baseline": {
                    name: ratio(value, baseline["metrics"][name])
                    for name, value in latency_metrics.items()
                },
                "peak_torch_allocated_mib": memory,
                "peak_memory_vs_baseline": ratio(memory, baseline_memory),
                "output_digest_matches_baseline": (
                    result["generated_token_ids"]["digest"] == baseline_digest
                ),
                "execution_valid": result["execution_validation"]["valid"],
                "generation_valid": result["generation_validation"]["valid"],
                "input_preparation": preparation,
                "input_preparation_vs_baseline": {
                    step_kind: ratio(
                        stats["average_time_s"],
                        baseline_preparation[step_kind]["average_time_s"],
                    )
                    for step_kind, stats in preparation.items()
                    if step_kind in baseline_preparation
                },
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
        for field in ("commit", *OPTIMIZATION_FIELDS, *STORAGE_FIELDS):
            if result[field] != baseline[field]:
                mismatches.append(f"{label}.{field}")
    if mismatches:
        raise ValueError(
            "benchmark repeats do not share one implementation: "
            + ", ".join(mismatches)
        )
    required_execution_paths = baseline["execution_validation"].get(
        "required_paths", []
    )
    if any(
        result["execution_validation"].get("required_paths", [])
        != required_execution_paths
        for result in results[1:]
    ):
        raise ValueError("benchmark repeats require different execution paths")
    observed_path_sets = [
        set(result["execution_validation"].get("observed_paths", []))
        for result in results
    ]
    observed_in_all_repeats = sorted(set.intersection(*observed_path_sets))
    preparation_by_run = [
        input_preparation_metrics(result, label=label)
        for label, result in zip(labels, results)
    ]
    preparation_step_sets = [set(item) for item in preparation_by_run]
    if any(
        steps != preparation_step_sets[0]
        for steps in preparation_step_sets[1:]
    ):
        raise ValueError("benchmark repeats recorded different input preparation paths")

    metrics = {
        "output_throughput_tok_s": [
            result["output_throughput_tok_s"] for result in results
        ],
        **{
            name: [result["metrics"][name] for result in results]
            for name in LATENCY_METRIC_NAMES
        },
        "peak_torch_allocated_mib": [
            result["peak_torch_allocated_mib"] for result in results
        ],
        **{
            f"host_{step_kind}_preparation_{metric_name}": [
                preparation[step_kind][metric_name]
                for preparation in preparation_by_run
            ]
            for step_kind in sorted(preparation_step_sets[0])
            for metric_name in (
                "call_count",
                "total_time_s",
                "max_time_s",
                "average_time_s",
            )
        },
        **{
            f"host_{step_kind}_preparation_rank_skew": [
                preparation[step_kind]["rank_skew"]
                for preparation in preparation_by_run
            ]
            for step_kind in sorted(preparation_step_sets[0])
            if all(
                preparation[step_kind].get("rank_skew") is not None
                for preparation in preparation_by_run
            )
        },
    }
    return {
        "commit": baseline["commit"],
        "workload": comparison["workload"],
        "environment": comparison["environment"],
        "configuration": {
            field: baseline[field] for field in OPTIMIZATION_FIELDS
        },
        "storage": {field: baseline[field] for field in STORAGE_FIELDS},
        "checkpoint_identity_strength": comparison[
            "checkpoint_identity_strength"
        ],
        "generated_token_ids_digest": baseline["generated_token_ids"]["digest"],
        "all_output_digests_match": comparison["all_output_digests_match"],
        "all_execution_paths_valid": comparison["all_execution_paths_valid"],
        "execution_paths": {
            "required": list(required_execution_paths),
            "observed_in_all_repeats": observed_in_all_repeats,
        },
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

    optional_metric_names = sorted(
        set.intersection(
            *(set(summary["statistics"]) for summary in summaries)
        )
        - {
            "output_throughput_tok_s",
            *LATENCY_METRIC_NAMES,
            "peak_torch_allocated_mib",
        }
    )
    metric_names = (
        "output_throughput_tok_s",
        *LATENCY_METRIC_NAMES,
        "peak_torch_allocated_mib",
        *optional_metric_names,
    )
    rows = []
    for label, summary in zip(labels, summaries):
        statistics_by_name = summary["statistics"]
        rows.append(
            {
                "label": label,
                **summary["configuration"],
                "storage": summary["storage"],
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
                "execution_paths": summary.get("execution_paths", {}),
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
            "latency": {
                name: ratio(median[name], baseline_median[name])
                for name in LATENCY_METRIC_NAMES
            },
            "input_preparation": {
                name: ratio(median[name], baseline_median[name])
                for name in optional_metric_names
                if name.startswith("host_")
            },
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
