"""Compare reproducible nano-vLLM benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def load_result(path: Path) -> dict:
    with path.open() as handle:
        result = json.load(handle)
    missing = [
        field
        for field in (
            *WORKLOAD_FIELDS,
            *ENVIRONMENT_FIELDS,
            "model",
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
    return result


def compare_results(results: list[dict], labels: list[str]) -> dict:
    if len(results) < 2 or len(results) != len(labels):
        raise ValueError("at least two results with matching labels are required")
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
        "environment": {
            field: baseline[field] for field in ENVIRONMENT_FIELDS
        },
        "all_output_digests_match": all(
            row["output_digest_matches_baseline"] for row in rows
        ),
        "all_execution_paths_valid": all(row["execution_valid"] for row in rows),
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
    args = parser.parse_args()
    labels = [path.stem for path in args.results]
    comparison = compare_results(
        [load_result(path) for path in args.results],
        labels,
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
