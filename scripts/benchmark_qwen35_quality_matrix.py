"""Run the Qwen3.6 TP/state/KV teacher-forcing quality matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALITY_SCRIPT = ROOT / "scripts" / "measure_kv_quality_teacher_forcing.py"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_checkpoint_mapping.py"
CROSS_TP_MAX_TARGET_LOGPROB_DIFF = 0.05
MIN_INT8_TOP1_AGREEMENT = 0.98
MAX_INT8_JS_DIVERGENCE = 0.01
MAX_INT8_PPL_RELATIVE_CHANGE = 0.10
MAX_MODEL_STATE_BF16_PPL_RELATIVE_CHANGE = 0.05
MAX_MODEL_STATE_INT8_PPL_RELATIVE_CHANGE = 0.10


@dataclass(frozen=True)
class QualityCase:
    tensor_parallel_size: int
    recurrent_state_dtype: str

    @property
    def name(self) -> str:
        return (
            f"qwen35_tp{self.tensor_parallel_size}_"
            f"state-{self.recurrent_state_dtype}"
        )


def comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def build_cases(tp_sizes: tuple[int, ...]) -> list[QualityCase]:
    return [
        QualityCase(tp_size, state_dtype)
        for tp_size in tp_sizes
        for state_dtype in ("float32", "model")
    ]


def command_for_case(
    args: argparse.Namespace,
    case: QualityCase,
    run_id: str,
) -> list[str]:
    case_name = f"{run_id}_{case.name}"
    case_dir = Path(args.result_dir) / case_name
    command = [
        sys.executable,
        str(QUALITY_SCRIPT),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(case.tensor_parallel_size),
        "--recurrent-state-dtype",
        case.recurrent_state_dtype,
        "--weight-quant-backend",
        getattr(args, "weight_quant_backend", "auto"),
        "--qwen35-decode-conv-backend",
        getattr(args, "qwen35_decode_conv_backend", "weighted"),
        "--qwen35-moe-decode-backend",
        getattr(args, "qwen35_moe_decode_backend", "sorted"),
        "--prompt-lengths",
        args.prompt_lengths,
        "--cases-per-length",
        str(args.cases_per_length),
        "--continuation-len",
        str(args.continuation_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--partition-threshold",
        str(args.partition_threshold),
        "--partition-size",
        str(args.partition_size),
        "--trace-max-events",
        str(args.trace_max_events),
        "--trace-max-index-values",
        str(args.trace_max_index_values),
        "--result-dir",
        str(case_dir),
        "--name",
        case_name,
    ]
    if args.cases_file:
        command.extend(("--cases-file", args.cases_file))
    return command


def result_path_for_case(
    result_dir: str | Path,
    run_id: str,
    case: QualityCase,
) -> Path:
    case_name = f"{run_id}_{case.name}"
    return Path(result_dir) / case_name / f"{case_name}.json"


def checkpoint_audit_command(
    args: argparse.Namespace,
    run_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--model",
        args.model,
        "--tp-sizes",
        ",".join(str(size) for size in args.tp_sizes),
        "--require-shards",
        "--output",
        str(Path(args.result_dir) / f"{run_id}_checkpoint_audit.json"),
    ]


def summarize_results(
    results: dict[QualityCase, dict],
) -> dict:
    checkpoint_digests = {
        result.get("checkpoint_manifest", {}).get("digest")
        for result in results.values()
    }
    if None in checkpoint_digests or len(checkpoint_digests) != 1:
        raise ValueError("quality matrix results use different checkpoints")
    commits = {result.get("commit") for result in results.values()}
    if None in commits or len(commits) != 1:
        raise ValueError("quality matrix results use different commits")
    if any(result.get("git_dirty") is not False for result in results.values()):
        raise ValueError("quality matrix requires clean worktrees")
    rows = []
    by_tp: dict[str, dict] = {}
    for case, result in results.items():
        configuration = result.get("configuration", {})
        if configuration.get("tensor_parallel_size") != case.tensor_parallel_size:
            raise ValueError(
                f"quality result TP does not match case {case.name}"
            )
        if configuration.get("recurrent_state_dtype") != case.recurrent_state_dtype:
            raise ValueError(
                f"quality result state dtype does not match case {case.name}"
            )
        if configuration.get("teacher_forcing") is not True:
            raise ValueError(f"quality result is not teacher-forced: {case.name}")
        summary = result["summary"]
        decode_ppl = summary["decode_ppl"]
        int8_attention_paths = {
            path
            for batch in result["batches"]
            for path in batch["execution_validation"]["int8"][
                "observed_paths"
            ]
        }
        rows.append(
            {
                "tensor_parallel_size": case.tensor_parallel_size,
                "recurrent_state_dtype": case.recurrent_state_dtype,
                "requested_weight_quant_backend": configuration.get(
                    "requested_weight_quant_backend"
                ),
                "weight_quant_backend": configuration.get(
                    "weight_quant_backend"
                ),
                "qwen35_decode_conv_backend": configuration.get(
                    "qwen35_decode_conv_backend"
                ),
                "qwen35_moe_decode_backend": configuration.get(
                    "qwen35_moe_decode_backend"
                ),
                "decode_ppl_bf16_kv": decode_ppl["bf16"],
                "decode_ppl_int8_kv": decode_ppl["int8"],
                "int8_kv_relative_change": decode_ppl["relative_change"],
                "decode_top1_agreement": summary["decode_aggregate"][
                    "top1_agreement"
                ],
                "decode_js_divergence": summary["decode_aggregate"][
                    "js_divergence"
                ],
                "kv_sensitive_token_rows": summary[
                    "kv_sensitive_token_rows_compared"
                ],
                "int8_partitioned_decode_observed": (
                    "int8_partitioned_decode" in int8_attention_paths
                ),
            }
        )

    tp_sizes = sorted({case.tensor_parallel_size for case in results})
    for tp_size in tp_sizes:
        fp32 = results[QualityCase(tp_size, "float32")]["summary"]["decode_ppl"]
        model = results[QualityCase(tp_size, "model")]["summary"]["decode_ppl"]
        baseline = fp32["bf16"]
        by_tp[f"tp{tp_size}"] = {
            "baseline": "float32 recurrent state + BF16 KV",
            "baseline_decode_ppl": baseline,
            "model_state_bf16_kv_relative_change": model["bf16"] / baseline - 1.0,
            "float32_state_int8_kv_relative_change": fp32["int8"] / baseline - 1.0,
            "model_state_int8_kv_relative_change": model["int8"] / baseline - 1.0,
        }
    case_digests = {result["case_token_digest"] for result in results.values()}
    if len(case_digests) != 1:
        raise ValueError("quality matrix cases differ across TP configurations")

    def trajectory(result: dict, name: str) -> list:
        return [
            value
            for batch in result["batches"]
            for step in batch["decode_trajectories"][name]
            for value in step
        ]

    cross_tp_comparisons = []
    baseline_tp = tp_sizes[0]
    for state_dtype in ("float32", "model"):
        baseline = results[QualityCase(baseline_tp, state_dtype)]
        for tp_size in tp_sizes[1:]:
            candidate = results[QualityCase(tp_size, state_dtype)]
            mode_results = {}
            for mode in ("bf16", "int8"):
                top1_name = f"{mode}_top1_token_ids"
                logprob_name = f"{mode}_target_logprobs"
                baseline_top1 = trajectory(baseline, top1_name)
                candidate_top1 = trajectory(candidate, top1_name)
                baseline_logprobs = trajectory(baseline, logprob_name)
                candidate_logprobs = trajectory(candidate, logprob_name)
                shapes_match = (
                    bool(baseline_top1)
                    and bool(baseline_logprobs)
                    and len(baseline_top1) == len(candidate_top1)
                    and len(baseline_logprobs) == len(candidate_logprobs)
                )
                max_logprob_diff = (
                    max(
                        (
                            abs(left - right)
                            for left, right in zip(
                                baseline_logprobs,
                                candidate_logprobs,
                            )
                        ),
                        default=0.0,
                    )
                    if shapes_match
                    else None
                )
                top1_match = shapes_match and baseline_top1 == candidate_top1
                mode_results[mode] = {
                    "top1_match": top1_match,
                    "max_target_logprob_diff": max_logprob_diff,
                    "passed": (
                        top1_match
                        and max_logprob_diff
                        <= CROSS_TP_MAX_TARGET_LOGPROB_DIFF
                    ),
                }
            cross_tp_comparisons.append(
                {
                    "baseline_tp": baseline_tp,
                    "candidate_tp": tp_size,
                    "recurrent_state_dtype": state_dtype,
                    "modes": mode_results,
                    "passed": all(
                        item["passed"] for item in mode_results.values()
                    ),
                }
            )
    per_case_quality = [
        {
            "tensor_parallel_size": row["tensor_parallel_size"],
            "recurrent_state_dtype": row["recurrent_state_dtype"],
            "int8_ppl": abs(row["int8_kv_relative_change"])
            <= MAX_INT8_PPL_RELATIVE_CHANGE,
            "int8_top1": row["decode_top1_agreement"]
            >= MIN_INT8_TOP1_AGREEMENT,
            "int8_js": row["decode_js_divergence"]
            <= MAX_INT8_JS_DIVERGENCE,
        }
        for row in rows
    ]
    per_tp_quality = {
        tp_name: {
            "model_state_bf16_ppl": abs(
                comparison["model_state_bf16_kv_relative_change"]
            )
            <= MAX_MODEL_STATE_BF16_PPL_RELATIVE_CHANGE,
            "model_state_int8_ppl": abs(
                comparison["model_state_int8_kv_relative_change"]
            )
            <= MAX_MODEL_STATE_INT8_PPL_RELATIVE_CHANGE,
        }
        for tp_name, comparison in by_tp.items()
    }
    cross_tp_passed = all(
        item["passed"] for item in cross_tp_comparisons
    )
    quality_gates_passed = all(
        all(checks.values()) for checks in per_case_quality
    ) and all(
        all(checks.values()) for checks in per_tp_quality.values()
    ) and cross_tp_passed
    return {
        "commit": next(iter(commits)),
        "quality_scope": "teacher-forced decode tokens that read stored KV cache",
        "cases": rows,
        "comparisons_by_tp": by_tp,
        "case_token_digest": next(iter(case_digests)),
        "cross_tp": {
            "max_target_logprob_diff": CROSS_TP_MAX_TARGET_LOGPROB_DIFF,
            "all_passed": cross_tp_passed,
            "comparisons": cross_tp_comparisons,
        },
        "quality_gates": {
            "all_passed": quality_gates_passed,
            "thresholds": {
                "min_int8_top1_agreement": MIN_INT8_TOP1_AGREEMENT,
                "max_int8_js_divergence": MAX_INT8_JS_DIVERGENCE,
                "max_abs_int8_ppl_relative_change": (
                    MAX_INT8_PPL_RELATIVE_CHANGE
                ),
                "max_abs_model_state_bf16_ppl_relative_change": (
                    MAX_MODEL_STATE_BF16_PPL_RELATIVE_CHANGE
                ),
                "max_abs_model_state_int8_ppl_relative_change": (
                    MAX_MODEL_STATE_INT8_PPL_RELATIVE_CHANGE
                ),
            },
            "per_case": per_case_quality,
            "per_tp": per_tp_quality,
        },
    }


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return 0 if devices == ["-1"] else len(devices)
    import torch

    return torch.cuda.device_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-sizes", type=comma_separated_ints, default=(4, 8))
    parser.add_argument("--cases-file")
    parser.add_argument("--prompt-lengths", default="128,1024,3072")
    parser.add_argument("--cases-per-length", type=int, default=2)
    parser.add_argument("--continuation-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--weight-quant-backend",
        choices=("auto", "reference", "resident", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--qwen35-decode-conv-backend",
        choices=("weighted", "channel_accumulate"),
        default="weighted",
    )
    parser.add_argument(
        "--qwen35-moe-decode-backend",
        choices=("sorted", "batched"),
        default="sorted",
    )
    parser.add_argument("--partition-threshold", type=int, default=8192)
    parser.add_argument("--partition-size", type=int, default=512)
    parser.add_argument("--trace-max-events", type=int, default=2048)
    parser.add_argument("--trace-max-index-values", type=int, default=64)
    parser.add_argument("--result-dir", default="benchmark_results/qwen35_quality_matrix")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--checkpoint-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require complete checkpoint shards and validate TP weight shapes.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.cases_per_length <= 0 or args.continuation_len < 2:
        raise ValueError(
            "cases-per-length must be positive and continuation-len must be at least 2"
        )
    if min(
        args.max_model_len,
        args.max_num_batched_tokens,
        args.partition_threshold,
        args.partition_size,
    ) <= 0:
        raise ValueError("model and batch token limits must be positive")
    if args.trace_max_events <= 0 or args.trace_max_index_values <= 0:
        raise ValueError("trace limits must be positive")
    if args.run_id is not None and (
        not args.run_id
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in args.run_id
        )
    ):
        raise ValueError(
            "--run-id may contain only letters, digits, dot, dash, and underscore"
        )
    return args


def main() -> None:
    try:
        args = normalize_args(parse_args())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = build_cases(args.tp_sizes)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "qwen35_quality_%Y%m%dT%H%M%S%fZ"
    )
    if not args.dry_run:
        required_gpus = max(args.tp_sizes)
        available_gpus = visible_gpu_count()
        if available_gpus < required_gpus:
            raise SystemExit(
                f"quality matrix requires {required_gpus} visible GPUs, "
                f"but found {available_gpus}; checkpoint audit was not started"
            )
    if args.checkpoint_audit:
        command = checkpoint_audit_command(args, run_id)
        print("[preflight] checkpoint mapping audit", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)
    results = {}
    for index, case in enumerate(cases, start=1):
        command = command_for_case(args, case, run_id)
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        path = result_path_for_case(args.result_dir, run_id, case)
        results[case] = json.loads(path.read_text())

    if args.dry_run:
        return
    summary = {
        "run_id": run_id,
        "model": args.model,
        "tensor_parallel_sizes": list(args.tp_sizes),
        **summarize_results(results),
    }
    output = Path(args.result_dir) / f"{run_id}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
