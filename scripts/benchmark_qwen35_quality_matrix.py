"""Run the Qwen3.5 TP/state/KV teacher-forcing quality matrix."""

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
    rows = []
    by_tp: dict[str, dict] = {}
    for case, result in results.items():
        summary = result["summary"]
        decode_ppl = summary["decode_ppl"]
        rows.append(
            {
                "tensor_parallel_size": case.tensor_parallel_size,
                "recurrent_state_dtype": case.recurrent_state_dtype,
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
            }
        )

    for tp_size in sorted({case.tensor_parallel_size for case in results}):
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
    return {
        "quality_scope": "teacher-forced decode tokens that read stored KV cache",
        "cases": rows,
        "comparisons_by_tp": by_tp,
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
    if args.max_model_len <= 0 or args.max_num_batched_tokens <= 0:
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
    if args.checkpoint_audit:
        command = checkpoint_audit_command(args, run_id)
        print("[preflight] checkpoint mapping audit", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        required_gpus = max(args.tp_sizes)
        available_gpus = visible_gpu_count()
        if available_gpus < required_gpus:
            raise SystemExit(
                f"quality matrix requires {required_gpus} visible GPUs, "
                f"but found {available_gpus}"
            )

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
