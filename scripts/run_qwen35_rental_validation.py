"""Run the fail-fast Qwen3.5-35B-A3B rental-GPU validation suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_matrix.py"
KERNEL_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_kernels.py"
QUALITY_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_quality_matrix.py"


def parse_tp_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size not in (1, 2, 4, 8) for size in sizes):
        raise argparse.ArgumentTypeError("TP sizes must be selected from 1,2,4,8")
    return sizes


def validate_run_id(value: str) -> str:
    if not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise argparse.ArgumentTypeError(
            "run id may contain only letters, digits, dot, dash, and underscore"
        )
    return value


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    tp_sizes = ",".join(str(size) for size in args.tp_sizes)
    root = Path(args.result_dir) / args.run_id
    common_matrix = [
        "--model",
        args.model,
        "--tp-sizes",
        tp_sizes,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--repeats",
        str(args.repeats),
        "--run-id",
        args.run_id,
    ]
    result = [
        (
            "preflight",
            [
                sys.executable,
                str(MATRIX_SCRIPT),
                *common_matrix,
                "--result-dir",
                str(root / "preflight"),
                "--preflight-only",
            ],
        )
    ]
    for tp_size in args.tp_sizes:
        result.append(
            (
                f"kernels-tp{tp_size}",
                [
                    sys.executable,
                    str(KERNEL_SCRIPT),
                    "--device",
                    "cuda",
                    "--tp-size",
                    str(tp_size),
                    "--output",
                    str(root / "kernels" / f"tp{tp_size}.json"),
                ],
            )
        )
    result.extend(
        (
            (
                "performance-matrix",
                [
                    sys.executable,
                    str(MATRIX_SCRIPT),
                    *common_matrix,
                    "--result-dir",
                    str(root / "performance"),
                    "--no-checkpoint-audit",
                    "--no-memory-preflight",
                ],
            ),
            (
                "quality-matrix",
                [
                    sys.executable,
                    str(QUALITY_SCRIPT),
                    "--model",
                    args.model,
                    "--tp-sizes",
                    tp_sizes,
                    "--run-id",
                    args.run_id,
                    "--result-dir",
                    str(root / "quality"),
                    "--no-checkpoint-audit",
                ],
            ),
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, default=(4, 8))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--run-id",
        type=validate_run_id,
        default=None,
    )
    parser.add_argument(
        "--result-dir",
        default="benchmark_results/qwen35_rental",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "rental_%Y%m%dT%H%M%S%fZ"
    )
    if min(
        args.num_seqs,
        args.input_len,
        args.output_len,
        args.max_num_seqs,
        args.repeats,
    ) <= 0:
        parser.error("workload sizes and repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    stages = commands(args)
    for index, (name, command) in enumerate(stages, start=1):
        print(f"[{index}/{len(stages)}] {name}", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
