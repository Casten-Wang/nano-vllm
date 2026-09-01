"""Run the fail-fast Qwen3.5-35B-A3B rental-GPU validation suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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


def manifest_plan(
    args: argparse.Namespace,
    stages: list[tuple[str, list[str]]],
) -> dict:
    return {
        "run_id": args.run_id,
        "model": str(Path(args.model).expanduser().resolve()),
        "stages": [
            {"name": name, "command": command}
            for name, command in stages
        ],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def prepare_manifest(
    path: Path,
    plan: dict,
    *,
    resume: bool,
) -> dict:
    if resume:
        if not path.is_file():
            raise ValueError(f"resume manifest does not exist: {path}")
        manifest = json.loads(path.read_text())
        if {
            "run_id": manifest.get("run_id"),
            "model": manifest.get("model"),
            "stages": manifest.get("stages"),
        } != plan:
            raise ValueError("resume manifest does not match requested run")
        completed = manifest.get("completed_stages", [])
        if not isinstance(completed, list):
            raise ValueError("resume manifest has invalid completed stages")
        return manifest
    if path.exists():
        raise ValueError(
            f"run manifest already exists: {path}; use --resume or a new run id"
        )
    manifest = {**plan, "completed_stages": []}
    write_manifest(path, manifest)
    return manifest


def mark_stage_completed(path: Path, manifest: dict, stage_name: str) -> None:
    completed = manifest["completed_stages"]
    if stage_name not in completed:
        completed.append(stage_name)
    write_manifest(path, manifest)


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only stages recorded by an identical run manifest.",
    )
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
    manifest = None
    manifest_path = Path(args.result_dir) / args.run_id / "manifest.json"
    if not args.dry_run:
        try:
            manifest = prepare_manifest(
                manifest_path,
                manifest_plan(args, stages),
                resume=args.resume,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    for index, (name, command) in enumerate(stages, start=1):
        if manifest is not None and name in manifest["completed_stages"]:
            print(f"[{index}/{len(stages)}] {name} (already completed)", flush=True)
            continue
        print(f"[{index}/{len(stages)}] {name}", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            mark_stage_completed(manifest_path, manifest, name)


if __name__ == "__main__":
    main()
