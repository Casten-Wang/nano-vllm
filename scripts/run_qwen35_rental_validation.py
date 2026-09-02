"""Run the fail-fast Qwen3.5-35B-A3B rental-GPU validation suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_matrix.py"
KERNEL_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_kernels.py"
ATTENTION_KERNEL_SCRIPT = ROOT / "scripts" / "benchmark_attention_kernel.py"
QUALITY_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_quality_matrix.py"
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_qwen35_rental.py"
CUDAGRAPH_PARITY_SCRIPT = ROOT / "scripts" / "verify_cudagraph_parity.py"


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
        "--max-model-len",
        str(args.max_model_len),
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
                "--include-moe-candidate",
            ],
        )
    ]
    for tp_size in args.tp_sizes:
        local_query_heads = 16 // tp_size
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
        result.append(
            (
                f"kernels-long-prefill-tp{tp_size}",
                [
                    sys.executable,
                    str(KERNEL_SCRIPT),
                    "--device",
                    "cuda",
                    "--tp-size",
                    str(tp_size),
                    "--prefill-only",
                    "--prefill-batch",
                    "1",
                    "--prefill-tokens",
                    "8192",
                    "--delta-prefill-chunk-sizes",
                    "32",
                    "64",
                    "128",
                    "--warmup",
                    "2",
                    "--iterations",
                    "5",
                    "--repeats",
                    "3",
                    "--output",
                    str(root / "kernels_long" / f"tp{tp_size}.json"),
                ],
            )
        )
        for context_name, context_len in (("short", 4096), ("long", 16384)):
            command = [
                sys.executable,
                str(ATTENTION_KERNEL_SCRIPT),
                "--batch-size",
                "4",
                "--context-len",
                str(context_len),
                "--num-heads",
                str(local_query_heads),
                "--num-kv-heads",
                "1",
                "--head-dim",
                "256",
                "--result-dir",
                str(root / "attention" / f"tp{tp_size}"),
                "--name",
                context_name,
            ]
            if context_name == "long":
                command.extend(
                    ("--include-partitioned", "--partition-sizes", "256,512")
                )
            result.append(
                (f"attention-{context_name}-tp{tp_size}", command)
            )
        for context_name, base_length, batch_sizes in (
            ("short", 33, f"3,9,{args.max_num_seqs}"),
            ("long", 8192, "3"),
        ):
            result.append(
                (
                    f"cudagraph-{context_name}-tp{tp_size}",
                    [
                        sys.executable,
                        str(CUDAGRAPH_PARITY_SCRIPT),
                        "--model",
                        args.model,
                        "--tensor-parallel-size",
                        str(tp_size),
                        "--batch-sizes",
                        batch_sizes,
                        "--input-length-base",
                        str(base_length),
                        "--max-model-len",
                        str(args.max_model_len),
                        "--max-num-batched-tokens",
                        str(args.max_model_len),
                        "--max-num-seqs",
                        str(args.max_num_seqs),
                        "--qwen35-moe-decode-backend",
                        "batched",
                        "--result-dir",
                        str(
                            root
                            / "cudagraph"
                            / f"tp{tp_size}"
                            / context_name
                        ),
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
                    "--include-moe-candidate",
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
                    "--prompt-lengths",
                    "128,1024,3072,8192",
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-batched-tokens",
                    str(args.max_model_len),
                ],
            ),
            (
                "final-summary",
                [
                    sys.executable,
                    str(SUMMARY_SCRIPT),
                    "--run-dir",
                    str(root),
                    "--run-id",
                    args.run_id,
                    "--output",
                    str(root / "summary.json"),
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_stage_artifacts(
    args: argparse.Namespace,
    stage_name: str,
) -> list[Path]:
    root = Path(args.result_dir) / args.run_id
    if stage_name == "preflight":
        required = [
            root / "preflight" / "checkpoint_mapping_audit.json",
            root / "preflight" / "memory_preflight.json",
        ]
        search_root = root / "preflight"
    elif stage_name.startswith("kernels-long-prefill-tp"):
        tp_name = stage_name.removeprefix("kernels-long-prefill-")
        required = [root / "kernels_long" / f"{tp_name}.json"]
        search_root = root / "kernels_long"
    elif stage_name.startswith("kernels-tp"):
        required = [root / "kernels" / f"{stage_name.removeprefix('kernels-')}.json"]
        search_root = root / "kernels"
    elif stage_name.startswith("attention-"):
        _, context_name, tp_name = stage_name.split("-")
        search_root = root / "attention" / tp_name
        required = [search_root / f"{context_name}.json"]
    elif stage_name.startswith("cudagraph-"):
        _, context_name, tp_name = stage_name.split("-")
        search_root = root / "cudagraph" / tp_name / context_name
        required = sorted(search_root.glob("run_*/summary.json"))
        if len(required) != 1:
            raise RuntimeError(
                f"stage {stage_name} must produce exactly one summary"
            )
    elif stage_name == "performance-matrix":
        search_root = root / "performance"
        required = (
            [search_root / f"{args.run_id}_matrix_summary.json"]
            if args.repeats > 1
            else []
        )
    elif stage_name == "quality-matrix":
        search_root = root / "quality"
        required = [search_root / f"{args.run_id}_summary.json"]
    elif stage_name == "final-summary":
        required = [root / "summary.json"]
        search_root = root
    else:
        raise ValueError(f"unknown validation stage: {stage_name}")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"stage {stage_name} did not produce required artifact: {missing[0]}"
        )
    artifacts = (
        required
        if stage_name in ("preflight", "final-summary")
        or stage_name.startswith("kernels-")
        or stage_name.startswith("attention-")
        or stage_name.startswith("cudagraph-")
        else sorted(search_root.rglob("*.json"))
    )
    if not artifacts:
        raise RuntimeError(f"stage {stage_name} produced no JSON artifacts")
    return artifacts


def artifact_records(paths: list[Path], root: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def validate_completed_artifacts(path: Path, manifest: dict) -> None:
    records_by_stage = manifest.get("completed_stage_artifacts")
    if not isinstance(records_by_stage, dict):
        raise ValueError("resume manifest has no artifact integrity records")
    for stage_name in manifest.get("completed_stages", []):
        records = records_by_stage.get(stage_name)
        if not isinstance(records, list) or not records:
            raise ValueError(f"resume stage has no artifacts: {stage_name}")
        for record in records:
            artifact = path.parent / record["path"]
            if (
                not artifact.is_file()
                or artifact.stat().st_size != record["size"]
                or file_sha256(artifact) != record["sha256"]
            ):
                raise ValueError(
                    f"resume artifact is missing or changed: {artifact}"
                )


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
        validate_completed_artifacts(path, manifest)
        return manifest
    if path.exists():
        raise ValueError(
            f"run manifest already exists: {path}; use --resume or a new run id"
        )
    manifest = {
        **plan,
        "completed_stages": [],
        "completed_stage_artifacts": {},
    }
    write_manifest(path, manifest)
    return manifest


def mark_stage_completed(
    path: Path,
    manifest: dict,
    stage_name: str,
    artifacts: list[Path],
) -> None:
    completed = manifest["completed_stages"]
    if stage_name not in completed:
        completed.append(stage_name)
    manifest["completed_stage_artifacts"][stage_name] = artifact_records(
        artifacts,
        path.parent,
    )
    write_manifest(path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, default=(4, 8))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=16_384)
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
        args.max_model_len,
        args.max_num_seqs,
        args.repeats,
    ) <= 0:
        parser.error("workload sizes and repeats must be positive")
    if args.repeats < 2:
        parser.error("rental validation requires at least two repeats")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input_len plus output_len cannot exceed max_model_len")
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
            artifacts = collect_stage_artifacts(args, name)
            mark_stage_completed(manifest_path, manifest, name, artifacts)


if __name__ == "__main__":
    main()
