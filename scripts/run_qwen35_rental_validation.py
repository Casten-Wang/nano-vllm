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
ONLINE_MIXED_SCRIPT = ROOT / "scripts" / "benchmark_online_mixed.py"
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_qwen35_rental.py"
CUDAGRAPH_PARITY_SCRIPT = ROOT / "scripts" / "verify_cudagraph_parity.py"
REMOTE_CHECKPOINT_AUDIT_SCRIPT = (
    ROOT / "scripts" / "audit_remote_checkpoint_headers.py"
)
OFFICIAL_CHECKPOINT_REPO = "Qwen/Qwen3.5-35B-A3B"
OFFICIAL_CHECKPOINT_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
QUALITY_MAX_PROMPT_LENGTH = 8_192
QUALITY_CONTINUATION_LENGTH = 16
MIXED_CONCURRENT_SEQUENCES = 16
SOURCE_ROOTS = (ROOT / "nanovllm", ROOT / "scripts")
SOURCE_FILES = (ROOT / "pyproject.toml",)


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


def source_tree_sha256() -> str:
    """Fingerprint executable project sources used by rental validation."""

    paths = list(SOURCE_FILES)
    for source_root in SOURCE_ROOTS:
        paths.extend(source_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


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
            "official-checkpoint-audit",
            [
                sys.executable,
                str(REMOTE_CHECKPOINT_AUDIT_SCRIPT),
                "--repo",
                OFFICIAL_CHECKPOINT_REPO,
                "--revision",
                OFFICIAL_CHECKPOINT_REVISION,
                "--tp-sizes",
                tp_sizes,
                "--output",
                str(
                    root
                    / "preflight"
                    / "official_checkpoint_header_audit.json"
                ),
            ],
        ),
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
                "--verify-checkpoint-shards",
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
        for repeat in range(1, args.repeats + 1):
            result.append(
                (
                    f"mixed-tp{tp_size}-r{repeat}",
                    [
                        sys.executable,
                        str(ONLINE_MIXED_SCRIPT),
                        "--model",
                        args.model,
                        "--tensor-parallel-size",
                        str(tp_size),
                        "--qwen35-moe-decode-backend",
                        "batched",
                        "--initial-seqs",
                        "8",
                        "--injected-seqs",
                        "8",
                        "--initial-input-len",
                        "128",
                        "--injected-input-len",
                        "1024",
                        "--output-len",
                        "64",
                        "--temperature",
                        "0",
                        "--inject-after-decode-steps",
                        "8",
                        "--max-model-len",
                        str(args.max_model_len),
                        "--max-num-batched-tokens",
                        "2048",
                        "--max-num-seqs",
                        str(args.max_num_seqs),
                        "--enable-dynamic-chunked-prefill",
                        "--require-paths",
                        "mixed_eager",
                        "--output",
                        str(
                            root
                            / "mixed"
                            / f"tp{tp_size}"
                            / f"r{repeat}.json"
                        ),
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
        # Keep the long case one token past both the 256-token cache block and
        # the 256/512-token partition boundaries so the CUDA gate exercises a
        # partially filled final block and partition.
        for context_name, context_len, batch_size in (
            ("short", 4096, 4),
            ("long", 16385, 4),
            ("max", 262143, 1),
        ):
            command = [
                sys.executable,
                str(ATTENTION_KERNEL_SCRIPT),
                "--batch-size",
                str(batch_size),
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
            if context_name != "short":
                command.extend(
                    ("--include-partitioned", "--partition-sizes", "256,512")
                )
            if context_name == "max":
                command.extend(
                    (
                        "--variants",
                        "v3",
                        "--block-tokens",
                        "256",
                        "--num-warps",
                        "8",
                        "--num-stages",
                        "2",
                        "--warmup",
                        "2",
                        "--iters",
                        "5",
                        "--repeats",
                        "3",
                    )
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
                    f"128,1024,3072,{QUALITY_MAX_PROMPT_LENGTH}",
                    "--continuation-len",
                    str(QUALITY_CONTINUATION_LENGTH),
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
        "source_tree_sha256": source_tree_sha256(),
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
    if stage_name == "official-checkpoint-audit":
        required = [
            root / "preflight" / "official_checkpoint_header_audit.json"
        ]
        search_root = root / "preflight"
    elif stage_name == "preflight":
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
    elif stage_name.startswith("mixed-tp"):
        tp_name, repeat_name = stage_name.removeprefix("mixed-").rsplit("-", 1)
        search_root = root / "mixed" / tp_name
        required = [search_root / f"{repeat_name}.json"]
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
        if stage_name in (
            "official-checkpoint-audit",
            "preflight",
            "final-summary",
        )
        or stage_name.startswith("kernels-")
        or stage_name.startswith("attention-")
        or stage_name.startswith("mixed-")
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


def validate_source_tree(manifest: dict) -> None:
    expected = manifest.get("source_tree_sha256")
    actual = source_tree_sha256()
    if not expected or actual != expected:
        raise ValueError("validation source tree changed during the run")


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
            "source_tree_sha256": manifest.get("source_tree_sha256"),
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
    minimum_model_len = QUALITY_MAX_PROMPT_LENGTH + QUALITY_CONTINUATION_LENGTH
    if args.max_model_len < minimum_model_len:
        parser.error(
            "max_model_len must be at least "
            f"{minimum_model_len} for the fixed long-context quality gate"
        )
    if args.max_num_seqs < MIXED_CONCURRENT_SEQUENCES:
        parser.error(
            "max_num_seqs must be at least "
            f"{MIXED_CONCURRENT_SEQUENCES} for the mixed-workload gate"
        )
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
            try:
                validate_source_tree(manifest)
            except ValueError as error:
                raise SystemExit(str(error)) from error
            subprocess.run(command, cwd=ROOT, check=True)
            artifacts = collect_stage_artifacts(args, name)
            mark_stage_completed(manifest_path, manifest, name, artifacts)


if __name__ == "__main__":
    main()
