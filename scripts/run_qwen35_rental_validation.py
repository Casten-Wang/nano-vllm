"""Run the fail-fast Qwen3.6-35B-A3B rental-GPU validation suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_matrix.py"
KERNEL_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_kernels.py"
ATTENTION_KERNEL_SCRIPT = ROOT / "scripts" / "benchmark_attention_kernel.py"
CACHE_TRANSFER_SCRIPT = ROOT / "scripts" / "benchmark_cache_transfer.py"
CACHE_EXPORT_SCRIPT = ROOT / "scripts" / "benchmark_cache_export.py"
QUALITY_SCRIPT = ROOT / "scripts" / "benchmark_qwen35_quality_matrix.py"
CHECKPOINT_QUALITY_SCRIPT = (
    ROOT / "scripts" / "compare_qwen35_checkpoint_quality.py"
)
ONLINE_MIXED_SCRIPT = ROOT / "scripts" / "benchmark_online_mixed.py"
SCHEDULER_WORKLOAD_SCRIPT = (
    ROOT / "scripts" / "benchmark_scheduler_workload.py"
)
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_qwen35_rental.py"
CUDAGRAPH_PARITY_SCRIPT = ROOT / "scripts" / "verify_cudagraph_parity.py"
REMOTE_CHECKPOINT_AUDIT_SCRIPT = (
    ROOT / "scripts" / "audit_remote_checkpoint_headers.py"
)
OFFICIAL_CHECKPOINT_REPO = "Qwen/Qwen3.6-35B-A3B"
OFFICIAL_CHECKPOINT_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
# No official Qwen3.6 GPTQ-Int4 checkpoint is currently available. This optional
# lane remains only for explicitly requested Qwen3.5 compatibility experiments.
OFFICIAL_GPTQ_CHECKPOINT_REPO = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
OFFICIAL_GPTQ_CHECKPOINT_REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
OFFICIAL_FP8_CHECKPOINT_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
OFFICIAL_FP8_CHECKPOINT_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
QUALITY_MAX_PROMPT_LENGTH = 8_192
QUALITY_CONTINUATION_LENGTH = 16
MIXED_CONCURRENT_SEQUENCES = 16
FAIRNESS_INITIAL_SEQUENCES = 32
FAIRNESS_INJECTED_SEQUENCES = 8
FAIRNESS_INITIAL_INPUT_LENGTH = 128
FAIRNESS_INJECTED_INPUT_LENGTH = 1024
FAIRNESS_OUTPUT_LENGTH = 64
FAIRNESS_INJECT_AFTER_DECODE_STEPS = 4
FAIRNESS_MAX_BATCHED_TOKENS = 32
FAIRNESS_THRESHOLD = 4
FAIRNESS_TOKEN_BUDGET = 256
PRESSURE_KV_BLOCKS = 5
PRESSURE_INITIAL_SEQUENCES = 2
PRESSURE_INJECTED_SEQUENCES = 2
PRESSURE_INITIAL_LENGTHS = (256, 1024)
PRESSURE_INJECTED_LENGTHS = (512, 512)
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

    paths = source_tree_paths()
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_tree_paths() -> list[Path]:
    """Return runnable sources, excluding files ignored by repository policy."""

    fallback = list(SOURCE_FILES)
    for source_root in SOURCE_ROOTS:
        fallback.extend(source_root.rglob("*.py"))
    try:
        pathspecs = [str(path.relative_to(ROOT)) for path in SOURCE_ROOTS]
        pathspecs.extend(str(path.relative_to(ROOT)) for path in SOURCE_FILES)
    except ValueError:
        return fallback
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *pathspecs,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return fallback
    relative_paths = [
        Path(value.decode())
        for value in result.stdout.split(b"\0")
        if value
    ]
    return [
        ROOT / path
        for path in relative_paths
        if path.suffix == ".py" or path == Path("pyproject.toml")
    ]


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    tp_sizes = ",".join(str(size) for size in args.tp_sizes)
    root = Path(args.result_dir) / args.run_id
    fp8_runtime_backend = getattr(args, "fp8_runtime_backend", "reference")
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
        "--sampling-chunk-size",
        str(getattr(args, "sampling_chunk_size", 32)),
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
        for kv_dtype, state_dtype in (
            ("auto", "float32"),
            ("int8", "model"),
        ):
            profile_name = f"{kv_dtype}-{state_dtype}"
            result.append(
                (
                    f"pd-export-{profile_name}-tp{tp_size}",
                    [
                        sys.executable,
                        str(CACHE_EXPORT_SCRIPT),
                        "--memory-preflight",
                        str(root / "preflight" / "memory_preflight.json"),
                        "--tp-size",
                        str(tp_size),
                        "--kv-dtype",
                        kv_dtype,
                        "--state-dtype",
                        state_dtype,
                        "--warmup",
                        "2",
                        "--repeats",
                        "10",
                        "--output",
                        str(
                            root
                            / "pd_export"
                            / f"tp{tp_size}"
                            / f"{profile_name}.json"
                        ),
                    ],
                )
            )
            result.append(
                (
                    f"pd-export-bounded-{profile_name}-tp{tp_size}",
                    [
                        sys.executable,
                        str(CACHE_EXPORT_SCRIPT),
                        "--memory-preflight",
                        str(root / "preflight" / "memory_preflight.json"),
                        "--tp-size",
                        str(tp_size),
                        "--kv-dtype",
                        kv_dtype,
                        "--state-dtype",
                        state_dtype,
                        "--warmup",
                        "2",
                        "--repeats",
                        "10",
                        "--max-cached-bytes",
                        "0",
                        "--output",
                        str(
                            root
                            / "pd_export_bounded"
                            / f"tp{tp_size}"
                            / f"{profile_name}.json"
                        ),
                    ],
                )
            )
            result.append(
                (
                    f"pd-transfer-{profile_name}-tp{tp_size}",
                    [
                        sys.executable,
                        str(CACHE_TRANSFER_SCRIPT),
                        "--memory-preflight",
                        str(root / "preflight" / "memory_preflight.json"),
                        "--tp-size",
                        str(tp_size),
                        "--kv-dtype",
                        kv_dtype,
                        "--state-dtype",
                        state_dtype,
                        "--warmup",
                        "2",
                        "--repeats",
                        "10",
                        "--output",
                        str(
                            root
                            / "pd_transfer"
                            / f"tp{tp_size}"
                            / f"{profile_name}.json"
                        ),
                    ],
                )
            )
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
        fairness_modes = (("disabled", 0), ("enabled", FAIRNESS_THRESHOLD))
        for mode, threshold in fairness_modes:
            for repeat in range(1, args.repeats + 1):
                result.append(
                    (
                        f"fairness-{mode}-tp{tp_size}-r{repeat}",
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
                            str(FAIRNESS_INITIAL_SEQUENCES),
                            "--injected-seqs",
                            str(FAIRNESS_INJECTED_SEQUENCES),
                            "--initial-input-len",
                            str(FAIRNESS_INITIAL_INPUT_LENGTH),
                            "--injected-input-len",
                            str(FAIRNESS_INJECTED_INPUT_LENGTH),
                            "--output-len",
                            str(FAIRNESS_OUTPUT_LENGTH),
                            "--temperature",
                            "0",
                            "--inject-after-decode-steps",
                            str(FAIRNESS_INJECT_AFTER_DECODE_STEPS),
                            "--max-model-len",
                            str(args.max_model_len),
                            "--max-num-batched-tokens",
                            str(FAIRNESS_MAX_BATCHED_TOKENS),
                            "--max-num-seqs",
                            str(
                                FAIRNESS_INITIAL_SEQUENCES
                                + FAIRNESS_INJECTED_SEQUENCES
                            ),
                            "--enable-dynamic-chunked-prefill",
                            "--prefill-starvation-threshold",
                            str(threshold),
                            "--prefill-starvation-token-budget",
                            str(FAIRNESS_TOKEN_BUDGET),
                            "--require-paths",
                            (
                                "prefill_eager"
                                if mode == "disabled"
                                else "mixed_eager"
                            ),
                            "--output",
                            str(
                                root
                                / "fairness"
                                / mode
                                / f"tp{tp_size}"
                                / f"r{repeat}.json"
                            ),
                        ],
                    )
                )
        for pressure_name, policy, decode_reservation in (
            ("fcfs", "fcfs", False),
            ("min_recompute", "min_recompute", False),
            ("min_recompute_reserved", "min_recompute", True),
        ):
            for repeat in range(1, args.repeats + 1):
                reservation_args = (
                    ["--enable-decode-kv-reservation"]
                    if decode_reservation
                    else []
                )
                result.append(
                    (
                        f"pressure-{pressure_name}-tp{tp_size}-r{repeat}",
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
                            str(PRESSURE_INITIAL_SEQUENCES),
                            "--injected-seqs",
                            str(PRESSURE_INJECTED_SEQUENCES),
                            "--initial-input-len",
                            "256",
                            "--initial-input-lens",
                            ",".join(
                                str(value)
                                for value in PRESSURE_INITIAL_LENGTHS
                            ),
                            "--injected-input-len",
                            "512",
                            "--injected-input-lens",
                            ",".join(
                                str(value)
                                for value in PRESSURE_INJECTED_LENGTHS
                            ),
                            "--output-len",
                            "16",
                            "--temperature",
                            "0",
                            "--inject-after-decode-steps",
                            "1",
                            "--max-model-len",
                            str(args.max_model_len),
                            "--max-num-batched-tokens",
                            "2048",
                            "--max-num-seqs",
                            str(
                                PRESSURE_INITIAL_SEQUENCES
                                + PRESSURE_INJECTED_SEQUENCES
                            ),
                            "--num-kvcache-blocks-override",
                            str(PRESSURE_KV_BLOCKS),
                            "--preemption-policy",
                            policy,
                            *reservation_args,
                            "--enable-dynamic-chunked-prefill",
                            "--require-paths",
                            "mixed_eager",
                            "--output",
                            str(
                                root
                                / "pressure"
                                / f"tp{tp_size}"
                                / pressure_name
                                / f"r{repeat}.json"
                            ),
                        ],
                    )
                )
        for scheduler_mode in ("baseline", "optimized"):
            for repeat in range(1, args.repeats + 1):
                policy_args = []
                required_paths = "prefill_eager"
                if scheduler_mode == "optimized":
                    policy_args = [
                        "--enable-dynamic-chunked-prefill",
                        "--enable-decode-kv-reservation",
                        "--prefill-starvation-threshold",
                        str(FAIRNESS_THRESHOLD),
                        "--prefill-starvation-token-budget",
                        str(FAIRNESS_TOKEN_BUDGET),
                        "--preemption-policy",
                        "min_recompute",
                    ]
                    required_paths = "mixed_eager"
                result.append(
                    (
                        f"scheduler-{scheduler_mode}-tp{tp_size}-r{repeat}",
                        [
                            sys.executable,
                            str(SCHEDULER_WORKLOAD_SCRIPT),
                            "--model",
                            args.model,
                            "--profile",
                            "mixed",
                            "--tensor-parallel-size",
                            str(tp_size),
                            "--qwen35-moe-decode-backend",
                            "batched",
                            "--temperature",
                            "0",
                            "--max-model-len",
                            str(args.max_model_len),
                            "--max-num-batched-tokens",
                            "2048",
                            "--max-num-seqs",
                            str(args.max_num_seqs),
                            *policy_args,
                            "--require-paths",
                            required_paths,
                            "--output",
                            str(
                                root
                                / "scheduler"
                                / scheduler_mode
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
            result.append(
                (
                    f"cudagraph-conv-channel_accumulate-{context_name}-tp{tp_size}",
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
                        "--qwen35-decode-conv-backend",
                        "channel_accumulate",
                        "--result-dir",
                        str(
                            root
                            / "cudagraph_conv"
                            / "channel_accumulate"
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
                    "--include-decode-conv-candidate",
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
                "quality-conv-channel_accumulate",
                [
                    sys.executable,
                    str(QUALITY_SCRIPT),
                    "--model",
                    args.model,
                    "--tp-sizes",
                    tp_sizes,
                    "--run-id",
                    f"{args.run_id}-conv-channel_accumulate",
                    "--result-dir",
                    str(root / "quality_conv" / "channel_accumulate"),
                    "--no-checkpoint-audit",
                    "--prompt-lengths",
                    f"128,1024,3072,{QUALITY_MAX_PROMPT_LENGTH}",
                    "--continuation-len",
                    str(QUALITY_CONTINUATION_LENGTH),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-batched-tokens",
                    str(args.max_model_len),
                    "--qwen35-decode-conv-backend",
                    "channel_accumulate",
                ],
            ),
        )
    )
    if args.gptq_model is not None:
        gptq_run_id = f"{args.run_id}-gptq"
        gptq_root = root / "gptq"
        result.extend(
            (
                (
                    "official-gptq-checkpoint-audit",
                    [
                        sys.executable,
                        str(REMOTE_CHECKPOINT_AUDIT_SCRIPT),
                        "--repo",
                        OFFICIAL_GPTQ_CHECKPOINT_REPO,
                        "--revision",
                        args.gptq_revision,
                        "--tp-sizes",
                        tp_sizes,
                        "--output",
                        str(gptq_root / "official_checkpoint_header_audit.json"),
                    ],
                ),
                (
                    "gptq-preflight",
                    [
                        sys.executable,
                        str(MATRIX_SCRIPT),
                        "--model",
                        args.gptq_model,
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
                        "--run-id",
                        gptq_run_id,
                        "--result-dir",
                        str(gptq_root / "preflight"),
                        "--weight-quant-backend",
                        "auto",
                        "--preflight-only",
                        "--verify-checkpoint-shards",
                    ],
                ),
                (
                    "gptq-performance-matrix",
                    [
                        sys.executable,
                        str(MATRIX_SCRIPT),
                        "--model",
                        args.gptq_model,
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
                        gptq_run_id,
                        "--result-dir",
                        str(gptq_root / "performance"),
                        "--weight-quant-backend",
                        "auto",
                        "--no-checkpoint-audit",
                        "--no-memory-preflight",
                    ],
                ),
                (
                    "gptq-quality-matrix",
                    [
                        sys.executable,
                        str(QUALITY_SCRIPT),
                        "--model",
                        args.gptq_model,
                        "--tp-sizes",
                        tp_sizes,
                        "--run-id",
                        gptq_run_id,
                        "--result-dir",
                        str(gptq_root / "quality"),
                        "--weight-quant-backend",
                        "auto",
                        "--qwen35-moe-decode-backend",
                        "sorted",
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
                    "gptq-vs-bf16-quality",
                    [
                        sys.executable,
                        str(CHECKPOINT_QUALITY_SCRIPT),
                        "--baseline-dir",
                        str(root / "quality"),
                        "--baseline-run-id",
                        args.run_id,
                        "--candidate-dir",
                        str(gptq_root / "quality"),
                        "--candidate-run-id",
                        gptq_run_id,
                        "--tp-sizes",
                        tp_sizes,
                        "--output",
                        str(gptq_root / "quality" / "bf16_vs_gptq.json"),
                    ],
                ),
            )
        )
    fp8_audit_model = args.fp8_audit_model
    if args.fp8_model is not None and fp8_audit_model is None:
        fp8_audit_model = OFFICIAL_FP8_CHECKPOINT_REPO
    if fp8_audit_model is not None:
        result.append(
            (
                "official-fp8-checkpoint-audit",
                [
                    sys.executable,
                    str(REMOTE_CHECKPOINT_AUDIT_SCRIPT),
                    "--repo",
                    fp8_audit_model,
                    "--revision",
                    args.fp8_revision,
                    "--tp-sizes",
                    tp_sizes,
                    "--fp8-runtime-backend",
                    fp8_runtime_backend,
                    "--output",
                    str(root / "fp8" / "official_checkpoint_header_audit.json"),
                ],
            )
        )
    if args.fp8_model is not None:
        fp8_run_id = f"{args.run_id}-fp8"
        fp8_root = root / "fp8"
        result.extend(
            (
                (
                    "fp8-preflight",
                    [
                        sys.executable,
                        str(MATRIX_SCRIPT),
                        "--model",
                        args.fp8_model,
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
                        "--run-id",
                        fp8_run_id,
                        "--result-dir",
                        str(fp8_root / "preflight"),
                        "--weight-quant-backend",
                        fp8_runtime_backend,
                        "--preflight-only",
                        "--verify-checkpoint-shards",
                    ],
                ),
                (
                    "fp8-performance-matrix",
                    [
                        sys.executable,
                        str(MATRIX_SCRIPT),
                        "--model",
                        args.fp8_model,
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
                        fp8_run_id,
                        "--result-dir",
                        str(fp8_root / "performance"),
                        "--weight-quant-backend",
                        fp8_runtime_backend,
                        "--no-checkpoint-audit",
                        "--no-memory-preflight",
                    ],
                ),
                (
                    "fp8-quality-matrix",
                    [
                        sys.executable,
                        str(QUALITY_SCRIPT),
                        "--model",
                        args.fp8_model,
                        "--tp-sizes",
                        tp_sizes,
                        "--run-id",
                        fp8_run_id,
                        "--result-dir",
                        str(fp8_root / "quality"),
                        "--weight-quant-backend",
                        fp8_runtime_backend,
                        "--qwen35-moe-decode-backend",
                        "sorted",
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
                    "fp8-vs-bf16-quality",
                    [
                        sys.executable,
                        str(CHECKPOINT_QUALITY_SCRIPT),
                        "--baseline-dir",
                        str(root / "quality"),
                        "--baseline-run-id",
                        args.run_id,
                        "--candidate-dir",
                        str(fp8_root / "quality"),
                        "--candidate-run-id",
                        fp8_run_id,
                        "--tp-sizes",
                        tp_sizes,
                        "--output",
                        str(fp8_root / "quality" / "bf16_vs_fp8.json"),
                    ],
                ),
            )
        )
    result.append(
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
            )
    )
    return result


def manifest_plan(
    args: argparse.Namespace,
    stages: list[tuple[str, list[str]]],
) -> dict:
    return {
        "run_id": args.run_id,
        "model": canonical_model_reference(args.model),
        "gptq_model": (
            canonical_model_reference(args.gptq_model)
            if args.gptq_model is not None
            else None
        ),
        "gptq_revision": args.gptq_revision if args.gptq_model else None,
        "fp8_audit_model": (
            args.fp8_audit_model
            if args.fp8_audit_model is not None
            else (
                OFFICIAL_FP8_CHECKPOINT_REPO
                if args.fp8_model is not None
                else None
            )
        ),
        "fp8_model": (
            canonical_model_reference(args.fp8_model)
            if args.fp8_model is not None
            else None
        ),
        "fp8_revision": (
            args.fp8_revision
            if args.fp8_audit_model is not None or args.fp8_model is not None
            else None
        ),
        "fp8_runtime_backend": getattr(
            args, "fp8_runtime_backend", "reference"
        ),
        "source_tree_sha256": source_tree_sha256(),
        "stages": [
            {"name": name, "command": command}
            for name, command in stages
        ],
    }


def canonical_model_reference(model: str) -> str:
    """Resolve local checkpoints without rewriting Hub repository IDs."""

    expanded = Path(model).expanduser()
    if expanded.exists() or expanded.is_absolute() or model.startswith((".", "~")):
        return str(expanded.resolve())
    return model


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return 0 if not devices or devices == ["-1"] else len(devices)

    import torch

    return torch.cuda.device_count()


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
    elif stage_name == "official-gptq-checkpoint-audit":
        required = [root / "gptq" / "official_checkpoint_header_audit.json"]
        search_root = root / "gptq"
    elif stage_name == "official-fp8-checkpoint-audit":
        required = [root / "fp8" / "official_checkpoint_header_audit.json"]
        search_root = root / "fp8"
    elif stage_name == "fp8-preflight":
        search_root = root / "fp8" / "preflight"
        required = [
            search_root / "checkpoint_mapping_audit.json",
            search_root / "memory_preflight.json",
        ]
    elif stage_name == "gptq-preflight":
        search_root = root / "gptq" / "preflight"
        required = [
            search_root / "checkpoint_mapping_audit.json",
            search_root / "memory_preflight.json",
        ]
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
    elif stage_name.startswith("pd-transfer-"):
        profile_name, tp_name = stage_name.removeprefix("pd-transfer-").rsplit(
            "-", 1
        )
        search_root = root / "pd_transfer" / tp_name
        required = [search_root / f"{profile_name}.json"]
    elif stage_name.startswith("pd-export-"):
        profile_name, tp_name = stage_name.removeprefix("pd-export-").rsplit(
            "-", 1
        )
        search_root = root / "pd_export" / tp_name
        required = [search_root / f"{profile_name}.json"]
    elif stage_name.startswith("attention-"):
        _, context_name, tp_name = stage_name.split("-")
        search_root = root / "attention" / tp_name
        required = [search_root / f"{context_name}.json"]
    elif stage_name.startswith("mixed-tp"):
        tp_name, repeat_name = stage_name.removeprefix("mixed-").rsplit("-", 1)
        search_root = root / "mixed" / tp_name
        required = [search_root / f"{repeat_name}.json"]
    elif stage_name.startswith("scheduler-"):
        mode, tp_name, repeat_name = stage_name.removeprefix(
            "scheduler-"
        ).rsplit("-", 2)
        search_root = root / "scheduler" / mode / tp_name
        required = [search_root / f"{repeat_name}.json"]
    elif stage_name.startswith("pressure-"):
        policy, tp_name, repeat_name = stage_name.removeprefix(
            "pressure-"
        ).rsplit("-", 2)
        search_root = root / "pressure" / tp_name / policy
        required = [search_root / f"{repeat_name}.json"]
    elif stage_name.startswith("fairness-"):
        mode, tp_name, repeat_name = stage_name.removeprefix(
            "fairness-"
        ).rsplit("-", 2)
        search_root = root / "fairness" / mode / tp_name
        required = [search_root / f"{repeat_name}.json"]
    elif stage_name.startswith("cudagraph-conv-channel_accumulate-"):
        context_name, tp_name = stage_name.removeprefix(
            "cudagraph-conv-channel_accumulate-"
        ).rsplit("-", 1)
        search_root = (
            root
            / "cudagraph_conv"
            / "channel_accumulate"
            / tp_name
            / context_name
        )
        required = sorted(search_root.glob("run_*/summary.json"))
        if len(required) != 1:
            raise RuntimeError(
                f"stage {stage_name} must produce exactly one summary"
            )
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
    elif stage_name == "quality-conv-channel_accumulate":
        search_root = root / "quality_conv" / "channel_accumulate"
        required = [
            search_root
            / f"{args.run_id}-conv-channel_accumulate_summary.json"
        ]
    elif stage_name == "gptq-performance-matrix":
        search_root = root / "gptq" / "performance"
        required = [search_root / f"{args.run_id}-gptq_matrix_summary.json"]
    elif stage_name == "gptq-quality-matrix":
        search_root = root / "gptq" / "quality"
        required = [search_root / f"{args.run_id}-gptq_summary.json"]
    elif stage_name == "gptq-vs-bf16-quality":
        search_root = root / "gptq" / "quality"
        required = [search_root / "bf16_vs_gptq.json"]
    elif stage_name == "fp8-performance-matrix":
        search_root = root / "fp8" / "performance"
        required = [search_root / f"{args.run_id}-fp8_matrix_summary.json"]
    elif stage_name == "fp8-quality-matrix":
        search_root = root / "fp8" / "quality"
        required = [search_root / f"{args.run_id}-fp8_summary.json"]
    elif stage_name == "fp8-vs-bf16-quality":
        search_root = root / "fp8" / "quality"
        required = [search_root / "bf16_vs_fp8.json"]
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
            "official-gptq-checkpoint-audit",
            "official-fp8-checkpoint-audit",
            "fp8-preflight",
            "fp8-vs-bf16-quality",
            "gptq-preflight",
            "gptq-vs-bf16-quality",
            "preflight",
            "final-summary",
        )
        or stage_name.startswith("kernels-")
        or stage_name.startswith("pd-transfer-")
        or stage_name.startswith("attention-")
        or stage_name.startswith("mixed-")
        or stage_name.startswith("scheduler-")
        or stage_name.startswith("pressure-")
        or stage_name.startswith("fairness-")
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
            "gptq_model": manifest.get("gptq_model"),
            "gptq_revision": manifest.get("gptq_revision"),
            "fp8_audit_model": manifest.get("fp8_audit_model"),
            "fp8_model": manifest.get("fp8_model"),
            "fp8_revision": manifest.get("fp8_revision"),
            "fp8_runtime_backend": manifest.get(
                "fp8_runtime_backend", "reference"
            ),
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
    parser.add_argument(
        "--model",
        required=True,
        help="Local Qwen3.6-35B-A3B BF16 checkpoint directory.",
    )
    parser.add_argument(
        "--gptq-model",
        default=None,
        help=(
            "Optional legacy Qwen3.5 GPTQ-Int4 compatibility checkpoint; "
            "this is not an official Qwen3.6 validation track."
        ),
    )
    parser.add_argument(
        "--gptq-revision",
        default=OFFICIAL_GPTQ_CHECKPOINT_REVISION,
    )
    parser.add_argument(
        "--fp8-audit-model",
        default=None,
        help=(
            "Optional official FP8 checkpoint whose remote headers are audited; "
            "this does not enable FP8 execution."
        ),
    )
    parser.add_argument(
        "--fp8-model",
        default=None,
        help=(
            "Optional local official FP8 checkpoint to validate with the "
            "BF16 reference-dequantization backend."
        ),
    )
    parser.add_argument(
        "--fp8-revision",
        default=OFFICIAL_FP8_CHECKPOINT_REVISION,
    )
    parser.add_argument(
        "--fp8-runtime-backend",
        choices=("reference", "resident"),
        default="reference",
        help="FP8 execution layout used by audit, performance, and quality stages.",
    )
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, default=(4, 8))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--sampling-chunk-size", type=int, default=32)
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
        args.sampling_chunk_size,
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
    if not args.dry_run:
        required_gpus = max(args.tp_sizes)
        available_gpus = visible_gpu_count()
        if available_gpus < required_gpus:
            raise SystemExit(
                f"rental validation requires {required_gpus} visible GPUs, "
                f"but found {available_gpus}; no checkpoint or benchmark work started"
            )
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
