import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "benchmark_baseline.py"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_checkpoint_mapping.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_benchmark_runs.py"
INT8_PARTITION_THRESHOLD = 8192
KVCACHE_BLOCK_SIZE = 256


@dataclass(frozen=True)
class BenchmarkCase:
    tensor_parallel_size: int
    recurrent_state_dtype: str
    kv_cache_dtype: str
    moe_decode_backend: str = "sorted"

    @property
    def name(self) -> str:
        kv = "bf16" if self.kv_cache_dtype == "auto" else self.kv_cache_dtype
        suffix = (
            ""
            if self.moe_decode_backend == "sorted"
            else f"_moe-{self.moe_decode_backend}"
        )
        return (
            f"qwen35_tp{self.tensor_parallel_size}_"
            f"state-{self.recurrent_state_dtype}_kv-{kv}{suffix}"
        )


def comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def build_cases(
    tp_sizes: tuple[int, ...],
    *,
    include_moe_candidate: bool = False,
) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase(tp, state_dtype, kv_dtype)
        for tp, state_dtype, kv_dtype in itertools.product(
            tp_sizes,
            ("float32", "model"),
            ("auto", "int8"),
        )
    ]
    if include_moe_candidate:
        cases.extend(
            BenchmarkCase(
                tp,
                "model",
                kv_dtype,
                moe_decode_backend="batched",
            )
            for tp, kv_dtype in itertools.product(
                tp_sizes,
                ("auto", "int8"),
            )
        )
    return cases


def command_for_case(
    args: argparse.Namespace,
    case: BenchmarkCase,
    repeat: int = 1,
    run_id: str | None = None,
) -> list[str]:
    repeats = getattr(args, "repeats", 1)
    name = f"{case.name}_r{repeat}" if repeats > 1 else case.name
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(case.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--recurrent-state-dtype",
        case.recurrent_state_dtype,
        "--qwen35-moe-decode-backend",
        case.moe_decode_backend,
        "--weight-quant-backend",
        getattr(args, "weight_quant_backend", "auto"),
        "--kv-cache-dtype",
        case.kv_cache_dtype,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--sampling-chunk-size",
        str(getattr(args, "sampling_chunk_size", 32)),
        "--seed",
        str(args.seed),
        "--name",
        name,
        "--result-dir",
        args.result_dir,
        "--require-paths",
        ",".join(required_paths(args, case)),
    ]
    if case.moe_decode_backend == "sorted":
        command.append("--enforce-eager")
    if not args.warmup:
        command.append("--no-warmup")
    if run_id is not None:
        command.extend(("--output-stem", result_stem(run_id, case, repeat)))
    return command


def result_stem(run_id: str, case: BenchmarkCase, repeat: int) -> str:
    return f"{run_id}_{case.name}_r{repeat}"


def summary_command(
    args: argparse.Namespace,
    case: BenchmarkCase,
    run_id: str,
) -> list[str]:
    result_dir = Path(args.result_dir)
    paths = [
        result_dir / f"{result_stem(run_id, case, repeat)}.json"
        for repeat in range(1, args.repeats + 1)
    ]
    return [
        sys.executable,
        str(COMPARE_SCRIPT),
        *(str(path) for path in paths),
        "--summarize-repeats",
        "--require-output-parity",
        "--output",
        str(result_dir / f"{run_id}_{case.name}_summary.json"),
    ]


def matrix_summary_command(
    args: argparse.Namespace,
    cases: list[BenchmarkCase],
    run_id: str,
) -> list[str]:
    result_dir = Path(args.result_dir)
    summaries = [
        result_dir / f"{run_id}_{case.name}_summary.json" for case in cases
    ]
    return [
        sys.executable,
        str(COMPARE_SCRIPT),
        *(str(path) for path in summaries),
        "--compare-repeat-summaries",
        "--output",
        str(result_dir / f"{run_id}_matrix_summary.json"),
    ]


def required_paths(
    args: argparse.Namespace,
    case: BenchmarkCase,
) -> tuple[str, ...]:
    paths = [
        "prefill_eager",
        (
            "decode_cuda_graph"
            if case.moe_decode_backend == "batched"
            else "decode_eager"
        ),
        "prefill_contiguous_view",
        (
            "decode_graph_indexed"
            if case.moe_decode_backend == "batched"
            else "decode_contiguous_view"
        ),
    ]
    if case.kv_cache_dtype == "auto":
        paths.extend(("float_flash_prefill", "float_flash_decode"))
    else:
        paths.append("int8_prefill")
        paths.append(
            "int8_partitioned_decode"
            if args.input_len >= INT8_PARTITION_THRESHOLD
            else "int8_fused_decode"
        )
    return tuple(paths)


def checkpoint_audit_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--model",
        args.model,
        "--tp-sizes",
        ",".join(str(size) for size in args.tp_sizes),
        "--require-shards",
        "--weight-quant-backend",
        getattr(args, "weight_quant_backend", "auto"),
        "--output",
        str(Path(args.result_dir) / "checkpoint_mapping_audit.json"),
    ]
    if args.verify_checkpoint_shards:
        command.append("--verify-shard-hashes")
    return command


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return 0 if devices == ["-1"] else len(devices)

    import torch

    return torch.cuda.device_count()


def gpu_memory_info_bytes() -> list[dict[str, int]]:
    import torch

    return [
        {"free": int(free), "total": int(total)}
        for device in range(torch.cuda.device_count())
        for free, total in (torch.cuda.mem_get_info(device),)
    ]


def validate_memory_capacity(
    audit_report: dict,
    tp_sizes: tuple[int, ...],
    memory_by_device: list[dict[str, int]],
    headroom_bytes: int,
    max_num_seqs: int,
    num_seqs: int,
    sequence_length: int,
    gpu_memory_utilization: float,
    recurrent_padding_slots: int = 0,
    configured_max_model_len: int | None = None,
    transfer_context_length: int | None = None,
) -> dict:
    if headroom_bytes < 0:
        raise ValueError("memory headroom must be non-negative")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if recurrent_padding_slots < 0:
        raise ValueError("recurrent padding slots must be non-negative")
    if num_seqs <= 0 or sequence_length <= 0:
        raise ValueError("workload dimensions must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if configured_max_model_len is not None and configured_max_model_len <= 0:
        raise ValueError("configured_max_model_len must be positive")
    if transfer_context_length is None:
        transfer_context_length = sequence_length
    if transfer_context_length <= 0 or transfer_context_length > sequence_length:
        raise ValueError(
            "transfer_context_length must be positive and no greater than sequence_length"
        )
    results = {}
    for tp_size in tp_sizes:
        if len(memory_by_device) < tp_size:
            raise ValueError(f"TP={tp_size} requires {tp_size} visible GPUs")
        audit = audit_report.get("results", {}).get(f"tp{tp_size}", {})
        parameter_bytes = audit.get("local_parameter_bytes")
        if not isinstance(parameter_bytes, int) or parameter_bytes <= 0:
            raise ValueError(f"checkpoint audit has no TP={tp_size} parameter size")
        runtime_model_bytes = audit.get(
            "local_parameter_and_resident_runtime_bytes",
            parameter_bytes,
        )
        if (
            not isinstance(runtime_model_bytes, int)
            or runtime_model_bytes < parameter_bytes
        ):
            raise ValueError(
                f"checkpoint audit has invalid TP={tp_size} runtime model size"
            )
        state_sizes = audit.get("state_bytes_per_sequence", {})
        if not all(
            isinstance(state_sizes.get(dtype), int)
            and state_sizes[dtype] >= 0
            for dtype in ("float32", "model")
        ):
            raise ValueError(f"checkpoint audit has no TP={tp_size} state size")
        state_bytes_per_sequence = max(
            state_sizes["float32"],
            state_sizes["model"],
        )
        state_slot_count = max_num_seqs + recurrent_padding_slots
        state_bytes_by_dtype = {
            dtype: state_sizes[dtype] * state_slot_count
            for dtype in ("float32", "model")
        }
        state_bytes = state_bytes_per_sequence * state_slot_count
        model_max_position_embeddings = audit.get(
            "model_max_position_embeddings"
        )
        if (
            not isinstance(model_max_position_embeddings, int)
            or model_max_position_embeddings <= 0
        ):
            raise ValueError(
                f"checkpoint audit has no TP={tp_size} model context limit"
            )
        runtime_limit = min(
            configured_max_model_len
            if configured_max_model_len is not None
            else model_max_position_embeddings,
            model_max_position_embeddings,
        )
        rotary_bytes = audit.get("rotary_cache_bytes_per_position", 0) * runtime_limit
        kv_sizes = audit.get("kv_bytes_per_token_by_dtype")
        if kv_sizes is None:
            kv_sizes = {"auto": audit.get("kv_bytes_per_token")}
        if not isinstance(kv_sizes, dict) or not kv_sizes or not all(
            isinstance(value, int) and value > 0
            for value in kv_sizes.values()
        ):
            raise ValueError(f"checkpoint audit has no TP={tp_size} KV size")
        kv_bytes_per_token = max(kv_sizes.values())
        kv_data_sizes = audit.get("kv_data_bytes_per_token_by_dtype", kv_sizes)
        kv_scale_sizes = audit.get(
            "kv_scale_bytes_per_token_by_dtype",
            {dtype: 0 for dtype in kv_sizes},
        )
        if (
            set(kv_data_sizes) != set(kv_sizes)
            or set(kv_scale_sizes) != set(kv_sizes)
            or any(
                not isinstance(kv_data_sizes[dtype], int)
                or kv_data_sizes[dtype] <= 0
                or not isinstance(kv_scale_sizes[dtype], int)
                or kv_scale_sizes[dtype] < 0
                or kv_data_sizes[dtype] + kv_scale_sizes[dtype]
                != kv_sizes[dtype]
                for dtype in kv_sizes
            )
        ):
            raise ValueError(
                f"checkpoint audit has invalid TP={tp_size} KV components"
            )
        convolution_bytes_per_sequence = state_sizes.get("convolution", 0)
        if (
            not isinstance(convolution_bytes_per_sequence, int)
            or convolution_bytes_per_sequence < 0
            or any(
                convolution_bytes_per_sequence > state_sizes[dtype]
                for dtype in ("float32", "model")
            )
        ):
            raise ValueError(
                f"checkpoint audit has invalid TP={tp_size} state components"
            )
        blocks_per_sequence = (
            sequence_length + KVCACHE_BLOCK_SIZE - 1
        ) // KVCACHE_BLOCK_SIZE
        transfer_blocks_per_sequence = (
            transfer_context_length + KVCACHE_BLOCK_SIZE - 1
        ) // KVCACHE_BLOCK_SIZE
        aligned_transfer_tokens = (
            transfer_blocks_per_sequence * KVCACHE_BLOCK_SIZE
        )
        concurrent_sequences = min(num_seqs, max_num_seqs)
        kv_bytes = (
            concurrent_sequences
            * blocks_per_sequence
            * KVCACHE_BLOCK_SIZE
            * kv_bytes_per_token
        )
        pd_transfer_bytes_per_sequence_by_dtype = {
            kv_dtype: {
                state_dtype: (
                    aligned_transfer_tokens * bytes_per_token
                    + state_sizes[state_dtype]
                )
                for state_dtype in ("float32", "model")
            }
            for kv_dtype, bytes_per_token in kv_sizes.items()
        }
        pd_transfer_components_per_sequence_by_dtype = {
            kv_dtype: {
                state_dtype: {
                    "kv": aligned_transfer_tokens
                    * kv_data_sizes[kv_dtype],
                    "kv_scales": aligned_transfer_tokens
                    * kv_scale_sizes[kv_dtype],
                    "recurrent": state_sizes[state_dtype]
                    - convolution_bytes_per_sequence,
                    "convolution": convolution_bytes_per_sequence,
                    "total": pd_transfer_bytes_per_sequence_by_dtype[kv_dtype][
                        state_dtype
                    ],
                }
                for state_dtype in ("float32", "model")
            }
            for kv_dtype in kv_sizes
        }
        pd_transfer_bytes_all_tp_ranks_by_dtype = {
            kv_dtype: {
                state_dtype: bytes_per_rank * tp_size
                for state_dtype, bytes_per_rank in by_state_dtype.items()
            }
            for kv_dtype, by_state_dtype in (
                pd_transfer_bytes_per_sequence_by_dtype.items()
            )
        }
        required_bytes = (
            runtime_model_bytes
            + state_bytes
            + rotary_bytes
            + kv_bytes
            + headroom_bytes
        )
        memory = memory_by_device[:tp_size]
        available_budgets = [
            max(
                int(item["total"] * gpu_memory_utilization)
                - (item["total"] - item["free"]),
                0,
            )
            for item in memory
        ]
        minimum_available_budget = min(available_budgets)
        fixed_bytes = (
            runtime_model_bytes + state_bytes + rotary_bytes + headroom_bytes
        )
        limiting_rank = min(
            range(tp_size), key=available_budgets.__getitem__
        )
        fixed_budget_margin_by_rank = [
            available - fixed_bytes for available in available_budgets
        ]
        available_kv_bytes = max(minimum_available_budget - fixed_bytes, 0)
        kv_capacity_by_dtype = {}
        for dtype, bytes_per_token in kv_sizes.items():
            capacity_blocks = available_kv_bytes // (
                KVCACHE_BLOCK_SIZE * bytes_per_token
            )
            kv_capacity_by_dtype[dtype] = {
                "memory_limited_total_token_slots": (
                    capacity_blocks * KVCACHE_BLOCK_SIZE
                ),
                "memory_limited_context_tokens_per_sequence": (
                    capacity_blocks // concurrent_sequences
                ) * KVCACHE_BLOCK_SIZE,
                "memory_limited_concurrent_sequences_at_workload_length": (
                    capacity_blocks // blocks_per_sequence
                ),
            }
            memory_limit = kv_capacity_by_dtype[dtype][
                "memory_limited_context_tokens_per_sequence"
            ]
            kv_capacity_by_dtype[dtype]["effective_context_tokens_per_sequence"] = min(
                memory_limit,
                model_max_position_embeddings,
                runtime_limit,
            )
            kv_capacity_by_dtype[dtype][
                "effective_concurrent_sequences_at_workload_length"
            ] = min(
                capacity_blocks // blocks_per_sequence,
                max_num_seqs,
            )
        workload_budget_margin_by_rank = [
            available - required_bytes for available in available_budgets
        ]
        shortfall_bytes_by_rank = [
            max(-margin, 0) for margin in workload_budget_margin_by_rank
        ]
        insufficient = [
            rank for rank, available in enumerate(available_budgets)
            if available < required_bytes
        ]
        fixed_insufficient = [
            rank for rank, margin in enumerate(fixed_budget_margin_by_rank)
            if margin < 0
        ]
        results[f"tp{tp_size}"] = {
            "local_parameter_bytes": parameter_bytes,
            "runtime_model_bytes": runtime_model_bytes,
            "resident_runtime_overhead_bytes": runtime_model_bytes
            - parameter_bytes,
            "max_state_bytes_per_rank": state_bytes,
            "state_bytes_per_rank_by_dtype": state_bytes_by_dtype,
            "rotary_cache_bytes_per_rank": rotary_bytes,
            "minimum_workload_kv_bytes_per_rank": kv_bytes,
            "kv_bytes_per_token_by_dtype": kv_sizes,
            "pd_transfer_bytes_per_sequence_by_dtype": (
                pd_transfer_bytes_per_sequence_by_dtype
            ),
            "pd_transfer_bytes_all_tp_ranks_by_dtype": (
                pd_transfer_bytes_all_tp_ranks_by_dtype
            ),
            "pd_transfer_components_per_sequence_by_dtype": (
                pd_transfer_components_per_sequence_by_dtype
            ),
            "pd_transfer_context_tokens": transfer_context_length,
            "pd_transfer_allocated_tokens": aligned_transfer_tokens,
            "kv_capacity_by_dtype": kv_capacity_by_dtype,
            "capacity_concurrent_sequences": concurrent_sequences,
            "model_max_position_embeddings": model_max_position_embeddings,
            "configured_max_model_len": configured_max_model_len,
            "max_num_seqs": max_num_seqs,
            "recurrent_padding_slots": recurrent_padding_slots,
            "allocated_state_slot_count": state_slot_count,
            "headroom_bytes": headroom_bytes,
            "required_free_bytes_per_rank": required_bytes,
            "gpu_memory_utilization": gpu_memory_utilization,
            "memory_by_rank": memory,
            "available_budget_bytes_by_rank": available_budgets,
            "fixed_required_bytes_per_rank": fixed_bytes,
            "fixed_budget_margin_bytes_by_rank": fixed_budget_margin_by_rank,
            "workload_budget_margin_bytes_by_rank": (
                workload_budget_margin_by_rank
            ),
            "shortfall_bytes_by_rank": shortfall_bytes_by_rank,
            "limiting_rank": limiting_rank,
            "limiting_available_budget_bytes": minimum_available_budget,
            "valid": not insufficient,
        }
        if insufficient:
            ranks = ", ".join(str(rank) for rank in insufficient)
            shortfalls = ", ".join(
                f"rank {rank}: {shortfall_bytes_by_rank[rank]} bytes"
                for rank in insufficient
            )
            if fixed_insufficient:
                fixed_ranks = ", ".join(str(rank) for rank in fixed_insufficient)
                fixed_shortfalls = ", ".join(
                    f"rank {rank}: {-fixed_budget_margin_by_rank[rank]} bytes"
                    for rank in fixed_insufficient
                )
                raise ValueError(
                    f"TP={tp_size} ranks {fixed_ranks}: fixed "
                    "model/state/headroom exceeds the usable budget; "
                    f"fixed shortfall by rank: {fixed_shortfalls}"
                )
            raise ValueError(
                f"TP={tp_size} ranks {ranks} lack free memory for model "
                "parameters, recurrent state, KV cache, and configured "
                f"headroom; shortfall by rank: {shortfalls}"
            )
    return {"valid": True, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Qwen3.6 TP4/TP8 benchmark matrix."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--verify-checkpoint-shards", action="store_true")
    parser.add_argument("--tp-sizes", type=comma_separated_ints, default=(4, 8))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--sampling-chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--weight-quant-backend",
        choices=("auto", "reference", "resident", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Concurrent state slots; defaults to --num-seqs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--include-moe-candidate",
        action="store_true",
        help=(
            "Add one graph-safe batched MoE decode case per TP size using "
            "model-dtype recurrent state and BF16 KV cache."
        ),
    )
    parser.add_argument("--result-dir", default="benchmark_results/qwen35_matrix")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable identifier for one complete matrix; generated by default.",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--checkpoint-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate checkpoint name mappings before allocating GPU weights.",
    )
    parser.add_argument(
        "--memory-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject GPUs that cannot fit local parameters plus headroom.",
    )
    parser.add_argument("--memory-headroom-gib", type=float, default=2.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run checkpoint, GPU-count, and memory checks without benchmarks.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_num_seqs is None:
        args.max_num_seqs = args.num_seqs
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.sampling_chunk_size <= 0:
        raise ValueError("--sampling-chunk-size must be positive")
    if args.num_seqs <= 0 or args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("workload sizes must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError("input_len plus output_len cannot exceed max_model_len")
    if args.memory_headroom_gib < 0:
        raise ValueError("--memory-headroom-gib must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.run_id is not None and (
        not args.run_id
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in args.run_id)
    ):
        raise ValueError("--run-id may contain only letters, digits, dot, dash, and underscore")
    return args


def main() -> None:
    try:
        args = normalize_args(parse_args())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = build_cases(
        args.tp_sizes,
        include_moe_candidate=args.include_moe_candidate,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "qwen35_%Y%m%dT%H%M%S%fZ"
    )
    if not args.dry_run:
        required_gpus = max(case.tensor_parallel_size for case in cases)
        available_gpus = visible_gpu_count()
        if available_gpus < required_gpus:
            raise SystemExit(
                f"benchmark matrix requires {required_gpus} visible GPUs, "
                f"but found {available_gpus}; use --tp-sizes to select a runnable subset"
            )

    if args.checkpoint_audit:
        command = checkpoint_audit_command(args)
        print("[preflight] checkpoint mapping audit", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)
            if args.memory_preflight:
                audit_path = Path(args.result_dir) / "checkpoint_mapping_audit.json"
                audit_report = json.loads(audit_path.read_text())
                memory_report = validate_memory_capacity(
                    audit_report,
                    args.tp_sizes,
                    gpu_memory_info_bytes(),
                    int(args.memory_headroom_gib * 1024**3),
                    args.max_num_seqs,
                    args.num_seqs,
                    args.input_len + args.output_len,
                    args.gpu_memory_utilization,
                    int(args.include_moe_candidate),
                    args.max_model_len,
                    args.input_len,
                )
                memory_path = Path(args.result_dir) / "memory_preflight.json"
                memory_path.write_text(json.dumps(memory_report, indent=2) + "\n")
                print(f"[preflight] wrote {memory_path}", flush=True)

    if args.preflight_only:
        return

    total_runs = len(cases) * args.repeats
    run_index = 0
    for case in cases:
        for repeat in range(1, args.repeats + 1):
            run_index += 1
            command = command_for_case(args, case, repeat, run_id)
            print(
                f"[{run_index}/{total_runs}] {case.name} repeat {repeat}",
                flush=True,
            )
            if args.dry_run:
                print(subprocess.list2cmdline(command))
            else:
                subprocess.run(command, cwd=ROOT, check=True)
        if args.repeats > 1:
            command = summary_command(args, case, run_id)
            print(f"[summary] {case.name}", flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command))
            else:
                subprocess.run(command, cwd=ROOT, check=True)
    if args.repeats > 1 and len(cases) > 1:
        command = matrix_summary_command(args, cases, run_id)
        print("[summary] complete matrix", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
