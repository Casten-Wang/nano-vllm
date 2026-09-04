"""Replay deterministic arrival traces against the nano-vLLM scheduler."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.benchmark_metadata import (
    checkpoint_manifest_metadata,
    collect_benchmark_metadata,
    model_config_metadata,
    validate_execution_stats,
)
from nanovllm.scheduler_workload import (
    PROFILE_NAMES,
    SchedulerWorkload,
    built_in_workload,
    prompt_token_ids,
    replay_scheduler_workload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a deterministic scheduler-arrival workload.",
    )
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3.6-35B-A3B/"),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workload", type=Path)
    source.add_argument("--profile", choices=PROFILE_NAMES, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-engine-steps", type=int, default=100000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--num-kvcache-blocks-override", type=int)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=4,
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--recurrent-state-dtype",
        choices=("float32", "model"),
        default="float32",
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
    parser.add_argument("--qwen35-moe-decode-chunk-size", type=int, default=8)
    parser.add_argument("--sampling-chunk-size", type=int, default=32)
    parser.add_argument(
        "--weight-quant-backend",
        choices=("auto", "reference", "resident", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "int8"),
        default="auto",
    )
    parser.add_argument(
        "--kv-dequant-backend",
        choices=("fused", "triton", "torch"),
        default="fused",
    )
    parser.add_argument(
        "--int8-partitioned-decode-threshold",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--int8-partitioned-decode-partition-size",
        type=int,
        default=512,
    )
    parser.add_argument("--enable-dynamic-chunked-prefill", action="store_true")
    parser.add_argument("--enable-decode-kv-reservation", action="store_true")
    parser.add_argument("--prefill-starvation-threshold", type=int, default=0)
    parser.add_argument(
        "--prefill-starvation-token-budget",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--preemption-policy",
        choices=("fcfs", "min_recompute"),
        default="fcfs",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-paths", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-dir", type=Path, default=Path("benchmark_results"))
    return parser.parse_args()


def load_workload(args: argparse.Namespace) -> SchedulerWorkload:
    if args.workload is None:
        workload = built_in_workload(args.profile)
    else:
        try:
            payload = json.loads(args.workload.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read workload: {error}") from error
        workload = SchedulerWorkload.from_dict(
            payload,
            max_model_len=args.max_model_len,
        )
    # Built-ins use the same validation gate as external traces.
    return SchedulerWorkload.from_dict(
        workload.to_dict(),
        max_model_len=args.max_model_len,
    )


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "vocab_size": args.vocab_size,
        "max_engine_steps": args.max_engine_steps,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "prefill_starvation_token_budget": args.prefill_starvation_token_budget,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.prefill_starvation_threshold < 0:
        raise ValueError("prefill_starvation_threshold must be non-negative")
    if not 0.0 < args.temperature:
        raise ValueError("temperature must be positive")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        workload = load_workload(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint_manifest = checkpoint_manifest_metadata(args.model)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks_override=args.num_kvcache_blocks_override,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        recurrent_state_dtype=args.recurrent_state_dtype,
        qwen35_decode_conv_backend=args.qwen35_decode_conv_backend,
        qwen35_moe_decode_backend=args.qwen35_moe_decode_backend,
        qwen35_moe_decode_chunk_size=args.qwen35_moe_decode_chunk_size,
        sampling_chunk_size=args.sampling_chunk_size,
        weight_quant_backend=args.weight_quant_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
        int8_partitioned_decode_threshold=args.int8_partitioned_decode_threshold,
        int8_partitioned_decode_partition_size=(
            args.int8_partitioned_decode_partition_size
        ),
        enable_dynamic_chunked_prefill=args.enable_dynamic_chunked_prefill,
        enable_decode_kv_reservation=args.enable_decode_kv_reservation,
        prefill_starvation_threshold=args.prefill_starvation_threshold,
        prefill_starvation_token_budget=args.prefill_starvation_token_budget,
        preemption_policy=args.preemption_policy,
    )
    if args.warmup:
        llm.generate(
            [[0]],
            SamplingParams(max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
    prompts = {
        request.request_id: prompt_token_ids(
            request,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        for request in workload.requests
    }
    llm.model_runner.call("reset_execution_stats")
    llm.model_runner.call("reset_shape_trace")
    llm.scheduler.block_manager.reset_cache_stats()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.synchronize()
    llm.model_runner.call("reset_cuda_peak_memory_stats")
    started_at = time.perf_counter()
    replay = replay_scheduler_workload(
        llm,
        workload,
        prompt_factory=lambda request: prompts[request.request_id],
        sampling_params_factory=lambda request: SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=request.output_len,
        ),
        synchronize=torch.cuda.synchronize,
        max_engine_steps=args.max_engine_steps,
    )
    total_time_s = time.perf_counter() - started_at

    final_manifest = checkpoint_manifest_metadata(args.model)
    if final_manifest["digest"] != checkpoint_manifest["digest"]:
        raise RuntimeError("checkpoint files changed during the benchmark run")
    execution_stats = llm.model_runner.call("get_execution_stats")
    required_paths = [
        item.strip() for item in args.require_paths.split(",") if item.strip()
    ]
    execution_validation = validate_execution_stats(
        execution_stats,
        required_paths,
    )
    cuda_memory_by_rank = llm.model_runner.call("get_cuda_memory_stats")
    runtime_buffer_by_rank = llm.model_runner.call("get_runtime_buffer_stats_by_rank")
    kv_cache_by_rank = llm.model_runner.call("get_kv_cache_stats_by_rank")
    result = {
        **collect_benchmark_metadata(),
        "model": args.model,
        "checkpoint_manifest": checkpoint_manifest,
        "model_config": model_config_metadata(llm.model_runner.config.hf_config),
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "num_kvcache_blocks_override": args.num_kvcache_blocks_override,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "recurrent_state_dtype": args.recurrent_state_dtype,
        "qwen35_decode_conv_backend": args.qwen35_decode_conv_backend,
        "qwen35_moe_decode_backend": args.qwen35_moe_decode_backend,
        "qwen35_moe_decode_chunk_size": args.qwen35_moe_decode_chunk_size,
        "sampling_chunk_size": args.sampling_chunk_size,
        "requested_weight_quant_backend": args.weight_quant_backend,
        "weight_quant_backend": llm.model_runner.config.weight_quant_backend,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_dequant_backend": args.kv_dequant_backend,
        "enable_dynamic_chunked_prefill": args.enable_dynamic_chunked_prefill,
        "enable_decode_kv_reservation": args.enable_decode_kv_reservation,
        "prefill_starvation_threshold": args.prefill_starvation_threshold,
        "prefill_starvation_token_budget": args.prefill_starvation_token_budget,
        "preemption_policy": args.preemption_policy,
        "enforce_eager": args.enforce_eager,
        "warmup": args.warmup,
        "require_paths": required_paths,
        "total_time_s": total_time_s,
        "output_throughput_tok_s": (
            replay["output_token_ids"]["token_count"] / total_time_s
            if total_time_s > 0.0
            else 0.0
        ),
        "replay": replay,
        "cuda_memory_by_rank": cuda_memory_by_rank,
        "peak_torch_allocated_mib": max(
            item["peak_allocated_bytes"] for item in cuda_memory_by_rank
        )
        / 1024
        / 1024,
        "peak_torch_reserved_mib": max(
            item["peak_reserved_bytes"] for item in cuda_memory_by_rank
        )
        / 1024
        / 1024,
        "runtime_buffer_storage_by_rank": runtime_buffer_by_rank,
        "kv_cache_storage_by_rank": kv_cache_by_rank,
        "prefix_cache": llm.scheduler.block_manager.cache_stats(),
        "execution_stats": execution_stats,
        "shape_trace": llm.model_runner.call("get_shape_trace"),
        "cudagraph_capture_stats": llm.model_runner.call("get_cudagraph_capture_stats"),
        "execution_validation": execution_validation,
    }
    output = args.output
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = args.result_dir / f"scheduler_{workload.name}_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    if not execution_validation["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
