import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.benchmark_metadata import (
    checkpoint_manifest_metadata,
    collect_benchmark_metadata,
    kv_cache_storage_metadata,
    model_config_metadata,
    token_ids_digest,
    validate_execution_stats,
)


def build_prompts(num_seqs: int, input_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, vocab_size - 1) for _ in range(input_len)] for _ in range(num_seqs)]


def make_result_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# nano-vLLM Benchmark",
        "",
        "## Configuration",
        "",
        f"- commit: `{result['commit']}`",
        f"- branch: `{result['branch']}`",
        f"- model: `{result['model']}`",
        f"- checkpoint_digest: `{result['checkpoint_manifest']['digest']}`",
        f"- checkpoint_identity_strength: `{result['checkpoint_manifest']['strength']}`",
        f"- num_seqs: `{result['num_seqs']}`",
        f"- input_len: `{result['input_len']}`",
        f"- output_len: `{result['output_len']}`",
        f"- seed: `{result['seed']}`",
        f"- enforce_eager: `{result['enforce_eager']}`",
        f"- max_model_len: `{result['max_model_len']}`",
        f"- max_num_batched_tokens: `{result['max_num_batched_tokens']}`",
        f"- max_num_seqs: `{result['max_num_seqs']}`",
        f"- tensor_parallel_size: `{result['tensor_parallel_size']}`",
        f"- recurrent_state_dtype: `{result['recurrent_state_dtype']}`",
        f"- kv_cache_dtype: `{result['kv_cache_dtype']}`",
        f"- kv_dequant_backend: `{result['kv_dequant_backend']}`",
        f"- int8_partitioned_decode_threshold: `{result['int8_partitioned_decode_threshold']}`",
        f"- int8_partitioned_decode_partition_size: `{result['int8_partitioned_decode_partition_size']}`",
        f"- sliding_window_size: `{result['sliding_window_size']}`",
        f"- enable_dynamic_chunked_prefill: `{result['enable_dynamic_chunked_prefill']}`",
        f"- execution_valid: `{result['execution_validation']['valid']}`",
        f"- execution_paths: `{','.join(result['execution_validation']['observed_paths'])}`",
        f"- dropped_execution_signature_steps: `{result['execution_validation']['dropped_execution_signature_steps']}`",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_time_s | {result['total_time_s']:.4f} |",
        f"| input_tokens | {result['input_tokens']} |",
        f"| output_tokens | {result['output_tokens']} |",
        f"| generated_token_sha256 | `{result['generated_token_ids']['digest']}` |",
        f"| output_throughput_tok_s | {result['output_throughput_tok_s']:.4f} |",
        f"| peak_torch_allocated_mib | {result['peak_torch_allocated_mib']:.2f} |",
        f"| peak_torch_reserved_mib | {result['peak_torch_reserved_mib']:.2f} |",
        f"| kv_storage_local_rank_mib | {result['kv_cache_storage']['total_mib']:.2f} |",
        f"| kv_storage_estimated_all_ranks_mib | {result['kv_cache_storage']['estimated_all_ranks_mib']:.2f} |",
        f"| recurrent_state_local_rank_mib | {result['recurrent_state_storage']['total_bytes_local_rank'] / 1024 / 1024:.2f} |",
        f"| recurrent_state_all_ranks_mib | {result['recurrent_state_total_all_ranks_bytes'] / 1024 / 1024:.2f} |",
        f"| num_kvcache_blocks | {result['num_kvcache_blocks']} |",
        f"| final_used_kvcache_blocks | {result['final_used_kvcache_blocks']} |",
        f"| final_free_kvcache_blocks | {result['final_free_kvcache_blocks']} |",
        f"| final_kv_block_usage | {result['final_kv_block_usage']:.6f} |",
        f"| peak_used_kvcache_blocks | {result['metrics']['peak_used_kvcache_blocks']} |",
        f"| peak_kv_block_usage | {result['metrics']['peak_kv_block_usage']:.6f} |",
        f"| pure_prefill_throughput_tok_s | {result['metrics']['pure_prefill_throughput_tok_s']:.4f} |",
        f"| pure_decode_throughput_tok_s | {result['metrics']['pure_decode_throughput_tok_s']:.4f} |",
        f"| avg_ttft_s | {result['metrics']['avg_ttft_s']:.6f} |",
        f"| avg_tpot_s | {result['metrics']['avg_tpot_s']:.6f} |",
        f"| avg_request_latency_s | {result['metrics']['avg_request_latency_s']:.6f} |",
        f"| prefix_cache_queries | {result['prefix_cache']['prefix_cache_queries']} |",
        f"| prefix_cache_checked_blocks | {result['prefix_cache']['prefix_cache_checked_blocks']} |",
        f"| prefix_cache_hit_blocks | {result['prefix_cache']['prefix_cache_hit_blocks']} |",
        f"| prefix_cache_hit_rate | {result['prefix_cache']['prefix_cache_hit_rate']:.6f} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable nano-vLLM benchmark.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--recurrent-state-dtype",
        choices=("float32", "model"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--kv-dequant-backend", choices=("fused", "triton", "torch"), default="fused")
    parser.add_argument("--int8-partitioned-decode-threshold", type=int, default=8192)
    parser.add_argument("--int8-partitioned-decode-partition-size", type=int, default=512)
    parser.add_argument("--sliding-window-size", type=int, default=None)
    parser.add_argument("--enable-dynamic-chunked-prefill", action="store_true")
    parser.add_argument("--name", default=None, help="Result file prefix. Defaults to a prefix derived from enabled features.")
    parser.add_argument("--result-dir", default="benchmark_results")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-paths",
        default="",
        help="Comma-separated runtime paths required for a valid result.",
    )
    return parser.parse_args()


def default_result_prefix(args: argparse.Namespace) -> str:
    parts = []
    parts.append(f"int8_{args.kv_dequant_backend}" if args.kv_cache_dtype == "int8" else "baseline")
    if args.sliding_window_size is not None:
        parts.append(f"sw{args.sliding_window_size}")
    if args.enable_dynamic_chunked_prefill:
        parts.append("dynchunk")
    if args.enforce_eager:
        parts.append("eager")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(args.num_seqs, args.input_len, args.vocab_size, args.seed)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
        for _ in range(args.num_seqs)
    ]

    checkpoint_manifest = checkpoint_manifest_metadata(args.model)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        recurrent_state_dtype=args.recurrent_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
        int8_partitioned_decode_threshold=args.int8_partitioned_decode_threshold,
        int8_partitioned_decode_partition_size=args.int8_partitioned_decode_partition_size,
        sliding_window_size=args.sliding_window_size,
        enable_dynamic_chunked_prefill=args.enable_dynamic_chunked_prefill,
    )

    if args.warmup:
        llm.generate([[0]], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)

    # Warmup consumes sampler RNG. Reset immediately before the measured run
    # so --warmup and repeated benchmark invocations generate comparable IDs.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    block_manager = llm.scheduler.block_manager
    block_manager.reset_cache_stats()
    llm.model_runner.call("reset_execution_stats")
    llm.model_runner.call("reset_shape_trace")
    torch.cuda.synchronize()
    llm.model_runner.call("reset_cuda_peak_memory_stats")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    workload_end = time.perf_counter()
    end_event.record()
    torch.cuda.synchronize()
    synchronized_end = time.perf_counter()
    final_checkpoint_manifest = checkpoint_manifest_metadata(args.model)
    if final_checkpoint_manifest["digest"] != checkpoint_manifest["digest"]:
        raise RuntimeError("checkpoint files changed during the benchmark run")
    gpu_elapsed_s = start_event.elapsed_time(end_event) / 1000.0
    submit_time_s = workload_end - start
    total_time = synchronized_end - start

    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    input_tokens = args.num_seqs * args.input_len
    output_throughput = output_tokens / total_time if total_time > 0 else 0.0
    execution_stats = llm.model_runner.call("get_execution_stats")
    shape_trace = llm.model_runner.call("get_shape_trace")
    cudagraph_capture_stats = llm.model_runner.call("get_cudagraph_capture_stats")
    cuda_memory_by_rank = llm.model_runner.call("get_cuda_memory_stats")
    recurrent_state_by_rank = llm.model_runner.call(
        "get_recurrent_state_stats_by_rank"
    )
    peak_allocated_bytes = max(
        item["peak_allocated_bytes"] for item in cuda_memory_by_rank
    )
    peak_reserved_bytes = max(
        item["peak_reserved_bytes"] for item in cuda_memory_by_rank
    )
    required_paths = [item.strip() for item in args.require_paths.split(",") if item.strip()]
    execution_validation = validate_execution_stats(execution_stats, required_paths)

    result = {
        **collect_benchmark_metadata(),
        "model": args.model,
        "checkpoint_manifest": checkpoint_manifest,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "recurrent_state_dtype": args.recurrent_state_dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_dequant_backend": args.kv_dequant_backend,
        "int8_partitioned_decode_threshold": args.int8_partitioned_decode_threshold,
        "int8_partitioned_decode_partition_size": args.int8_partitioned_decode_partition_size,
        "sliding_window_size": args.sliding_window_size,
        "enable_dynamic_chunked_prefill": args.enable_dynamic_chunked_prefill,
        "require_paths": required_paths,
        "warmup": args.warmup,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generated_token_ids": token_ids_digest(outputs),
        "total_time_s": total_time,
        "gpu_elapsed_s": gpu_elapsed_s,
        "cpu_submit_time_s": submit_time_s,
        "final_cuda_synchronize_s": synchronized_end - workload_end,
        "output_throughput_tok_s": output_throughput,
        "peak_torch_allocated_mib": peak_allocated_bytes / 1024 / 1024,
        "peak_torch_reserved_mib": peak_reserved_bytes / 1024 / 1024,
        "cuda_memory_by_rank": cuda_memory_by_rank,
        "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
        "kv_cache_storage": kv_cache_storage_metadata(llm.model_runner),
        "recurrent_state_storage": recurrent_state_by_rank[0],
        "recurrent_state_storage_by_rank": recurrent_state_by_rank,
        "recurrent_state_total_all_ranks_bytes": sum(
            item["total_bytes_local_rank"] for item in recurrent_state_by_rank
        ),
        "model_config": model_config_metadata(llm.model_runner.config.hf_config),
        "final_used_kvcache_blocks": block_manager.num_used_blocks,
        "final_free_kvcache_blocks": block_manager.num_free_blocks,
        "final_kv_block_usage": block_manager.usage,
        "waiting_queue_len": llm.scheduler.num_waiting,
        "running_queue_len": llm.scheduler.num_running,
        "metrics": llm.metrics.to_dict(),
        "prefix_cache": block_manager.cache_stats(),
        "execution_stats": execution_stats,
        "shape_trace": shape_trace,
        "cudagraph_capture_stats": cudagraph_capture_stats,
        "execution_validation": execution_validation,
    }

    name = make_result_name(args.name or default_result_prefix(args))
    json_path = result_dir / f"{name}.json"
    md_path = result_dir / f"{name}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_markdown(md_path, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if not execution_validation["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
