import argparse
import json
import os
import random
import statistics
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
    validate_generation_completion,
)


def build_prompts(num_seqs: int, input_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(0, vocab_size - 1) for _ in range(input_len)] for _ in range(num_seqs)]


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def resolve_prompt_lengths(
    count: int,
    default_length: int,
    explicit_lengths: tuple[int, ...] | None,
    label: str,
) -> tuple[int, ...]:
    lengths = explicit_lengths or (default_length,) * count
    if len(lengths) != count:
        raise ValueError(f"{label} prompt lengths must contain exactly {count} values")
    return lengths


def build_prompts_for_lengths(
    lengths: tuple[int, ...],
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(length)]
        for length in lengths
    ]


def validate_workload(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    numeric = (
        args.initial_seqs,
        args.injected_seqs,
        args.initial_input_len,
        args.injected_input_len,
        args.output_len,
        args.inject_after_decode_steps,
        args.max_model_len,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        getattr(args, "prefill_starvation_token_budget", 256),
        args.vocab_size,
    )
    if any(value <= 0 for value in numeric):
        raise ValueError("workload sizes must be positive")
    initial = resolve_prompt_lengths(
        args.initial_seqs,
        args.initial_input_len,
        args.initial_input_lens,
        "initial",
    )
    injected = resolve_prompt_lengths(
        args.injected_seqs,
        args.injected_input_len,
        args.injected_input_lens,
        "injected",
    )
    if max((*initial, *injected)) + args.output_len > args.max_model_len:
        raise ValueError("prompt length plus output length exceeds max_model_len")
    return initial, injected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark online mixed decode/prefill scheduling.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--initial-seqs", type=int, default=8)
    parser.add_argument("--injected-seqs", type=int, default=8)
    parser.add_argument("--initial-input-len", type=int, default=128)
    parser.add_argument("--injected-input-len", type=int, default=1024)
    parser.add_argument("--initial-input-lens", type=parse_positive_int_list)
    parser.add_argument("--injected-input-lens", type=parse_positive_int_list)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--inject-after-decode-steps", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--num-kvcache-blocks-override",
        type=int,
        default=None,
        help="Cap KV blocks to create a reproducible memory-pressure workload.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
    )
    parser.add_argument(
        "--qwen35-moe-decode-backend",
        choices=("sorted", "batched"),
        default="sorted",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--kv-dequant-backend", choices=("fused", "triton", "torch"), default="fused")
    parser.add_argument("--int8-partitioned-decode-threshold", type=int, default=8192)
    parser.add_argument("--int8-partitioned-decode-partition-size", type=int, default=512)
    parser.add_argument("--sliding-window-size", type=int, default=None)
    parser.add_argument("--enable-dynamic-chunked-prefill", action="store_true")
    parser.add_argument("--enable-decode-kv-reservation", action="store_true")
    parser.add_argument(
        "--prefill-starvation-threshold",
        type=int,
        default=0,
        help=(
            "After this many consecutive decode-only steps, reserve one token "
            "for waiting prefill work; zero disables the fairness policy."
        ),
    )
    parser.add_argument(
        "--prefill-starvation-token-budget",
        type=int,
        default=256,
        help=(
            "Prefill tokens reserved after the starvation threshold; capped "
            "at half of a multi-token scheduler step."
        ),
    )
    parser.add_argument(
        "--preemption-policy",
        choices=("fcfs", "min_recompute"),
        default="fcfs",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--result-dir", default="benchmark_results")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-paths",
        default="",
        help="Comma-separated runtime paths required for a valid result.",
    )
    return parser.parse_args()


def request_group(seq_id: int, initial_ids: set[int]) -> str:
    return "initial" if seq_id in initial_ids else "injected"


def main() -> None:
    args = parse_args()
    try:
        initial_input_lens, injected_input_lens = validate_workload(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    initial_prompts = build_prompts_for_lengths(
        initial_input_lens,
        args.vocab_size,
        args.seed,
    )
    injected_prompts = build_prompts_for_lengths(
        injected_input_lens,
        args.vocab_size,
        args.seed + 1,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks_override=args.num_kvcache_blocks_override,
        tensor_parallel_size=args.tensor_parallel_size,
        qwen35_moe_decode_backend=args.qwen35_moe_decode_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
        int8_partitioned_decode_threshold=args.int8_partitioned_decode_threshold,
        int8_partitioned_decode_partition_size=args.int8_partitioned_decode_partition_size,
        sliding_window_size=args.sliding_window_size,
        enable_dynamic_chunked_prefill=args.enable_dynamic_chunked_prefill,
        enable_decode_kv_reservation=args.enable_decode_kv_reservation,
        prefill_starvation_threshold=args.prefill_starvation_threshold,
        prefill_starvation_token_budget=args.prefill_starvation_token_budget,
        preemption_policy=args.preemption_policy,
    )
    llm.model_runner.call("reset_execution_stats")
    llm.model_runner.call("reset_shape_trace")
    llm.metrics.reset()
    llm.scheduler.block_manager.reset_cache_stats()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for prompt in initial_prompts:
        llm.add_request(prompt, sampling_params)
    initial_ids = {seq.seq_id for seq in llm.scheduler.waiting}

    injected = False
    decode_steps_before_injection = 0
    token_times: dict[int, list[float]] = {}
    finished_request_ids: set[int] = set()
    finished_output_lengths: dict[int, int] = {}
    finished_token_ids: dict[int, list[int]] = {}
    groups: dict[int, str] = {seq_id: "initial" for seq_id in initial_ids}
    torch.cuda.synchronize()
    llm.model_runner.call("reset_cuda_peak_memory_stats")
    start = time.perf_counter()
    request_arrival_times = {seq_id: start for seq_id in initial_ids}
    step_count = 0

    while not llm.is_finished():
        step_start = time.perf_counter()
        output, _num_tokens, prefill_tokens, decode_tokens = llm.step()
        # CPU perf_counter is used for serving-observed step/token gaps.  A
        # CUDA event pair would only measure GPU work and would not preserve
        # request-level arrival/completion semantics.
        torch.cuda.synchronize()
        step_elapsed = time.perf_counter() - step_start
        now = time.perf_counter()
        step_count += 1
        llm.metrics.record_step(
            _num_tokens,
            step_elapsed,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )

        if decode_tokens > 0 and not injected:
            decode_steps_before_injection += 1
            if decode_steps_before_injection >= args.inject_after_decode_steps:
                injection_time = time.perf_counter()
                for prompt in injected_prompts:
                    llm.add_request(prompt, sampling_params)
                for seq in llm.scheduler.waiting:
                    group = request_group(seq.seq_id, initial_ids)
                    groups.setdefault(seq.seq_id, group)
                    if group == "injected":
                        request_arrival_times.setdefault(
                            seq.seq_id,
                            injection_time,
                        )
                injected = True

        for seq_id, token_ids in output:
            finished_request_ids.add(seq_id)
            finished_output_lengths[seq_id] = len(token_ids)
            finished_token_ids[seq_id] = token_ids
            # step() returns complete outputs only when a request finishes. For
            # gap metrics we need per-token timestamps, so collect them from
            # live sequence objects below as well.
            times = token_times.setdefault(seq_id, [])
            # Finished sequences leave the running queue before the loop
            # below can observe their last token. Fill all missing completion
            # timestamps with the current step time; normally this is exactly
            # one token, while preserving the complete output length keeps
            # gap accounting internally consistent.
            while len(times) < len(token_ids):
                times.append(now)

        for seq in list(llm.scheduler.running):
            if seq.num_completion_tokens > len(token_times.get(seq.seq_id, [])):
                # Timestamp each generated token at the end of its scheduler
                # step.  All token-level gap metrics are therefore
                # step-completion gaps, not kernel-only latency.
                token_times.setdefault(seq.seq_id, []).append(now)
                groups.setdefault(seq.seq_id, request_group(seq.seq_id, initial_ids))

    total_time = time.perf_counter() - start
    initial_gaps = []
    injected_gaps = []
    initial_ttfts = []
    injected_ttfts = []
    for seq_id, times in token_times.items():
        if times and seq_id in request_arrival_times:
            ttft = times[0] - request_arrival_times[seq_id]
            if groups.get(seq_id) == "initial":
                initial_ttfts.append(ttft)
            else:
                injected_ttfts.append(ttft)
        gaps = [b - a for a, b in zip(times, times[1:])]
        if not gaps:
            continue
        if groups.get(seq_id) == "initial":
            initial_gaps.extend(gaps)
        else:
            injected_gaps.extend(gaps)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def max_or_zero(values: list[float]) -> float:
        return max(values) if values else 0.0

    def percentile(values: list[float], percentile_rank: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int((percentile_rank / 100.0) * (len(ordered) - 1))),
        )
        return ordered[index]

    execution_stats = llm.model_runner.call("get_execution_stats")
    shape_trace = llm.model_runner.call("get_shape_trace")
    cudagraph_capture_stats = llm.model_runner.call("get_cudagraph_capture_stats")
    cuda_memory_by_rank = llm.model_runner.call("get_cuda_memory_stats")
    recurrent_state_by_rank = llm.model_runner.call(
        "get_recurrent_state_stats_by_rank"
    )
    runtime_buffer_by_rank = llm.model_runner.call(
        "get_runtime_buffer_stats_by_rank"
    )
    kv_cache_by_rank = llm.model_runner.call("get_kv_cache_stats_by_rank")
    peak_allocated_bytes = max(
        item["peak_allocated_bytes"] for item in cuda_memory_by_rank
    )
    peak_reserved_bytes = max(
        item["peak_reserved_bytes"] for item in cuda_memory_by_rank
    )
    required_paths = [item.strip() for item in args.require_paths.split(",") if item.strip()]
    execution_validation = validate_execution_stats(execution_stats, required_paths)
    expected_requests = args.initial_seqs + args.injected_seqs
    generation_validation = validate_generation_completion(
        list(finished_output_lengths.values()),
        expected_num_seqs=expected_requests,
        expected_output_len=args.output_len,
        waiting_queue_len=llm.scheduler.num_waiting,
        running_queue_len=llm.scheduler.num_running,
    )
    scenario_errors = []
    if not injected:
        scenario_errors.append("injected requests were never submitted")
    if not generation_validation["valid"]:
        scenario_errors.extend(generation_validation["errors"])
    if scenario_errors:
        execution_validation["valid"] = False
        execution_validation["reason"] = "; ".join(
            filter(
                None,
                [execution_validation.get("reason"), *scenario_errors],
            )
        )

    result = {
        **collect_benchmark_metadata(),
        "model": args.model,
        "checkpoint_manifest": checkpoint_manifest_metadata(args.model),
        "initial_seqs": args.initial_seqs,
        "injected_seqs": args.injected_seqs,
        "initial_input_len": args.initial_input_len,
        "injected_input_len": args.injected_input_len,
        "initial_input_lens": list(initial_input_lens),
        "injected_input_lens": list(injected_input_lens),
        "output_len": args.output_len,
        "inject_after_decode_steps": args.inject_after_decode_steps,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "num_kvcache_blocks_override": args.num_kvcache_blocks_override,
        "tensor_parallel_size": args.tensor_parallel_size,
        "qwen35_moe_decode_backend": args.qwen35_moe_decode_backend,
        "seed": args.seed,
        "temperature": args.temperature,
        "generated_token_ids": token_ids_digest(
            [
                {"token_ids": finished_token_ids[seq_id]}
                for seq_id in sorted(finished_token_ids)
            ]
        ),
        "enforce_eager": args.enforce_eager,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_dequant_backend": args.kv_dequant_backend,
        "int8_partitioned_decode_threshold": args.int8_partitioned_decode_threshold,
        "int8_partitioned_decode_partition_size": args.int8_partitioned_decode_partition_size,
        "sliding_window_size": args.sliding_window_size,
        "enable_dynamic_chunked_prefill": args.enable_dynamic_chunked_prefill,
        "enable_decode_kv_reservation": args.enable_decode_kv_reservation,
        "prefill_starvation_threshold": args.prefill_starvation_threshold,
        "prefill_starvation_token_budget": args.prefill_starvation_token_budget,
        "preemption_policy": args.preemption_policy,
        "require_paths": required_paths,
        "injected": injected,
        "expected_requests": expected_requests,
        "finished_requests": len(finished_request_ids),
        "total_time_s": total_time,
        "peak_torch_allocated_mib": peak_allocated_bytes / 1024 / 1024,
        "peak_torch_reserved_mib": peak_reserved_bytes / 1024 / 1024,
        "cuda_memory_by_rank": cuda_memory_by_rank,
        "recurrent_state_storage_by_rank": recurrent_state_by_rank,
        "runtime_buffer_storage_by_rank": runtime_buffer_by_rank,
        "kv_cache_storage_by_rank": kv_cache_by_rank,
        "recurrent_state_total_all_ranks_bytes": sum(
            item["total_bytes_local_rank"] for item in recurrent_state_by_rank
        ),
        "runtime_buffer_total_all_ranks_bytes": sum(
            item["total_bytes_local_rank"] for item in runtime_buffer_by_rank
        ),
        "step_count": step_count,
        "initial_decode_gap_count": len(initial_gaps),
        "injected_decode_gap_count": len(injected_gaps),
        "initial_ttft_count": len(initial_ttfts),
        "injected_ttft_count": len(injected_ttfts),
        "initial_avg_ttft_s": avg(initial_ttfts),
        "initial_p95_ttft_s": percentile(initial_ttfts, 95),
        "initial_max_ttft_s": max_or_zero(initial_ttfts),
        "injected_avg_ttft_s": avg(injected_ttfts),
        "injected_p95_ttft_s": percentile(injected_ttfts, 95),
        "injected_max_ttft_s": max_or_zero(injected_ttfts),
        "initial_avg_decode_gap_s": avg(initial_gaps),
        "initial_median_decode_gap_s": statistics.median(initial_gaps) if initial_gaps else 0.0,
        "initial_p95_decode_gap_s": percentile(initial_gaps, 95),
        "initial_max_decode_gap_s": max_or_zero(initial_gaps),
        "injected_avg_decode_gap_s": avg(injected_gaps),
        "injected_median_decode_gap_s": statistics.median(injected_gaps) if injected_gaps else 0.0,
        "injected_p95_decode_gap_s": percentile(injected_gaps, 95),
        "injected_max_decode_gap_s": max_or_zero(injected_gaps),
        "output_throughput_tok_s": (
            sum(finished_output_lengths.values()) / total_time
            if total_time > 0
            else 0.0
        ),
        "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
        "kv_cache_storage": kv_cache_storage_metadata(llm.model_runner),
        "model_config": model_config_metadata(llm.model_runner.config.hf_config),
        "metrics": llm.metrics.to_dict(),
        "prefix_cache": llm.scheduler.block_manager.cache_stats(),
        "execution_stats": execution_stats,
        "shape_trace": shape_trace,
        "cudagraph_capture_stats": cudagraph_capture_stats,
        "execution_validation": execution_validation,
        "generation_validation": generation_validation,
    }

    prefix = args.name
    if prefix is None:
        prefix = "online_mixed"
        if args.kv_cache_dtype == "int8":
            prefix += f"_int8_{args.kv_dequant_backend}"
        if args.enable_dynamic_chunked_prefill:
            prefix += "_dynchunk"
        if args.enable_decode_kv_reservation:
            prefix += "_decode_kv_reserve"
        if args.prefill_starvation_threshold:
            prefix += (
                f"_fair{args.prefill_starvation_threshold}"
                f"x{args.prefill_starvation_token_budget}"
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output or result_dir / f"{prefix}_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    if not execution_validation["valid"] or not generation_validation["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
