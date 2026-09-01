"""Compare eager and CUDA Graph decode with identical deterministic inputs.

The parent process launches eager and graph workers separately so two model
copies never need to coexist on the GPU. Each worker records generated tokens,
per-step logits, and actual execution paths. The parent verifies numerical and
token parity and writes a reusable JSON summary plus raw torch artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Make worker subprocesses independent of the caller's current directory.
    sys.path.insert(0, str(ROOT))


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one integer is required")
    return result


def scenario_lengths(batch_size: int, scenario_index: int) -> list[int]:
    # Vary lengths within a batch so context_lens change independently and
    # requests use different block tables. The second scenario includes a
    # 250-token prompt; with eight generated tokens, its final decode replay
    # allocates and observes a new 256-token KV block.
    base = 33 + scenario_index * 25
    stride = 32
    return [base + stride * index for index in range(batch_size)]


def build_prompts(lengths: list[int], vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(length)] for length in lengths
    ]


def extract_decode_steps(artifact: dict) -> list[dict]:
    """Return only pure-decode logits captured after prefill."""

    return [step for step in artifact["logits_steps"] if not step["is_prefill"]]


def run_worker(args: argparse.Namespace) -> None:
    import torch

    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA Graph parity verification")

    lengths = parse_int_list(args.worker_input_lengths)
    prompts = build_prompts(lengths, args.vocab_size, args.seed)
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
        for _ in prompts
    ]
    enforce_eager = args.worker_mode == "eager"
    llm = LLM(
        args.model,
        enforce_eager=enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        qwen35_moe_decode_backend=args.qwen35_moe_decode_backend,
        qwen35_moe_decode_chunk_size=args.qwen35_moe_decode_chunk_size,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
        int8_partitioned_decode_threshold=args.partition_threshold,
        int8_partitioned_decode_partition_size=args.partition_size,
        sliding_window_size=None,
        enable_dynamic_chunked_prefill=False,
    )

    hidden_steps = []
    logits_steps = []
    original_run_model = llm.model_runner.run_model
    original_compute_logits = llm.model_runner.model.compute_logits
    current_is_prefill = None

    def capture_compute_logits(hidden_states):
        hidden_steps.append(
            {
                "is_prefill": bool(current_is_prefill),
                "shape": list(hidden_states.shape),
                "hidden_states": hidden_states.detach().float().cpu(),
            }
        )
        return original_compute_logits(hidden_states)

    def capture_run_model(input_ids, positions, is_prefill):
        nonlocal current_is_prefill
        current_is_prefill = bool(is_prefill)
        logits = original_run_model(input_ids, positions, is_prefill)
        logits_steps.append(
            {
                "is_prefill": bool(is_prefill),
                "shape": list(logits.shape),
                "logits": logits.detach().float().cpu(),
            }
        )
        current_is_prefill = None
        return logits

    llm.model_runner.model.compute_logits = capture_compute_logits
    llm.model_runner.run_model = capture_run_model
    llm.model_runner.call("reset_execution_stats")
    llm.model_runner.call("reset_shape_trace")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    execution_stats = llm.model_runner.call("get_execution_stats")
    shape_trace = llm.model_runner.call("get_shape_trace")
    artifact = {
        "mode": args.worker_mode,
        "input_lengths": lengths,
        "output_tokens": [item["token_ids"] for item in outputs],
        "hidden_steps": hidden_steps,
        "logits_steps": logits_steps,
        "execution_stats": execution_stats,
        "shape_trace": shape_trace,
        "config": {
            "kv_cache_dtype": args.kv_cache_dtype,
            "kv_dequant_backend": args.kv_dequant_backend,
            "partition_threshold": args.partition_threshold,
            "partition_size": args.partition_size,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "tensor_parallel_size": args.tensor_parallel_size,
            "qwen35_moe_decode_backend": args.qwen35_moe_decode_backend,
            "qwen35_moe_decode_chunk_size": args.qwen35_moe_decode_chunk_size,
            "output_len": args.output_len,
            "seed": args.seed,
        },
    }
    output_path = Path(args.worker_artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)


def worker_command(
    args: argparse.Namespace,
    *,
    mode: str,
    input_lengths: list[int],
    artifact: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-artifact",
        str(artifact),
        "--worker-input-lengths",
        ",".join(map(str, input_lengths)),
        "--model",
        args.model,
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--qwen35-moe-decode-backend",
        args.qwen35_moe_decode_backend,
        "--qwen35-moe-decode-chunk-size",
        str(args.qwen35_moe_decode_chunk_size),
        "--vocab-size",
        str(args.vocab_size),
        "--seed",
        str(seed),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--kv-dequant-backend",
        args.kv_dequant_backend,
        "--partition-threshold",
        str(args.partition_threshold),
        "--partition-size",
        str(args.partition_size),
    ]


def compare_artifacts(
    eager: dict,
    graph: dict,
    *,
    atol: float,
    rtol: float,
    expected_graph_bucket: int,
) -> dict:
    import torch

    token_match = eager["output_tokens"] == graph["output_tokens"]
    eager_hidden_steps = extract_decode_steps(
        {"logits_steps": eager["hidden_steps"]}
    )
    graph_hidden_steps = extract_decode_steps(
        {"logits_steps": graph["hidden_steps"]}
    )
    eager_steps = extract_decode_steps(eager)
    graph_steps = extract_decode_steps(graph)
    hidden_step_count_match = len(eager_hidden_steps) == len(graph_hidden_steps)
    step_count_match = len(eager_steps) == len(graph_steps)
    hidden_step_results = []
    step_results = []
    hidden_match = hidden_step_count_match
    if hidden_step_count_match:
        for index, (eager_step, graph_step) in enumerate(
            zip(eager_hidden_steps, graph_hidden_steps)
        ):
            shape_match = eager_step["shape"] == graph_step["shape"]
            mode_match = eager_step["is_prefill"] == graph_step["is_prefill"]
            if shape_match:
                difference = (
                    eager_step["hidden_states"] - graph_step["hidden_states"]
                ).abs()
                max_abs = difference.max().item()
                mean_abs = difference.mean().item()
                close = torch.allclose(
                    eager_step["hidden_states"],
                    graph_step["hidden_states"],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                max_abs = None
                mean_abs = None
                close = False
            step_passed = shape_match and mode_match and close
            hidden_match = hidden_match and step_passed
            hidden_step_results.append(
                {
                    "step": index,
                    "is_prefill": eager_step["is_prefill"],
                    "shape_match": shape_match,
                    "mode_match": mode_match,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "allclose": close,
                    "passed": step_passed,
                }
            )

    logits_match = step_count_match
    if step_count_match:
        for index, (eager_step, graph_step) in enumerate(zip(eager_steps, graph_steps)):
            shape_match = eager_step["shape"] == graph_step["shape"]
            mode_match = eager_step["is_prefill"] == graph_step["is_prefill"]
            if shape_match:
                difference = (eager_step["logits"] - graph_step["logits"]).abs()
                max_abs = difference.max().item()
                mean_abs = difference.mean().item()
                close = torch.allclose(
                    eager_step["logits"],
                    graph_step["logits"],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                max_abs = None
                mean_abs = None
                close = False
            passed = shape_match and mode_match and close
            logits_match = logits_match and passed
            step_results.append(
                {
                    "step": index,
                    "is_prefill": eager_step["is_prefill"],
                    "shape_match": shape_match,
                    "mode_match": mode_match,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "allclose": close,
                    "passed": passed,
                }
            )

    eager_model_paths = eager["execution_stats"]["model_path_counts"]
    graph_model_paths = graph["execution_stats"]["model_path_counts"]
    expected_eager_paths = (
        "prefill_eager" in eager_model_paths
        and "decode_eager" in eager_model_paths
        and "decode_cuda_graph" not in eager_model_paths
    )
    expected_graph_paths = (
        "prefill_eager" in graph_model_paths
        and "decode_cuda_graph" in graph_model_paths
        and "decode_eager" not in graph_model_paths
    )
    graph_buckets = {
        item["graph_bucket"]
        for item in graph["execution_stats"]["execution_signatures"]
        if item["model_path"] == "decode_cuda_graph"
    }
    expected_bucket_observed = expected_graph_bucket in graph_buckets
    passed = (
        token_match
        and hidden_match
        and logits_match
        and expected_eager_paths
        and expected_graph_paths
        and expected_bucket_observed
    )
    return {
        "passed": passed,
        "token_match": token_match,
        "hidden_step_count_match": hidden_step_count_match,
        "hidden_match": hidden_match,
        "step_count_match": step_count_match,
        "logits_match": logits_match,
        "expected_eager_paths": expected_eager_paths,
        "expected_graph_paths": expected_graph_paths,
        "expected_graph_bucket": expected_graph_bucket,
        "observed_graph_buckets": sorted(
            bucket for bucket in graph_buckets if bucket is not None
        ),
        "expected_bucket_observed": expected_bucket_observed,
        "hidden_step_results": hidden_step_results,
        "step_results": step_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify eager/CUDA Graph decode logits and token parity."
    )
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
    )
    parser.add_argument("--batch-sizes", default="3,9")
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--qwen35-moe-decode-backend",
        choices=("sorted", "batched"),
        default="batched",
    )
    parser.add_argument("--qwen35-moe-decode-chunk-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "int8"),
        default="int8",
    )
    parser.add_argument(
        "--kv-dequant-backend",
        choices=("fused",),
        default="fused",
    )
    parser.add_argument("--partition-threshold", type=int, default=8192)
    parser.add_argument("--partition-size", type=int, default=512)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--result-dir",
        default="benchmark_results/cudagraph_parity",
    )
    parser.add_argument(
        "--worker-mode",
        choices=("eager", "graph"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-artifact",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-input-lengths",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_mode is not None:
        if not args.worker_artifact or not args.worker_input_lengths:
            parser.error("worker mode requires artifact and input lengths")
        run_worker(args)
        return

    if not os.path.isdir(args.model):
        parser.error(f"model directory does not exist: {args.model}")
    batch_sizes = parse_int_list(args.batch_sizes)
    if any(batch_size <= 0 for batch_size in batch_sizes):
        parser.error("batch sizes must be positive")
    if any(batch_size > 512 for batch_size in batch_sizes):
        parser.error(
            "batch sizes must be <= 512 for the current CUDA Graph implementation"
        )
    if max(batch_sizes) > args.max_num_seqs:
        parser.error("max_num_seqs must cover every scenario batch size")
    if args.tensor_parallel_size <= 0:
        parser.error("tensor_parallel_size must be positive")
    if args.qwen35_moe_decode_chunk_size <= 0:
        parser.error("qwen35_moe_decode_chunk_size must be positive")
    if args.partition_threshold <= args.max_model_len:
        parser.error(
            "partition-threshold must exceed max-model-len so the parity run "
            "stays on the short-context fused Graph path"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA Graph parity verification")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    scenario_results = []
    all_passed = True

    from nanovllm.benchmark_metadata import collect_benchmark_metadata
    from nanovllm.engine.execution import cuda_graph_buckets

    graph_buckets = cuda_graph_buckets(min(args.max_num_seqs, 512))
    for scenario_index, batch_size in enumerate(batch_sizes):
        lengths = scenario_lengths(batch_size, scenario_index)
        if max(lengths) + args.output_len > args.max_model_len:
            raise ValueError(
                f"scenario input length {max(lengths)} plus output_len "
                f"{args.output_len} exceeds max_model_len {args.max_model_len}"
            )
        expected_bucket = next(
            bucket for bucket in graph_buckets if bucket >= batch_size
        )
        scenario_dir = run_dir / f"scenario_{scenario_index}_b{batch_size}"
        eager_artifact = scenario_dir / "eager.pt"
        graph_artifact = scenario_dir / "graph.pt"
        scenario_seed = args.seed + scenario_index
        for mode, artifact in (
            ("eager", eager_artifact),
            ("graph", graph_artifact),
        ):
            command = worker_command(
                args,
                mode=mode,
                input_lengths=lengths,
                artifact=artifact,
                seed=scenario_seed,
            )
            subprocess.run(command, check=True)

        eager = torch.load(eager_artifact, map_location="cpu", weights_only=False)
        graph = torch.load(graph_artifact, map_location="cpu", weights_only=False)
        comparison = compare_artifacts(
            eager,
            graph,
            atol=args.atol,
            rtol=args.rtol,
            expected_graph_bucket=expected_bucket,
        )
        all_passed = all_passed and comparison["passed"]
        scenario_results.append(
            {
                "scenario": scenario_index,
                "batch_size": batch_size,
                "input_lengths": lengths,
                "expected_graph_bucket": expected_bucket,
                "eager_artifact": str(eager_artifact),
                "graph_artifact": str(graph_artifact),
                "comparison": comparison,
                "eager_execution_stats": eager["execution_stats"],
                "graph_execution_stats": graph["execution_stats"],
            }
        )

    summary = {
        **collect_benchmark_metadata(torch),
        "passed": all_passed,
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "kv_dequant_backend": args.kv_dequant_backend,
        "tensor_parallel_size": args.tensor_parallel_size,
        "qwen35_moe_decode_backend": args.qwen35_moe_decode_backend,
        "qwen35_moe_decode_chunk_size": args.qwen35_moe_decode_chunk_size,
        "output_len": args.output_len,
        "atol": args.atol,
        "rtol": args.rtol,
        "scenarios": scenario_results,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
