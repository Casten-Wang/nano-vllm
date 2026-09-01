"""Reproducible microbenchmarks for Qwen3.5 routing and DeltaNet paths."""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_source_module(name: str, relative_path: str):
    spec = spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GDN = load_source_module(
    "qwen35_gated_delta_benchmark",
    "nanovllm/models/qwen35_gated_delta.py",
)
METADATA = load_source_module(
    "qwen35_benchmark_metadata",
    "nanovllm/benchmark_metadata.py",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    peak_extra_bytes = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = 0
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000 / iterations)
        if device.type == "cuda":
            peak_extra_bytes.append(
                max(torch.cuda.max_memory_allocated(device) - baseline, 0)
            )

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "peak_extra_mib": (
            max(peak_extra_bytes) / 1024 / 1024 if peak_extra_bytes else 0.0
        ),
    }


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs_error": difference.abs().max().item(),
        "max_relative_error": (difference.abs() / denominator).max().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def compare(
    reference: Callable[[], tuple[torch.Tensor, ...]],
    candidate: Callable[[], tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    expected = reference()
    actual = candidate()
    if len(actual) != len(expected):
        raise RuntimeError("candidate and reference output counts differ")
    for index, (value, target) in enumerate(zip(actual, expected)):
        if value.shape != target.shape:
            raise RuntimeError(
                f"output {index} shape differs: {tuple(value.shape)} != "
                f"{tuple(target.shape)}"
            )
    errors = [error(value, target) for value, target in zip(actual, expected)]
    reference_timing = measure(
        reference,
        device=device,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    candidate_timing = measure(
        candidate,
        device=device,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    return {
        "reference": reference_timing,
        "candidate": candidate_timing,
        "speedup": (
            reference_timing["median_ms"] / candidate_timing["median_ms"]
        ),
        "errors": errors,
    }


def benchmark_router(args, device, dtype) -> dict:
    logits = torch.randn(
        args.router_tokens,
        args.num_experts,
        device=device,
        dtype=dtype,
    )

    def reference():
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, args.top_k, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), ids

    def candidate():
        values, ids = torch.topk(logits, args.top_k, dim=-1)
        return torch.softmax(values.float(), dim=-1), ids

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["full_probability_mib"] = (
        args.router_tokens * args.num_experts * 4 / 1024 / 1024
    )
    result["selected_probability_mib"] = (
        args.router_tokens * args.top_k * 4 / 1024 / 1024
    )
    return result


def benchmark_convolution(args, device, dtype, local_conv_channels) -> dict:
    x = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_conv_channels,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.prefill_batch,
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    return compare(
        lambda: GDN.causal_conv1d_scan(x, state, weight),
        lambda: GDN.causal_conv1d_prefill(x, state, weight),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )


def benchmark_delta_decode(args, device, dtype, local_value_heads) -> dict:
    shape = (args.decode_batch, local_value_heads, args.key_head_dim)
    query = torch.randn(*shape, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(
        args.decode_batch,
        local_value_heads,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    decay = -torch.rand(
        args.decode_batch,
        local_value_heads,
        device=device,
    )
    beta = torch.rand(
        args.decode_batch,
        local_value_heads,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.decode_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    def reference():
        output, next_state = GDN.recurrent_gated_delta_rule(
            query.unsqueeze(1),
            key.unsqueeze(1),
            value.unsqueeze(1),
            decay.unsqueeze(1),
            beta.unsqueeze(1),
            state,
        )
        return output.squeeze(1), next_state

    def candidate():
        return GDN.recurrent_gated_delta_step(
            query,
            key,
            value,
            decay,
            beta,
            state,
        )

    return compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--router-tokens", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--total-key-heads", type=int, default=16)
    parser.add_argument("--total-value-heads", type=int, default=32)
    parser.add_argument("--key-head-dim", type=int, default=128)
    parser.add_argument("--value-head-dim", type=int, default=128)
    parser.add_argument("--conv-kernel-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/qwen35_kernels.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = {
        "router_tokens": args.router_tokens,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "prefill_batch": args.prefill_batch,
        "prefill_tokens": args.prefill_tokens,
        "decode_batch": args.decode_batch,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"benchmark values must be positive: {', '.join(invalid)}")
    if args.top_k > args.num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and args.dtype == "float16":
        raise ValueError("float16 benchmark requires CUDA")
    dtype = getattr(torch, args.dtype)
    if args.total_key_heads % args.tp_size or args.total_value_heads % args.tp_size:
        raise ValueError("Qwen3.5 linear-attention heads must divide TP size")
    local_key_heads = args.total_key_heads // args.tp_size
    local_value_heads = args.total_value_heads // args.tp_size
    local_conv_channels = (
        2 * local_key_heads * args.key_head_dim
        + local_value_heads * args.value_head_dim
    )

    torch.manual_seed(47)
    result = {
        **METADATA.collect_benchmark_metadata(torch),
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "resolved_device": str(device),
            "local_key_heads": local_key_heads,
            "local_value_heads": local_value_heads,
            "local_conv_channels": local_conv_channels,
        },
        "results": {
            "router_topk_first": benchmark_router(args, device, dtype),
            "vectorized_prefill_convolution": benchmark_convolution(
                args,
                device,
                dtype,
                local_conv_channels,
            ),
            "specialized_delta_decode": benchmark_delta_decode(
                args,
                device,
                dtype,
                local_value_heads,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
