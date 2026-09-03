"""Benchmark fused GPTQ W4A16 linear against the torch dequantization oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.benchmark_metadata import collect_benchmark_metadata
from nanovllm.models.qwen35_gptq import dequantize_gptq_int4


def latency_ms(function, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=2048)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/gptq_w4a16.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    from nanovllm.layers.gptq_w4a16 import gptq_w4a16_linear

    if (
        min(
            args.tokens,
            args.input_size,
            args.output_size,
            args.group_size,
            args.warmup,
            args.repeats,
        )
        <= 0
    ):
        parser.error("all numeric arguments must be positive")
    if args.input_size % 8 or args.output_size % 8:
        parser.error("input and output sizes must be divisible by 8")

    device = torch.device("cuda")
    inputs = torch.randn(
        args.tokens,
        args.input_size,
        device=device,
        dtype=torch.bfloat16,
    )
    qweight = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (args.input_size // 8, args.output_size),
        device=device,
        dtype=torch.int32,
    )
    group_count = (args.input_size + args.group_size - 1) // args.group_size
    qzeros = torch.full(
        (group_count, args.output_size // 8),
        -2004318072,
        device=device,
        dtype=torch.int32,
    )
    scales = (
        torch.rand(
            group_count,
            args.output_size,
            device=device,
            dtype=torch.float16,
        )
        / 32
    )
    g_idx = torch.arange(args.input_size, device=device, dtype=torch.int32)
    g_idx.div_(args.group_size, rounding_mode="floor")

    def fused():
        return gptq_w4a16_linear(inputs, qweight, qzeros, scales, g_idx)

    reference_weight = dequantize_gptq_int4(
        qweight,
        qzeros,
        scales,
        g_idx,
        output_dtype=inputs.dtype,
    )

    def reference():
        return torch.nn.functional.linear(inputs, reference_weight)

    actual = fused()
    expected = reference()
    difference = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    correctness = {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
    }
    del actual, difference, expected, reference_weight
    torch.cuda.empty_cache()

    fused_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fused_samples = latency_ms(fused, args.warmup, args.repeats)
    fused_peak = torch.cuda.max_memory_allocated() - fused_baseline

    reference_weight = dequantize_gptq_int4(
        qweight,
        qzeros,
        scales,
        g_idx,
        output_dtype=inputs.dtype,
    )
    reference_weight_bytes = reference_weight.numel() * reference_weight.element_size()
    reference_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    reference_samples = latency_ms(reference, args.warmup, args.repeats)
    reference_peak = torch.cuda.max_memory_allocated() - reference_baseline
    report = {
        "metadata": collect_benchmark_metadata(torch),
        "shape": {
            "tokens": args.tokens,
            "input_size": args.input_size,
            "output_size": args.output_size,
            "group_size": args.group_size,
        },
        "correctness": correctness,
        "fused": {
            **summary(fused_samples),
            "incremental_peak_bytes": fused_peak,
        },
        "reference_gemm_only": {
            **summary(reference_samples),
            "incremental_peak_bytes": reference_peak,
            "dequantized_weight_bytes": reference_weight_bytes,
            "note": "dequantization is excluded from timed iterations",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
