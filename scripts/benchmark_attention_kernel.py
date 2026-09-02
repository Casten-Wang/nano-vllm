import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.benchmark_metadata import collect_benchmark_metadata
from nanovllm.layers.int8_fused_attention import (
    allocate_partitioned_workspace,
    fused_int8_decode_attention,
    fused_int8_decode_attention_latev,
    fused_int8_decode_attention_v3,
    partitioned_fused_int8_decode_attention,
)
from nanovllm.layers.kv_cache_quant import dequant_packed_kvcache


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def tensor_manifest(
    name: str,
    tensor: torch.Tensor | None,
    *,
    include_values: bool = False,
    max_values: int = 256,
) -> dict | None:
    """Serialize tensor layout metadata without copying large tensor values."""

    if tensor is None:
        return None
    item = {
        "name": name,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "bytes": tensor.numel() * tensor.element_size(),
        "contiguous": tensor.is_contiguous(),
    }
    if include_values and tensor.numel() <= max_values:
        item["values"] = tensor.detach().cpu().reshape(-1).tolist()
    return item


def build_shape_manifest(
    args: argparse.Namespace,
    *,
    q: torch.Tensor,
    k_reference: torch.Tensor,
    v_reference: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> dict:
    """Build the runtime shape/index/launch manifest for this microbenchmark."""

    max_blocks = k_int8.size(0)
    block_head_dim = 1 << (args.head_dim - 1).bit_length()
    q_heads_per_kv_head = args.num_heads // args.num_kv_heads
    tensors = {
        name: tensor_manifest(
            name,
            tensor,
            include_values=name in {"block_tables", "context_lens"},
        )
        for name, tensor in {
            "q": q,
            "q_for_flash_kvcache": q.unsqueeze(1),
            "k_reference_bf16_or_fp16": k_reference,
            "v_reference_bf16_or_fp16": v_reference,
            "k_int8_cache": k_int8,
            "v_int8_cache": v_int8,
            "k_scale_fp16": k_scale,
            "v_scale_fp16": v_scale,
            "block_tables": block_tables,
            "context_lens": context_lens,
        }.items()
    }

    launches = [
        {
            "name": "flash_attn_with_kvcache_reference",
            "role": "reference",
            "grid": None,
            "threads_per_program": None,
            "source": "provider_opaque_runtime",
            "note": "FlashAttention provider controls its internal launch.",
        }
    ]
    if args.include_packed_dequant_flash:
        launches.append({
            "name": "dequant_packed_kvcache",
            "role": "baseline_preprocess",
            "grid": [
                max_blocks,
                args.num_kv_heads,
                math.ceil(args.block_size / args.dequant_block_tokens),
            ],
            "threads_per_program": None,
            "source": "static_inference",
            "meta": {
                "BLOCK_SIZE": args.block_size,
                "BLOCK_TOKENS": args.dequant_block_tokens,
                "NUM_KV_HEADS": args.num_kv_heads,
                "HEAD_DIM": args.head_dim,
                "BLOCK_HEAD_DIM": block_head_dim,
            },
        })
    for variant in parse_str_list(args.variants):
        for block_tokens in parse_int_list(args.block_tokens):
            if block_tokens > args.block_size or args.block_size % block_tokens:
                continue
            for num_warps in parse_int_list(args.num_warps):
                for num_stages in parse_int_list(args.num_stages):
                    launches.append(
                        {
                            "name": f"int8_{variant}_bt{block_tokens}"
                            f"_w{num_warps}_s{num_stages}",
                            "role": "candidate",
                            "grid": [args.batch_size, args.num_heads],
                            "threads_per_program": num_warps * 32,
                            "source": "static_inference",
                            "meta": {
                                "NUM_KV_HEADS": args.num_kv_heads,
                                "HEAD_DIM": args.head_dim,
                                "BLOCK_HEAD_DIM": block_head_dim,
                                "BLOCK_SIZE": args.block_size,
                                "BLOCK_TOKENS": block_tokens,
                                "MAX_BLOCKS": max_blocks,
                                "Q_HEADS_PER_KV_HEAD": q_heads_per_kv_head,
                                "NUM_WARPS": num_warps,
                                "NUM_STAGES": num_stages,
                                "WINDOW_SIZE": args.sliding_window_size or 0,
                            },
                        }
                    )

    if args.include_partitioned:
        for partition_size in parse_int_list(args.partition_sizes):
            num_partitions = max(
                math.ceil(
                    min(
                        args.context_len,
                        args.sliding_window_size
                        if args.sliding_window_size is not None
                        else args.context_len,
                    )
                    / partition_size
                ),
                1,
            )
            launches.extend(
                [
                    {
                        "name": f"int8_partitioned_ps{partition_size}",
                        "role": "candidate",
                        "grid": [
                            args.batch_size,
                            args.num_heads,
                            num_partitions,
                        ],
                        "threads_per_program": 4 * 32
                        if args.head_dim <= 128
                        else 8 * 32,
                        "source": "static_inference",
                        "meta": {
                            "PARTITION_SIZE": partition_size,
                            "NUM_PARTITIONS": num_partitions,
                            "BLOCK_SIZE": args.block_size,
                            "BLOCK_TOKENS": min(args.block_size, 256),
                            "HEAD_DIM": args.head_dim,
                            "BLOCK_HEAD_DIM": block_head_dim,
                        },
                    },
                    {
                        "name": f"int8_partitioned_ps{partition_size}_reduce",
                        "role": "candidate_reduce",
                        "grid": [args.batch_size, args.num_heads],
                        "threads_per_program": 4 * 32
                        if args.head_dim <= 128
                        else 8 * 32,
                        "source": "static_inference",
                        "meta": {
                            "NUM_PARTITIONS": num_partitions,
                            "HEAD_DIM": args.head_dim,
                            "BLOCK_HEAD_DIM": block_head_dim,
                        },
                    },
                ]
            )

    partial_workspace = {}
    if args.include_partitioned:
        for partition_size in parse_int_list(args.partition_sizes):
            num_partitions = max(
                math.ceil(
                    min(
                        args.context_len,
                        args.sliding_window_size
                        if args.sliding_window_size is not None
                        else args.context_len,
                    )
                    / partition_size
                ),
                1,
            )
            partial_workspace[str(partition_size)] = {
                "allocation_count": 1,
                "shared_storage": True,
                "partial_acc_shape": [
                    args.batch_size,
                    args.num_heads,
                    num_partitions,
                    block_head_dim,
                ],
                "partial_acc_dtype": "torch.float32",
                "partial_m_shape": [
                    args.batch_size,
                    args.num_heads,
                    num_partitions,
                ],
                "partial_l_shape": [
                    args.batch_size,
                    args.num_heads,
                    num_partitions,
                ],
                "partial_bytes": (
                    args.batch_size
                    * args.num_heads
                    * num_partitions
                    * block_head_dim
                    * 4
                    + 2
                    * args.batch_size
                    * args.num_heads
                    * num_partitions
                    * 4
                ),
                "source": "static_inference",
            }

    return {
        "scenario": (
            f"int8_decode_b{args.batch_size}_ctx{args.context_len}"
            f"_dtype{args.dtype}"
        ),
        "source": "runtime_tensor_metadata_plus_static_launch_inference",
        "configuration": {
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "block_size": args.block_size,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "q_heads_per_kv_head": q_heads_per_kv_head,
            "max_blocks": max_blocks,
            "dtype": args.dtype,
            "cache_dtype": "torch.int8",
            "scale_dtype": "torch.float16",
            "sliding_window_size": args.sliding_window_size,
        },
        "indexing": {
            "q_head_to_kv_head": (
                f"q_head // {q_heads_per_kv_head}"
            ),
            "physical_block_id": (
                "block_tables[sequence, logical_block]"
            ),
            "token_slot": (
                "physical_block_id * block_size + token_offset"
            ),
        },
        "tensors": tensors,
        "workspace": {
            "packed_k_shape": (
                [
                    max_blocks,
                    args.block_size,
                    args.num_kv_heads,
                    args.head_dim,
                ]
                if args.include_packed_dequant_flash
                else None
            ),
            "packed_k_dtype": (
                str(q.dtype)
                if args.include_packed_dequant_flash
                else None
            ),
            "packed_v_shape": (
                [
                    max_blocks,
                    args.block_size,
                    args.num_kv_heads,
                    args.head_dim,
                ]
                if args.include_packed_dequant_flash
                else None
            ),
            "partitioned": partial_workspace,
        },
        "kernel_launches": launches,
    }


def make_case(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    max_blocks = (args.context_len + args.block_size - 1) // args.block_size

    q = torch.randn(
        args.batch_size,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k_int8 = torch.randint(
        -127,
        128,
        (max_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    v_int8 = torch.randint(
        -127,
        128,
        (max_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        device=device,
        dtype=torch.int8,
    )

    # Match the production cache granularity: one FP16 scale per token and KV
    # head, with independent K and V scales. Random quantized data isolates the
    # attention kernel; store/quantization correctness is tested separately.
    k_scale = (
        torch.rand(
            max_blocks,
            args.block_size,
            args.num_kv_heads,
            device=device,
            dtype=torch.float16,
        )
        * 0.03
        + 0.001
    )
    v_scale = (
        torch.rand(
            max_blocks,
            args.block_size,
            args.num_kv_heads,
            device=device,
            dtype=torch.float16,
        )
        * 0.03
        + 0.001
    )
    k_reference = (k_int8.float() * k_scale.float().unsqueeze(-1)).to(dtype)
    v_reference = (v_int8.float() * v_scale.float().unsqueeze(-1)).to(dtype)

    block_tables = torch.arange(
        max_blocks,
        device=device,
        dtype=torch.int32,
    ).repeat(args.batch_size, 1)
    context_lens = torch.full(
        (args.batch_size,),
        args.context_len,
        device=device,
        dtype=torch.int32,
    )
    return (
        q,
        k_reference,
        v_reference,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        block_tables,
        context_lens,
    )


def measure_once(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, int]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = fn()
    end.record()
    torch.cuda.synchronize()
    assert output is not None
    peak_extra_bytes = max(
        0,
        torch.cuda.max_memory_allocated() - baseline_bytes,
    )
    return start.elapsed_time(end) / iters, output, peak_extra_bytes


def summarize_samples(samples_ms: list[float]) -> dict[str, float | list[float]]:
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    median_ms = statistics.median(samples_ms)
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    relative_range = (max_ms - min_ms) / median_ms if median_ms > 0 else 0.0
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "stdev_ms": stdev_ms,
        "relative_range": relative_range,
    }


def measure_repeated(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[dict, torch.Tensor]:
    samples_ms = []
    peak_extra_bytes = []
    output = None
    for _ in range(repeats):
        elapsed_ms, output, peak_bytes = measure_once(fn, warmup, iters)
        samples_ms.append(elapsed_ms)
        peak_extra_bytes.append(peak_bytes)
    assert output is not None
    summary = summarize_samples(samples_ms)
    summary["peak_extra_mib"] = max(peak_extra_bytes) / 1024 / 1024
    return summary, output


def numerical_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = actual.float() - reference.float()
    abs_diff = diff.abs()
    return {
        "max_abs_diff_vs_flash_reference": abs_diff.max().item(),
        "mean_abs_diff_vs_flash_reference": abs_diff.mean().item(),
        "rmse_vs_flash_reference": diff.square().mean().sqrt().item(),
    }


def benchmark_backend(
    fn: Callable[[], torch.Tensor],
    *,
    role: str,
    warmup: int,
    iters: int,
    repeats: int,
    reference_out: torch.Tensor | None,
) -> tuple[dict, torch.Tensor]:
    timing, output = measure_repeated(
        fn,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
    )
    item = {"role": role, "status": "ok", **timing}
    if reference_out is None:
        item.update(
            {
                "max_abs_diff_vs_flash_reference": 0.0,
                "mean_abs_diff_vs_flash_reference": 0.0,
                "rmse_vs_flash_reference": 0.0,
            }
        )
    else:
        item.update(numerical_error(output, reference_out))
    return item, output


def unavailable_backend(role: str, exc: BaseException) -> tuple[dict, None]:
    """Represent a backend that could not be compiled or executed.

    A single Triton configuration should not prevent the rest of a sweep from
    producing evidence.  Keeping the exception in the JSON makes the failed
    candidate auditable instead of silently dropping it.
    """

    return {
        "role": role,
        "status": "unavailable",
        "error": f"{type(exc).__name__}: {exc}",
    }, None


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# INT8 Attention Kernel Benchmark",
        "",
        "## Environment",
        "",
        f"- commit: `{result['commit']}`",
        f"- branch: `{result['branch']}`",
        f"- git_dirty: `{result['git_dirty']}`",
        f"- command: `{result['command']}`",
        f"- timestamp: `{result['benchmark_timestamp']}`",
        f"- device: `{result['device']}`",
        f"- torch: `{result['torch_version']}`",
        f"- CUDA: `{result['cuda_version']}`",
        f"- Triton: `{result['triton_version']}`",
        f"- FlashAttention: `{result['flash_attn_version']}`",
        "",
        "## Configuration",
        "",
        f"- dtype: `{result['dtype']}`",
        f"- batch_size: `{result['batch_size']}`",
        f"- context_len: `{result['context_len']}`",
        f"- block_size: `{result['block_size']}`",
        f"- num_heads: `{result['num_heads']}`",
        f"- num_kv_heads: `{result['num_kv_heads']}`",
        f"- head_dim: `{result['head_dim']}`",
        f"- sliding_window_size: `{result['sliding_window_size']}`",
        f"- variants: `{','.join(result['variants'])}`",
        f"- block_tokens: `{','.join(map(str, result['block_tokens']))}`",
        f"- num_warps: `{','.join(map(str, result['num_warps']))}`",
        f"- production_default_num_warps: `{result['production_default_num_warps']}`",
        f"- num_stages: `{','.join(map(str, result['num_stages']))}`",
        f"- warmup: `{result['warmup']}`",
        f"- iters: `{result['iters']}`",
        f"- repeats: `{result['repeats']}`",
        "",
        "## Results",
        "",
        "| Backend | Role | median ms | min ms | max ms | stdev ms | rel. range | speedup vs Flash | max abs diff | peak extra MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reference_ms = result["results"]["flash_reference"]["median_ms"]
    for name, item in result["results"].items():
        if item["status"] != "ok":
            lines.append(
                f"| {name} | {item['role']} | unavailable | unavailable | "
                f"unavailable | unavailable | unavailable | unavailable | "
                f"unavailable | unavailable |"
            )
            lines.append(f"  - `{name}` error: `{item['error']}`")
            continue
        speedup = reference_ms / item["median_ms"] if item["median_ms"] > 0 else 0.0
        lines.append(
            f"| {name} | {item['role']} | {item['median_ms']:.6f} | "
            f"{item['min_ms']:.6f} | {item['max_ms']:.6f} | "
            f"{item['stdev_ms']:.6f} | {item['relative_range']:.4f} | "
            f"{speedup:.4f} | "
            f"{item['max_abs_diff_vs_flash_reference']:.6f} | "
            f"{item['peak_extra_mib']:.2f} |"
        )

    candidates = [
        (name, item)
        for name, item in result["results"].items()
        if item["role"] == "candidate" and item["status"] == "ok"
    ]
    if candidates:
        best_name, best_item = min(
            candidates,
            key=lambda pair: pair[1]["median_ms"],
        )
        lines.extend(
            [
                "",
                "## Best Custom Candidate",
                "",
                f"- backend: `{best_name}`",
                f"- median_ms: `{best_item['median_ms']:.6f}`",
                f"- speedup_vs_flash_reference: `{reference_ms / best_item['median_ms']:.4f}`",
                f"- max_abs_diff_vs_flash_reference: `{best_item['max_abs_diff_vs_flash_reference']:.6f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark INT8 fused decode attention kernels."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=3584)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--sliding-window-size", type=int, default=None)
    parser.add_argument("--variants", default="v1,latev,v3")
    parser.add_argument("--block-tokens", default="16,64,128,256")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--num-stages", default="2")
    parser.add_argument("--partition-sizes", default="128,256")
    parser.add_argument("--include-partitioned", action="store_true")
    parser.add_argument("--include-packed-dequant-flash", action="store_true")
    parser.add_argument("--dequant-block-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--result-dir",
        default="benchmark_results/attention_kernel",
    )
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    assert args.num_heads % args.num_kv_heads == 0
    assert args.block_size > 0
    assert args.context_len > 0
    assert args.repeats > 0
    assert args.sliding_window_size is None or args.sliding_window_size > 0
    variants = parse_str_list(args.variants)
    block_token_values = parse_int_list(args.block_tokens)
    num_warps_values = parse_int_list(args.num_warps)
    num_stages_values = parse_int_list(args.num_stages)
    partition_sizes = parse_int_list(args.partition_sizes)
    valid_variants = {"v1", "latev", "v3"}
    assert variants, "at least one variant is required"
    assert all(
        variant in valid_variants for variant in variants
    ), f"variants must be in {sorted(valid_variants)}"
    assert block_token_values, "at least one block_tokens value is required"
    assert num_warps_values, "at least one num_warps value is required"
    assert num_stages_values, "at least one num_stages value is required"

    (
        q,
        k_reference,
        v_reference,
        k_int8,
        v_int8,
        k_scale,
        v_scale,
        block_tables,
        context_lens,
    ) = make_case(args)
    softmax_scale = args.head_dim**-0.5

    def run_flash_reference():
        kwargs = dict(
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )
        if args.sliding_window_size is not None:
            kwargs["window_size"] = (args.sliding_window_size - 1, 0)
        output = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_reference,
            v_reference,
            **kwargs,
        )
        return output.squeeze(1) if output.dim() == 4 else output

    flash_item, flash_out = benchmark_backend(
        run_flash_reference,
        role="reference",
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        reference_out=None,
    )
    results = {"flash_reference": flash_item}

    if args.include_packed_dequant_flash:
        if args.sliding_window_size is None:
            first_logical_block = 0
        else:
            window_start = max(0, args.context_len - args.sliding_window_size)
            first_logical_block = window_start // args.block_size
        max_blocks = k_int8.size(0)
        selected_block_ids = torch.arange(
            first_logical_block,
            max_blocks,
            device=q.device,
            dtype=torch.int32,
        )
        num_selected_blocks = selected_block_ids.numel()
        packed_k = torch.empty(
            (
                num_selected_blocks,
                args.block_size,
                args.num_kv_heads,
                args.head_dim,
            ),
            device=q.device,
            dtype=q.dtype,
        )
        packed_v = torch.empty_like(packed_k)
        packed_block_tables = torch.arange(
            -first_logical_block,
            max_blocks - first_logical_block,
            device=q.device,
            dtype=torch.int32,
        ).repeat(args.batch_size, 1)
        if first_logical_block > 0:
            packed_block_tables[:, :first_logical_block] = -1

        def run_packed_dequant_flash():
            dequant_packed_kvcache(
                k_int8,
                v_int8,
                k_scale,
                v_scale,
                selected_block_ids,
                packed_k,
                packed_v,
                block_tokens=args.dequant_block_tokens,
            )
            kwargs = dict(
                cache_seqlens=context_lens,
                block_table=packed_block_tables,
                softmax_scale=softmax_scale,
                causal=True,
            )
            if args.sliding_window_size is not None:
                kwargs["window_size"] = (
                    args.sliding_window_size - 1,
                    0,
                )
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),
                packed_k,
                packed_v,
                **kwargs,
            )
            return output.squeeze(1) if output.dim() == 4 else output

        name = f"packed_dequant_flash_bt{args.dequant_block_tokens}"
        try:
            item, _ = benchmark_backend(
                run_packed_dequant_flash,
                role="baseline",
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                reference_out=flash_out,
            )
        except Exception as exc:
            item, _ = unavailable_backend("baseline", exc)
        results[name] = item

    variant_fns = {
        "v1": fused_int8_decode_attention,
        "latev": fused_int8_decode_attention_latev,
        "v3": fused_int8_decode_attention_v3,
    }
    for block_tokens in block_token_values:
        if block_tokens > args.block_size or args.block_size % block_tokens != 0:
            continue
        for variant in variants:
            kernel_fn = variant_fns[variant]
            for num_warps in num_warps_values:
                for num_stages in num_stages_values:

                    def run_fused(
                        kernel_fn=kernel_fn,
                        block_tokens=block_tokens,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    ):
                        return kernel_fn(
                            q,
                            k_int8,
                            v_int8,
                            k_scale,
                            v_scale,
                            block_tables,
                            context_lens,
                            softmax_scale,
                            args.sliding_window_size,
                            block_tokens=block_tokens,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )

                    name = (
                        f"int8_{variant}_bt{block_tokens}"
                        f"_w{num_warps}_s{num_stages}"
                    )
                    try:
                        item, _ = benchmark_backend(
                            run_fused,
                            role="candidate",
                            warmup=args.warmup,
                            iters=args.iters,
                            repeats=args.repeats,
                            reference_out=flash_out,
                        )
                    except Exception as exc:
                        item, _ = unavailable_backend("candidate", exc)
                    results[name] = item

    if args.include_partitioned:
        for partition_size in partition_sizes:
            effective_context = min(
                args.context_len,
                args.sliding_window_size
                if args.sliding_window_size is not None
                else args.context_len,
            )
            num_partitions = max(math.ceil(effective_context / partition_size), 1)
            block_head_dim = 1 << (args.head_dim - 1).bit_length()
            reusable_workspace = allocate_partitioned_workspace(
                q,
                num_partitions,
                block_head_dim,
            )
            reusable_output = torch.empty_like(q)

            def run_partitioned(partition_size=partition_size):
                return partitioned_fused_int8_decode_attention(
                    q,
                    k_int8,
                    v_int8,
                    k_scale,
                    v_scale,
                    block_tables,
                    context_lens,
                    softmax_scale,
                    args.sliding_window_size,
                    block_tokens=min(args.block_size, 256),
                    partition_size=partition_size,
                    max_context_len=args.context_len,
                )

            def run_partitioned_reuse(
                partition_size=partition_size,
                workspace=reusable_workspace,
                output=reusable_output,
            ):
                return partitioned_fused_int8_decode_attention(
                    q,
                    k_int8,
                    v_int8,
                    k_scale,
                    v_scale,
                    block_tables,
                    context_lens,
                    softmax_scale,
                    args.sliding_window_size,
                    block_tokens=min(args.block_size, 256),
                    partition_size=partition_size,
                    max_context_len=args.context_len,
                    workspace=workspace,
                    output=output,
                )

            name = f"int8_partitioned_ps{partition_size}"
            try:
                item, _ = benchmark_backend(
                    run_partitioned,
                    role="candidate",
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    reference_out=flash_out,
                )
            except Exception as exc:
                item, _ = unavailable_backend("candidate", exc)
            results[name] = item
            reuse_name = f"{name}_workspace_reuse"
            try:
                reuse_item, _ = benchmark_backend(
                    run_partitioned_reuse,
                    role="candidate",
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    reference_out=flash_out,
                )
            except Exception as exc:
                reuse_item, _ = unavailable_backend("candidate", exc)
            if item["status"] == "ok" and reuse_item["status"] == "ok":
                reuse_item["speedup_vs_allocating"] = (
                    item["median_ms"] / reuse_item["median_ms"]
                )
                reuse_item["avoided_peak_extra_mib"] = max(
                    item["peak_extra_mib"] - reuse_item["peak_extra_mib"],
                    0.0,
                )
            results[reuse_name] = reuse_item

    result = {
        **collect_benchmark_metadata(torch),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "block_size": args.block_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "sliding_window_size": args.sliding_window_size,
        "variants": variants,
        "block_tokens": block_token_values,
        "num_warps": num_warps_values,
        "production_default_num_warps": 4 if args.head_dim < 128 else 8,
        "num_stages": num_stages_values,
        "partition_sizes": partition_sizes,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
        "shape_manifest": build_shape_manifest(
            args,
            q=q,
            k_reference=k_reference,
            v_reference=v_reference,
            k_int8=k_int8,
            v_int8=v_int8,
            k_scale=k_scale,
            v_scale=v_scale,
            block_tables=block_tables,
            context_lens=context_lens,
        ),
        "results": results,
    }

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        args.name
        or f"attention_kernel_b{args.batch_size}_ctx{args.context_len}_{timestamp}"
    )
    json_path = result_dir / f"{name}.json"
    md_path = result_dir / f"{name}.md"
    shape_path = result_dir / f"{name}.shape_manifest.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    shape_path.write_text(
        json.dumps(result["shape_manifest"], indent=2, ensure_ascii=False) + "\n"
    )
    write_markdown(md_path, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {shape_path}")


if __name__ == "__main__":
    main()
