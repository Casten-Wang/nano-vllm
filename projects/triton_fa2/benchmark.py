"""Benchmark the Triton FlashAttention-2-style forward kernel.

The benchmark keeps PyTorch SDPA baselines explicit:

* ``sdpa_default`` lets PyTorch choose its normal backend and records the
  concrete backend selected by PyTorch.
* ``sdpa_forced_flash`` enables only the FlashAttention SDPA backend when the
  current runtime supports it.

For GQA, the benchmark also records a materialized-repeat-K/V baseline so that
the memory and latency effect of avoiding repeated K/V can be measured
separately from the attention algorithm itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import triton

from .reference import (
    error_summary,
    repeat_kv_for_gqa,
    sdpa_backend_context,
    sdpa_reference,
    selected_sdpa_backend,
    torch_attention_reference,
)
from .triton_fa2 import triton_flash_attention_forward
from .triton_fa2_v2 import (
    V2_AUTOTUNE_CONFIGS,
    get_last_v2_autotune_config,
    triton_flash_attention_forward_v2,
    triton_flash_attention_forward_v2_configured,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def to_dao_flash_attn_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Convert project layout [B, H, N, D] to Dao FA layout [B, N, H, D]."""

    if tensor.ndim != 4:
        raise ValueError("attention tensor must have shape [B, H, N, D]")
    return tensor.transpose(1, 2).contiguous()


def from_dao_flash_attn_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Convert Dao FA output [B, N, H, D] back to project layout [B, H, N, D]."""

    if tensor.ndim != 4:
        raise ValueError("Dao FlashAttention output must have shape [B, N, H, D]")
    return tensor.transpose(1, 2).contiguous()


def load_dao_flash_attn() -> tuple[Callable | None, str | None]:
    """Load the optional Dao FlashAttention provider without hiding failures."""

    try:
        from flash_attn import flash_attn_func
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return flash_attn_func, None


def dao_flash_attn_version() -> str | None:
    try:
        import flash_attn
    except Exception:
        return None
    return getattr(flash_attn, "__version__", "unknown")


def serialize_v2_configs() -> list[dict[str, int]]:
    return [
        {
            "block_m": int(config.kwargs["BLOCK_M"]),
            "block_n": int(config.kwargs["BLOCK_N"]),
            "num_warps": int(config.num_warps),
            "num_stages": int(config.num_stages),
        }
        for config in V2_AUTOTUNE_CONFIGS
    ]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    result: list[bool] = []
    for item in items:
        if item in {"1", "true", "yes", "causal"}:
            result.append(True)
        elif item in {"0", "false", "no", "noncausal"}:
            result.append(False)
        else:
            raise ValueError(f"invalid boolean value: {item}")
    return result


def nonempty_int_list(
    parser: argparse.ArgumentParser,
    option: str,
    value: str,
) -> list[int]:
    try:
        result = parse_int_list(value)
    except ValueError as exc:
        parser.error(f"{option} must be a comma-separated integer list: {exc}")
    if not result:
        parser.error(f"{option} must contain at least one value")
    return result


def measure_ms(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor, int, int]:
    out: torch.Tensor | None = None
    for _ in range(warmup):
        out = fn()
        del out
        out = None

    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if out is not None:
            del out
        out = fn()
    end.record()
    torch.cuda.synchronize()

    if out is None:
        raise RuntimeError("benchmark produced no output")
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_extra_bytes = max(0, peak_bytes - baseline_bytes)
    return start.elapsed_time(end) / iters, out, peak_bytes, peak_extra_bytes


def variant_name(
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    loop_num_stages: int,
    causal_early_stop: bool,
) -> str:
    return (
        f"triton_fa2_bm{block_m}_bn{block_n}_w{num_warps}"
        f"_ks{num_stages}_ls{loop_num_stages}"
        f"_ces{int(causal_early_stop)}"
    )


def attention_tflops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    ms: float,
) -> float:
    if ms <= 0:
        return 0.0
    score_elements = seq_len * (seq_len + 1) / 2 if causal else seq_len * seq_len
    # Count one multiply-add as two FLOPs.
    flops = batch_size * num_heads * score_elements * head_dim * 4
    return flops / (ms * 1.0e-3) / 1.0e12


def benchmark_one_backend(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    reference_out: torch.Tensor | None = None,
    selected_backend: str | None = None,
    output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict, torch.Tensor]:
    ms, raw_out, peak, peak_extra = measure_ms(fn, warmup, iters)
    out = output_transform(raw_out) if output_transform is not None else raw_out
    if output_transform is not None:
        del raw_out
    result = {
        "status": "ok",
        **(
            {"selected_backend": selected_backend}
            if selected_backend is not None
            else {}
        ),
        "ms": ms,
        "tflops": attention_tflops(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            causal,
            ms,
        ),
        "peak_alloc_mib": peak / 1024 / 1024,
        "peak_extra_mib": peak_extra / 1024 / 1024,
    }
    if reference_out is not None:
        result.update(
            {
                f"{key}_vs_sdpa_default": value
                for key, value in error_summary(out, reference_out).items()
            }
        )
    return result, out


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Triton FA2-style Forward Benchmark",
        "",
        "## Environment",
        "",
        f"- commit: `{result['commit']}`",
        f"- branch: `{result['branch']}`",
        f"- git_dirty: `{result['git_dirty']}`",
        f"- device: `{result['device']}`",
        f"- torch: `{result['torch_version']}`",
        f"- cuda: `{result['cuda_version']}`",
        f"- triton: `{result['triton_version']}`",
        "",
        "## Configuration",
        "",
        f"- batch_size: `{result['batch_size']}`",
        f"- num_q_heads: `{result['num_q_heads']}`",
        f"- num_kv_heads: `{result['num_kv_heads']}`",
        f"- seq_lens: `{','.join(map(str, result['seq_lens']))}`",
        f"- head_dims: `{','.join(map(str, result['head_dims']))}`",
        f"- dtype: `{result['dtype']}`",
        f"- causal_values: `{','.join('causal' if x else 'noncausal' for x in result['causal_values'])}`",
        f"- causal_early_stop_values: `{','.join(map(str, result['causal_early_stop_values']))}`",
        f"- block_ms: `{','.join(map(str, result['block_ms']))}`",
        f"- block_ns: `{','.join(map(str, result['block_ns']))}`",
        f"- num_warps_list: `{','.join(map(str, result['num_warps_list']))}`",
        f"- num_stages_list: `{','.join(map(str, result['num_stages_list']))}`",
        f"- loop_num_stages_list: `{','.join(map(str, result['loop_num_stages_list']))}`",
        f"- warmup: `{result['warmup']}`",
        f"- iters: `{result['iters']}`",
        f"- seed: `{result['seed']}`",
        f"- include_dao_flash_attn: `{result['include_dao_flash_attn']}`",
        f"- flash_attn_version: `{result['flash_attn_version']}`",
        "",
        "## Results",
        "",
        "| causal | seq_len | head_dim | backend | concrete_backend | status | ms | TFLOPS | speedup_vs_default | speedup_vs_forced_flash | max_abs_vs_default | peak_extra_mib |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        default_item = row["backends"]["sdpa_default"]
        default_ms = default_item["ms"]
        forced_item = row["backends"].get("sdpa_forced_flash")
        forced_ms = (
            forced_item["ms"]
            if forced_item is not None and forced_item["status"] == "ok"
            else None
        )
        for name, item in row["backends"].items():
            label = "causal" if row["causal"] else "noncausal"
            if item["status"] != "ok":
                lines.append(
                    f"| {label} | {row['seq_len']} | {row['head_dim']} | {name} | "
                    f"{item.get('selected_backend', '-')} | {item['status']} | "
                    "- | - | - | - | - | - |"
                )
                continue
            speedup_default = default_ms / item["ms"]
            speedup_forced = (
                f"{forced_ms / item['ms']:.4f}" if forced_ms is not None else "-"
            )
            lines.append(
                f"| {label} | {row['seq_len']} | {row['head_dim']} | {name} | "
                f"{item.get('selected_backend', '-')} | ok | {item['ms']:.6f} | "
                f"{item['tflops']:.2f} | {speedup_default:.4f} | "
                f"{speedup_forced} | {item.get('max_abs_vs_sdpa_default', 0.0):.6f} | "
                f"{item['peak_extra_mib']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Best Triton Variants",
            "",
            "| causal | seq_len | head_dim | best_variant | ms | TFLOPS | speedup_vs_default | speedup_vs_forced_flash |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        default_ms = row["backends"]["sdpa_default"]["ms"]
        forced_item = row["backends"].get("sdpa_forced_flash")
        forced_ms = (
            forced_item["ms"]
            if forced_item is not None and forced_item["status"] == "ok"
            else None
        )
        triton_items = [
            (name, item)
            for name, item in row["backends"].items()
            if name.startswith("triton_fa2_") and item["status"] == "ok"
        ]
        if not triton_items:
            continue
        best_name, best_item = min(triton_items, key=lambda pair: pair[1]["ms"])
        forced_speedup = (
            f"{forced_ms / best_item['ms']:.4f}" if forced_ms is not None else "-"
        )
        label = "causal" if row["causal"] else "noncausal"
        lines.append(
            f"| {label} | {row['seq_len']} | {row['head_dim']} | {best_name} | "
            f"{best_item['ms']:.6f} | {best_item['tflops']:.2f} | "
            f"{default_ms / best_item['ms']:.4f} | {forced_speedup} |"
        )
    lines.extend(
        [
            "",
            "## Selected Providers",
            "",
            "| causal | seq_len | head_dim | provider | selected_choice | ms | speedup_vs_default | speedup_vs_forced_flash |",
            "|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        default_ms = row["backends"]["sdpa_default"]["ms"]
        forced_item = row["backends"].get("sdpa_forced_flash")
        forced_ms = (
            forced_item["ms"]
            if forced_item is not None and forced_item["status"] == "ok"
            else None
        )
        label = "causal" if row["causal"] else "noncausal"
        for provider in (
            "triton_v1_best",
            "triton_v2_autotuned",
            "triton_v2_configured_bm128_bn128_w4_s2",
            "dao_flash_attn",
        ):
            item = row["backends"].get(provider)
            if item is None:
                continue
            if item["status"] != "ok":
                lines.append(
                    f"| {label} | {row['seq_len']} | {row['head_dim']} | {provider} | "
                    f"- | {item['status']} | - | - |"
                )
                continue
            forced_speedup = (
                f"{forced_ms / item['ms']:.4f}" if forced_ms is not None else "-"
            )
            lines.append(
                f"| {label} | {row['seq_len']} | {row['head_dim']} | {provider} | "
                f"{item.get('selected_config', item.get('selected_variant', '-'))} | "
                f"{item['ms']:.6f} | "
                f"{default_ms / item['ms']:.4f} | {forced_speedup} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton FlashAttention-2-style forward attention."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="K/V head count. Defaults to --num-heads; use a divisor for GQA.",
    )
    parser.add_argument("--seq-lens", default="512,1024,2048,4096")
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal-values", default="causal,noncausal")
    parser.add_argument("--causal-early-stop-values", default="true")
    parser.add_argument("--block-ms", default="16,32")
    parser.add_argument("--block-ns", default="32,64")
    parser.add_argument("--num-warps-list", default="4")
    parser.add_argument("--num-stages-list", default="3")
    parser.add_argument(
        "--loop-num-stages-list",
        default=None,
        help="tl.range pipeline stages; defaults to --num-stages-list.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-naive", action="store_true")
    parser.add_argument(
        "--include-dao-flash-attn",
        action="store_true",
        help="Include optional Dao FlashAttention CUDA provider.",
    )
    parser.add_argument("--max-naive-seq-len", type=int, default=2048)
    parser.add_argument("--result-dir", default="benchmark_results/triton_fa2")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_heads <= 0:
        parser.error("--num-heads must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.max_naive_seq_len <= 0:
        parser.error("--max-naive-seq-len must be positive")

    seq_lens = nonempty_int_list(parser, "--seq-lens", args.seq_lens)
    head_dims = nonempty_int_list(parser, "--head-dims", args.head_dims)
    block_ms = nonempty_int_list(parser, "--block-ms", args.block_ms)
    block_ns = nonempty_int_list(parser, "--block-ns", args.block_ns)
    num_warps_list = nonempty_int_list(parser, "--num-warps-list", args.num_warps_list)
    num_stages_list = nonempty_int_list(
        parser, "--num-stages-list", args.num_stages_list
    )
    loop_num_stages_list = (
        nonempty_int_list(parser, "--loop-num-stages-list", args.loop_num_stages_list)
        if args.loop_num_stages_list is not None
        else num_stages_list
    )
    try:
        causal_values = parse_bool_list(args.causal_values)
        causal_early_stop_values = parse_bool_list(args.causal_early_stop_values)
    except ValueError as exc:
        parser.error(str(exc))
    if not causal_values:
        parser.error("--causal-values must contain at least one value")
    if not causal_early_stop_values:
        parser.error("--causal-early-stop-values must contain at least one value")
    if any(value <= 0 for value in seq_lens):
        parser.error("--seq-lens values must be positive")
    if any(value not in (64, 128) for value in head_dims):
        parser.error("--head-dims currently supports only 64 and 128")
    for option, values in (
        ("--block-ms", block_ms),
        ("--block-ns", block_ns),
        ("--num-warps-list", num_warps_list),
        ("--num-stages-list", num_stages_list),
        ("--loop-num-stages-list", loop_num_stages_list),
    ):
        if any(value <= 0 for value in values):
            parser.error(f"{option} values must be positive")
    if any(value & (value - 1) for value in block_ms):
        parser.error("--block-ms values must be powers of two")
    if any(value & (value - 1) for value in block_ns):
        parser.error("--block-ns values must be powers of two")

    num_kv_heads = (
        args.num_kv_heads if args.num_kv_heads is not None else args.num_heads
    )
    if num_kv_heads <= 0 or args.num_heads % num_kv_heads != 0:
        parser.error("--num-kv-heads must be positive and divide --num-heads")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Triton FA2 benchmark")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    dao_flash_fn: Callable | None = None
    dao_flash_error: str | None = None
    if args.include_dao_flash_attn:
        dao_flash_fn, dao_flash_error = load_dao_flash_attn()
    rows: list[dict] = []

    for causal in causal_values:
        for seq_len in seq_lens:
            for head_dim in head_dims:
                torch.manual_seed(args.seed)
                q = torch.randn(
                    args.batch_size,
                    args.num_heads,
                    seq_len,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                )
                k = torch.randn(
                    args.batch_size,
                    num_kv_heads,
                    seq_len,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                )
                v = torch.randn_like(k)
                softmax_scale = head_dim**-0.5
                default_backend = selected_sdpa_backend(
                    q, k, v, causal=causal, softmax_scale=softmax_scale
                )
                sdpa_default, sdpa_out = benchmark_one_backend(
                    lambda: sdpa_reference(
                        q,
                        k,
                        v,
                        causal=causal,
                        softmax_scale=softmax_scale,
                        backend="default",
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    batch_size=args.batch_size,
                    num_heads=args.num_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    causal=causal,
                    selected_backend=f"default:{default_backend}",
                )
                sdpa_default.update(
                    {
                        "max_abs_vs_sdpa_default": 0.0,
                        "mean_abs_vs_sdpa_default": 0.0,
                        "rms_vs_sdpa_default": 0.0,
                    }
                )
                backends: dict[str, dict] = {"sdpa_default": sdpa_default}

                try:
                    with sdpa_backend_context("flash"):
                        forced_backend = selected_sdpa_backend(
                            q, k, v, causal=causal, softmax_scale=softmax_scale
                        )
                        if forced_backend != "FLASH_ATTENTION":
                            raise RuntimeError(
                                f"forced context selected {forced_backend}"
                            )
                        forced_result, forced_out = benchmark_one_backend(
                            lambda: sdpa_reference(
                                q,
                                k,
                                v,
                                causal=causal,
                                softmax_scale=softmax_scale,
                                backend="default",
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                            batch_size=args.batch_size,
                            num_heads=args.num_heads,
                            seq_len=seq_len,
                            head_dim=head_dim,
                            causal=causal,
                            reference_out=sdpa_out,
                            selected_backend="forced:FLASH_ATTENTION",
                        )
                        del forced_out
                except (RuntimeError, NotImplementedError) as exc:
                    warnings.warn(f"forced FlashAttention SDPA unavailable: {exc}")
                    backends["sdpa_forced_flash"] = {
                        "status": "unavailable",
                        "selected_backend": "forced:FLASH_ATTENTION",
                        "reason": str(exc),
                    }
                else:
                    backends["sdpa_forced_flash"] = forced_result

                if args.include_dao_flash_attn:
                    if dao_flash_fn is None:
                        backends["dao_flash_attn"] = {
                            "status": "unavailable",
                            "selected_backend": "dao:flash_attn_func",
                            "reason": dao_flash_error,
                        }
                    else:
                        try:
                            dao_q = to_dao_flash_attn_layout(q)
                            dao_k = to_dao_flash_attn_layout(k)
                            dao_v = to_dao_flash_attn_layout(v)
                            dao_result, dao_out = benchmark_one_backend(
                                lambda: dao_flash_fn(
                                    dao_q,
                                    dao_k,
                                    dao_v,
                                    dropout_p=0.0,
                                    softmax_scale=softmax_scale,
                                    causal=causal,
                                ),
                                warmup=args.warmup,
                                iters=args.iters,
                                batch_size=args.batch_size,
                                num_heads=args.num_heads,
                                seq_len=seq_len,
                                head_dim=head_dim,
                                causal=causal,
                                reference_out=sdpa_out,
                                selected_backend="dao:flash_attn_func",
                                output_transform=from_dao_flash_attn_layout,
                            )
                            dao_result["layout_conversion"] = (
                                "[B,H,N,D] <-> [B,N,H,D] outside timing"
                            )
                            del dao_out
                            backends["dao_flash_attn"] = dao_result
                            del dao_q, dao_k, dao_v
                        except Exception as exc:
                            warnings.warn(f"Dao FlashAttention unavailable: {exc}")
                            backends["dao_flash_attn"] = {
                                "status": "unavailable",
                                "selected_backend": "dao:flash_attn_func",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }

                if num_kv_heads != args.num_heads:
                    repeated_k, repeated_v = repeat_kv_for_gqa(q, k, v)
                    repeated_backend = selected_sdpa_backend(
                        q,
                        repeated_k,
                        repeated_v,
                        causal=causal,
                        softmax_scale=softmax_scale,
                    )
                    repeated_result, repeated_out = benchmark_one_backend(
                        lambda: sdpa_reference(
                            q,
                            repeated_k,
                            repeated_v,
                            causal=causal,
                            softmax_scale=softmax_scale,
                            backend="default",
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        causal=causal,
                        reference_out=sdpa_out,
                        selected_backend=f"default:{repeated_backend}",
                    )
                    backends["sdpa_repeat_kv"] = repeated_result
                    repeated_kv_mib = (
                        (repeated_k.nbytes + repeated_v.nbytes - k.nbytes - v.nbytes)
                        / 1024
                        / 1024
                    )
                    del repeated_k, repeated_v
                    del repeated_out
                else:
                    repeated_kv_mib = 0.0

                for block_m in block_ms:
                    for block_n in block_ns:
                        for num_warps in num_warps_list:
                            for num_stages in num_stages_list:
                                for loop_num_stages in loop_num_stages_list:
                                    effective_early_stop_values = (
                                        causal_early_stop_values if causal else [False]
                                    )
                                    for early_stop in effective_early_stop_values:
                                        name = variant_name(
                                            block_m,
                                            block_n,
                                            num_warps,
                                            num_stages,
                                            loop_num_stages,
                                            early_stop,
                                        )
                                        try:
                                            triton_result, triton_out = (
                                                benchmark_one_backend(
                                                    lambda block_m=block_m, block_n=block_n, num_warps=num_warps, num_stages=num_stages, loop_num_stages=loop_num_stages, early_stop=early_stop: triton_flash_attention_forward(
                                                        q,
                                                        k,
                                                        v,
                                                        causal=causal,
                                                        causal_early_stop=early_stop,
                                                        softmax_scale=softmax_scale,
                                                        block_m=block_m,
                                                        block_n=block_n,
                                                        num_warps=num_warps,
                                                        num_stages=num_stages,
                                                        loop_num_stages=loop_num_stages,
                                                    ),
                                                    warmup=args.warmup,
                                                    iters=args.iters,
                                                    batch_size=args.batch_size,
                                                    num_heads=args.num_heads,
                                                    seq_len=seq_len,
                                                    head_dim=head_dim,
                                                    causal=causal,
                                                    reference_out=sdpa_out,
                                                    selected_backend="custom:triton",
                                                )
                                            )
                                        except Exception as exc:
                                            warnings.warn(
                                                f"Triton V1 candidate {name} unavailable: {exc}"
                                            )
                                            backends[name] = {
                                                "status": "unavailable",
                                                "selected_backend": "custom:triton",
                                                "reason": f"{type(exc).__name__}: {exc}",
                                            }
                                            torch.cuda.empty_cache()
                                        else:
                                            del triton_out
                                            backends[name] = triton_result

                v1_items = [
                    (name, item)
                    for name, item in backends.items()
                    if name.startswith("triton_fa2_") and item["status"] == "ok"
                ]
                if v1_items:
                    best_v1_name, best_v1_item = min(
                        v1_items, key=lambda pair: pair[1]["ms"]
                    )
                    backends["triton_v1_best"] = dict(
                        best_v1_item,
                        selected_variant=best_v1_name,
                        selected_backend="custom:triton-v1",
                    )
                else:
                    backends["triton_v1_best"] = {
                        "status": "unavailable",
                        "selected_backend": "custom:triton-v1",
                        "reason": "no successful V1 variant",
                    }

                try:
                    v2_result, v2_out = benchmark_one_backend(
                        lambda: triton_flash_attention_forward_v2(
                            q,
                            k,
                            v,
                            causal=causal,
                            softmax_scale=softmax_scale,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        causal=causal,
                        reference_out=sdpa_out,
                        selected_backend="custom:triton-v2-autotuned",
                    )
                    v2_result["selected_config"] = get_last_v2_autotune_config()
                    del v2_out
                    backends["triton_v2_autotuned"] = v2_result
                except Exception as exc:
                    warnings.warn(f"Triton V2 unavailable: {exc}")
                    backends["triton_v2_autotuned"] = {
                        "status": "unavailable",
                        "selected_backend": "custom:triton-v2-autotuned",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    torch.cuda.empty_cache()

                configured_v2_name = "triton_v2_configured_bm128_bn128_w4_s2"
                if seq_len < 128:
                    backends[configured_v2_name] = {
                        "status": "unavailable",
                        "selected_backend": "custom:triton-v2-configured",
                        "reason": "sequence length is smaller than BLOCK_M/BLOCK_N=128",
                    }
                else:
                    try:
                        configured_v2_result, configured_v2_out = benchmark_one_backend(
                            lambda: triton_flash_attention_forward_v2_configured(
                                q,
                                k,
                                v,
                                causal=causal,
                                softmax_scale=softmax_scale,
                                block_m=128,
                                block_n=128,
                                num_warps=4,
                                num_stages=2,
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                            batch_size=args.batch_size,
                            num_heads=args.num_heads,
                            seq_len=seq_len,
                            head_dim=head_dim,
                            causal=causal,
                            reference_out=sdpa_out,
                            selected_backend="custom:triton-v2-configured",
                        )
                        configured_v2_result["selected_config"] = {
                            "block_m": 128,
                            "block_n": 128,
                            "num_warps": 4,
                            "num_stages": 2,
                        }
                        del configured_v2_out
                        backends[configured_v2_name] = configured_v2_result
                    except Exception as exc:
                        warnings.warn(
                            f"Configured Triton V2 candidate unavailable: {exc}"
                        )
                        backends[configured_v2_name] = {
                            "status": "unavailable",
                            "selected_backend": "custom:triton-v2-configured",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                        torch.cuda.empty_cache()

                if args.include_naive and seq_len <= args.max_naive_seq_len:
                    naive_result, naive_out = benchmark_one_backend(
                        lambda: torch_attention_reference(
                            q,
                            k,
                            v,
                            causal=causal,
                            softmax_scale=softmax_scale,
                        ),
                        warmup=args.warmup,
                        iters=max(1, min(args.iters, 20)),
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        seq_len=seq_len,
                        head_dim=head_dim,
                        causal=causal,
                        reference_out=sdpa_out,
                        selected_backend="reference:torch_matmul",
                    )
                    del naive_out
                    backends["torch_naive"] = naive_result

                rows.append(
                    {
                        "causal": causal,
                        "seq_len": seq_len,
                        "head_dim": head_dim,
                        "gqa": num_kv_heads != args.num_heads,
                        "materialized_repeat_kv_mib": repeated_kv_mib,
                        "backends": backends,
                    }
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"triton_fa2_{timestamp}"
    result = {
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["branch", "--show-current"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "command": list(sys.argv),
        "working_directory": os.getcwd(),
        "python_version": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "transformers_version": module_version("transformers"),
        "batch_size": args.batch_size,
        "num_q_heads": args.num_heads,
        "num_kv_heads": num_kv_heads,
        "seq_lens": seq_lens,
        "head_dims": head_dims,
        "dtype": args.dtype,
        "causal_values": causal_values,
        "causal_early_stop_values": causal_early_stop_values,
        "block_ms": block_ms,
        "block_ns": block_ns,
        "num_warps_list": num_warps_list,
        "num_stages_list": num_stages_list,
        "loop_num_stages_list": loop_num_stages_list,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "include_dao_flash_attn": args.include_dao_flash_attn,
        "flash_attn_version": (
            dao_flash_attn_version() if args.include_dao_flash_attn else None
        ),
        "v2_autotune_candidates": serialize_v2_configs(),
        "benchmark_timestamp": datetime.now().astimezone().isoformat(),
        "rows": rows,
    }
    json_path = result_dir / f"{stem}.json"
    md_path = result_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_markdown(md_path, result)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
