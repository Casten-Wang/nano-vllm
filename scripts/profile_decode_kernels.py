import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.int8_fused_attention import (
    fused_int8_decode_attention,
    fused_int8_decode_attention_latev,
    fused_int8_decode_attention_v3,
)
from nanovllm.layers.kv_cache_quant import store_kvcache, store_kvcache_int8


def measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile decode-time KV store and attention kernels.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=3584)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--int8-attention-variant", choices=("v1", "latev", "v3"), default="v1")
    parser.add_argument("--block-tokens", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", default="benchmark_results/profile_decode")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    assert args.num_heads % args.num_kv_heads == 0
    assert args.block_size > 0
    assert args.block_tokens > 0 and args.block_size % args.block_tokens == 0

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    max_blocks = (args.context_len + args.block_size - 1) // args.block_size
    softmax_scale = args.head_dim ** -0.5

    q = torch.randn(args.batch_size, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k_new = torch.randn(args.batch_size, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    # Use valid cache slots near the end of the allocated cache. The store
    # microbench only measures write cost, so these slots do not need to match
    # the per-sequence block table used by the attention microbench.
    num_slots = max_blocks * args.block_size
    slot_mapping = torch.arange(args.batch_size, device=device, dtype=torch.int32) + (num_slots - args.batch_size)

    k_cache_fp = torch.randn(max_blocks, args.block_size, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v_cache_fp = torch.randn_like(k_cache_fp)
    k_cache_int8 = torch.randint(
        -127,
        128,
        (max_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    v_cache_int8 = torch.randint(
        -127,
        128,
        (max_blocks, args.block_size, args.num_kv_heads, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    k_scale = torch.rand(max_blocks, args.block_size, args.num_kv_heads, device=device, dtype=torch.float16) * 0.03 + 0.001
    v_scale = torch.rand_like(k_scale)
    block_tables = torch.arange(max_blocks, device=device, dtype=torch.int32).repeat(args.batch_size, 1)
    context_lens = torch.full((args.batch_size,), args.context_len, device=device, dtype=torch.int32)

    def run_store_fp():
        store_kvcache(k_new, v_new, k_cache_fp, v_cache_fp, slot_mapping)

    def run_store_int8():
        store_kvcache_int8(k_new, v_new, k_cache_int8, v_cache_int8, k_scale, v_scale, slot_mapping)

    def run_flash_attention():
        out = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache_fp,
            v_cache_fp,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )
        return out.squeeze(1) if out.dim() == 4 else out

    int8_attention_fns = {
        "v1": fused_int8_decode_attention,
        "latev": fused_int8_decode_attention_latev,
        "v3": fused_int8_decode_attention_v3,
    }
    int8_attention_fn = int8_attention_fns[args.int8_attention_variant]

    def run_int8_attention():
        return int8_attention_fn(
            q,
            k_cache_int8,
            v_cache_int8,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            block_tokens=args.block_tokens,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

    results = {
        "store_fp_ms": measure_ms(run_store_fp, args.warmup, args.iters),
        "store_int8_quant_ms": measure_ms(run_store_int8, args.warmup, args.iters),
        "flash_attention_ms": measure_ms(run_flash_attention, args.warmup, args.iters),
        "int8_fused_attention_ms": measure_ms(run_int8_attention, args.warmup, args.iters),
    }
    results["store_int8_vs_fp_ratio"] = results["store_int8_quant_ms"] / results["store_fp_ms"]
    results["int8_attention_vs_flash_ratio"] = results["int8_fused_attention_ms"] / results["flash_attention_ms"]
    results["per_model_step_store_fp_ms"] = results["store_fp_ms"] * args.num_layers
    results["per_model_step_store_int8_ms"] = results["store_int8_quant_ms"] * args.num_layers
    results["per_model_step_flash_attention_ms"] = results["flash_attention_ms"] * args.num_layers
    results["per_model_step_int8_attention_ms"] = results["int8_fused_attention_ms"] * args.num_layers

    payload = {
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "context_len": args.context_len,
        "block_size": args.block_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "num_layers": args.num_layers,
        "dtype": args.dtype,
        "int8_attention_variant": args.int8_attention_variant,
        "block_tokens": args.block_tokens,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"profile_decode_kernels_b{args.batch_size}_ctx{args.context_len}_{timestamp}"
    json_path = result_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
