"""Run an auditable GPU matrix for the INT8 KV-cache store kernel.

This is a correctness/data-collection script, not a latency benchmark.  It
keeps the actual input, quantized cache, scales, slot mapping, tensor layouts,
and error metrics so the result can be explained in an interview without
reconstructing anything from source code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanovllm.benchmark_metadata import collect_benchmark_metadata
from nanovllm.layers.kv_cache_quant import store_kvcache_int8


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected a non-empty list of positive integers")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty list")
    return values


def tensor_metadata(name: str, tensor: torch.Tensor) -> dict:
    return {
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


def quantize_reference(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = tensor.float()
    absmax = source.abs().amax(dim=-1)
    scale = torch.clamp(absmax / 127.0, min=1.0e-6)
    scaled = source / scale.unsqueeze(-1)
    rounded = torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        torch.ceil(scaled - 0.5),
    )
    return rounded.clamp(-127, 127).to(torch.int8), scale


def make_slot_mapping(
    num_tokens: int,
    pattern: str,
    *,
    seed: int,
) -> torch.Tensor:
    if pattern == "contiguous":
        values = list(range(num_tokens))
    elif pattern == "shuffled":
        values = list(range(num_tokens))
        random.Random(seed).shuffle(values)
    elif pattern == "with_invalid":
        values = list(range(num_tokens))
        # Keep token 0 valid so N=1 still exercises the kernel and produces
        # non-empty error statistics.
        for index in range(1, num_tokens, 7):
            values[index] = -1
    else:
        raise ValueError(
            f"unsupported slot pattern {pattern!r}; "
            "use contiguous, shuffled, or with_invalid"
        )
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def scalar_metrics(
    dequantized: torch.Tensor,
    source: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    source_f = source.float()
    dequant_f = dequantized.float()
    diff = dequant_f - source_f
    abs_diff = diff.abs()
    denom = source_f.abs().clamp_min(1.0e-6)
    flat_source = source_f.reshape(-1)
    flat_dequant = dequant_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_source.unsqueeze(0),
        flat_dequant.unsqueeze(0),
    ).item()
    return {
        "scale_min": scale.float().min().item(),
        "scale_max": scale.float().max().item(),
        "scale_mean": scale.float().mean().item(),
        "abs_error_max": abs_diff.max().item(),
        "abs_error_mean": abs_diff.mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_error_mean": (abs_diff / denom).mean().item(),
        "relative_error_max": (abs_diff / denom).max().item(),
        "cosine_similarity": cosine,
        "saturation_ratio": (quantized.abs() == 127).float().mean().item(),
        "dequant_error_bound_passed": bool(
            (abs_diff <= scale.float().unsqueeze(-1) * 0.55 + 2.0e-4)
            .all()
            .item()
        ),
    }


def run_case(
    *,
    num_tokens: int,
    pattern: str,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seed: int,
    raw_dir: Path | None,
) -> dict:
    torch.manual_seed(seed)
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    # Include deterministic edge values in every matrix.  This exercises the
    # zero-vector EPS path and the largest representable symmetric INT8 code.
    key[0].zero_()
    value[0].zero_()
    if num_tokens > 1:
        key[1, 0, 0] = 32
        value[1, 0, 0] = -32

    slot_mapping = make_slot_mapping(num_tokens, pattern, seed=seed + 17)
    valid = slot_mapping >= 0
    has_valid = bool(valid.any().item())
    max_slot = max(
        num_tokens - 1,
        int(slot_mapping[valid].max().item()) if has_valid else 0,
    )
    num_blocks = max(1, math.ceil((max_slot + 1) / block_size))
    k_cache = torch.full(
        (num_blocks, block_size, num_kv_heads, head_dim),
        -99,
        dtype=torch.int8,
        device="cuda",
    )
    v_cache = torch.full_like(k_cache, -99)
    k_scale = torch.full(
        (num_blocks, block_size, num_kv_heads),
        -1.0,
        dtype=torch.float16,
        device="cuda",
    )
    v_scale = torch.full_like(k_scale, -1.0)

    store_kvcache_int8(
        key,
        value,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping,
    )
    torch.cuda.synchronize()

    expected_k, expected_k_scale = quantize_reference(key)
    expected_v, expected_v_scale = quantize_reference(value)
    flat_k = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v = v_cache.view(-1, num_kv_heads, head_dim)
    flat_ks = k_scale.view(-1, num_kv_heads)
    flat_vs = v_scale.view(-1, num_kv_heads)
    valid_slots = slot_mapping[valid].long()
    valid_key = key[valid]
    valid_value = value[valid]
    valid_kq = flat_k[valid_slots]
    valid_vq = flat_v[valid_slots]
    valid_ks = flat_ks[valid_slots]
    valid_vs = flat_vs[valid_slots]
    expected_kq = expected_k[valid]
    expected_vq = expected_v[valid]
    expected_ks = expected_k_scale[valid]
    expected_vs = expected_v_scale[valid]
    k_dequant = valid_kq.float() * valid_ks.float().unsqueeze(-1)
    v_dequant = valid_vq.float() * valid_vs.float().unsqueeze(-1)

    # Every physical slot not targeted by a valid mapping must retain the
    # sentinel, regardless of whether this case includes -1 input mappings.
    # This catches stray/out-of-bounds writes in every slot pattern.
    used = set(valid_slots.detach().cpu().tolist())
    untouched_slot_count = num_blocks * block_size - len(used)
    invalid_untouched = True
    for slot in set(range(num_blocks * block_size)) - used:
        invalid_untouched = invalid_untouched and bool(
            torch.equal(
                flat_k[slot],
                torch.full_like(flat_k[slot], -99),
            )
        )
        invalid_untouched = invalid_untouched and bool(
            torch.equal(
                flat_v[slot],
                torch.full_like(flat_v[slot], -99),
            )
        )
        invalid_untouched = invalid_untouched and bool(
            torch.equal(
                flat_ks[slot],
                torch.full_like(flat_ks[slot], -1.0),
            )
        )
        invalid_untouched = invalid_untouched and bool(
            torch.equal(
                flat_vs[slot],
                torch.full_like(flat_vs[slot], -1.0),
            )
        )

    result = {
        "num_tokens": num_tokens,
        "pattern": pattern,
        "seed": seed,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "valid_token_count": int(valid.sum().item()),
        "invalid_token_count": int((~valid).sum().item()),
        "untouched_slot_count": untouched_slot_count,
        "slot_mapping": slot_mapping.cpu().tolist(),
        "tensors": {
            "key": tensor_metadata("key", key),
            "value": tensor_metadata("value", value),
            "k_cache": tensor_metadata("k_cache", k_cache),
            "v_cache": tensor_metadata("v_cache", v_cache),
            "k_scale": tensor_metadata("k_scale", k_scale),
            "v_scale": tensor_metadata("v_scale", v_scale),
            "slot_mapping": tensor_metadata("slot_mapping", slot_mapping),
        },
        "k_quantization": scalar_metrics(
            k_dequant,
            valid_key,
            valid_kq,
            valid_ks,
        ),
        "v_quantization": scalar_metrics(
            v_dequant,
            valid_value,
            valid_vq,
            valid_vs,
        ),
        "reference_integer_match": {
            "k": bool(torch.equal(valid_kq, expected_kq)),
            "v": bool(torch.equal(valid_vq, expected_vq)),
            "k_scale": bool(torch.allclose(valid_ks.float(), expected_ks, atol=2e-3, rtol=2e-3)),
            "v_scale": bool(torch.allclose(valid_vs.float(), expected_vs, atol=2e-3, rtol=2e-3)),
        },
        "reference_integer_compatibility": {
            # The kernel performs the scale and round arithmetic on device
            # before the FP16 scale is stored.  Values exactly at an INT8
            # half-integer tie can therefore differ from a host FP32
            # reference by one code without changing the bounded dequantized
            # error.  Treat exact equality as diagnostic, not as the only
            # correctness criterion for random stress data.
            "k_max_abs_code_diff": int(
                (valid_kq.to(torch.int16) - expected_kq.to(torch.int16))
                .abs()
                .max()
                .item()
            ),
            "v_max_abs_code_diff": int(
                (valid_vq.to(torch.int16) - expected_vq.to(torch.int16))
                .abs()
                .max()
                .item()
            ),
            "k_code_mismatch_ratio": float(
                (
                    valid_kq.to(torch.int16) != expected_kq.to(torch.int16)
                )
                .float()
                .mean()
                .item()
            ),
            "v_code_mismatch_ratio": float(
                (
                    valid_vq.to(torch.int16) != expected_vq.to(torch.int16)
                )
                .float()
                .mean()
                .item()
            ),
            "passed": bool(
                (
                    (valid_kq.to(torch.int16) - expected_kq.to(torch.int16))
                    .abs()
                    .max()
                    <= 1
                ).item()
                and (
                    (valid_vq.to(torch.int16) - expected_vq.to(torch.int16))
                    .abs()
                    .max()
                    <= 1
                ).item()
            ),
        },
        "invalid_slots_untouched": invalid_untouched,
    }
    result["passed"] = bool(
        result["reference_integer_compatibility"]["passed"]
        and result["k_quantization"]["dequant_error_bound_passed"]
        and result["v_quantization"]["dequant_error_bound_passed"]
        and result["reference_integer_match"]["k_scale"]
        and result["reference_integer_match"]["v_scale"]
        and invalid_untouched
    )

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "key_bf16": key.cpu(),
                "value_bf16": value.cpu(),
                "slot_mapping": slot_mapping.cpu(),
                "k_cache_int8": k_cache.cpu(),
                "v_cache_int8": v_cache.cpu(),
                "k_scale_fp16": k_scale.cpu(),
                "v_scale_fp16": v_scale.cpu(),
            },
            raw_dir / f"N{num_tokens}_{pattern}.pt",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect INT8 KV-cache quantization correctness evidence."
    )
    parser.add_argument("--n-values", default="1,16,128,256,257,512")
    parser.add_argument(
        "--patterns",
        default="contiguous,shuffled,with_invalid",
    )
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--result-dir",
        default="benchmark_results/kv_quantization",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--save-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INT8 KV quantization collection")
    n_values = parse_int_list(args.n_values)
    patterns = parse_str_list(args.patterns)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.head_dim <= 0 or args.num_kv_heads <= 0:
        parser.error("head dimensions and head count must be positive")

    result_dir = Path(args.result_dir)
    raw_dir = result_dir / "raw" if args.save_tensors else None
    result = {
        **collect_benchmark_metadata(torch),
        "configuration": {
            "n_values": n_values,
            "patterns": patterns,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "input_dtype": "torch.bfloat16",
            "cache_dtype": "torch.int8",
            "scale_dtype": "torch.float16",
            "quantization_range": [-127, 127],
            "zero_point": 0,
            "granularity": "per-token-per-KV-head",
        },
        "cases": [],
    }
    for n_index, num_tokens in enumerate(n_values):
        for p_index, pattern in enumerate(patterns):
            case = run_case(
                num_tokens=num_tokens,
                pattern=pattern,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                seed=args.seed + n_index * 100 + p_index,
                raw_dir=raw_dir,
            )
            result["cases"].append(case)

    result["passed"] = all(case["passed"] for case in result["cases"])
    result_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or "kv_quantization_matrix"
    path = result_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
