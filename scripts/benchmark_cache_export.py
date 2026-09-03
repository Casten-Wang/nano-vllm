"""Measure GPU-to-host cache export cost for one Qwen3.5 TP rank."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.engine.cache_transfer import RankCacheTransfer, export_rank_cache


BLOCK_SIZE = 256
FULL_ATTENTION_LAYERS = 10
LINEAR_ATTENTION_LAYERS = 30


def _profile(path: Path, tp_size: int, kv_dtype: str, state_dtype: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        result = report["results"][f"tp{tp_size}"]
        components = result["pd_transfer_components_per_sequence_by_dtype"][
            kv_dtype
        ][state_dtype]
        allocated_tokens = result["pd_transfer_allocated_tokens"]
        cached_tokens = result["pd_transfer_context_tokens"]
    except (KeyError, TypeError) as exc:
        raise ValueError("memory preflight has no matching export profile") from exc
    if (
        not isinstance(allocated_tokens, int)
        or allocated_tokens <= 0
        or allocated_tokens % BLOCK_SIZE
        or not isinstance(cached_tokens, int)
        or not 0 < cached_tokens <= allocated_tokens
    ):
        raise ValueError("memory preflight transfer token counts are invalid")
    if any(
        isinstance(components.get(name), bool)
        or not isinstance(components.get(name), int)
        or components[name] < 0
        for name in ("kv", "kv_scales", "recurrent", "convolution", "total")
    ):
        raise ValueError("memory preflight transfer components are invalid")
    if sum(components[name] for name in ("kv", "kv_scales", "recurrent", "convolution")) != components["total"]:
        raise ValueError("memory preflight transfer component total is invalid")
    return {
        "components": components,
        "allocated_tokens": allocated_tokens,
        "cached_tokens": cached_tokens,
        "tp_size": tp_size,
    }


def _flat_layers(
    total_bytes: int,
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if total_bytes == 0:
        return ()
    divisor = LINEAR_ATTENTION_LAYERS * torch.empty((), dtype=dtype).element_size()
    if total_bytes % divisor:
        raise ValueError("state bytes do not divide evenly across linear layers")
    elements = total_bytes // divisor
    return tuple(
        torch.zeros(elements, dtype=dtype, device=device)
        for _ in range(LINEAR_ATTENTION_LAYERS)
    )


def make_source(
    profile: dict,
    *,
    kv_dtype: str,
    state_dtype: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    components = profile["components"]
    allocated_tokens = profile["allocated_tokens"]
    blocks = allocated_tokens // BLOCK_SIZE
    dtype = torch.int8 if kv_dtype == "int8" else torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()
    divisor = 2 * FULL_ATTENTION_LAYERS * allocated_tokens * element_size
    if components["kv"] % divisor:
        raise ValueError("KV bytes do not map to the Qwen3.5 cache layout")
    head_dim = components["kv"] // divisor
    kv_cache = torch.zeros(
        2,
        FULL_ATTENTION_LAYERS,
        blocks,
        BLOCK_SIZE,
        1,
        head_dim,
        dtype=dtype,
        device=device,
    )
    for block_id in range(blocks):
        kv_cache[:, :, block_id].fill_(block_id % 97)
    kv_scale = None
    if kv_dtype == "int8":
        kv_scale = torch.zeros(
            2,
            FULL_ATTENTION_LAYERS,
            blocks,
            BLOCK_SIZE,
            1,
            dtype=torch.float16,
            device=device,
        )
        if kv_scale.numel() * kv_scale.element_size() != components["kv_scales"]:
            raise ValueError("KV scale bytes do not map to the Qwen3.5 cache layout")
        for block_id in range(blocks):
            kv_scale[:, :, block_id].fill_(block_id + 1)
    elif components["kv_scales"]:
        raise ValueError("floating-point KV profile unexpectedly contains scales")
    recurrent_dtype = torch.float32 if state_dtype == "float32" else torch.bfloat16
    recurrent = _flat_layers(
        components["recurrent"], recurrent_dtype, device=device
    )
    convolution = _flat_layers(
        components["convolution"], torch.bfloat16, device=device
    )
    for layer, tensor in enumerate(recurrent):
        tensor.fill_(layer + 1)
    for layer, tensor in enumerate(convolution):
        tensor.fill_(layer + 2)
    return kv_cache, kv_scale, recurrent, convolution


def _host_payload(payload: RankCacheTransfer) -> RankCacheTransfer:
    def copy(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu").contiguous()

    return replace(
        payload,
        kv_blocks=copy(payload.kv_blocks),
        kv_scales=copy(payload.kv_scales) if payload.kv_scales is not None else None,
        recurrent_states=tuple(copy(tensor) for tensor in payload.recurrent_states),
        convolution_states=tuple(copy(tensor) for tensor in payload.convolution_states),
    )


def _payload_host_layout(payload: RankCacheTransfer) -> dict[str, int | bool]:
    tensors = (
        payload.kv_blocks,
        *((payload.kv_scales,) if payload.kv_scales is not None else ()),
        *payload.recurrent_states,
        *payload.convolution_states,
    )
    return {
        "tensor_count": len(tensors),
        "storage_count": len(
            {tensor.untyped_storage().data_ptr() for tensor in tensors}
        ),
        "all_cpu": all(tensor.device.type == "cpu" for tensor in tensors),
        "all_pinned": all(tensor.is_pinned() for tensor in tensors),
    }


def _export(
    source,
    profile: dict,
    *,
    direct_host: bool,
) -> RankCacheTransfer:
    kv_cache, kv_scale, recurrent, convolution = source
    payload = export_rank_cache(
        kv_cache,
        kv_scale,
        list(reversed(range(kv_cache.shape[2]))),
        transfer_id="benchmark/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=profile["tp_size"],
        block_size=BLOCK_SIZE,
        cached_tokens=profile["cached_tokens"],
        recurrent_states=recurrent,
        convolution_states=convolution,
        to_host=direct_host,
    )
    return payload if direct_host else _host_payload(payload)


def _measure(source, profile: dict, *, direct_host: bool, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        _export(source, profile, direct_host=direct_host)
    torch.cuda.synchronize()
    samples = []
    peaks = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter_ns()
        payload = _export(source, profile, direct_host=direct_host)
        torch.cuda.synchronize()
        samples.append((perf_counter_ns() - start) / 1e6)
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
        del payload
    return {
        "latency_ms_samples": samples,
        "latency_ms_p50": median(samples),
        "peak_extra_device_bytes_samples": peaks,
        "peak_extra_device_bytes_max": max(peaks),
    }


def _assert_payload_equal(
    reference: RankCacheTransfer,
    candidate: RankCacheTransfer,
) -> None:
    if not torch.equal(reference.kv_blocks, candidate.kv_blocks):
        raise RuntimeError("direct-host KV export does not match reference")
    if (reference.kv_scales is None) != (candidate.kv_scales is None):
        raise RuntimeError("direct-host KV scale presence does not match reference")
    if reference.kv_scales is not None and not torch.equal(
        reference.kv_scales, candidate.kv_scales
    ):
        raise RuntimeError("direct-host KV scales do not match reference")
    for name, expected, actual in (
        ("recurrent", reference.recurrent_states, candidate.recurrent_states),
        ("convolution", reference.convolution_states, candidate.convolution_states),
    ):
        if len(expected) != len(actual) or any(
            not torch.equal(left, right) for left, right in zip(expected, actual)
        ):
            raise RuntimeError(f"direct-host {name} states do not match reference")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-preflight", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--kv-dtype", choices=("auto", "int8"), required=True)
    parser.add_argument("--state-dtype", choices=("float32", "model"), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cache export benchmark requires CUDA")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    profile = _profile(
        args.memory_preflight,
        args.tp_size,
        args.kv_dtype,
        args.state_dtype,
    )
    source = make_source(
        profile,
        kv_dtype=args.kv_dtype,
        state_dtype=args.state_dtype,
        device=torch.device("cuda"),
    )
    reference_payload = _export(source, profile, direct_host=False)
    candidate_payload = _export(source, profile, direct_host=True)
    _assert_payload_equal(reference_payload, candidate_payload)
    reference_layout = _payload_host_layout(reference_payload)
    candidate_layout = _payload_host_layout(candidate_payload)
    del reference_payload, candidate_payload
    reference = _measure(
        source, profile, direct_host=False, warmup=args.warmup, repeats=args.repeats
    )
    candidate = _measure(
        source, profile, direct_host=True, warmup=args.warmup, repeats=args.repeats
    )
    reference["host_layout"] = reference_layout
    candidate["host_layout"] = candidate_layout
    result = {
        "schema_version": 1,
        "scope": "single-rank Qwen3.5 GPU-to-host cache export",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        },
        "profile": {
            "tp_size": args.tp_size,
            "kv_dtype": args.kv_dtype,
            "state_dtype": args.state_dtype,
            **profile,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "reference_gpu_gather_then_host_copy": reference,
        "candidate_direct_host_staging": candidate,
        "correctness": {"candidate_matches_reference": True},
        "limitations": [
            "synthetic tensors measure export and D2H staging, not model execution",
            "the benchmark excludes socket transfer and receiver installation",
            "results apply only to the recorded GPU, software stack, and payload",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
