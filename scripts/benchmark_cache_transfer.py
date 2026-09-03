"""Benchmark the synchronous rank-local TCP cache-transfer baseline."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from pathlib import Path
from statistics import median
from threading import Thread
from time import perf_counter_ns

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.engine.cache_transfer import RankCacheTransfer, import_rank_cache
from nanovllm.engine.cache_transfer_wire import (
    RankCacheReceiver,
    send_rank_cache_to_endpoint,
)


MIB = 1024**2


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _allocate_bytes(
    size: int,
    dtype: torch.dtype,
    *,
    multiple: int = 1,
) -> torch.Tensor:
    if size < 0:
        raise ValueError("tensor byte size must be non-negative")
    if size == 0:
        return torch.empty(0, dtype=dtype)
    if multiple <= 0:
        raise ValueError("tensor element multiple must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    elements = (size + element_size - 1) // element_size
    elements = ((elements + multiple - 1) // multiple) * multiple
    return torch.zeros(elements, dtype=dtype)


def make_payload(
    *,
    kv_bytes: int,
    scale_bytes: int,
    recurrent_bytes: int,
    convolution_bytes: int,
    kv_dtype: torch.dtype,
) -> RankCacheTransfer:
    if kv_bytes <= 0:
        raise ValueError("kv_bytes must be positive")
    if kv_dtype not in (torch.float16, torch.bfloat16, torch.int8):
        raise ValueError("unsupported benchmark KV dtype")
    if kv_dtype is torch.int8 and scale_bytes <= 0:
        raise ValueError("INT8 KV benchmark requires scale bytes")
    if kv_dtype is not torch.int8 and scale_bytes:
        raise ValueError("floating-point KV benchmark cannot include scales")
    if scale_bytes:
        scale_flat = _allocate_bytes(scale_bytes, torch.float16, multiple=2)
        kv_flat = _allocate_bytes(
            kv_bytes,
            kv_dtype,
            multiple=scale_flat.numel(),
        )
        head_dim = kv_flat.numel() // scale_flat.numel()
        kv_blocks = kv_flat.reshape(2, 1, 1, 1, -1, head_dim)
        kv_scales = scale_flat.reshape(*kv_blocks.shape[:-1])
    else:
        kv_flat = _allocate_bytes(kv_bytes, kv_dtype, multiple=2)
        kv_blocks = kv_flat.reshape(2, 1, 1, 1, 1, -1)
        kv_scales = None
    recurrent = (
        (_allocate_bytes(recurrent_bytes, torch.float32),)
        if recurrent_bytes
        else ()
    )
    convolution = (
        (_allocate_bytes(convolution_bytes, torch.bfloat16),)
        if convolution_bytes
        else ()
    )
    return RankCacheTransfer(
        format_version=1,
        transfer_id="benchmark/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=256,
        cached_tokens=256,
        kv_blocks=kv_blocks,
        kv_scales=kv_scales,
        recurrent_states=recurrent,
        convolution_states=convolution,
    )


def payload_bytes(payload: RankCacheTransfer) -> dict[str, int]:
    components = {
        "kv": _tensor_bytes(payload.kv_blocks),
        "kv_scales": _tensor_bytes(payload.kv_scales),
        "recurrent": sum(_tensor_bytes(tensor) for tensor in payload.recurrent_states),
        "convolution": sum(
            _tensor_bytes(tensor) for tensor in payload.convolution_states
        ),
    }
    components["total"] = sum(components.values())
    return components


def payload_from_memory_preflight(
    path: Path,
    *,
    tp_size: int,
    kv_dtype: str,
    state_dtype: str,
) -> tuple[RankCacheTransfer, dict]:
    if kv_dtype not in ("auto", "int8"):
        raise ValueError("preflight KV dtype must be auto or int8")
    if state_dtype not in ("float32", "model"):
        raise ValueError("preflight state dtype must be float32 or model")
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        components = report["results"][f"tp{tp_size}"][
            "pd_transfer_components_per_sequence_by_dtype"
        ][kv_dtype][state_dtype]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "memory preflight has no matching PD transfer component profile"
        ) from exc
    required = ("kv", "kv_scales", "recurrent", "convolution", "total")
    if any(
        isinstance(components.get(name), bool)
        or not isinstance(components.get(name), int)
        or components[name] < 0
        for name in required
    ):
        raise ValueError("memory preflight PD transfer components are invalid")
    if (
        components["kv"] <= 0
        or sum(components[name] for name in required[:-1])
        != components["total"]
    ):
        raise ValueError("memory preflight PD transfer component total is invalid")
    payload = make_payload(
        kv_bytes=components["kv"],
        scale_bytes=components["kv_scales"],
        recurrent_bytes=components["recurrent"],
        convolution_bytes=components["convolution"],
        kv_dtype=torch.int8 if kv_dtype == "int8" else torch.bfloat16,
    )
    if payload_bytes(payload) != components:
        raise ValueError(
            "memory preflight component bytes do not map exactly to tensor storage"
        )
    return payload, {
        "memory_preflight": str(path),
        "tp_size": tp_size,
        "kv_dtype": kv_dtype,
        "state_dtype": state_dtype,
    }


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("latency samples must not be empty")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def transfer_once(payload: RankCacheTransfer, timeout_s: float) -> tuple[float, int]:
    received = []
    failures = []
    with RankCacheReceiver("127.0.0.1", 0, timeout_s=timeout_s) as receiver:
        def receive():
            try:
                received.append(receiver.receive(timeout_s=timeout_s))
            except BaseException as exc:
                failures.append(exc)

        thread = Thread(target=receive)
        thread.start()
        start = perf_counter_ns()
        wire_bytes = send_rank_cache_to_endpoint(
            *receiver.address,
            payload,
            timeout_s=timeout_s,
        )
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise TimeoutError("cache transfer benchmark receiver did not finish")
    if failures:
        raise RuntimeError("cache transfer benchmark receiver failed") from failures[0]
    if len(received) != 1:
        raise RuntimeError("cache transfer benchmark did not receive one payload")
    if not torch.equal(received[0].kv_blocks, payload.kv_blocks):
        raise RuntimeError("cache transfer benchmark KV round-trip mismatch")
    return elapsed_ms, wire_bytes


def benchmark_cuda_install(
    payload: RankCacheTransfer,
    *,
    warmup: int,
    repeats: int,
) -> dict:
    """Compare full-payload staging with bounded direct cache installation."""

    if not torch.cuda.is_available():
        return {
            "enabled": False,
            "valid": True,
            "reason": "CUDA is unavailable",
        }
    device = torch.device("cuda")
    destination_kv = torch.empty_like(payload.kv_blocks, device=device)
    destination_scales = (
        torch.empty_like(payload.kv_scales, device=device)
        if payload.kv_scales is not None
        else None
    )
    destination_recurrent = tuple(
        torch.empty_like(tensor, device=device)
        for tensor in payload.recurrent_states
    )
    destination_convolution = tuple(
        torch.empty_like(tensor, device=device)
        for tensor in payload.convolution_states
    )
    index = torch.tensor([0], dtype=torch.int64, device=device)

    def reference() -> None:
        destination_kv.index_copy_(
            2,
            index,
            payload.kv_blocks.to(device),
        )
        if destination_scales is not None:
            assert payload.kv_scales is not None
            destination_scales.index_copy_(
                2,
                index,
                payload.kv_scales.to(device),
            )
        for source, destination in zip(
            (*payload.recurrent_states, *payload.convolution_states),
            (*destination_recurrent, *destination_convolution),
        ):
            destination.copy_(source.to(device))

    def candidate() -> None:
        import_rank_cache(
            payload,
            destination_kv,
            destination_scales,
            [0],
            transfer_id=payload.transfer_id,
            tensor_parallel_rank=payload.tensor_parallel_rank,
            tensor_parallel_size=payload.tensor_parallel_size,
            block_size=payload.block_size,
            recurrent_states=destination_recurrent,
            convolution_states=destination_convolution,
        )

    def measure(operation) -> dict:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        latency = []
        peaks = []
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            latency.append(start.elapsed_time(end))
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
        return {
            "latency_ms_samples": latency,
            "latency_ms_p50": median(latency),
            "peak_extra_device_bytes_samples": peaks,
            "peak_extra_device_bytes_max": max(peaks),
        }

    reference_result = measure(reference)
    candidate_result = measure(candidate)
    torch.testing.assert_close(destination_kv.cpu(), payload.kv_blocks)
    if destination_scales is not None:
        assert payload.kv_scales is not None
        torch.testing.assert_close(destination_scales.cpu(), payload.kv_scales)
    return {
        "enabled": True,
        "valid": True,
        "measured_on_cuda": True,
        "avoids_full_payload_device_conversion": True,
        "reference_full_payload_staging": reference_result,
        "candidate_direct_block_install": candidate_result,
        "peak_device_bytes_reduction": (
            reference_result["peak_extra_device_bytes_max"]
            - candidate_result["peak_extra_device_bytes_max"]
        ),
        "latency_ratio_vs_reference": (
            candidate_result["latency_ms_p50"]
            / reference_result["latency_ms_p50"]
        ),
    }


def run_benchmark(
    payload: RankCacheTransfer,
    *,
    warmup: int,
    repeats: int,
    timeout_s: float,
) -> dict:
    if warmup < 0 or repeats <= 0 or timeout_s <= 0:
        raise ValueError("warmup, repeats, and timeout must be valid")
    for _ in range(warmup):
        transfer_once(payload, timeout_s)
    samples = []
    wire_bytes = None
    peak_before = _peak_rss_bytes()
    for _ in range(repeats):
        elapsed_ms, current_wire_bytes = transfer_once(payload, timeout_s)
        samples.append(elapsed_ms)
        if wire_bytes is None:
            wire_bytes = current_wire_bytes
        elif wire_bytes != current_wire_bytes:
            raise RuntimeError("cache transfer wire size changed between repeats")
    components = payload_bytes(payload)
    median_ms = median(samples)
    return {
        "schema_version": 1,
        "scope": "single-rank synchronous TCP loopback correctness baseline",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "workload": {
            "warmup": warmup,
            "repeats": repeats,
            "timeout_s": timeout_s,
            "components_bytes": components,
            "payload_frame_bytes_sent": wire_bytes,
            "receiver_ack_bytes": 1,
            "framing_overhead_bytes": wire_bytes - components["total"],
        },
        "results": {
            "latency_ms_samples": samples,
            "latency_ms_p50": median_ms,
            "latency_ms_p95": _percentile(samples, 0.95),
            "effective_payload_gib_s_p50": (
                components["total"] / 1024**3 / (median_ms / 1000)
            ),
            "process_peak_rss_before_bytes": peak_before,
            "process_peak_rss_after_bytes": _peak_rss_bytes(),
        },
        "cuda_install": benchmark_cuda_install(
            payload,
            warmup=warmup,
            repeats=repeats,
        ),
        "limitations": [
            "loopback TCP is not cross-node network evidence",
            "source tensors are already on CPU, so GPU-to-host staging is excluded",
            "process peak RSS is a lifetime high-water mark, not per-transfer allocation",
            "the benchmark does not measure model execution or end-to-end TTFT",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-mib", type=float, default=8.0)
    parser.add_argument("--scale-mib", type=float, default=0.0)
    parser.add_argument("--recurrent-mib", type=float, default=4.0)
    parser.add_argument("--convolution-mib", type=float, default=1.0)
    parser.add_argument(
        "--kv-dtype",
        choices=("auto", "float16", "bfloat16", "int8"),
        default="bfloat16",
    )
    parser.add_argument("--memory-preflight", type=Path)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument(
        "--state-dtype",
        choices=("float32", "model"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/cache_transfer.json"),
    )
    args = parser.parse_args()
    dtype = {
        "auto": torch.bfloat16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }[args.kv_dtype]
    profile = None
    if args.memory_preflight is not None:
        if args.tp_size is None:
            parser.error("--tp-size is required with --memory-preflight")
        if args.kv_dtype not in ("auto", "int8"):
            parser.error(
                "preflight profiles use --kv-dtype auto or int8"
            )
        payload, profile = payload_from_memory_preflight(
            args.memory_preflight,
            tp_size=args.tp_size,
            kv_dtype=args.kv_dtype,
            state_dtype=args.state_dtype,
        )
    else:
        payload = make_payload(
            kv_bytes=round(args.kv_mib * MIB),
            scale_bytes=round(args.scale_mib * MIB),
            recurrent_bytes=round(args.recurrent_mib * MIB),
            convolution_bytes=round(args.convolution_mib * MIB),
            kv_dtype=dtype,
        )
    result = run_benchmark(
        payload,
        warmup=args.warmup,
        repeats=args.repeats,
        timeout_s=args.timeout_s,
    )
    result["profile"] = profile or {"source": "manual byte sizes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
