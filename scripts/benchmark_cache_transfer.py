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

from nanovllm.engine.cache_transfer import RankCacheTransfer
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
    kv_flat = _allocate_bytes(kv_bytes, kv_dtype, multiple=2)
    kv_blocks = kv_flat.reshape(2, 1, 1, 1, 1, -1)
    kv_scales = _allocate_bytes(scale_bytes, torch.float16) if scale_bytes else None
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
    parser.add_argument("--kv-dtype", choices=("float16", "bfloat16", "int8"), default="bfloat16")
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
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }[args.kv_dtype]
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
