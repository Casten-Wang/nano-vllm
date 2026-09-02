"""Pure execution-policy helpers and bounded runtime path statistics.

The helpers in this module deliberately do not import torch, Triton, or model
code.  They are used both by the runtime and by CPU-only tests to keep path
selection rules in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable


def cuda_graph_buckets(max_batch_size: int) -> tuple[int, ...]:
    """Return the graph batch buckets supported by ModelRunner.

    The regular buckets preserve the existing policy.  A final partial bucket
    is added when ``max_batch_size`` is not itself one of the regular bucket
    sizes, so every valid request batch up to the configured maximum has a
    corresponding graph.  Keeping this helper pure prevents ModelRunner and
    integration tests from maintaining subtly different bucket policies.
    """

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    buckets = {size for size in (1, 2, 4, 8) if size <= max_batch_size}
    buckets.update(range(16, max_batch_size + 1, 16))
    buckets.add(max_batch_size)
    return tuple(sorted(buckets))


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def select_int8_decode_attention_path(
    *,
    kv_dequant_backend: str,
    max_context_len: int,
    partition_threshold: int,
    sliding_window_size: int | None,
) -> str:
    """Return the attention backend for one INT8 decode step."""

    _validate_nonnegative("max_context_len", max_context_len)
    if partition_threshold <= 0:
        raise ValueError("partition_threshold must be positive")
    if sliding_window_size is not None and sliding_window_size <= 0:
        raise ValueError("sliding_window_size must be positive when provided")
    if kv_dequant_backend not in {"fused", "torch", "triton"}:
        raise ValueError(f"unsupported KV dequant backend: {kv_dequant_backend}")

    if kv_dequant_backend != "fused":
        return "int8_dequant_flash"
    if (
        sliding_window_size is None
        and max_context_len >= partition_threshold
    ):
        return "int8_partitioned_decode"
    return "int8_fused_decode"


def select_attention_paths(
    *,
    step_kind: str,
    kv_cache_dtype: str,
    kv_dequant_backend: str,
    max_context_len: int,
    partition_threshold: int,
    sliding_window_size: int | None,
) -> tuple[str, ...]:
    """Return the logical attention subpaths used by one engine step."""

    if step_kind not in {"prefill", "decode", "mixed"}:
        raise ValueError(f"unsupported step kind: {step_kind}")
    if kv_cache_dtype not in {"auto", "int8"}:
        raise ValueError(f"unsupported KV cache dtype: {kv_cache_dtype}")

    if kv_cache_dtype == "auto":
        prefill_path = "float_flash_prefill"
        decode_path = "float_flash_decode"
    else:
        prefill_path = "int8_prefill"
        decode_path = select_int8_decode_attention_path(
            kv_dequant_backend=kv_dequant_backend,
            max_context_len=max_context_len,
            partition_threshold=partition_threshold,
            sliding_window_size=sliding_window_size,
        )

    if step_kind == "prefill":
        return (prefill_path,)
    if step_kind == "decode":
        return (decode_path,)
    # Preserve the engine's decode-first ordering.  The resulting tuple is
    # normalized by ExecutionStats before it is used as a signature.
    return (decode_path, prefill_path)


def select_model_path(step_kind: str, *, use_cuda_graph: bool) -> str:
    """Return the model-level path for one scheduler step."""

    if step_kind not in {"prefill", "decode", "mixed"}:
        raise ValueError(f"unsupported step kind: {step_kind}")
    if use_cuda_graph and step_kind != "decode":
        raise ValueError("CUDA Graph is supported only for pure decode")
    if use_cuda_graph:
        return "decode_cuda_graph"
    return f"{step_kind}_eager"


def supports_cudagraph_policy(
    *,
    enforce_eager: bool,
    sliding_window_size: int | None,
    is_hybrid: bool,
    qwen35_moe_decode_backend: str,
    kv_cache_dtype: str,
    kv_dequant_backend: str,
    weight_quant_backend: str = "auto",
) -> bool:
    """Return whether every configured decode component is graph-safe."""

    if enforce_eager or sliding_window_size is not None:
        return False
    if is_hybrid and qwen35_moe_decode_backend != "batched":
        return False
    # GPTQ expert dispatch currently performs a device-to-host synchronization
    # and a data-dependent Python loop, so neither executor is graph-safe yet.
    if weight_quant_backend in {"reference", "triton"}:
        return False
    if kv_cache_dtype == "int8" and kv_dequant_backend != "fused":
        return False
    return True


def partition_count(
    *,
    max_context_len: int,
    partition_size: int,
    sliding_window_size: int | None = None,
) -> int:
    """Return the number of visible-token partitions for a decode step."""

    _validate_nonnegative("max_context_len", max_context_len)
    if partition_size <= 0:
        raise ValueError("partition_size must be positive")
    if sliding_window_size is not None and sliding_window_size <= 0:
        raise ValueError("sliding_window_size must be positive when provided")
    visible_tokens = max_context_len
    if sliding_window_size is not None:
        visible_tokens = min(visible_tokens, sliding_window_size)
    return max(ceil(visible_tokens / partition_size), 1)


@dataclass
class ExecutionStats:
    """Bounded aggregate statistics for actual runtime execution paths."""

    max_signatures: int = 4096
    model_path_counts: dict[str, int] = field(default_factory=dict)
    attention_path_counts: dict[str, int] = field(default_factory=dict)
    state_access_path_counts: dict[str, int] = field(default_factory=dict)
    _signature_counts: dict[tuple, int] = field(default_factory=dict)
    _dropped_signature_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_signatures <= 0:
            raise ValueError("max_signatures must be positive")

    @staticmethod
    def _normalize_paths(attention_paths: Iterable[str]) -> tuple[str, ...]:
        paths = tuple(dict.fromkeys(attention_paths))
        if not paths:
            raise ValueError("at least one attention path is required")
        if any(not isinstance(path, str) or not path for path in paths):
            raise ValueError("attention paths must be non-empty strings")
        return tuple(sorted(paths))

    def reset(self) -> None:
        self.model_path_counts.clear()
        self.attention_path_counts.clear()
        self.state_access_path_counts.clear()
        self._signature_counts.clear()
        self._dropped_signature_steps = 0

    def record(
        self,
        *,
        model_path: str,
        attention_paths: Iterable[str],
        actual_batch_size: int,
        actual_input_rows: int,
        graph_bucket: int | None,
        max_context_len: int,
        partition_threshold: int,
        sliding_window_size: int | None,
        state_access_path: str | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be non-empty")
        if actual_batch_size < 0:
            raise ValueError("actual_batch_size must be non-negative")
        if actual_input_rows < 0:
            raise ValueError("actual_input_rows must be non-negative")
        _validate_nonnegative("max_context_len", max_context_len)
        if partition_threshold <= 0:
            raise ValueError("partition_threshold must be positive")
        paths = self._normalize_paths(attention_paths)
        if state_access_path is not None and not state_access_path:
            raise ValueError("state_access_path must be non-empty when provided")
        if graph_bucket is not None and graph_bucket < actual_batch_size:
            raise ValueError("graph_bucket must cover actual_batch_size")

        self.model_path_counts[model_path] = (
            self.model_path_counts.get(model_path, 0) + 1
        )
        for path in paths:
            self.attention_path_counts[path] = (
                self.attention_path_counts.get(path, 0) + 1
            )
        if state_access_path is not None:
            self.state_access_path_counts[state_access_path] = (
                self.state_access_path_counts.get(state_access_path, 0) + 1
            )

        signature = (
            model_path,
            paths,
            actual_batch_size,
            actual_input_rows,
            graph_bucket,
            max_context_len,
            partition_threshold,
            sliding_window_size,
            state_access_path,
        )
        if signature in self._signature_counts:
            self._signature_counts[signature] += 1
        elif len(self._signature_counts) < self.max_signatures:
            self._signature_counts[signature] = 1
        else:
            self._dropped_signature_steps += 1

    def to_dict(self) -> dict:
        signatures = []
        for signature, count in self._signature_counts.items():
            (
                model_path,
                attention_paths,
                actual_batch_size,
                actual_input_rows,
                graph_bucket,
                max_context_len,
                partition_threshold,
                sliding_window_size,
                state_access_path,
            ) = signature
            signatures.append(
                {
                    "model_path": model_path,
                    "attention_paths": list(attention_paths),
                    "actual_batch_size": actual_batch_size,
                    "actual_input_rows": actual_input_rows,
                    "graph_bucket": graph_bucket,
                    "max_context_len": max_context_len,
                    "partition_threshold": partition_threshold,
                    "sliding_window_size": sliding_window_size,
                    "state_access_path": state_access_path,
                    "count": count,
                }
            )
        return {
            "model_path_counts": dict(self.model_path_counts),
            "attention_path_counts": dict(self.attention_path_counts),
            "state_access_path_counts": dict(self.state_access_path_counts),
            "execution_signatures": signatures,
            "max_execution_signatures": self.max_signatures,
            "dropped_execution_signature_steps": self._dropped_signature_steps,
        }
