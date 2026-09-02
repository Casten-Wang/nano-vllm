"""Optional runtime shape and data-flow tracing for formal GPU retests.

The tracer is deliberately metadata-only: it records tensor layout and small
index tensors, never model values or full KV-cache contents.  It is disabled
unless ``NANOVLLM_SHAPE_TRACE=1`` is set before the worker process starts.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive integer trace limit without making tracing fragile."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def tensor_metadata(
    name: str,
    tensor: torch.Tensor | None,
    *,
    include_values: bool = False,
    max_values: int = 64,
) -> dict[str, Any] | None:
    """Return JSON-safe shape/layout metadata for one tensor."""

    if tensor is None:
        return None
    result: dict[str, Any] = {
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
        result["values"] = (
            tensor.detach().to(device="cpu").reshape(-1).tolist()
        )
    return result


def context_metadata(
    context: Any,
    *,
    include_values: bool = True,
    max_values: int = 64,
) -> dict[str, Any]:
    """Serialize the runtime attention context without copying large tensors."""

    tensor_fields = (
        "slot_mapping",
        "context_lens",
        "block_tables",
        "dequant_block_ids",
        "dequant_block_tables",
        "decode_context_lens",
        "decode_block_tables",
        "decode_dequant_block_ids",
        "decode_dequant_block_tables",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "prefill_cu_seqlens_q",
        "prefill_cu_seqlens_k",
        "prefill_block_tables",
        "prefill_dequant_block_ids",
        "prefill_dequant_block_tables",
    )
    scalar_fields = (
        "is_prefill",
        "is_mixed",
        "max_seqlen_q",
        "max_seqlen_k",
        "max_context_len",
        "sliding_window_size",
        "decode_token_count",
        "prefill_token_count",
        "decode_max_context_len",
        "prefill_max_seqlen_q",
        "prefill_max_seqlen_k",
        "decode_state_span",
    )
    result: dict[str, Any] = {
        "scalars": {
            field: _json_safe(getattr(context, field, None))
            for field in scalar_fields
        },
        "tensors": {},
    }
    for field in tensor_fields:
        value = getattr(context, field, None)
        if value is not None:
            result["tensors"][field] = tensor_metadata(
                field,
                value,
                include_values=include_values,
                max_values=max_values,
            )
    return result


class ShapeTrace:
    """Bounded per-process trace used by benchmark result JSON files."""

    def __init__(
        self,
        *,
        max_events: int | None = None,
        max_index_values: int | None = None,
    ):
        self.enabled = os.environ.get("NANOVLLM_SHAPE_TRACE", "0") == "1"
        self.max_events = (
            max_events
            if max_events is not None
            else _env_positive_int("NANOVLLM_SHAPE_TRACE_MAX_EVENTS", 128)
        )
        self.max_index_values = (
            max_index_values
            if max_index_values is not None
            else _env_positive_int("NANOVLLM_SHAPE_TRACE_MAX_INDEX_VALUES", 64)
        )
        self.events: list[dict[str, Any]] = []
        self.dropped_events = 0

    def reset(self) -> None:
        self.events.clear()
        self.dropped_events = 0

    def record(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(_json_safe(event))

    def record_model_step(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: Any,
        model_path: str,
        attention_paths: tuple[str, ...],
        graph_bucket: int | None,
        state_access_path: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.record(
            {
                "event": "model_step_inputs",
                "model_path": model_path,
                "attention_paths": list(attention_paths),
                "graph_bucket": graph_bucket,
                "state_access_path": state_access_path,
                "tensors": {
                    "input_ids": tensor_metadata("input_ids", input_ids),
                    "positions": tensor_metadata("positions", positions),
                },
                "context": context_metadata(
                    context,
                    include_values=True,
                    max_values=self.max_index_values,
                ),
            }
        )

    def record_attention(
        self,
        *,
        layer_id: int | None,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
        context: Any,
    ) -> None:
        if not self.enabled:
            return
        self.record(
            {
                "event": "attention_forward",
                "layer_id": layer_id,
                "tensors": {
                    "q": tensor_metadata("q", q),
                    "k_new": tensor_metadata("k_new", k),
                    "v_new": tensor_metadata("v_new", v),
                    "k_cache": tensor_metadata("k_cache", k_cache),
                    "v_cache": tensor_metadata("v_cache", v_cache),
                    "k_scale": tensor_metadata("k_scale", k_scale),
                    "v_scale": tensor_metadata("v_scale", v_scale),
                },
                "context": context_metadata(
                    context,
                    include_values=True,
                    max_values=self.max_index_values,
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_events": self.max_events,
            "dropped_events": self.dropped_events,
            "events": list(self.events),
        }


_ACTIVE_TRACE: ShapeTrace | None = None


def activate(trace: ShapeTrace | None) -> ShapeTrace | None:
    """Set the process-local trace and return the previous value."""

    global _ACTIVE_TRACE
    previous = _ACTIVE_TRACE
    _ACTIVE_TRACE = trace
    return previous


def restore(previous: ShapeTrace | None) -> None:
    global _ACTIVE_TRACE
    _ACTIVE_TRACE = previous


def active_trace() -> ShapeTrace | None:
    return _ACTIVE_TRACE
