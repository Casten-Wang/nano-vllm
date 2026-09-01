"""Reference implementations for the Triton FA2-style project."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn.functional as F

SDPABackend = Literal["default", "flash"]


def _validate_gqa_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q/k/v batch sizes must match")
    if q.shape[2:] != k.shape[2:] or q.shape[2:] != v.shape[2:]:
        raise ValueError("q/k/v sequence lengths and head dimensions must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of heads")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("q heads must be divisible by k/v heads for GQA")
    return q.shape[1] != k.shape[1]


def repeat_kv_for_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat K/V heads to match Q for a readable GQA reference.

    This helper deliberately materializes repeated K/V and is intended for
    correctness checks and a memory-cost baseline, not for the optimized path.
    """
    is_gqa = _validate_gqa_inputs(q, k, v)
    if not is_gqa:
        return k, v
    repeat_factor = q.shape[1] // k.shape[1]
    return (
        k.repeat_interleave(repeat_factor, dim=1),
        v.repeat_interleave(repeat_factor, dim=1),
    )


def torch_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Readable PyTorch reference that explicitly builds the score matrix.

    This is intentionally slow and memory hungry for long sequences.  It exists
    to make correctness tests easy to inspect.
    """
    k, v = repeat_kv_for_gqa(q, k, v)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * softmax_scale
    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.ones(
            (seq_len, seq_len), device=q.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    backend: SDPABackend = "default",
) -> torch.Tensor:
    """PyTorch SDPA reference with an explicit backend selection.

    ``backend="default"`` allows PyTorch to select an implementation.
    ``backend="flash"`` forces the FlashAttention SDPA backend and raises if
    the current shape/device/runtime cannot execute it.
    """
    is_gqa = _validate_gqa_inputs(q, k, v)
    with sdpa_backend_context(backend):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=is_gqa,
        )


def sdpa_backend_context(backend: SDPABackend):
    """Return a context that selects the requested PyTorch SDPA backend."""
    if backend == "default":
        return nullcontext()
    if backend == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    raise ValueError(f"unsupported SDPA backend: {backend}")


def selected_sdpa_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> str:
    """Return the concrete backend selected by default PyTorch SDPA.

    PyTorch exposes this selector as a private diagnostic API.  Keeping it in
    the benchmark metadata prevents the public ``SDPA`` API name from being
    mistaken for a single fixed implementation.
    """
    is_gqa = _validate_gqa_inputs(q, k, v)
    choice = torch._fused_sdp_choice(
        q,
        k,
        v,
        None,
        0.0,
        causal,
        scale=softmax_scale,
        enable_gqa=is_gqa,
    )
    from torch.nn.attention import SDPBackend

    return SDPBackend(choice).name


def error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rms": torch.sqrt(torch.mean(diff * diff)).item(),
    }
