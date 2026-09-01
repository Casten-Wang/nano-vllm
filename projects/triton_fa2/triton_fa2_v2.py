"""Official-structure-aligned Triton FlashAttention-style forward kernel.

V2 deliberately keeps the V1 implementation in ``triton_fa2.py`` untouched.
The kernel follows the structure of Triton's fused-attention tutorial:

* causal attention is split into an off-band stage and an on-band stage;
* only the on-band stage applies the elementwise causal mask;
* the accumulator and online-softmax state stay in registers as far as the
  generated kernel permits;
* tile/warp/stage configurations are autotuned for the current shape.

The implementation targets prefill forward attention with tensors laid out as
``[batch, heads, seq_len, head_dim]``. It supports MHA and GQA without
materializing repeated K/V heads.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl

SUPPORTED_HEAD_DIMS = (64, 128)


V2_AUTOTUNE_CONFIGS = tuple(
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m in (64, 128)
    for block_n in (32, 64, 128)
    for num_warps in (4, 8)
    for num_stages in (2, 3, 4)
)


def _meta_value(
    named_args: dict[str, Any],
    kwargs: dict[str, Any],
    name: str,
) -> Any:
    """Read a meta-parameter from Triton's early-prune callback arguments."""

    if name in kwargs:
        return kwargs[name]
    if name in named_args:
        return named_args[name]
    return None


def _prune_v2_configs(
    configs: list[triton.Config],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[triton.Config]:
    """Apply the shape constraints used by the official causal implementation.

    Triton calls this function before benchmarking configurations. Keeping the
    function pure makes the pruning policy independently unit-testable on a
    machine without CUDA.
    """

    seq_len = _meta_value(named_args, kwargs, "SEQ_LEN")
    head_dim = _meta_value(named_args, kwargs, "HEAD_DIM")
    causal = _meta_value(named_args, kwargs, "CAUSAL")
    if seq_len is None or head_dim is None or causal is None:
        raise ValueError("SEQ_LEN, HEAD_DIM, and CAUSAL are required for V2 pruning")

    valid: list[triton.Config] = []
    for config in configs:
        block_m = int(config.kwargs["BLOCK_M"])
        block_n = int(config.kwargs["BLOCK_N"])
        if block_m > int(seq_len):
            continue
        # BLOCK_N tiles the sequence axis, not the head-dimension axis.
        # Restrict it by SEQ_LEN; tying it to HEAD_DIM incorrectly removes
        # legal candidates such as HEAD_DIM=64, BLOCK_N=128.
        if block_n > int(seq_len):
            continue
        if bool(causal) and block_m < block_n:
            continue
        valid.append(config)

    if not valid:
        raise ValueError(
            "no valid V2 autotune configuration for "
            f"SEQ_LEN={seq_len}, HEAD_DIM={head_dim}, CAUSAL={causal}"
        )
    return valid


def _config_to_dict(config: triton.Config | None) -> dict[str, int] | None:
    if config is None:
        return None
    return {
        "block_m": int(config.kwargs["BLOCK_M"]),
        "block_n": int(config.kwargs["BLOCK_N"]),
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
    }


_LAST_V2_AUTOTUNE_CONFIG: dict[str, int] | None = None


def get_last_v2_autotune_config() -> dict[str, int] | None:
    """Return a copy of the last selected V2 configuration, if available."""

    if _LAST_V2_AUTOTUNE_CONFIG is None:
        return None
    return dict(_LAST_V2_AUTOTUNE_CONFIG)


@triton.jit
def _flash_attention_fwd_inner_v2(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    k_base,
    v_base,
    q_offsets,
    q_valid,
    offs_n,
    offs_d,
    qk_scale_log2,
    Q_STRIDE_N,
    Q_STRIDE_D,
    K_STRIDE_N,
    K_STRIDE_D,
    V_STRIDE_N,
    V_STRIDE_D,
    start_m,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    P_DTYPE_BF16: tl.constexpr,
):
    """Process one official-style attention stage."""

    if STAGE == 1:
        lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    elif STAGE == 3:
        lo = 0
        hi = (N_CTX // BLOCK_N) * BLOCK_N
    else:
        # STAGE=4 is the non-causal tail. It is a no-op when N_CTX is already
        # aligned to BLOCK_N.
        lo = (N_CTX // BLOCK_N) * BLOCK_N
        hi = N_CTX

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_offsets = start_n + offs_n

        if STAGE == 1 or STAGE == 3:
            valid_k = tl.full((BLOCK_N,), True, dtype=tl.int1)
        elif STAGE == 2:
            valid_k = n_offsets < hi
        else:
            valid_k = n_offsets < N_CTX

        k = tl.load(
            k_ptr
            + k_base
            + n_offsets[None, :] * K_STRIDE_N
            + offs_d[:, None] * K_STRIDE_D,
            mask=valid_k[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale_log2

        if STAGE == 2:
            keep = (
                q_valid[:, None]
                & valid_k[None, :]
                & (q_offsets[:, None] >= n_offsets[None, :])
            )
        else:
            keep = q_valid[:, None] & valid_k[None, :]
        qk = tl.where(keep, qk, -float("inf"))

        tile_m = tl.max(qk, axis=1)
        tile_m = tl.where(q_valid, tile_m, m_i)
        m_ij = tl.maximum(m_i, tile_m)
        alpha = tl.where(q_valid, tl.exp2(m_i - m_ij), 0.0)

        # Avoid (-inf)-(-inf) for invalid query rows. For valid rows m_ij is
        # finite after the first legal K/V tile.
        safe_m_ij = tl.where(q_valid, m_ij, 0.0)
        p = tl.where(keep, tl.exp2(qk - safe_m_ij[:, None]), 0.0)
        l_ij = tl.sum(p, axis=1)

        # Keep this rescaling immediately before P@V, then update l_i/m_i at
        # the end of the loop as in Triton's official implementation.
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptr
            + v_base
            + n_offsets[:, None] * V_STRIDE_N
            + offs_d[None, :] * V_STRIDE_D,
            mask=valid_k[:, None],
            other=0.0,
        )
        if P_DTYPE_BF16:
            p = p.to(tl.bfloat16)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc=acc)
        l_i = tl.where(q_valid, l_i * alpha + l_ij, l_i)
        m_i = tl.where(q_valid, m_ij, m_i)

    return acc, l_i, m_i


@triton.jit
def _flash_attention_fwd_v2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    Q_STRIDE_B,
    Q_STRIDE_H,
    Q_STRIDE_N,
    Q_STRIDE_D,
    K_STRIDE_B,
    K_STRIDE_H,
    K_STRIDE_N,
    K_STRIDE_D,
    V_STRIDE_B,
    V_STRIDE_H,
    V_STRIDE_N,
    V_STRIDE_D,
    O_STRIDE_B,
    O_STRIDE_H,
    O_STRIDE_N,
    O_STRIDE_D,
    BATCH_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    P_DTYPE_BF16: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // NUM_Q_HEADS
    head_id = batch_head_id - batch_id * NUM_Q_HEADS
    kv_head_id = head_id // HEADS_PER_KV

    q_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = q_offsets < SEQ_LEN

    q_base = batch_id * Q_STRIDE_B + head_id * Q_STRIDE_H
    k_base = batch_id * K_STRIDE_B + kv_head_id * K_STRIDE_H
    v_base = batch_id * V_STRIDE_B + kv_head_id * V_STRIDE_H

    q = tl.load(
        q_ptr + q_base + q_offsets[:, None] * Q_STRIDE_N + offs_d[None, :] * Q_STRIDE_D,
        mask=q_valid[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    if CAUSAL:
        # Off-band: all keys precede every query row in this block.
        acc, l_i, m_i = _flash_attention_fwd_inner_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            q_offsets,
            q_valid,
            offs_n,
            offs_d,
            SOFTMAX_SCALE_LOG2,
            Q_STRIDE_N,
            Q_STRIDE_D,
            K_STRIDE_N,
            K_STRIDE_D,
            V_STRIDE_N,
            V_STRIDE_D,
            start_m,
            N_CTX=SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            STAGE=1,
            P_DTYPE_BF16=P_DTYPE_BF16,
        )
        # On-band: only this stage needs elementwise causal masking.
        acc, l_i, m_i = _flash_attention_fwd_inner_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            q_offsets,
            q_valid,
            offs_n,
            offs_d,
            SOFTMAX_SCALE_LOG2,
            Q_STRIDE_N,
            Q_STRIDE_D,
            K_STRIDE_N,
            K_STRIDE_D,
            V_STRIDE_N,
            V_STRIDE_D,
            start_m,
            N_CTX=SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            STAGE=2,
            P_DTYPE_BF16=P_DTYPE_BF16,
        )
    else:
        # Non-causal full tiles and a separate tail stage.
        acc, l_i, m_i = _flash_attention_fwd_inner_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            q_offsets,
            q_valid,
            offs_n,
            offs_d,
            SOFTMAX_SCALE_LOG2,
            Q_STRIDE_N,
            Q_STRIDE_D,
            K_STRIDE_N,
            K_STRIDE_D,
            V_STRIDE_N,
            V_STRIDE_D,
            start_m,
            N_CTX=SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            STAGE=3,
            P_DTYPE_BF16=P_DTYPE_BF16,
        )
        acc, l_i, m_i = _flash_attention_fwd_inner_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            k_base,
            v_base,
            q_offsets,
            q_valid,
            offs_n,
            offs_d,
            SOFTMAX_SCALE_LOG2,
            Q_STRIDE_N,
            Q_STRIDE_D,
            K_STRIDE_N,
            K_STRIDE_D,
            V_STRIDE_N,
            V_STRIDE_D,
            start_m,
            N_CTX=SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
            STAGE=4,
            P_DTYPE_BF16=P_DTYPE_BF16,
        )

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    o_base = batch_id * O_STRIDE_B + head_id * O_STRIDE_H
    tl.store(
        o_ptr + o_base + q_offsets[:, None] * O_STRIDE_N + offs_d[None, :] * O_STRIDE_D,
        out,
        mask=q_valid[:, None],
    )


_flash_attention_fwd_v2_autotuned = triton.autotune(
    configs=list(V2_AUTOTUNE_CONFIGS),
    key=[
        "BATCH_SIZE",
        "NUM_Q_HEADS",
        "SEQ_LEN",
        "HEAD_DIM",
        "CAUSAL",
        "HEADS_PER_KV",
        "P_DTYPE_BF16",
    ],
    prune_configs_by={"early_config_prune": _prune_v2_configs},
)(_flash_attention_fwd_v2_kernel)


def _v2_grid(meta: dict[str, Any]) -> tuple[int, int]:
    return (
        triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]),
        meta["BATCH_SIZE"] * meta["NUM_Q_HEADS"],
    )


def _check_v2_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape [batch, heads, seq_len, head_dim]"
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise AssertionError("V2 Triton kernel requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("V2 supports only float16 and bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v last dimensions must be contiguous")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v batch sizes must match")
    if q.shape[2:] != k.shape[2:] or q.shape[2:] != v.shape[2:]:
        raise ValueError("q, k, and v sequence lengths and head dimensions must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same number of heads")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("q heads must be divisible by k/v heads for GQA")
    if q.shape[3] not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}")
    if q.shape[2] <= 0:
        raise ValueError("sequence length must be positive")
    return q.shape[0], q.shape[1], k.shape[1], q.shape[2], q.shape[3]


def _launch_v2_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float | None,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = _check_v2_inputs(q, k, v)
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("block_m must be a positive power of two")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("block_n must be a positive power of two")
    if block_m > seq_len:
        raise ValueError("block_m must not exceed sequence length")
    if block_n > seq_len:
        raise ValueError("block_n must not exceed sequence length")
    if causal and block_m < block_n:
        raise ValueError("causal V2 requires block_m >= block_n")
    if num_warps <= 0 or num_stages <= 0:
        raise ValueError("num_warps and num_stages must be positive")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    scale_log2 = float(softmax_scale) / math.log(2.0)
    out = torch.empty_like(q)
    meta = {
        "BATCH_SIZE": batch_size,
        "NUM_Q_HEADS": num_q_heads,
        "HEADS_PER_KV": num_q_heads // num_kv_heads,
        "SEQ_LEN": seq_len,
        "HEAD_DIM": head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "CAUSAL": bool(causal),
        "P_DTYPE_BF16": q.dtype is torch.bfloat16,
    }
    _flash_attention_fwd_v2_kernel[_v2_grid(meta)](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BATCH_SIZE=batch_size,
        NUM_Q_HEADS=num_q_heads,
        HEADS_PER_KV=num_q_heads // num_kv_heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SOFTMAX_SCALE_LOG2=scale_log2,
        CAUSAL=bool(causal),
        P_DTYPE_BF16=q.dtype is torch.bfloat16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    global _LAST_V2_AUTOTUNE_CONFIG
    _LAST_V2_AUTOTUNE_CONFIG = {
        "block_m": block_m,
        "block_n": block_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return out


def triton_flash_attention_forward_v2_configured(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Launch V2 with an explicit configuration for tests and profiling."""

    return _launch_v2_configured(
        q,
        k,
        v,
        causal=causal,
        softmax_scale=softmax_scale,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def triton_flash_attention_forward_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Launch the autotuned official-structure-aligned V2 kernel."""

    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = _check_v2_inputs(q, k, v)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    scale_log2 = float(softmax_scale) / math.log(2.0)
    out = torch.empty_like(q)
    meta = {
        "BATCH_SIZE": batch_size,
        "NUM_Q_HEADS": num_q_heads,
        "HEADS_PER_KV": num_q_heads // num_kv_heads,
        "SEQ_LEN": seq_len,
        "HEAD_DIM": head_dim,
        "CAUSAL": bool(causal),
        "P_DTYPE_BF16": q.dtype is torch.bfloat16,
    }
    _flash_attention_fwd_v2_autotuned[_v2_grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BATCH_SIZE=batch_size,
        NUM_Q_HEADS=num_q_heads,
        HEADS_PER_KV=num_q_heads // num_kv_heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        SOFTMAX_SCALE_LOG2=scale_log2,
        CAUSAL=bool(causal),
        P_DTYPE_BF16=q.dtype is torch.bfloat16,
    )
    global _LAST_V2_AUTOTUNE_CONFIG
    _LAST_V2_AUTOTUNE_CONFIG = _config_to_dict(
        getattr(_flash_attention_fwd_v2_autotuned, "best_config", None)
    )
    return out
