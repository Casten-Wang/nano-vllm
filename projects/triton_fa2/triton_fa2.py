"""Triton FlashAttention-2 style forward attention.

This module intentionally targets the prefill path rather than decode. Q is
shaped [batch, q_heads, seq_len, head_dim], while K/V may use the same head
count (MHA) or fewer heads (GQA). The kernel avoids materializing both the full
seq_len x seq_len attention matrix and repeated GQA K/V tensors.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

SUPPORTED_HEAD_DIMS = (64, 128)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@triton.jit
def _flash_attention_fwd_kernel(
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
    NUM_Q_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
    CAUSAL: tl.constexpr,
    CAUSAL_EARLY_STOP: tl.constexpr,
    P_DTYPE_BF16: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
):
    q_block_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // NUM_Q_HEADS
    head_id = batch_head_id - batch_id * NUM_Q_HEADS
    kv_head_id = head_id // HEADS_PER_KV

    q_offsets = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_valid = q_offsets < SEQ_LEN

    q = tl.load(
        q_ptr
        + batch_id * Q_STRIDE_B
        + head_id * Q_STRIDE_H
        + q_offsets[:, None] * Q_STRIDE_N
        + d_offsets[None, :] * Q_STRIDE_D,
        mask=q_valid[:, None],
        other=0.0,
    )

    # One program owns BLOCK_M query rows.  m_i/l_i/acc are the online
    # softmax state for these rows after scanning all previous K/V tiles.
    m_i = tl.full((BLOCK_M,), -3.4028234663852886e38, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # In a causal kernel, the largest query row owned by this program is the
    # only row that can require keys.  The old implementation scanned all
    # SEQ_LEN K/V tiles and masked future positions afterwards.  Using a
    # runtime loop bound avoids loading and computing tiles that are entirely
    # in the future of this query block.
    if CAUSAL and CAUSAL_EARLY_STOP:
        kv_end = tl.minimum((q_block_id + 1) * BLOCK_M, SEQ_LEN)
    else:
        kv_end = SEQ_LEN

    for kv_start in tl.range(0, kv_end, BLOCK_N, num_stages=LOOP_NUM_STAGES):
        n_offsets = kv_start + kv_offsets
        valid_k = n_offsets < kv_end

        k = tl.load(
            k_ptr
            + batch_id * K_STRIDE_B
            + kv_head_id * K_STRIDE_H
            + n_offsets[None, :] * K_STRIDE_N
            + d_offsets[:, None] * K_STRIDE_D,
            mask=valid_k[None, :],
            other=0.0,
        )
        scores = tl.dot(q, k) * SOFTMAX_SCALE_LOG2

        valid_mask = valid_k[None, :] & q_valid[:, None]
        if CAUSAL:
            valid_mask = valid_mask & (n_offsets[None, :] <= q_offsets[:, None])
        scores = tl.where(valid_mask, scores, -float("inf"))

        # Online softmax:
        #   new_m rescales the old accumulator before this tile contributes.
        #   This is the key FlashAttention idea that avoids writing the full
        #   attention score/probability matrix to HBM.
        tile_m = tl.max(scores, axis=1)
        tile_m = tl.where(q_valid, tile_m, m_i)
        new_m = tl.maximum(m_i, tile_m)
        alpha = tl.exp2(m_i - new_m)
        p = tl.exp2(scores - new_m[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr
            + batch_id * V_STRIDE_B
            + kv_head_id * V_STRIDE_H
            + n_offsets[:, None] * V_STRIDE_N
            + d_offsets[None, :] * V_STRIDE_D,
            mask=valid_k[:, None],
            other=0.0,
        )
        if P_DTYPE_BF16:
            p_dot = p.to(tl.bfloat16)
        else:
            p_dot = p.to(tl.float16)
        acc = tl.dot(p_dot, v, acc=acc * alpha[:, None])
        m_i = new_m

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / safe_l[:, None]
    tl.store(
        o_ptr
        + batch_id * O_STRIDE_B
        + head_id * O_STRIDE_H
        + q_offsets[:, None] * O_STRIDE_N
        + d_offsets[None, :] * O_STRIDE_D,
        out,
        mask=q_valid[:, None],
    )


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    assert q.ndim == 4, "q must have shape [batch, heads, seq_len, head_dim]"
    assert (
        k.ndim == 4 and v.ndim == 4
    ), "k and v must have shape [batch, heads, seq_len, head_dim]"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel requires CUDA tensors"
    assert q.device == k.device == v.device, "q/k/v must be on the same CUDA device"
    assert q.dtype in (torch.float16, torch.bfloat16), "q must be float16 or bfloat16"
    assert k.dtype == q.dtype and v.dtype == q.dtype, "q/k/v dtypes must match"
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    ), "last dimension must be contiguous"
    assert q.shape[0] == k.shape[0] == v.shape[0], "q/k/v batch sizes must match"
    assert q.shape[2] == k.shape[2] == v.shape[2], "q/k/v sequence lengths must match"
    assert q.shape[3] == k.shape[3] == v.shape[3], "q/k/v head dimensions must match"
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert v.shape[1] == num_kv_heads, "k and v must have the same number of heads"
    assert num_kv_heads > 0, "k/v must contain at least one head"
    assert (
        num_q_heads % num_kv_heads == 0
    ), "q heads must be divisible by k/v heads for GQA"
    assert (
        head_dim in SUPPORTED_HEAD_DIMS
    ), f"head_dim must be one of {SUPPORTED_HEAD_DIMS}"
    assert seq_len > 0, "seq_len must be positive"
    return batch_size, num_q_heads, num_kv_heads, seq_len, head_dim


def triton_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    causal_early_stop: bool = True,
    softmax_scale: float | None = None,
    block_m: int = 32,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
    loop_num_stages: int | None = None,
) -> torch.Tensor:
    """Compute FlashAttention-style forward attention with Triton.

    Parameters are deliberately explicit because this is a learning and
    benchmark project.  The benchmark script sweeps block_m/block_n/warps/stages
    to show how tile shape affects latency.  K/V may have fewer heads than Q;
    in that case the kernel uses grouped-query attention (GQA) without
    materializing repeated K/V tensors.
    """
    batch_size, num_q_heads, num_kv_heads, seq_len, head_dim = _check_inputs(q, k, v)
    assert _is_power_of_two(block_m), "block_m must be a power of two"
    assert _is_power_of_two(block_n), "block_n must be a power of two"
    assert num_warps > 0, "num_warps must be positive"
    assert num_stages > 0, "num_stages must be positive"
    if loop_num_stages is None:
        loop_num_stages = num_stages
    assert loop_num_stages > 0, "loop_num_stages must be positive"
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    # The kernel uses exp2 for speed, so convert e-based softmax scale to
    # log2 space: exp(x * scale) == exp2(x * scale / ln(2)).
    softmax_scale_log2 = float(softmax_scale) / math.log(2.0)

    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), batch_size * num_q_heads)
    _flash_attention_fwd_kernel[grid](
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
        NUM_Q_HEADS=num_q_heads,
        HEADS_PER_KV=num_q_heads // num_kv_heads,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SOFTMAX_SCALE_LOG2=softmax_scale_log2,
        CAUSAL=causal,
        P_DTYPE_BF16=q.dtype is torch.bfloat16,
        LOOP_NUM_STAGES=loop_num_stages,
        CAUSAL_EARLY_STOP=causal_early_stop,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
