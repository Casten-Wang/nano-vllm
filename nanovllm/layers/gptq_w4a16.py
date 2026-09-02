"""Fused GPTQ INT4 dequantization and W4A16 matrix multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gptq_w4a16_kernel(
    x_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    g_idx_ptr,
    output_ptr,
    x_stride_m,
    qweight_stride_k,
    qzeros_stride_g,
    scales_stride_g,
    output_stride_m,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    column_block = tl.program_id(1)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = columns < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(x_ptr + row * x_stride_m + k, mask=k_mask, other=0.0)
        packed_weight = tl.load(
            qweight_ptr + (k[:, None] // 8) * qweight_stride_k + columns[None, :],
            mask=k_mask[:, None] & column_mask[None, :],
            other=0,
        )
        quantized = (packed_weight >> ((k[:, None] % 8) * 4)) & 15
        group = tl.load(g_idx_ptr + k, mask=k_mask, other=0)
        packed_zero = tl.load(
            qzeros_ptr + group[:, None] * qzeros_stride_g + columns[None, :] // 8,
            mask=k_mask[:, None] & column_mask[None, :],
            other=0,
        )
        zero = (packed_zero >> ((columns[None, :] % 8) * 4)) & 15
        scale = tl.load(
            scales_ptr + group[:, None] * scales_stride_g + columns[None, :],
            mask=k_mask[:, None] & column_mask[None, :],
            other=0.0,
        )
        weight = (quantized - zero).to(tl.float32) * scale
        accumulator += tl.sum(x[:, None].to(tl.float32) * weight, axis=0)

    tl.store(
        output_ptr + row * output_stride_m + columns,
        accumulator,
        mask=column_mask,
    )


def gptq_w4a16_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute ``inputs @ dequantize(qweight)`` without materializing weights."""

    if not inputs.is_cuda:
        raise ValueError("the Triton W4A16 backend requires CUDA tensors")
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("the Triton W4A16 backend requires FP16 or BF16 activations")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("GPTQ qweight and qzeros must use torch.int32")
    if g_idx.dtype != torch.int32:
        raise ValueError("GPTQ g_idx must use torch.int32")
    if inputs.ndim != 2 or qweight.ndim != 2:
        raise ValueError("W4A16 inputs and qweight must be rank 2")
    rows, input_size = inputs.shape
    if qweight.shape[0] * 8 != input_size:
        raise ValueError("packed qweight input size does not match activations")
    output_size = qweight.shape[1]
    if output_size % 8:
        raise ValueError("GPTQ output size must be divisible by pack factor 8")
    if tuple(scales.shape) != (qzeros.shape[0], output_size):
        raise ValueError("GPTQ scale shape is inconsistent with qzeros/qweight")
    if qzeros.shape[1] * 8 != output_size:
        raise ValueError("packed qzeros output size does not match qweight")
    if tuple(g_idx.shape) != (input_size,):
        raise ValueError("GPTQ g_idx size does not match activations")
    inputs = inputs.contiguous()
    output = torch.empty(
        (rows, output_size),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    _gptq_w4a16_kernel[(rows, triton.cdiv(output_size, 64))](
        inputs,
        qweight,
        qzeros,
        scales,
        g_idx,
        output,
        inputs.stride(0),
        qweight.stride(0),
        qzeros.stride(0),
        scales.stride(0),
        output.stride(0),
        K=input_size,
        N=output_size,
        BLOCK_K=32,
        BLOCK_N=64,
        num_warps=4,
    )
    return output
