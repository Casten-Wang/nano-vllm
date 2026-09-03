"""Correctness-first loading helpers for official Qwen3.6 block-FP8 weights."""

from __future__ import annotations

import re
from math import ceil

import torch

EXPERT_FP8_WEIGHT = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _tensor_shape(value) -> tuple[int, ...]:
    get_shape = getattr(value, "get_shape", None)
    return tuple(get_shape() if get_shape is not None else value.shape)


def resolve_fp8_expert_parameter(
    weight_name: str,
) -> tuple[str, tuple[int, str]] | None:
    """Map one serialized expert projection to the stacked runtime parameter."""

    match = EXPERT_FP8_WEIGHT.fullmatch(weight_name)
    if match is None:
        return None
    projection = match.group("projection").removesuffix("_proj")
    target_projection = "gate_up_proj" if projection in ("gate", "up") else "down_proj"
    target = f"{match.group('prefix')}.{target_projection}"
    return target, (int(match.group("expert")), projection)


def dequantize_fp8_block_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    *,
    output_dtype: torch.dtype,
    block_offset: tuple[int, int] = (0, 0),
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a rank-2 block-FP8 weight slice using its scale grid."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("block-FP8 weight and scale must both be rank 2")
    block_rows, block_columns = block_size
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("block-FP8 block dimensions must both be positive")
    row_offset, column_offset = block_offset
    if not 0 <= row_offset < block_rows or not 0 <= column_offset < block_columns:
        raise ValueError("block-FP8 offsets must lie within the first block")
    expected_scale_shape = (
        ceil((row_offset + weight.shape[0]) / block_rows),
        ceil((column_offset + weight.shape[1]) / block_columns),
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"invalid block-FP8 scale shape: {tuple(scale.shape)}; "
            f"expected {expected_scale_shape}"
        )
    if out is None:
        output = weight.to(output_dtype)
    else:
        if tuple(out.shape) != tuple(weight.shape):
            raise ValueError("block-FP8 output shape must match the weight")
        if out.dtype != output_dtype:
            raise ValueError("block-FP8 output dtype does not match output_dtype")
        output = out
        output.copy_(weight)
    scale = scale.to(dtype=output_dtype, device=output.device)
    rows, columns = weight.shape
    if (
        row_offset == 0
        and column_offset == 0
        and rows % block_rows == 0
        and columns % block_columns == 0
    ):
        # Keep the compact scale grid and broadcast it over a block view.  The
        # previous repeat_interleave implementation materialized another full
        # [rows, columns] tensor, temporarily adding one model-dtype copy of
        # every FP8 weight during CPU checkpoint loading.
        blocked_output = output.view(
            rows // block_rows,
            block_rows,
            columns // block_columns,
            block_columns,
        )
        blocked_output.mul_(scale[:, None, :, None])
        return output

    # Partial edge blocks cannot be represented by one regular block view.
    # Expand only one row of the scale grid at a time, bounding the temporary
    # to `columns` elements instead of `rows * columns` elements.
    row_start = 0
    while row_start < rows:
        block_row = (row_offset + row_start) // block_rows
        row_end = min(
            row_start + block_rows - (row_offset + row_start) % block_rows,
            rows,
        )
        row_scale = torch.repeat_interleave(
            scale[block_row],
            block_columns,
        )[column_offset : column_offset + columns]
        output[row_start:row_end].mul_(row_scale)
        row_start = row_end
    return output


def dequantize_fp8_block_weight_slice(
    weight,
    scale,
    block_size: tuple[int, int],
    row_range: tuple[int, int],
    column_range: tuple[int, int],
    *,
    output_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read and dequantize only one rectangular shard of an FP8 weight."""

    weight_shape = _tensor_shape(weight)
    scale_shape = _tensor_shape(scale)
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        raise ValueError("block-FP8 weight and scale must both be rank 2")
    block_rows, block_columns = block_size
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("block-FP8 block dimensions must both be positive")
    expected_scale_shape = (
        ceil(weight_shape[0] / block_rows),
        ceil(weight_shape[1] / block_columns),
    )
    if scale_shape != expected_scale_shape:
        raise ValueError(
            f"invalid block-FP8 scale shape: {scale_shape}; "
            f"expected {expected_scale_shape}"
        )
    row_start, row_end = row_range
    column_start, column_end = column_range
    if not 0 <= row_start < row_end <= weight_shape[0]:
        raise ValueError("block-FP8 row range is outside the weight")
    if not 0 <= column_start < column_end <= weight_shape[1]:
        raise ValueError("block-FP8 column range is outside the weight")

    scale_row_start = row_start // block_rows
    scale_row_end = ceil(row_end / block_rows)
    scale_column_start = column_start // block_columns
    scale_column_end = ceil(column_end / block_columns)
    local_weight = weight[row_start:row_end, column_start:column_end]
    local_scale = scale[
        scale_row_start:scale_row_end,
        scale_column_start:scale_column_end,
    ]
    return dequantize_fp8_block_weight(
        local_weight,
        local_scale,
        block_size,
        output_dtype=output_dtype,
        out=out,
        block_offset=(
            row_start % block_rows,
            column_start % block_columns,
        ),
    )
