"""Correctness-first loading helpers for official Qwen3.5 block-FP8 weights."""

from __future__ import annotations

import re
from math import ceil

import torch

EXPERT_FP8_WEIGHT = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


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
) -> torch.Tensor:
    """Dequantize a rank-2 block-FP8 weight using its inverse-scale grid."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("block-FP8 weight and scale must both be rank 2")
    block_rows, block_columns = block_size
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("block-FP8 block dimensions must both be positive")
    expected_scale_shape = (
        ceil(weight.shape[0] / block_rows),
        ceil(weight.shape[1] / block_columns),
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"invalid block-FP8 scale shape: {tuple(scale.shape)}; "
            f"expected {expected_scale_shape}"
        )
    output = weight.to(output_dtype)
    scale = scale.to(output_dtype)
    rows, columns = weight.shape
    if rows % block_rows == 0 and columns % block_columns == 0:
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
    for block_row, row_start in enumerate(range(0, rows, block_rows)):
        row_end = min(row_start + block_rows, rows)
        row_scale = torch.repeat_interleave(
            scale[block_row],
            block_columns,
        )[:columns]
        output[row_start:row_end].mul_(row_scale)
    return output
