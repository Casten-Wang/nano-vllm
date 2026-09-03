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
    expected_scale_shape = (
        ceil(weight.shape[0] / block_rows),
        ceil(weight.shape[1] / block_columns),
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"invalid block-FP8 scale shape: {tuple(scale.shape)}; "
            f"expected {expected_scale_shape}"
        )
    expanded_scale = scale.repeat_interleave(block_rows, 0).repeat_interleave(
        block_columns,
        1,
    )
    expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
    return weight.to(output_dtype).mul_(expanded_scale.to(output_dtype))
