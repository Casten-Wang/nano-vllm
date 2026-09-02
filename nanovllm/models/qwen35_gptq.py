"""Correctness-first GPTQ-Int4 routed experts for Qwen3.5."""

from __future__ import annotations

from functools import partial
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.models.moe_dispatch import weight_expert_output

GPTQ_BITS = 4
GPTQ_PACK_FACTOR = 32 // GPTQ_BITS
EXPERT_GPTQ_WEIGHT = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<component>qweight|qzeros|scales|g_idx)$"
)


def resolve_gptq_expert_parameter(weight_name: str) -> tuple[str, int] | None:
    match = EXPERT_GPTQ_WEIGHT.fullmatch(weight_name)
    if match is None:
        return None
    projection = match.group("projection").removesuffix("_proj")
    target = f"{match.group('prefix')}.{projection}_{match.group('component')}"
    return target, int(match.group("expert"))


def unpack_gptq_int4(packed: torch.Tensor, axis: int) -> torch.Tensor:
    """Unpack unsigned INT4 values stored eight-per-int32 by GPTQModel."""

    if packed.dtype != torch.int32:
        raise ValueError("GPTQ packed tensors must use torch.int32")
    shifts_shape = [1] * (packed.ndim + 1)
    shifts_shape[axis + 1] = GPTQ_PACK_FACTOR
    shifts = (
        torch.arange(GPTQ_PACK_FACTOR, device=packed.device, dtype=torch.int32)
        .mul_(GPTQ_BITS)
        .reshape(shifts_shape)
    )
    unpacked = torch.bitwise_right_shift(packed.unsqueeze(axis + 1), shifts)
    unpacked.bitwise_and_((1 << GPTQ_BITS) - 1)
    shape = list(packed.shape)
    shape[axis] *= GPTQ_PACK_FACTOR
    return unpacked.reshape(shape)


def dequantize_gptq_int4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Return a [out, in] weight for the official desc_act=false GPTQ layout."""

    quantized = unpack_gptq_int4(qweight, 0)
    zeros = unpack_gptq_int4(qzeros, 1)
    # The pinned official Qwen checkpoint stores the symmetric zero point
    # directly (every packed nibble is 8), rather than zero-minus-one.
    group_ids = g_idx.to(torch.long)
    if group_ids.ndim != 1 or group_ids.numel() != quantized.shape[0]:
        raise ValueError("GPTQ g_idx must contain one group id per input channel")
    if group_ids.numel() and (
        int(group_ids.min()) < 0 or int(group_ids.max()) >= scales.shape[0]
    ):
        raise ValueError("GPTQ g_idx refers to an unavailable scale group")
    values = quantized.to(scales.dtype)
    values.sub_(zeros[group_ids]).mul_(scales[group_ids])
    return values.transpose(0, 1).to(output_dtype)


class Qwen35GPTQExperts(nn.Module):
    """Stacked GPTQ experts with an intentionally slow reference executor."""

    checkpoint_components = ("qweight", "qzeros", "scales", "g_idx")

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        group_size: int,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.group_size = group_size
        if backend not in ("reference", "triton"):
            raise ValueError("GPTQ expert backend must be 'reference' or 'triton'")
        self.backend = backend
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        if intermediate_size % self.tp_size:
            raise ValueError("moe_intermediate_size must be divisible by TP size")
        self.local_intermediate_size = intermediate_size // self.tp_size
        if self.local_intermediate_size % GPTQ_PACK_FACTOR:
            raise ValueError("local expert width must be divisible by GPTQ pack factor")

        self._register_projection("gate", hidden_size, self.local_intermediate_size)
        self._register_projection("up", hidden_size, self.local_intermediate_size)
        local_group_count = (
            self.local_intermediate_size + group_size - 1
        ) // group_size
        self._register_projection(
            "down",
            self.local_intermediate_size,
            hidden_size,
            group_count=local_group_count,
        )

    def _register_projection(
        self,
        projection: str,
        input_size: int,
        output_size: int,
        *,
        group_count: int | None = None,
    ) -> None:
        group_count = (
            group_count or (input_size + self.group_size - 1) // self.group_size
        )
        shapes = {
            "qweight": (self.num_experts, input_size // GPTQ_PACK_FACTOR, output_size),
            "qzeros": (
                self.num_experts,
                group_count,
                output_size // GPTQ_PACK_FACTOR,
            ),
            "scales": (self.num_experts, group_count, output_size),
            "g_idx": (self.num_experts, input_size),
        }
        dtypes = {
            "qweight": torch.int32,
            "qzeros": torch.int32,
            "scales": torch.float16,
            "g_idx": torch.int32,
        }
        for component in self.checkpoint_components:
            parameter = nn.Parameter(
                torch.empty(shapes[component], dtype=dtypes[component]),
                requires_grad=False,
            )
            parameter.packed_safetensors_loader = partial(
                self._load_component,
                projection,
                component,
            )
            parameter.required_checkpoint_shards = frozenset(range(self.num_experts))
            self.register_parameter(f"{projection}_{component}", parameter)

    def _load_component(
        self,
        projection: str,
        component: str,
        parameter: nn.Parameter,
        source,
        expert_id: int,
    ) -> None:
        get_shape = getattr(source, "get_shape", None)
        shape = tuple(get_shape() if get_shape is not None else source.shape)
        if projection in ("gate", "up"):
            output_start = self.tp_rank * self.local_intermediate_size
            output_end = output_start + self.local_intermediate_size
            if component == "qweight" or component == "scales":
                value = source[:, output_start:output_end]
            elif component == "qzeros":
                value = source[
                    :,
                    output_start // GPTQ_PACK_FACTOR : output_end // GPTQ_PACK_FACTOR,
                ]
            else:
                value = source[:]
        else:
            input_start = self.tp_rank * self.local_intermediate_size
            input_end = input_start + self.local_intermediate_size
            first_group = input_start // self.group_size
            last_group = (input_end + self.group_size - 1) // self.group_size
            if component == "qweight":
                value = source[
                    input_start // GPTQ_PACK_FACTOR : input_end // GPTQ_PACK_FACTOR,
                    :,
                ]
            elif component in ("qzeros", "scales"):
                value = source[first_group:last_group, :]
            else:
                value = source[input_start:input_end] - first_group
        target = parameter[expert_id]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"invalid {projection}_{component} expert {expert_id} slice: "
                f"source {shape} produced {tuple(value.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        target.copy_(value)

    def _weight(self, projection: str, expert_id: int, dtype: torch.dtype):
        return dequantize_gptq_int4(
            getattr(self, f"{projection}_qweight")[expert_id],
            getattr(self, f"{projection}_qzeros")[expert_id],
            getattr(self, f"{projection}_scales")[expert_id],
            getattr(self, f"{projection}_g_idx")[expert_id],
            output_dtype=dtype,
        )

    def _linear(
        self,
        inputs: torch.Tensor,
        projection: str,
        expert_id: int,
    ) -> torch.Tensor:
        if self.backend == "triton":
            from nanovllm.layers.gptq_w4a16 import gptq_w4a16_linear

            return gptq_w4a16_linear(
                inputs,
                getattr(self, f"{projection}_qweight")[expert_id],
                getattr(self, f"{projection}_qzeros")[expert_id],
                getattr(self, f"{projection}_scales")[expert_id],
                getattr(self, f"{projection}_g_idx")[expert_id],
            )
        return F.linear(
            inputs,
            self._weight(projection, expert_id, inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        is_decode: bool | None = None,
        decode_token_count: int = 0,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        del is_decode, decode_token_count
        output = torch.zeros_like(hidden_states)
        for expert_id in torch.unique(topk_ids).cpu().tolist():
            routes = (topk_ids == expert_id).nonzero(as_tuple=False)
            token_ids = routes[:, 0]
            slots = routes[:, 1]
            expert_input = hidden_states[token_ids]
            gate = self._linear(expert_input, "gate", expert_id)
            up = self._linear(expert_input, "up", expert_id)
            value = self._linear(
                F.silu(gate) * up,
                "down",
                expert_id,
            )
            value = weight_expert_output(value, topk_weights[token_ids, slots])
            output.index_add_(0, token_ids, value.to(output.dtype))
        if reduce_output and self.tp_size > 1:
            dist.all_reduce(output)
        return output
