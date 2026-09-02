"""Qwen3.5 MoE layers shared by the text-only runtime."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, divide
from nanovllm.models.moe_dispatch import (
    batched_expert_dispatch,
    weight_expert_output,
)


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm whose checkpoint stores the residual weight delta."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Keep the materialized ``1 + checkpoint_weight`` gain in FP32. This
        # removes a cast and an addition from every norm invocation while
        # preserving the official accumulation order.
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.weight.weight_loader = self._load_weight
        self.weight.safetensors_loader = self._load_weight_slice

    @staticmethod
    def _load_weight(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight.float().add_(1.0))

    @staticmethod
    def _load_weight_slice(param: nn.Parameter, loaded_weight) -> None:
        Qwen35RMSNorm._load_weight(param, loaded_weight[:])

    def forward(
        self,
        x: torch.Tensor,
        *,
        inplace_output: bool = False,
    ) -> torch.Tensor:
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        if inplace_output and not torch.is_grad_enabled():
            x_float.mul_(inverse_rms)
            x_float.mul_(self.weight)
            if x_float is not x:
                x.copy_(x_float)
            return x
        if not torch.is_grad_enabled() and x_float is not x:
            x_float.mul_(inverse_rms)
            x_float.mul_(self.weight)
            return x_float.to(x.dtype)
        normalized = x_float * inverse_rms
        return (normalized * self.weight).to(x.dtype)


class Qwen35TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = F.linear(flat_states, self.weight)
        # The full-softmax denominator cancels when selected expert weights
        # are renormalized. Select first so FP32 softmax only materializes
        # ``top_k`` values per token instead of ``num_experts`` values.
        topk_logits, topk_ids = torch.topk(router_logits, self.top_k, dim=-1)
        topk_logits = topk_logits.float()
        if topk_logits.requires_grad:
            topk_weights = torch.softmax(topk_logits, dim=-1)
        else:
            torch.softmax(topk_logits, dim=-1, out=topk_logits)
            topk_weights = topk_logits
        return topk_weights.to(hidden_states.dtype), topk_ids


class Qwen35Experts(nn.Module):
    """Tensor-parallel expert weights with a correctness-first torch path."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        decode_backend: str = "sorted",
        decode_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        if decode_backend not in ("sorted", "batched"):
            raise ValueError("decode_backend must be 'sorted' or 'batched'")
        if decode_chunk_size <= 0:
            raise ValueError("decode_chunk_size must be positive")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.decode_backend = decode_backend
        self.decode_chunk_size = decode_chunk_size
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.local_intermediate_size = divide(intermediate_size, self.tp_size)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_experts,
                2 * self.local_intermediate_size,
                hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                self.local_intermediate_size,
            )
        )
        self.gate_up_proj.weight_loader = self._load_gate_up
        self.gate_up_proj.safetensors_loader = self._load_gate_up_slice
        self.down_proj.weight_loader = self._load_down
        self.down_proj.safetensors_loader = self._load_down_slice

    @staticmethod
    def _slice_shape(loaded_weight) -> tuple[int, ...]:
        get_shape = getattr(loaded_weight, "get_shape", None)
        return tuple(get_shape() if get_shape is not None else loaded_weight.shape)

    def _load_gate_up(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        expected_shape = (
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
        )
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"invalid expert gate_up_proj shape: {tuple(loaded_weight.shape)}; "
                f"expected {expected_shape}"
            )
        gate, up = loaded_weight.chunk(2, dim=1)
        start = self.tp_rank * self.local_intermediate_size
        end = start + self.local_intermediate_size
        param.data[:, : self.local_intermediate_size].copy_(gate[:, start:end])
        param.data[:, self.local_intermediate_size :].copy_(up[:, start:end])

    def _load_gate_up_slice(self, param: nn.Parameter, loaded_weight) -> None:
        expected_shape = (
            self.num_experts,
            2 * self.intermediate_size,
            self.hidden_size,
        )
        shape = self._slice_shape(loaded_weight)
        if shape != expected_shape:
            raise ValueError(
                f"invalid expert gate_up_proj shape: {shape}; "
                f"expected {expected_shape}"
            )
        start = self.tp_rank * self.local_intermediate_size
        end = start + self.local_intermediate_size
        param.data[:, : self.local_intermediate_size].copy_(
            loaded_weight[:, start:end, :]
        )
        param.data[:, self.local_intermediate_size :].copy_(
            loaded_weight[
                :,
                self.intermediate_size + start : self.intermediate_size + end,
                :,
            ]
        )

    def _load_down(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        expected_shape = (
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"invalid expert down_proj shape: {tuple(loaded_weight.shape)}; "
                f"expected {expected_shape}"
            )
        start = self.tp_rank * self.local_intermediate_size
        param.data.copy_(
            loaded_weight.narrow(2, start, self.local_intermediate_size)
        )

    def _load_down_slice(self, param: nn.Parameter, loaded_weight) -> None:
        expected_shape = (
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        shape = self._slice_shape(loaded_weight)
        if shape != expected_shape:
            raise ValueError(
                f"invalid expert down_proj shape: {shape}; expected {expected_shape}"
            )
        start = self.tp_rank * self.local_intermediate_size
        end = start + self.local_intermediate_size
        param.data.copy_(loaded_weight[:, :, start:end])

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
        if is_decode is None:
            is_decode = hidden_states.shape[0] == 1
        if not 0 <= decode_token_count <= hidden_states.shape[0]:
            raise ValueError("decode_token_count must fit within hidden_states")
        if self.decode_backend == "batched" and (
            is_decode or decode_token_count == hidden_states.shape[0]
        ):
            output = batched_expert_dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_up_proj,
                self.down_proj,
                self.decode_chunk_size,
            )
            if reduce_output and self.tp_size > 1:
                dist.all_reduce(output)
            return output
        if self.decode_backend == "batched" and decode_token_count:
            output = torch.zeros_like(hidden_states)
            batched_expert_dispatch(
                hidden_states[:decode_token_count],
                topk_ids[:decode_token_count],
                topk_weights[:decode_token_count],
                self.gate_up_proj,
                self.down_proj,
                self.decode_chunk_size,
                output=output[:decode_token_count],
            )
            self._forward_sorted(
                hidden_states[decode_token_count:],
                topk_ids[decode_token_count:],
                topk_weights[decode_token_count:],
                output=output[decode_token_count:],
            )
        else:
            output = self._forward_sorted(
                hidden_states,
                topk_ids,
                topk_weights,
            )
        if reduce_output and self.tp_size > 1:
            dist.all_reduce(output)
        return output

    def _forward_sorted(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.zeros_like(hidden_states)
        else:
            output.zero_()
        if hidden_states.shape[0] == 1:
            # Decode has only ``top_k`` routes. Preserve the expert-sorted
            # accumulation order without building the general flattened
            # token/group metadata used by prefill batches.
            route_order = torch.argsort(topk_ids[0], stable=True)
            routes = torch.stack(
                (topk_ids[0, route_order], route_order), dim=1
            ).cpu().tolist()
            for expert_id, route_index in routes:
                gate_up = F.linear(hidden_states, self.gate_up_proj[expert_id])
                gate, up = gate_up.chunk(2, dim=-1)
                expert_output = F.linear(
                    F.silu(gate) * up,
                    self.down_proj[expert_id],
                )
                output.add_(
                    weight_expert_output(
                        expert_output,
                        topk_weights[0, route_index],
                    )
                )
            return output

        assignments = topk_ids.reshape(-1)
        routing_weights = topk_weights.reshape(-1)
        order = torch.argsort(assignments, stable=True)
        sorted_experts = assignments[order]
        sorted_weights = routing_weights[order]
        # Flattened routes are token-major, so the original token index is
        # encoded directly in the sort permutation. The expert ids and route
        # weights have already consumed ``order``, so reuse its int64 storage
        # instead of allocating another route-sized token-index tensor.
        if torch.is_grad_enabled():
            sorted_tokens = torch.div(
                order,
                topk_ids.shape[1],
                rounding_mode="floor",
            )
        else:
            order.div_(topk_ids.shape[1], rounding_mode="floor")
            sorted_tokens = order
        active_experts, counts = torch.unique_consecutive(
            sorted_experts,
            return_counts=True,
        )
        # Copy only active expert metadata in one host synchronization. Decode
        # batches commonly route to far fewer than all 256 experts.
        groups = torch.stack((active_experts, counts), dim=1).cpu().tolist()
        offset = 0
        for expert_id, count in groups:
            end = offset + count
            token_index = sorted_tokens[offset:end]
            expert_input = hidden_states[token_index]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.down_proj[expert_id],
            )
            expert_output = weight_expert_output(
                expert_output,
                sorted_weights[offset:end],
            )
            output.index_add_(0, token_index, expert_output.to(output.dtype))
            offset = end
        return output


class Qwen35SharedExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.activation = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        activated = self.activation(self.gate_up_proj(hidden_states))
        if reduce_output:
            return self.down_proj(activated)
        return F.linear(
            activated,
            self.down_proj.weight,
            self.down_proj.bias if self.down_proj.tp_rank == 0 else None,
        )


class Qwen35SparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.gate = Qwen35TopKRouter(
            self.hidden_size,
            int(config.num_experts),
            int(config.num_experts_per_tok),
        )
        quantization = getattr(config, "nanovllm_quantization_spec", None)
        if quantization is not None and quantization.format == "gptq_int4":
            from nanovllm.models.qwen35_gptq import Qwen35GPTQExperts

            self.experts = Qwen35GPTQExperts(
                self.hidden_size,
                int(config.moe_intermediate_size),
                int(config.num_experts),
                int(quantization.group_size),
                getattr(config, "nanovllm_weight_quant_backend", "reference"),
            )
        else:
            self.experts = Qwen35Experts(
                self.hidden_size,
                int(config.moe_intermediate_size),
                int(config.num_experts),
                getattr(config, "qwen35_moe_decode_backend", "sorted"),
                int(getattr(config, "qwen35_moe_decode_chunk_size", 8)),
            )
        self.shared_expert = Qwen35SharedExpert(
            self.hidden_size,
            int(config.shared_expert_intermediate_size),
        )
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        topk_weights, topk_ids = self.gate(flat_states)
        from nanovllm.utils.context import get_context

        context = get_context()
        pure_decode = not context.is_prefill and not context.is_mixed

        expert_options = {
            "is_decode": pure_decode,
            "reduce_output": False,
        }
        if context.is_mixed:
            expert_options["decode_token_count"] = context.decode_token_count
        routed = self.experts(
            flat_states,
            topk_ids,
            topk_weights,
            **expert_options,
        )
        shared = self.shared_expert(flat_states, reduce_output=False)
        shared_gate = self.shared_expert_gate(flat_states)
        if torch.is_grad_enabled():
            shared = torch.sigmoid(shared_gate) * shared
            output = routed + shared
        else:
            # Both TP partials are dead after this merge. Reuse them instead
            # of materializing gated-shared and combined-output tensors for
            # every MoE layer in the inference hot path.
            shared_gate.sigmoid_()
            shared.mul_(shared_gate)
            routed.add_(shared)
            output = routed
        if self.experts.tp_size > 1:
            dist.all_reduce(output)
        return output.reshape(original_shape)
