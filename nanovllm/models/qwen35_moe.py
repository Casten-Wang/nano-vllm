"""Qwen3.5 MoE layers shared by the text-only runtime."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, divide


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm whose checkpoint stores the residual weight delta."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(x.dtype)


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
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(hidden_states.dtype), topk_ids


class Qwen35Experts(nn.Module):
    """Tensor-parallel expert weights with a correctness-first torch path."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
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
        self.down_proj.weight_loader = self._load_down

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
        local_weight = torch.cat((gate[:, start:end], up[:, start:end]), dim=1)
        param.data.copy_(local_weight)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        active_experts = torch.unique(topk_ids)
        for expert_id_tensor in active_experts:
            expert_id = int(expert_id_tensor.item())
            topk_slot, token_index = torch.where(topk_ids == expert_id)
            expert_input = hidden_states[token_index]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.down_proj[expert_id],
            )
            expert_output = expert_output * topk_weights[
                token_index,
                topk_slot,
            ].unsqueeze(-1)
            output.index_add_(0, token_index, expert_output.to(output.dtype))
        if self.tp_size > 1:
            dist.all_reduce(output)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_up_proj(hidden_states)))


class Qwen35SparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.gate = Qwen35TopKRouter(
            self.hidden_size,
            int(config.num_experts),
            int(config.num_experts_per_tok),
        )
        self.experts = Qwen35Experts(
            self.hidden_size,
            int(config.moe_intermediate_size),
            int(config.num_experts),
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
        routed = self.experts(flat_states, topk_ids, topk_weights)
        shared = self.shared_expert(flat_states)
        shared = torch.sigmoid(self.shared_expert_gate(flat_states)) * shared
        return (routed + shared).reshape(original_shape)
