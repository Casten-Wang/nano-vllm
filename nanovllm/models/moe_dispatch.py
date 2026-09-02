"""Graph-safe PyTorch primitives for mixture-of-experts dispatch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.requires_grad:
        return F.silu(gate) * up
    F.silu(gate, inplace=True)
    return gate.mul_(up)


def weighted_route_sum(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    routed = expert_output.reshape(*topk_weights.shape, -1)
    weights = topk_weights.unsqueeze(-1)
    if expert_output.requires_grad or topk_weights.requires_grad:
        return (routed * weights).sum(dim=1)
    routed.mul_(weights)
    return routed.sum(dim=1)


def batched_expert_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    chunk_size: int = 8,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch decode tokens without device-to-host synchronization.

    Selected expert weights are gathered in bounded token chunks so temporary
    storage does not scale with the full continuous-batching decode size.
    """

    if chunk_size <= 0:
        raise ValueError("decode chunk size must be positive")
    top_k = topk_ids.shape[1]
    if output is None:
        output = torch.empty_like(hidden_states)
    elif (
        output.shape != hidden_states.shape
        or output.dtype != hidden_states.dtype
        or output.device != hidden_states.device
    ):
        raise ValueError("output must match hidden_states shape, dtype, and device")
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        expert_ids = topk_ids[start:end].reshape(-1)
        selected_gate_up = gate_up_proj.index_select(0, expert_ids)
        route_hidden = (
            hidden_states[start:end]
            .unsqueeze(1)
            .expand(-1, top_k, -1)
            .reshape(expert_ids.numel(), -1, 1)
        )
        gate_up = torch.bmm(selected_gate_up, route_hidden).squeeze(-1)
        del selected_gate_up, route_hidden
        gate, up = gate_up.chunk(2, dim=-1)
        activated = silu_and_mul(gate, up)
        selected_down = down_proj.index_select(0, expert_ids)
        expert_output = torch.bmm(
            selected_down,
            activated.unsqueeze(-1),
        ).squeeze(-1)
        output[start:end] = weighted_route_sum(
            expert_output,
            topk_weights[start:end],
        )
    return output
