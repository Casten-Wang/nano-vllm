"""Graph-safe PyTorch primitives for mixture-of-experts dispatch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batched_expert_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Dispatch decode tokens without device-to-host synchronization.

    Selected expert weights are gathered in bounded token chunks so temporary
    storage does not scale with the full continuous-batching decode size.
    """

    if chunk_size <= 0:
        raise ValueError("decode chunk size must be positive")
    top_k = topk_ids.shape[1]
    chunks = []
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
        selected_down = down_proj.index_select(0, expert_ids)
        expert_output = torch.bmm(
            selected_down,
            (F.silu(gate) * up).unsqueeze(-1),
        ).squeeze(-1)
        chunks.append(
            (
                expert_output.reshape(end - start, top_k, -1)
                * topk_weights[start:end].unsqueeze(-1)
            ).sum(dim=1)
        )
    return torch.cat(chunks, dim=0)
