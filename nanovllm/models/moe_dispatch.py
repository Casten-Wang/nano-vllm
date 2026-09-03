"""Graph-safe PyTorch primitives for mixture-of-experts dispatch."""

from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F


class BatchedExpertWeightBufferPool:
    """Single-stream scratch storage for batched expert decode."""

    def __init__(self) -> None:
        self.storage: torch.Tensor | None = None
        self.allocation_count = 0
        self.reuse_count = 0
        self.workspace_storage: torch.Tensor | None = None
        self.workspace_allocation_count = 0
        self.workspace_reuse_count = 0

    def gather(
        self,
        weight: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("expert gather buffer is inference-only")
        if weight.ndim < 2 or expert_ids.ndim != 1:
            raise ValueError("expert gather expects weights and flat expert ids")
        required = expert_ids.numel() * weight[0].numel()
        if (
            self.storage is None
            or self.storage.device != weight.device
            or self.storage.dtype != weight.dtype
            or self.storage.numel() < required
        ):
            self.storage = torch.empty(
                required,
                dtype=weight.dtype,
                device=weight.device,
            )
            self.allocation_count += 1
        else:
            self.reuse_count += 1
        output = self.storage[:required].view(
            expert_ids.numel(),
            *weight.shape[1:],
        )
        torch.index_select(weight, 0, expert_ids, out=output)
        return output

    def workspaces(
        self,
        *shapes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Return disjoint views for simultaneously live MoE intermediates."""

        if torch.is_grad_enabled():
            raise RuntimeError("expert workspace buffer is inference-only")
        sizes = tuple(prod(shape) for shape in shapes)
        required = sum(sizes)
        if (
            self.workspace_storage is None
            or self.workspace_storage.device != device
            or self.workspace_storage.dtype != dtype
            or self.workspace_storage.numel() < required
        ):
            self.workspace_storage = torch.empty(
                required,
                dtype=dtype,
                device=device,
            )
            self.workspace_allocation_count += 1
        else:
            self.workspace_reuse_count += 1
        outputs = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            outputs.append(
                self.workspace_storage[offset : offset + size].view(shape)
            )
            offset += size
        return tuple(outputs)

    def storage_stats(self) -> dict[str, int]:
        storage_bytes = (
            0
            if self.storage is None
            else self.storage.numel() * self.storage.element_size()
        )
        workspace_bytes = (
            0
            if self.workspace_storage is None
            else self.workspace_storage.numel()
            * self.workspace_storage.element_size()
        )
        return {
            "storage_bytes": storage_bytes,
            "allocation_count": self.allocation_count,
            "reuse_count": self.reuse_count,
            "workspace_bytes": workspace_bytes,
            "workspace_allocation_count": self.workspace_allocation_count,
            "workspace_reuse_count": self.workspace_reuse_count,
        }


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.requires_grad:
        return F.silu(gate) * up
    F.silu(gate, inplace=True)
    return gate.mul_(up)


def weighted_route_sum(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine top-k routes, optionally writing inference output in place."""

    routed = expert_output.reshape(*topk_weights.shape, -1)
    weights = topk_weights.unsqueeze(-1)
    if expert_output.requires_grad or topk_weights.requires_grad:
        if output is not None:
            raise ValueError("output is unsupported when route sum requires grad")
        return (routed * weights).sum(dim=1)
    expected_shape = (topk_weights.shape[0], expert_output.shape[-1])
    if output is not None and (
        output.shape != expected_shape
        or output.dtype != expert_output.dtype
        or output.device != expert_output.device
    ):
        raise ValueError("output must match the weighted route sum")
    routed.mul_(weights)
    if output is not None:
        torch.sum(routed, dim=1, out=output)
        return output
    return routed.sum(dim=1)


def weight_expert_output(
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply route weights while reusing inference-only expert output."""

    weights = route_weights.unsqueeze(-1)
    if expert_output.requires_grad or route_weights.requires_grad:
        return expert_output * weights
    return expert_output.mul_(weights)


def batched_expert_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    chunk_size: int = 8,
    *,
    output: torch.Tensor | None = None,
    weight_buffer_pool: BatchedExpertWeightBufferPool | None = None,
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
        chunk_tokens = end - start
        expert_ids = topk_ids[start:end].reshape(-1)
        route_count = expert_ids.numel()
        gate_up_output = expert_output_buffer = None
        if weight_buffer_pool is not None:
            gate_up_output, expert_output_buffer = weight_buffer_pool.workspaces(
                (chunk_tokens, top_k, gate_up_proj.shape[1], 1),
                (route_count, down_proj.shape[1], 1),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        selected_gate_up = (
            gate_up_proj.index_select(0, expert_ids)
            if weight_buffer_pool is None
            else weight_buffer_pool.gather(gate_up_proj, expert_ids)
        ).view(
            chunk_tokens,
            top_k,
            gate_up_proj.shape[1],
            gate_up_proj.shape[2],
        )
        # Broadcast each token across its top-k routes inside matmul instead
        # of materializing a [chunk, top_k, hidden] repeated-input tensor.
        if gate_up_output is None:
            gate_up = torch.matmul(
                selected_gate_up,
                hidden_states[start:end, None, :, None],
            ).squeeze(-1).flatten(0, 1)
        else:
            torch.matmul(
                selected_gate_up,
                hidden_states[start:end, None, :, None],
                out=gate_up_output,
            )
            gate_up = gate_up_output.squeeze(-1).flatten(0, 1)
        del selected_gate_up
        gate, up = gate_up.chunk(2, dim=-1)
        activated = silu_and_mul(gate, up)
        selected_down = (
            down_proj.index_select(0, expert_ids)
            if weight_buffer_pool is None
            else weight_buffer_pool.gather(down_proj, expert_ids)
        )
        if expert_output_buffer is None:
            expert_output = torch.bmm(
                selected_down,
                activated.unsqueeze(-1),
            ).squeeze(-1)
        else:
            torch.bmm(
                selected_down,
                activated.unsqueeze(-1),
                out=expert_output_buffer,
            )
            expert_output = expert_output_buffer.squeeze(-1)
        route_weights = topk_weights[start:end]
        if expert_output.requires_grad or route_weights.requires_grad:
            output[start:end] = weighted_route_sum(
                expert_output,
                route_weights,
            )
        else:
            weighted_route_sum(
                expert_output,
                route_weights,
                output=output[start:end],
            )
        # Python keeps loop locals alive until their next assignment. Release
        # the consumed chunk so the next gathered expert weights do not
        # overlap with the previous chunk's large temporaries.
        del (
            activated,
            expert_ids,
            expert_output,
            gate,
            gate_up,
            selected_down,
            up,
        )
    return output
