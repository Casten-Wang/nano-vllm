"""Correctness-first state primitives for Qwen3.5 Gated DeltaNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match the FLA/Qwen3.5 L2 normalization convention."""

    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def causal_conv1d_step(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one depthwise causal-convolution step without in-place mutation.

    Args:
        x: Current input with shape ``[batch, channels]``.
        state: Previous raw inputs with shape ``[batch, channels, kernel]``.
        weight: Depthwise weights with shape ``[channels, kernel]``.
    """

    if x.ndim != 2 or state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid causal convolution tensor rank")
    if state.shape[0] != x.shape[0] or state.shape[1:] != weight.shape:
        raise ValueError("causal convolution shapes are inconsistent")
    next_state = torch.cat((state[..., 1:], x.unsqueeze(-1)), dim=-1)
    output = (next_state.to(weight.dtype) * weight.unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        output = output + bias
    return F.silu(output).to(x.dtype), next_state


def causal_conv1d_scan(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference causal convolution over ``[batch, sequence, channels]``."""

    if x.ndim != 3:
        raise ValueError("causal convolution input must be rank 3")
    outputs = []
    next_state = state
    for token_index in range(x.shape[1]):
        output, next_state = causal_conv1d_step(
            x[:, token_index], next_state, weight, bias
        )
        outputs.append(output)
    if not outputs:
        return x.new_empty(x.shape), next_state
    return torch.stack(outputs, dim=1), next_state


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-semantics recurrent oracle for prefill chunks and decode.

    Q/K/V use ``[batch, sequence, heads, dim]``. State is maintained in
    fp32 as ``[batch, heads, key_dim, value_dim]`` for numerical stability.
    """

    if query.shape != key.shape or query.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value shapes are inconsistent")
    if query.shape[:3] != value.shape[:3]:
        raise ValueError("query/key and value batch, sequence, heads must match")
    expected_gate_shape = query.shape[:3]
    if log_decay.shape != expected_gate_shape or beta.shape != expected_gate_shape:
        raise ValueError("decay and beta must have shape [batch, sequence, heads]")

    q = l2_normalize(query.float()) / (query.shape[-1] ** 0.5)
    k = l2_normalize(key.float())
    v = value.float()
    if initial_state is None:
        state = torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[3],
            value.shape[3],
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected_state_shape = (
            query.shape[0], query.shape[2], query.shape[3], value.shape[3]
        )
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"invalid recurrent state shape: {tuple(initial_state.shape)}; "
                f"expected {expected_state_shape}"
            )
        state = initial_state.float()

    outputs = []
    for token_index in range(query.shape[1]):
        q_t = q[:, token_index]
        k_t = k[:, token_index]
        v_t = v[:, token_index]
        state = state * log_decay[:, token_index].float().exp()[..., None, None]
        prediction = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        correction = (v_t - prediction) * beta[:, token_index].float().unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * correction.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    if not outputs:
        output = value.new_empty(value.shape)
    else:
        output = torch.stack(outputs, dim=1).to(value.dtype)
    return output, state


class Qwen35RecurrentStatePool:
    """Per-rank recurrent and convolution state indexed by scheduler slots."""

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        conv_channels: int,
        conv_kernel_size: int,
        *,
        device: torch.device | str,
    ) -> None:
        if min(num_layers, num_slots, num_heads, key_dim, value_dim) <= 0:
            raise ValueError("recurrent state dimensions must be positive")
        if conv_channels <= 0 or conv_kernel_size <= 0:
            raise ValueError("convolution state dimensions must be positive")
        self.recurrent = torch.zeros(
            num_layers,
            num_slots,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
        self.convolution = torch.zeros(
            num_layers,
            num_slots,
            conv_channels,
            conv_kernel_size,
            dtype=torch.float32,
            device=device,
        )

    def get(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.recurrent[layer, slots], self.convolution[layer, slots]

    def update(
        self,
        layer: int,
        slots: torch.Tensor,
        recurrent: torch.Tensor,
        convolution: torch.Tensor,
    ) -> None:
        self.recurrent[layer, slots] = recurrent
        self.convolution[layer, slots] = convolution

    def reset(self, slots: torch.Tensor) -> None:
        self.recurrent[:, slots] = 0
        self.convolution[:, slots] = 0
