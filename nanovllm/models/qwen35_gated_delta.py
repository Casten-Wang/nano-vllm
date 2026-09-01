"""Correctness-first state primitives for Qwen3.5 Gated DeltaNet."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


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
        recurrent_dtype: torch.dtype = torch.float32,
        convolution_dtype: torch.dtype = torch.float32,
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
            dtype=recurrent_dtype,
            device=device,
        )
        self.convolution = torch.zeros(
            num_layers,
            num_slots,
            conv_channels,
            conv_kernel_size,
            dtype=convolution_dtype,
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


class Qwen35GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # The official checkpoint intentionally stores this weight in FP32.
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (
            normalized.to(input_dtype)
            * self.weight
            * F.silu(gate.float())
        ).to(input_dtype)


class Qwen35GatedDeltaNet(nn.Module):
    """Tensor-parallel Qwen3.5 linear-attention reference layer.

    This path prioritizes official numerical semantics. It deliberately scans
    each packed sequence independently and is the oracle for later fused
    prefill and decode kernels.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.total_k_heads = int(config.linear_num_key_heads)
        self.total_v_heads = int(config.linear_num_value_heads)
        self.key_head_dim = int(config.linear_key_head_dim)
        self.value_head_dim = int(config.linear_value_head_dim)
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        if self.total_k_heads % self.tp_size or self.total_v_heads % self.tp_size:
            raise ValueError("linear attention heads must divide tensor parallel size")
        self.num_k_heads = self.total_k_heads // self.tp_size
        self.num_v_heads = self.total_v_heads // self.tp_size
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("value heads must be a multiple of key heads")
        self.local_key_dim = self.num_k_heads * self.key_head_dim
        self.local_value_dim = self.num_v_heads * self.value_head_dim
        self.local_conv_dim = 2 * self.local_key_dim + self.local_value_dim
        self.global_key_dim = self.total_k_heads * self.key_head_dim
        self.global_value_dim = self.total_v_heads * self.value_head_dim

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.local_conv_dim, bias=False
        )
        self.in_proj_qkv.weight.weight_loader = self._load_qkv
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.local_value_dim, bias=False
        )
        self.in_proj_z.weight.weight_loader = self._load_column
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_b.weight.weight_loader = self._load_column
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_a.weight.weight_loader = self._load_column
        self.conv1d = nn.Conv1d(
            self.local_conv_dim,
            self.local_conv_dim,
            self.conv_kernel_size,
            groups=self.local_conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self._load_conv
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias.weight_loader = self._load_vector
        # Keep the decay exponent in FP32; BF16 can turn large learned values
        # into unstable decay factors during long recurrent scans.
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads, dtype=torch.float32)
        )
        self.A_log.weight_loader = self._load_vector
        self.norm = Qwen35GatedRMSNorm(
            self.value_head_dim,
            eps=float(config.rms_norm_eps),
        )
        self.out_proj = nn.Linear(
            self.local_value_dim, self.hidden_size, bias=False
        )
        self.out_proj.weight.weight_loader = self._load_row
        self.state_pool: Qwen35RecurrentStatePool | None = None

    def _column_shard(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[0] % self.tp_size:
            raise ValueError("weight output dimension must divide TP size")
        width = weight.shape[0] // self.tp_size
        return weight.narrow(0, self.tp_rank * width, width)

    def _load_column(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(self._column_shard(weight))

    def _load_vector(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(self._column_shard(weight))

    def _load_row(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        if weight.shape[1] % self.tp_size:
            raise ValueError("weight input dimension must divide TP size")
        width = weight.shape[1] // self.tp_size
        param.data.copy_(weight.narrow(1, self.tp_rank * width, width))

    def _load_qkv(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        if tuple(weight.shape) != (expected, self.hidden_size):
            raise ValueError("invalid Qwen3.5 in_proj_qkv weight shape")
        query, key, value = weight.split(
            (self.global_key_dim, self.global_key_dim, self.global_value_dim),
            dim=0,
        )
        param.data.copy_(
            torch.cat(
                tuple(self._column_shard(part) for part in (query, key, value)),
                dim=0,
            )
        )

    def _load_conv(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        if weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError("invalid Qwen3.5 depthwise convolution weight shape")
        flat = weight.squeeze(1)
        expected = 2 * self.global_key_dim + self.global_value_dim
        if tuple(flat.shape) != (expected, self.conv_kernel_size):
            raise ValueError("invalid Qwen3.5 depthwise convolution weight shape")
        query, key, value = flat.split(
            (self.global_key_dim, self.global_key_dim, self.global_value_dim),
            dim=0,
        )
        local = torch.cat(
            tuple(self._column_shard(part) for part in (query, key, value)),
            dim=0,
        )
        param.data.copy_(local.unsqueeze(1))

    def allocate_state_cache(
        self,
        num_slots: int,
        device: torch.device | str,
        *,
        recurrent_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.state_pool = Qwen35RecurrentStatePool(
            1,
            num_slots,
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.local_conv_dim,
            self.conv_kernel_size,
            device=device,
            recurrent_dtype=recurrent_dtype,
            convolution_dtype=self.in_proj_qkv.weight.dtype,
        )

    @staticmethod
    def _sequence_ranges(context, token_count: int) -> list[tuple[int, int]]:
        if context.is_mixed:
            ranges = [(index, index + 1) for index in range(context.decode_token_count)]
            offsets = context.prefill_cu_seqlens_q.tolist()
            ranges.extend(
                (context.decode_token_count + start, context.decode_token_count + end)
                for start, end in zip(offsets, offsets[1:])
            )
            return ranges
        if context.is_prefill:
            offsets = context.cu_seqlens_q.tolist()
            return list(zip(offsets, offsets[1:]))
        return [(index, index + 1) for index in range(token_count)]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nanovllm.utils.context import get_context

        if self.state_pool is None:
            raise RuntimeError("Qwen3.5 recurrent state cache is not allocated")
        context = get_context()
        if context.state_slots is None:
            raise RuntimeError("Qwen3.5 execution context has no state slots")
        ranges = self._sequence_ranges(context, hidden_states.shape[0])
        slots = context.state_slots.tolist()
        if len(ranges) != len(slots):
            raise RuntimeError("sequence ranges and recurrent state slots differ")
        if context.state_reset_mask is not None:
            reset_slots = context.state_slots[context.state_reset_mask]
            if reset_slots.numel():
                self.state_pool.reset(reset_slots.to(torch.long))

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        beta = torch.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        log_decay = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias
        )
        outputs = torch.empty_like(z)
        repeat_factor = self.num_v_heads // self.num_k_heads

        for (start, end), slot in zip(ranges, slots):
            slot_tensor = context.state_slots.new_tensor([slot], dtype=torch.long)
            recurrent_state, conv_state = self.state_pool.get(0, slot_tensor)
            convolved, conv_state = causal_conv1d_scan(
                mixed_qkv[start:end].unsqueeze(0),
                conv_state,
                self.conv1d.weight.squeeze(1),
            )
            query, key, value = convolved.split(
                (self.local_key_dim, self.local_key_dim, self.local_value_dim),
                dim=-1,
            )
            query = query.view(1, end - start, self.num_k_heads, self.key_head_dim)
            key = key.view(1, end - start, self.num_k_heads, self.key_head_dim)
            value = value.view(1, end - start, self.num_v_heads, self.value_head_dim)
            if repeat_factor > 1:
                query = query.repeat_interleave(repeat_factor, dim=2)
                key = key.repeat_interleave(repeat_factor, dim=2)
            output, recurrent_state = recurrent_gated_delta_rule(
                query,
                key,
                value,
                log_decay[start:end].unsqueeze(0),
                beta[start:end].unsqueeze(0),
                recurrent_state,
            )
            gated = self.norm(
                output.reshape(-1, self.value_head_dim),
                z[start:end].reshape(-1, self.value_head_dim),
            )
            outputs[start:end] = gated.reshape(end - start, self.local_value_dim)
            self.state_pool.update(0, slot_tensor, recurrent_state, conv_state)

        projected = self.out_proj(outputs)
        if self.tp_size > 1:
            dist.all_reduce(projected)
        return projected
