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


def causal_conv1d_prefill(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized depthwise causal convolution for a prefill batch.

    The initial state contains the last ``kernel`` raw inputs. Dropping its
    oldest entry before appending the new sequence creates exactly one valid
    convolution window per new token.
    """

    if x.ndim != 3 or state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid causal convolution tensor rank")
    if state.shape[0] != x.shape[0] or state.shape[1:] != weight.shape:
        raise ValueError("causal convolution shapes are inconsistent")
    if x.shape[2] != weight.shape[0]:
        raise ValueError("causal convolution channel dimensions differ")
    if x.shape[1] == 0:
        return x.new_empty(x.shape), state

    history = torch.cat((state[..., 1:], x.transpose(1, 2)), dim=-1)
    output = F.conv1d(
        history.to(weight.dtype),
        weight.unsqueeze(1),
        bias=bias,
        groups=weight.shape[0],
    )
    next_state = history[..., -weight.shape[1] :]
    return F.silu(output.transpose(1, 2)).to(x.dtype), next_state


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


def recurrent_gated_delta_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one decode step, broadcasting key heads over value-head groups."""

    if query.shape != key.shape or query.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value shapes are inconsistent")
    if query.shape[0] != value.shape[0]:
        raise ValueError("query/key and value batch dimensions must match")
    if query.shape[1] <= 0 or value.shape[1] % query.shape[1]:
        raise ValueError("value heads must be a multiple of key heads")
    if log_decay.shape != value.shape[:2] or beta.shape != value.shape[:2]:
        raise ValueError("decay and beta must have shape [batch, value_heads]")
    expected_state_shape = (
        value.shape[0],
        value.shape[1],
        query.shape[2],
        value.shape[2],
    )
    if tuple(state.shape) != expected_state_shape:
        raise ValueError(
            f"invalid recurrent state shape: {tuple(state.shape)}; "
            f"expected {expected_state_shape}"
        )

    batch_size, key_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[1:]
    groups = value_heads // key_heads
    grouped_state = state.float().reshape(
        batch_size,
        key_heads,
        groups,
        key_dim,
        value_dim,
    )
    normalized_query = (
        l2_normalize(query.float()).unsqueeze(2) / (key_dim**0.5)
    )
    normalized_key = l2_normalize(key.float()).unsqueeze(2)
    grouped_decay = log_decay.float().reshape(batch_size, key_heads, groups)
    grouped_beta = beta.float().reshape(batch_size, key_heads, groups)
    grouped_value = value.float().reshape(
        batch_size,
        key_heads,
        groups,
        value_dim,
    )
    next_state = grouped_state * grouped_decay.exp()[..., None, None]
    prediction = (next_state * normalized_key.unsqueeze(-1)).sum(dim=-2)
    correction = (grouped_value - prediction) * grouped_beta.unsqueeze(-1)
    next_state = (
        next_state
        + normalized_key.unsqueeze(-1) * correction.unsqueeze(-2)
    )
    output = (next_state * normalized_query.unsqueeze(-1)).sum(dim=-2)
    return (
        output.reshape(batch_size, value_heads, value_dim).to(value.dtype),
        next_state.reshape(expected_state_shape),
    )


def effective_chunk_size(sequence_length: int, maximum: int) -> int:
    """Avoid padding short DeltaNet prefills to the full chunk size."""

    if sequence_length <= 0 or maximum <= 0:
        raise ValueError("sequence length and maximum chunk size must be positive")
    next_power_of_two = 1 << (sequence_length - 1).bit_length()
    return min(maximum, next_power_of_two)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrix-based official Qwen3.5 prefill reference.

    Work inside each chunk is parallelized into matrix operations; only the
    much shorter inter-chunk state dependency remains sequential.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if query.shape != key.shape or query.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value shapes are inconsistent")
    if query.shape[:2] != value.shape[:2]:
        raise ValueError("query/key and value batch and sequence must match")
    if query.shape[2] <= 0 or value.shape[2] % query.shape[2]:
        raise ValueError("value heads must be a multiple of key heads")
    if log_decay.shape != value.shape[:3] or beta.shape != value.shape[:3]:
        raise ValueError(
            "decay and beta must have shape [batch, sequence, value_heads]"
        )
    if query.shape[1] == 0:
        expected_state_shape = (
            query.shape[0],
            value.shape[2],
            query.shape[3],
            value.shape[3],
        )
        if initial_state is None:
            state = torch.zeros(
                expected_state_shape,
                dtype=torch.float32,
                device=query.device,
            )
        elif tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"invalid recurrent state shape: {tuple(initial_state.shape)}; "
                f"expected {expected_state_shape}"
            )
        else:
            state = initial_state.float()
        return value.new_empty(value.shape), state

    chunk_size = effective_chunk_size(query.shape[1], chunk_size)

    input_dtype = query.dtype
    batch_size, sequence_length, key_heads, key_dim = query.shape
    value_heads = value.shape[2]
    groups = value_heads // key_heads
    value_dim = value.shape[-1]
    query, key = (
        tensor.transpose(1, 2).float().contiguous()
        for tensor in (query, key)
    )
    value = value.transpose(1, 2).float().reshape(
        batch_size,
        key_heads,
        groups,
        sequence_length,
        value_dim,
    )
    beta = beta.transpose(1, 2).float().reshape(
        batch_size,
        key_heads,
        groups,
        sequence_length,
    )
    decay = log_decay.transpose(1, 2).float().reshape_as(beta)
    query = l2_normalize(query) / (key_dim**0.5)
    key = l2_normalize(key)

    pad_size = (-sequence_length) % chunk_size
    query, key = (
        F.pad(tensor, (0, 0, 0, pad_size)) for tensor in (query, key)
    )
    value = F.pad(value, (0, 0, 0, pad_size))
    beta, decay = (
        F.pad(tensor, (0, pad_size)) for tensor in (beta, decay)
    )
    total_length = sequence_length + pad_size
    num_chunks = total_length // chunk_size
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key.unsqueeze(2) * beta.unsqueeze(-1)
    query, key = (
        tensor.reshape(
            batch_size,
            key_heads,
            num_chunks,
            chunk_size,
            tensor.shape[-1],
        )
        .unsqueeze(2)
        for tensor in (query, key)
    )
    key_beta, value_beta = (
        tensor.reshape(
            batch_size,
            key_heads,
            groups,
            num_chunks,
            chunk_size,
            tensor.shape[-1],
        )
        for tensor in (key_beta, value_beta)
    )
    decay = decay.reshape(
        batch_size,
        key_heads,
        groups,
        num_chunks,
        chunk_size,
    )
    upper_mask = torch.ones(
        chunk_size,
        chunk_size,
        dtype=torch.bool,
        device=query.device,
    ).triu(1)
    cumulative_decay = decay.cumsum(dim=-1)
    pairwise_decay = (
        cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)
    )
    if pairwise_decay.requires_grad:
        pairwise_decay = pairwise_decay.masked_fill(
            upper_mask,
            float("-inf"),
        ).exp()
    else:
        pairwise_decay.masked_fill_(upper_mask, float("-inf")).exp_()
    transform = key_beta @ key.transpose(-1, -2)
    intra_attention = query @ key.transpose(-1, -2)
    if transform.requires_grad:
        transform = transform * pairwise_decay
    else:
        transform.mul_(pairwise_decay)
    # The query matmul keeps a singleton group dimension and relies on
    # broadcasting across value-head groups, so it cannot be multiplied
    # in-place unless the shapes already match.
    if (
        intra_attention.requires_grad
        or intra_attention.shape != pairwise_decay.shape
    ):
        intra_attention = intra_attention * pairwise_decay
    else:
        intra_attention.mul_(pairwise_decay)
    key_beta_scale = cumulative_decay.exp().unsqueeze(-1)
    if key_beta.requires_grad:
        decayed_key_beta = key_beta * key_beta_scale
    else:
        key_beta.mul_(key_beta_scale)
        decayed_key_beta = key_beta
    new_values = torch.linalg.solve_triangular(
        transform,
        value_beta,
        upper=False,
        unitriangular=True,
    )
    cumulative_keys = torch.linalg.solve_triangular(
        transform,
        decayed_key_beta,
        upper=False,
        unitriangular=True,
    )
    if initial_state is None:
        state = torch.zeros(
            batch_size,
            value_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected_state_shape = (
            batch_size,
            value_heads,
            key_dim,
            value_dim,
        )
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"invalid recurrent state shape: {tuple(initial_state.shape)}; "
                f"expected {expected_state_shape}"
            )
        state = initial_state.float()
    state = state.reshape(batch_size, key_heads, groups, key_dim, value_dim)
    # In inference ``new_values`` is consumed one chunk at a time, so its
    # storage can hold the corresponding output after the correction has been
    # formed. Keep a separate buffer only for autograd, whose backward graph
    # may still need the original solve result.
    output = (
        torch.empty_like(new_values)
        if new_values.requires_grad
        else new_values
    )
    query = query * cumulative_decay.exp().unsqueeze(-1)
    key = key * (
        cumulative_decay[..., -1:] - cumulative_decay
    ).exp().unsqueeze(-1)
    chunk_decay = cumulative_decay[..., -1].exp()[..., None, None]
    for chunk_index in range(num_chunks):
        corrected_value = (
            new_values[:, :, :, chunk_index]
            - cumulative_keys[:, :, :, chunk_index] @ state
        )
        output[:, :, :, chunk_index] = (
            query[:, :, :, chunk_index] @ state
            + intra_attention[:, :, :, chunk_index] @ corrected_value
        )
        state = (
            state * chunk_decay[:, :, :, chunk_index]
            + key[:, :, :, chunk_index].transpose(-1, -2) @ corrected_value
        )
    output = output.reshape(batch_size, value_heads, total_length, value_dim)
    output = output[:, :, :sequence_length].transpose(1, 2).contiguous()
    return output.to(input_dtype), state.reshape(
        batch_size,
        value_heads,
        key_dim,
        value_dim,
    )


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
        self.recurrent[layer, slots] = recurrent.to(self.recurrent.dtype)
        self.convolution[layer, slots] = convolution.to(self.convolution.dtype)

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
        hidden_float = hidden_states.float()
        normalized = hidden_float * torch.rsqrt(
            hidden_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
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
        self.in_proj_qkv.weight.safetensors_loader = self._load_qkv_slice
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.local_value_dim, bias=False
        )
        self.in_proj_z.weight.weight_loader = self._load_column
        self.in_proj_z.weight.safetensors_loader = self._load_column_slice
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_b.weight.weight_loader = self._load_column
        self.in_proj_b.weight.safetensors_loader = self._load_column_slice
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_a.weight.weight_loader = self._load_column
        self.in_proj_a.weight.safetensors_loader = self._load_column_slice
        self.conv1d = nn.Conv1d(
            self.local_conv_dim,
            self.local_conv_dim,
            self.conv_kernel_size,
            groups=self.local_conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self._load_conv
        self.conv1d.weight.safetensors_loader = self._load_conv_slice
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias.weight_loader = self._load_vector
        self.dt_bias.safetensors_loader = self._load_column_slice
        # Keep the decay exponent in FP32; BF16 can turn large learned values
        # into unstable decay factors during long recurrent scans.
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads, dtype=torch.float32)
        )
        self.A_log.weight_loader = self._load_vector
        self.A_log.safetensors_loader = self._load_column_slice
        self.norm = Qwen35GatedRMSNorm(
            self.value_head_dim,
            eps=float(config.rms_norm_eps),
        )
        self.out_proj = nn.Linear(
            self.local_value_dim, self.hidden_size, bias=False
        )
        self.out_proj.weight.weight_loader = self._load_row
        self.out_proj.weight.safetensors_loader = self._load_row_slice
        self.state_pool: Qwen35RecurrentStatePool | None = None

    def _column_shard(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[0] % self.tp_size:
            raise ValueError("weight output dimension must divide TP size")
        width = weight.shape[0] // self.tp_size
        return weight.narrow(0, self.tp_rank * width, width)

    @staticmethod
    def _slice_shape(weight) -> tuple[int, ...]:
        get_shape = getattr(weight, "get_shape", None)
        return tuple(get_shape() if get_shape is not None else weight.shape)

    def _column_bounds(self, size: int) -> tuple[int, int]:
        if size % self.tp_size:
            raise ValueError("weight output dimension must divide TP size")
        width = size // self.tp_size
        start = self.tp_rank * width
        return start, start + width

    def _load_column(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(self._column_shard(weight))

    def _load_column_slice(self, param: nn.Parameter, weight) -> None:
        shape = self._slice_shape(weight)
        if not shape:
            raise ValueError("invalid tensor-parallel column weight shape")
        start, end = self._column_bounds(shape[0])
        if tuple(param.shape) != (end - start, *shape[1:]):
            raise ValueError("invalid tensor-parallel column weight shape")
        index = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
        param.data.copy_(weight[index])

    def _load_vector(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(self._column_shard(weight))

    def _load_row(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        if weight.shape[1] % self.tp_size:
            raise ValueError("weight input dimension must divide TP size")
        width = weight.shape[1] // self.tp_size
        param.data.copy_(weight.narrow(1, self.tp_rank * width, width))

    def _load_row_slice(self, param: nn.Parameter, weight) -> None:
        shape = self._slice_shape(weight)
        if len(shape) != 2 or shape[0] != param.shape[0]:
            raise ValueError("invalid tensor-parallel row weight shape")
        start, end = self._column_bounds(shape[1])
        if end - start != param.shape[1]:
            raise ValueError("invalid tensor-parallel row weight shape")
        param.data.copy_(weight[:, start:end])

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

    def _load_qkv_slice(self, param: nn.Parameter, weight) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        shape = self._slice_shape(weight)
        if shape != (expected, self.hidden_size):
            raise ValueError("invalid Qwen3.5 in_proj_qkv weight shape")
        parts = []
        offset = 0
        for width in (
            self.global_key_dim,
            self.global_key_dim,
            self.global_value_dim,
        ):
            start, end = self._column_bounds(width)
            parts.append(weight[offset + start : offset + end, :])
            offset += width
        param.data.copy_(torch.cat(parts, dim=0))

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

    def _load_conv_slice(self, param: nn.Parameter, weight) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        shape = self._slice_shape(weight)
        if shape != (expected, 1, self.conv_kernel_size):
            raise ValueError("invalid Qwen3.5 depthwise convolution weight shape")
        parts = []
        offset = 0
        for width in (
            self.global_key_dim,
            self.global_key_dim,
            self.global_value_dim,
        ):
            start, end = self._column_bounds(width)
            parts.append(weight[offset + start : offset + end, :, :])
            offset += width
        param.data.copy_(torch.cat(parts, dim=0))

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
    def _prefill_ranges(context) -> list[tuple[int, int]]:
        return list(context.state_token_ranges)

    def _decode_batch(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        assert self.state_pool is not None
        recurrent_state, conv_state = self.state_pool.get(0, slots)
        convolved, conv_state = causal_conv1d_step(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
        )
        convolved = convolved.unsqueeze(1)
        query, key, value = convolved.split(
            (self.local_key_dim, self.local_key_dim, self.local_value_dim),
            dim=-1,
        )
        batch_size = mixed_qkv.shape[0]
        query = query.view(batch_size, 1, self.num_k_heads, self.key_head_dim)
        key = key.view(batch_size, 1, self.num_k_heads, self.key_head_dim)
        value = value.view(batch_size, 1, self.num_v_heads, self.value_head_dim)
        output, recurrent_state = recurrent_gated_delta_step(
            query.squeeze(1),
            key.squeeze(1),
            value.squeeze(1),
            log_decay,
            beta,
            recurrent_state,
        )
        self.state_pool.update(0, slots, recurrent_state, conv_state)
        return self.norm(
            output.reshape(-1, self.value_head_dim),
            z.reshape(-1, self.value_head_dim),
        ).reshape(batch_size, self.local_value_dim)

    def _prefill_batch(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        assert self.state_pool is not None
        recurrent_state, conv_state = self.state_pool.get(0, slots)
        convolved, conv_state = causal_conv1d_prefill(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
        )
        query, key, value = convolved.split(
            (self.local_key_dim, self.local_key_dim, self.local_value_dim),
            dim=-1,
        )
        batch_size, sequence_length = mixed_qkv.shape[:2]
        query = query.view(
            batch_size, sequence_length, self.num_k_heads, self.key_head_dim
        )
        key = key.view(
            batch_size, sequence_length, self.num_k_heads, self.key_head_dim
        )
        value = value.view(
            batch_size, sequence_length, self.num_v_heads, self.value_head_dim
        )
        output, recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            log_decay,
            beta,
            recurrent_state,
        )
        self.state_pool.update(0, slots, recurrent_state, conv_state)
        return self.norm(
            output.reshape(-1, self.value_head_dim),
            z.reshape(-1, self.value_head_dim),
        ).reshape(batch_size, sequence_length, self.local_value_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nanovllm.utils.context import get_context

        if self.state_pool is None:
            raise RuntimeError("Qwen3.5 recurrent state cache is not allocated")
        context = get_context()
        if context.state_slots is None:
            raise RuntimeError("Qwen3.5 execution context has no state slots")
        ranges = self._prefill_ranges(context)
        decode_count = (
            context.decode_token_count
            if context.is_mixed
            else (0 if context.is_prefill else hidden_states.shape[0])
        )
        if len(ranges) + decode_count != context.state_slots.numel():
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
        if decode_count:
            decode_slots = context.state_slots[:decode_count].to(torch.long)
            outputs[:decode_count] = self._decode_batch(
                mixed_qkv[:decode_count],
                z[:decode_count],
                beta[:decode_count],
                log_decay[:decode_count],
                decode_slots,
            )

        prefill_groups = getattr(context, "state_prefill_groups", None)
        if prefill_groups is None:
            ranges_by_length: dict[int, list[tuple[int, int, int]]] = {}
            for range_index, (start, end) in enumerate(ranges):
                ranges_by_length.setdefault(end - start, []).append(
                    (start, end, decode_count + range_index)
                )
            prefill_groups = tuple(
                (
                    sequence_length,
                    tuple(group),
                    context.state_slots.index_select(
                        0,
                        context.state_slots.new_tensor(
                            [slot_index for _, _, slot_index in group],
                            dtype=torch.long,
                        ),
                    ).to(torch.long),
                )
                for sequence_length, group in ranges_by_length.items()
            )
        for sequence_length, group, group_slots in prefill_groups:
            group_qkv = torch.stack(
                [mixed_qkv[start:end] for start, end, _ in group]
            )
            group_z = torch.stack([z[start:end] for start, end, _ in group])
            group_beta = torch.stack(
                [beta[start:end] for start, end, _ in group]
            )
            group_decay = torch.stack(
                [log_decay[start:end] for start, end, _ in group]
            )
            group_output = self._prefill_batch(
                group_qkv,
                group_z,
                group_beta,
                group_decay,
                group_slots,
            )
            for batch_index, (start, end, _) in enumerate(group):
                outputs[start:end] = group_output[batch_index, :sequence_length]

        projected = self.out_proj(outputs)
        if self.tp_size > 1:
            dist.all_reduce(projected)
        return projected
