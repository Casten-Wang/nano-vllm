"""Correctness-first state primitives for Qwen3.5 Gated DeltaNet."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.linear import MergedColumnParallelLinear


_MAX_CACHED_CAUSAL_MASK_SIZE = 1024


def _make_causal_upper_mask(
    chunk_size: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    # A cache miss can occur during CUDA Graph warmup under inference_mode.
    # Create a regular tensor so the same immutable mask remains safe if the
    # process later runs an autograd-enabled correctness or training path.
    with torch.inference_mode(False):
        return torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=device,
        ).triu_(1)


@lru_cache(maxsize=32)
def _cached_causal_upper_mask(
    chunk_size: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    """Share immutable DeltaNet masks across layers on the same device."""

    return _make_causal_upper_mask(chunk_size, device_type, device_index)


def causal_upper_mask(chunk_size: int, device: torch.device) -> torch.Tensor:
    """Return a bounded, device-local causal mask for chunked DeltaNet."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_size > _MAX_CACHED_CAUSAL_MASK_SIZE:
        return _make_causal_upper_mask(
            chunk_size,
            device.type,
            device.index,
        )
    return _cached_causal_upper_mask(
        chunk_size,
        device.type,
        device.index,
    )


def l2_normalize(
    x: torch.Tensor,
    eps: float = 1e-6,
    *,
    inplace_output: bool = False,
) -> torch.Tensor:
    """Match the FLA/Qwen3.5 L2 normalization convention."""

    inverse_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    if inplace_output and not torch.is_grad_enabled():
        return x.mul_(inverse_norm)
    return x * inverse_norm


def causal_conv1d_step(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    inplace_state: bool = False,
    inplace_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one depthwise causal-convolution step with optional storage reuse.

    Args:
        x: Current input with shape ``[batch, channels]``.
        state: Previous raw inputs with shape ``[batch, channels, kernel]``.
        weight: Depthwise weights with shape ``[channels, kernel]``.
    """

    if x.ndim != 2 or state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid causal convolution tensor rank")
    if state.shape[0] != x.shape[0] or state.shape[1:] != weight.shape:
        raise ValueError("causal convolution shapes are inconsistent")
    if inplace_state and not torch.is_grad_enabled():
        state[..., :-1].copy_(state[..., 1:])
        state[..., -1].copy_(x)
        next_state = state
    else:
        next_state = torch.cat((state[..., 1:], x.unsqueeze(-1)), dim=-1)
    weighted_state = next_state.to(weight.dtype) * weight.unsqueeze(0)
    can_reuse_input = (
        inplace_output
        and not torch.is_grad_enabled()
        and x.dtype == weight.dtype
    )
    if can_reuse_input:
        # ``x`` has already been copied into ``next_state`` and has no later
        # consumer in decode. Reuse it for both the reduction and activation.
        torch.sum(weighted_state, dim=-1, out=x)
        if bias is not None:
            x.add_(bias)
        return F.silu(x, inplace=True), next_state
    output = weighted_state.sum(dim=-1)
    if bias is not None:
        output = output + bias
    return F.silu(output).to(x.dtype), next_state


def causal_conv1d_step_accumulate_(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference candidate avoiding the full weighted-state temporary.

    This deliberately remains an explicit candidate until CUDA benchmarks show
    that its small sequence of pointwise launches beats the allocating baseline.
    Both ``x`` and ``state`` are consumed and updated in place.
    """

    if torch.is_grad_enabled():
        raise RuntimeError("in-place convolution accumulation is inference-only")
    if x.ndim != 2 or state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid causal convolution tensor rank")
    if state.shape[0] != x.shape[0] or state.shape[1:] != weight.shape:
        raise ValueError("causal convolution shapes are inconsistent")
    if x.dtype != state.dtype or x.dtype != weight.dtype:
        raise ValueError("in-place convolution tensors must have the same dtype")

    state[..., :-1].copy_(state[..., 1:])
    state[..., -1].copy_(x)
    x.copy_(state[..., 0]).mul_(weight[:, 0])
    for offset in range(1, weight.shape[1]):
        x.addcmul_(state[..., offset], weight[:, offset])
    if bias is not None:
        x.add_(bias)
    return F.silu(x, inplace=True), state


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
    *,
    inplace_state: bool = False,
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
    # Detach the tiny recurrent tail from the full prefill history. Returning
    # a view would keep the entire [batch, channels, sequence] allocation live
    # until the later DeltaNet state update finishes. In inference the gathered
    # input state has no later consumer, so reuse it instead of allocating a
    # second compact state tensor.
    history_tail = history[..., -weight.shape[1] :]
    if inplace_state and not torch.is_grad_enabled():
        state.copy_(history_tail)
        next_state = state
    else:
        next_state = history_tail.clone()
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
    *,
    inplace_state: bool = False,
    inplace_decay: bool = False,
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
    query_float = query.float()
    key_float = key.float()
    reuse_query = query_float is not query and not torch.is_grad_enabled()
    reuse_key = key_float is not key and not torch.is_grad_enabled()
    normalized_query = l2_normalize(
        query_float,
        inplace_output=reuse_query,
    ).unsqueeze(2)
    if normalized_query.requires_grad:
        normalized_query = normalized_query / (key_dim**0.5)
    else:
        normalized_query.div_(key_dim**0.5)
    normalized_key = l2_normalize(
        key_float,
        inplace_output=reuse_key,
    ).unsqueeze(2)
    grouped_decay = log_decay.float().reshape(batch_size, key_heads, groups)
    grouped_beta = beta.float().reshape(batch_size, key_heads, groups)
    grouped_value = value.float().reshape(
        batch_size,
        key_heads,
        groups,
        value_dim,
    )
    reuse_state = inplace_state and not torch.is_grad_enabled()
    reuse_decay = (
        inplace_decay
        and not torch.is_grad_enabled()
        and grouped_decay.data_ptr() == log_decay.data_ptr()
    )
    decay_factor = grouped_decay.exp_() if reuse_decay else grouped_decay.exp()
    if reuse_state:
        grouped_state.mul_(decay_factor[..., None, None])
        next_state = grouped_state
    else:
        next_state = grouped_state * decay_factor[..., None, None]
    prediction = torch.matmul(
        normalized_key.unsqueeze(-2),
        next_state,
    ).squeeze(-2)
    if reuse_state:
        # Prediction is dead after the correction is formed. Reuse its FP32
        # storage instead of allocating another [batch, value_heads,
        # value_dim] decode temporary in every linear-attention layer.
        prediction.neg_().add_(grouped_value)
        prediction.mul_(grouped_beta.unsqueeze(-1))
        correction = prediction
    else:
        correction = (grouped_value - prediction) * grouped_beta.unsqueeze(-1)
    if reuse_state:
        next_state.addcmul_(
            normalized_key.unsqueeze(-1),
            correction.unsqueeze(-2),
        )
    else:
        next_state = (
            next_state
            + normalized_key.unsqueeze(-1) * correction.unsqueeze(-2)
        )
    output = torch.matmul(
        normalized_query.unsqueeze(-2),
        next_state,
    ).squeeze(-2)
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


def _gather_prefill_group(
    tensor: torch.Tensor,
    sequence_length: int,
    group: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Batch contiguous ranges as a view and copy only interleaved groups."""

    first_start = group[0][0]
    contiguous = all(
        start == first_start + index * sequence_length
        and end == start + sequence_length
        for index, (start, end, _) in enumerate(group)
    )
    if contiguous:
        return tensor.narrow(
            0,
            first_start,
            len(group) * sequence_length,
        ).view(len(group), sequence_length, *tensor.shape[1:])
    return torch.stack([tensor[start:end] for start, end, _ in group])


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    inplace_state: bool = True,
    materialize_decay_scaled_qk: bool = False,
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
    query = l2_normalize(
        query,
        inplace_output=not torch.is_grad_enabled(),
    )
    if not query.requires_grad:
        query.div_(key_dim**0.5)
    else:
        query = query / (key_dim**0.5)
    key = l2_normalize(
        key,
        inplace_output=not torch.is_grad_enabled(),
    )

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
    upper_mask = causal_upper_mask(chunk_size, query.device)
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
    if torch.is_grad_enabled():
        cumulative_decay_exp = cumulative_decay.exp()
        key_decay = (
            cumulative_decay[..., -1:] - cumulative_decay
        ).exp()
    else:
        # Pairwise decay is fully materialized above, so the cumulative
        # workspace can now hold its exponential. Keep the key decay separate
        # because it needs the original cumulative values.
        key_decay = cumulative_decay[..., -1:] - cumulative_decay
        key_decay.exp_()
        cumulative_decay.exp_()
        cumulative_decay_exp = cumulative_decay
    key_beta_scale = cumulative_decay_exp.unsqueeze(-1)
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
        if (
            inplace_state
            and not torch.is_grad_enabled()
            and state.data_ptr() == initial_state.data_ptr()
        ):
            # The caller still owns its recurrent state. Copy it once so the
            # chunk scan can reuse one state allocation without mutating the
            # input or allocating a replacement for every chunk. A compressed
            # input already received private storage from the fp32 conversion.
            state = state.clone()
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
    retained_decay_scaled_qk = None
    if torch.is_grad_enabled():
        query = query * cumulative_decay_exp.unsqueeze(-1)
        key = key * key_decay.unsqueeze(-1)
    elif materialize_decay_scaled_qk:
        # Benchmark-only allocation baseline: retain the former expanded Q/K
        # workspaces while the optimized contractions run unchanged.
        retained_decay_scaled_qk = (
            query * cumulative_decay_exp.unsqueeze(-1),
            key * key_decay.unsqueeze(-1),
        )
    chunk_decay = cumulative_decay_exp[..., -1, None, None]
    # The solves and decay projections no longer need these large
    # intermediates. Dropping the Python references before the sequential scan
    # lets the inference allocator reuse their storage for per-chunk results.
    del (
        decayed_key_beta,
        decay,
        key_beta,
        key_beta_scale,
        pairwise_decay,
        transform,
        upper_mask,
        value_beta,
    )
    reuse_state = inplace_state and not torch.is_grad_enabled()
    for chunk_index in range(num_chunks):
        chunk_values = new_values[:, :, :, chunk_index]
        state_prediction = cumulative_keys[:, :, :, chunk_index] @ state
        if torch.is_grad_enabled():
            corrected_value = chunk_values - state_prediction
        else:
            # ``output`` already aliases ``new_values`` during inference.
            # Reuse the current chunk for its correction instead of keeping
            # another [batch, value_heads, chunk, value_dim] allocation live.
            chunk_values.sub_(state_prediction)
            corrected_value = chunk_values
        if torch.is_grad_enabled():
            state_update = (
                key[:, :, :, chunk_index].transpose(-1, -2)
                @ corrected_value
            )
            output[:, :, :, chunk_index] = (
                query[:, :, :, chunk_index] @ state
                + intra_attention[:, :, :, chunk_index] @ corrected_value
            )
        else:
            # Applying decay after the query contraction avoids expanding Q
            # from one key-head group to every value-head group. Materialize
            # the intra-chunk contraction before reusing the correction buffer
            # for key decay and the state update.
            intra_output = (
                intra_attention[:, :, :, chunk_index] @ corrected_value
            )
            corrected_value.mul_(
                key_decay[:, :, :, chunk_index].unsqueeze(-1)
            )
            state_update = (
                key[:, :, :, chunk_index].transpose(-1, -2)
                @ corrected_value
            )
            chunk_output = query[:, :, :, chunk_index] @ state
            chunk_output.mul_(
                cumulative_decay_exp[:, :, :, chunk_index].unsqueeze(-1)
            )
            chunk_output.add_(intra_output)
            output[:, :, :, chunk_index] = chunk_output
        if not reuse_state:
            state = (
                state * chunk_decay[:, :, :, chunk_index] + state_update
            )
        else:
            state.mul_(chunk_decay[:, :, :, chunk_index]).add_(state_update)
    del retained_decay_scaled_qk
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

    def get_contiguous(
        self,
        layer: int,
        start: int,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start < 0 or count <= 0 or start + count > self.recurrent.shape[1]:
            raise ValueError("contiguous state span is out of bounds")
        return (
            self.recurrent[layer].narrow(0, start, count),
            self.convolution[layer].narrow(0, start, count),
        )

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
        inverse_rms = torch.rsqrt(
            hidden_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        if (
            not torch.is_grad_enabled()
            and hidden_float is not hidden_states
            and gate.dtype != torch.float32
        ):
            hidden_float.mul_(inverse_rms)
            normalized = hidden_float.to(input_dtype)
            gate_float = gate.float()
            F.silu(gate_float, inplace=True)
            torch.mul(normalized, self.weight, out=hidden_float)
            hidden_float.mul_(gate_float)
            return hidden_float.to(input_dtype)
        normalized = hidden_float * inverse_rms
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
        self.in_proj_qkv.weight.fp8_safetensors_loader = self._load_qkv_fp8_slice
        # These official checkpoint projections share the same input. Pack
        # them so decode launches one GEMM instead of three small GEMMs.
        self.in_proj_zba = MergedColumnParallelLinear(
            self.hidden_size,
            [self.global_value_dim, self.total_v_heads, self.total_v_heads],
            bias=False,
        )
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
        self.register_buffer("_decay_rate", None, persistent=False)
        self.A_log.weight_loader = self._load_a_log
        self.A_log.safetensors_loader = self._load_a_log_slice
        self.norm = Qwen35GatedRMSNorm(
            self.value_head_dim,
            eps=float(config.rms_norm_eps),
        )
        self.out_proj = nn.Linear(
            self.local_value_dim, self.hidden_size, bias=False
        )
        self.out_proj.weight.weight_loader = self._load_row
        self.out_proj.weight.safetensors_loader = self._load_row_slice
        self.out_proj.weight.fp8_safetensors_loader = self._load_row_fp8_slice
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

    def _load_a_log(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        self._load_vector(param, weight)
        self._decay_rate = -param.detach().float().exp()

    def _load_a_log_slice(self, param: nn.Parameter, weight) -> None:
        self._load_column_slice(param, weight)
        self._decay_rate = -param.detach().float().exp()

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

    def _load_row_fp8_slice(
        self,
        param: nn.Parameter,
        weight,
        scale,
        block_size: tuple[int, int],
    ) -> None:
        from nanovllm.models.qwen35_fp8 import (
            dequantize_fp8_block_weight_slice,
        )

        shape = self._slice_shape(weight)
        if len(shape) != 2 or shape[0] != param.shape[0]:
            raise ValueError("invalid tensor-parallel row weight shape")
        start, end = self._column_bounds(shape[1])
        if end - start != param.shape[1]:
            raise ValueError("invalid tensor-parallel row weight shape")
        dequantize_fp8_block_weight_slice(
            weight,
            scale,
            block_size,
            (0, shape[0]),
            (start, end),
            output_dtype=param.dtype,
            out=param.data,
        )

    @staticmethod
    def _copy_packed_rows(
        param: nn.Parameter,
        parts,
    ) -> None:
        destination_offset = 0
        for part in parts:
            rows = part.shape[0]
            param.data.narrow(0, destination_offset, rows).copy_(part)
            destination_offset += rows
        if destination_offset != param.shape[0]:
            raise ValueError("packed Qwen3.5 weight rows do not match parameter")

    def _load_qkv(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        if tuple(weight.shape) != (expected, self.hidden_size):
            raise ValueError("invalid Qwen3.5 in_proj_qkv weight shape")
        query, key, value = weight.split(
            (self.global_key_dim, self.global_key_dim, self.global_value_dim),
            dim=0,
        )
        self._copy_packed_rows(
            param,
            (self._column_shard(part) for part in (query, key, value)),
        )

    def _load_qkv_slice(self, param: nn.Parameter, weight) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        shape = self._slice_shape(weight)
        if shape != (expected, self.hidden_size):
            raise ValueError("invalid Qwen3.5 in_proj_qkv weight shape")
        def local_parts():
            offset = 0
            for width in (
                self.global_key_dim,
                self.global_key_dim,
                self.global_value_dim,
            ):
                start, end = self._column_bounds(width)
                yield weight[offset + start : offset + end, :]
                offset += width

        self._copy_packed_rows(param, local_parts())

    def _load_qkv_fp8_slice(
        self,
        param: nn.Parameter,
        weight,
        scale,
        block_size: tuple[int, int],
    ) -> None:
        from nanovllm.models.qwen35_fp8 import (
            dequantize_fp8_block_weight_slice,
        )

        expected = 2 * self.global_key_dim + self.global_value_dim
        shape = self._slice_shape(weight)
        if shape != (expected, self.hidden_size):
            raise ValueError("invalid Qwen3.5 in_proj_qkv weight shape")

        destination_offset = 0
        source_offset = 0
        for width in (
            self.global_key_dim,
            self.global_key_dim,
            self.global_value_dim,
        ):
            start, end = self._column_bounds(width)
            target = param.data.narrow(
                0,
                destination_offset,
                end - start,
            )
            dequantize_fp8_block_weight_slice(
                weight,
                scale,
                block_size,
                (source_offset + start, source_offset + end),
                (0, self.hidden_size),
                output_dtype=param.dtype,
                out=target,
            )
            destination_offset += end - start
            source_offset += width
        if destination_offset != param.shape[0]:
            raise ValueError("packed Qwen3.5 weight rows do not match parameter")

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
        self._copy_packed_rows(
            param,
            (
                self._column_shard(part).unsqueeze(1)
                for part in (query, key, value)
            ),
        )

    def _load_conv_slice(self, param: nn.Parameter, weight) -> None:
        expected = 2 * self.global_key_dim + self.global_value_dim
        shape = self._slice_shape(weight)
        if shape != (expected, 1, self.conv_kernel_size):
            raise ValueError("invalid Qwen3.5 depthwise convolution weight shape")
        def local_parts():
            offset = 0
            for width in (
                self.global_key_dim,
                self.global_key_dim,
                self.global_value_dim,
            ):
                start, end = self._column_bounds(width)
                yield weight[offset + start : offset + end, :, :]
                offset += width

        self._copy_packed_rows(param, local_parts())

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
        state_span: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        assert self.state_pool is not None
        use_contiguous_state = (
            state_span is not None
            and not torch.is_grad_enabled()
        )
        if use_contiguous_state:
            recurrent_state, conv_state = self.state_pool.get_contiguous(
                0,
                *state_span,
            )
            cached_recurrent_state = recurrent_state
        else:
            recurrent_state, conv_state = self.state_pool.get(0, slots)
        convolved, conv_state = causal_conv1d_step(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            inplace_state=True,
            inplace_output=True,
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
            inplace_state=True,
            inplace_decay=True,
        )
        if use_contiguous_state:
            if recurrent_state.data_ptr() != cached_recurrent_state.data_ptr():
                cached_recurrent_state.copy_(recurrent_state)
        else:
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
        state_span: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        assert self.state_pool is not None
        use_contiguous_state = (
            state_span is not None and not torch.is_grad_enabled()
        )
        if use_contiguous_state:
            recurrent_state, conv_state = self.state_pool.get_contiguous(
                0,
                *state_span,
            )
            cached_recurrent_state = recurrent_state
            cached_conv_state = conv_state
        else:
            recurrent_state, conv_state = self.state_pool.get(0, slots)
        convolved, conv_state = causal_conv1d_prefill(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            inplace_state=True,
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
        if use_contiguous_state:
            if recurrent_state.data_ptr() != cached_recurrent_state.data_ptr():
                cached_recurrent_state.copy_(recurrent_state)
            if conv_state.data_ptr() != cached_conv_state.data_ptr():
                cached_conv_state.copy_(conv_state)
        else:
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
        reset_slots = getattr(context, "state_reset_slots", None)
        if reset_slots is None and context.state_reset_mask is not None:
            reset_slots = context.state_slots[context.state_reset_mask]
        if reset_slots is not None and reset_slots.numel():
            self.state_pool.reset(reset_slots)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z, beta, a = self.in_proj_zba(hidden_states).split(
            (self.local_value_dim, self.num_v_heads, self.num_v_heads),
            dim=-1,
        )
        if beta.requires_grad:
            beta = torch.sigmoid(beta)
        else:
            # The projection is dead after sigmoid in inference. Reuse it to
            # avoid one token_count x local_value_heads allocation per layer.
            beta.sigmoid_()
        decay_rate = (
            self._decay_rate
            if not torch.is_grad_enabled() and self._decay_rate is not None
            else -self.A_log.float().exp()
        )
        if torch.is_grad_enabled():
            log_decay = decay_rate * F.softplus(
                a.float() + self.dt_bias
            )
        else:
            # Reuse both short-lived FP32 intermediates: the converted
            # projection can absorb the bias, and the softplus output can
            # become the final log-decay tensor.
            a_float = a.float()
            a_float.add_(self.dt_bias)
            log_decay = F.softplus(a_float)
            log_decay.mul_(decay_rate)
        # The gate projection is consumed before each slice is written. During
        # inference its storage can therefore hold normalized GDN outputs and
        # avoid another token_count x local_value_dim allocation. Autograd may
        # retain the original gate values for backward, so keep it separate.
        outputs = torch.empty_like(z) if z.requires_grad else z
        if decode_count:
            decode_slots = context.state_slots[:decode_count]
            outputs[:decode_count] = self._decode_batch(
                mixed_qkv[:decode_count],
                z[:decode_count],
                beta[:decode_count],
                log_decay[:decode_count],
                decode_slots,
                getattr(context, "decode_state_span", None),
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
                    None,
                )
                for sequence_length, group in ranges_by_length.items()
            )
        for sequence_length, group, group_slots, state_span in prefill_groups:
            group_qkv = _gather_prefill_group(
                mixed_qkv, sequence_length, group
            )
            group_z = _gather_prefill_group(z, sequence_length, group)
            group_beta = _gather_prefill_group(
                beta, sequence_length, group
            )
            group_decay = _gather_prefill_group(
                log_decay, sequence_length, group
            )
            group_output = self._prefill_batch(
                group_qkv,
                group_z,
                group_beta,
                group_decay,
                group_slots,
                state_span,
            )
            for batch_index, (start, end, _) in enumerate(group):
                outputs[start:end] = group_output[batch_index, :sequence_length]

        projected = self.out_proj(outputs)
        if self.tp_size > 1:
            dist.all_reduce(projected)
        return projected
