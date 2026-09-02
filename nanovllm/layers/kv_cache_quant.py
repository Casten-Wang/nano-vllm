import torch
import triton
import triton.language as tl


def dequant_selected_kvcache_torch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    selected_block_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected INT8 blocks and dequantize in their output buffers."""

    if selected_block_ids.device != k_cache.device:
        block_ids = selected_block_ids.to(device=k_cache.device)
    else:
        block_ids = selected_block_ids
    if block_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected block ids must use int32 or int64")
    packed_k = k_cache.index_select(0, block_ids).to(dtype)
    packed_v = v_cache.index_select(0, block_ids).to(dtype)
    packed_k.mul_(k_scale.index_select(0, block_ids).unsqueeze(-1).to(dtype))
    packed_v.mul_(v_scale.index_select(0, block_ids).unsqueeze(-1).to(dtype))
    return packed_k, packed_v


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _round_away_from_zero(x):
    # Triton casts float -> int by truncating toward zero. Adding/subtracting
    # 0.5 first gives the usual nearest integer behavior for KV quantization.
    return tl.where(x >= 0.0, x + 0.5, x - 0.5)


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, tl.load(key_ptr + key_offsets))
    tl.store(v_cache_ptr + cache_offsets, tl.load(value_ptr + value_offsets))


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Store FP16/BF16 K/V tensors into the paged KV cache."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    _store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


def store_kvcache_range(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
):
    store_kvcache(
        key[start:end],
        value[start:end],
        k_cache,
        v_cache,
        slot_mapping[start:end],
    )


@triton.jit
def _store_kvcache_int8_kernel(
    key_ptr,
    key_stride_token,
    value_ptr,
    value_stride_token,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return

    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    key_offsets = token_idx * key_stride_token + kv_head_idx * HEAD_DIM + dim_offsets
    value_offsets = token_idx * value_stride_token + kv_head_idx * HEAD_DIM + dim_offsets

    key = tl.load(key_ptr + key_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + value_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # One scale per token per KV head. This is the same granularity used by
    # many practical INT8 KV cache implementations: compact, cheap to index,
    # and accurate enough because each head vector gets its own dynamic range.
    k_absmax = tl.max(tl.abs(key), axis=0)
    v_absmax = tl.max(tl.abs(value), axis=0)
    k_scale = tl.maximum(k_absmax / 127.0, EPS)
    v_scale = tl.maximum(v_absmax / 127.0, EPS)

    k_q = _round_away_from_zero(key / k_scale)
    v_q = _round_away_from_zero(value / v_scale)
    k_q = tl.minimum(tl.maximum(k_q, -127.0), 127.0).to(tl.int8)
    v_q = tl.minimum(tl.maximum(v_q, -127.0), 127.0).to(tl.int8)

    cache_offsets = slot * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM + dim_offsets
    scale_offset = slot * NUM_KV_HEADS + kv_head_idx
    tl.store(k_cache_ptr + cache_offsets, k_q, mask=dim_mask)
    tl.store(v_cache_ptr + cache_offsets, v_q, mask=dim_mask)
    tl.store(k_scale_ptr + scale_offset, k_scale)
    tl.store(v_scale_ptr + scale_offset, v_scale)


def store_kvcache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Quantize current K/V tensors and store them in INT8 paged KV cache.

    key/value shape: [num_tokens, num_kv_heads, head_dim]
    cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    scale shape: [num_blocks, block_size, num_kv_heads]
    """
    N, num_kv_heads, head_dim = key.shape
    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256
    assert value.shape == key.shape
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.shape[2:] == (num_kv_heads, head_dim)
    assert v_cache.shape[2:] == (num_kv_heads, head_dim)
    assert k_scale.shape == k_cache.shape[:3]
    assert v_scale.shape == v_cache.shape[:3]
    assert slot_mapping.numel() == N

    # The cache is contiguous in [token_slot, kv_head, head_dim] order after
    # flattening the first two dimensions. Passing k_cache[:, :, h, :] as
    # slot * num_kv_heads * head_dim + h * head_dim + dim keeps indexing simple.
    flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
    flat_k_scale = k_scale.view(-1, num_kv_heads)
    flat_v_scale = v_scale.view(-1, num_kv_heads)
    _store_kvcache_int8_kernel[(N, num_kv_heads)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        slot_mapping,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        EPS=1.0e-6,
    )


def store_kvcache_int8_range(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
):
    store_kvcache_int8(
        key[start:end],
        value[start:end],
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping[start:end],
    )


@triton.jit
def _dequant_packed_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    selected_block_ids_ptr,
    packed_k_ptr,
    packed_v_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    packed_block_id = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    token_tile_id = tl.program_id(2)

    physical_block_id = tl.load(selected_block_ids_ptr + packed_block_id)
    token_offsets = token_tile_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    token_mask = token_offsets < BLOCK_SIZE
    dim_mask = dim_offsets < HEAD_DIM
    mask = token_mask[:, None] & dim_mask[None, :]

    src_token_slots = physical_block_id * BLOCK_SIZE + token_offsets
    dst_token_slots = packed_block_id * BLOCK_SIZE + token_offsets

    scale_offsets = src_token_slots * NUM_KV_HEADS + kv_head_idx
    src_offsets = (
        src_token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
        + kv_head_idx * HEAD_DIM
        + dim_offsets[None, :]
    )
    dst_offsets = (
        dst_token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
        + kv_head_idx * HEAD_DIM
        + dim_offsets[None, :]
    )

    k_scale = tl.load(k_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
    k_q = tl.load(k_cache_ptr + src_offsets, mask=mask, other=0).to(tl.float32)
    v_q = tl.load(v_cache_ptr + src_offsets, mask=mask, other=0).to(tl.float32)

    tl.store(packed_k_ptr + dst_offsets, k_q * k_scale[:, None], mask=mask)
    tl.store(packed_v_ptr + dst_offsets, v_q * v_scale[:, None], mask=mask)


def dequant_packed_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    selected_block_ids: torch.Tensor,
    packed_k_cache: torch.Tensor,
    packed_v_cache: torch.Tensor,
    block_tokens: int = 16,
):
    """Dequantize selected physical KV blocks into compact FP16/BF16 buffers."""
    num_selected_blocks = selected_block_ids.numel()
    if num_selected_blocks == 0:
        return

    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256
    assert block_tokens > 0 and (block_tokens & (block_tokens - 1)) == 0
    assert v_cache.shape == k_cache.shape
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.shape == (num_blocks, block_size, num_kv_heads)
    assert v_scale.shape == (num_blocks, block_size, num_kv_heads)
    assert selected_block_ids.dtype == torch.int32
    assert packed_k_cache.shape == (num_selected_blocks, block_size, num_kv_heads, head_dim)
    assert packed_v_cache.shape == packed_k_cache.shape
    assert packed_k_cache.dtype in (torch.float16, torch.bfloat16)
    assert packed_v_cache.dtype == packed_k_cache.dtype

    grid = (
        num_selected_blocks,
        num_kv_heads,
        triton.cdiv(block_size, block_tokens),
    )
    _dequant_packed_kvcache_kernel[grid](
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        selected_block_ids,
        packed_k_cache,
        packed_v_cache,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        BLOCK_TOKENS=block_tokens,
    )
