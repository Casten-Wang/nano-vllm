import torch
import triton
import triton.language as tl

from nanovllm.engine.execution import partition_count


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _fused_int8_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    o_ptr,
    Q_SEQ_STRIDE,
    Q_HEAD_STRIDE,
    O_SEQ_STRIDE,
    O_HEAD_STRIDE,
    BLOCK_TABLE_STRIDE,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    Q_HEADS_PER_KV_HEAD: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    kv_head_id = q_head_id // Q_HEADS_PER_KV_HEAD

    context_len = tl.load(context_lens_ptr + seq_id)
    q_dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    q_dim_mask = q_dim_offsets < HEAD_DIM
    q_offsets = seq_id * Q_SEQ_STRIDE + q_head_id * Q_HEAD_STRIDE + q_dim_offsets
    q = tl.load(q_ptr + q_offsets, mask=q_dim_mask, other=0.0).to(tl.float32)

    # Keep the running max finite. With -inf, a fully masked tile would compute
    # exp(-inf - -inf), which can poison the online softmax with NaNs.
    m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)
    if WINDOW_SIZE > 0:
        window_start = tl.maximum(context_len - WINDOW_SIZE, 0)
    else:
        window_start = 0
    start_block = window_start // BLOCK_SIZE
    end_block = tl.minimum((context_len + BLOCK_SIZE - 1) // BLOCK_SIZE, MAX_BLOCKS)

    # One Triton program computes one request and one query head. It scans the
    # paged KV cache block by block, tile by tile. With a sliding window, begin
    # at the first block intersecting the visible window instead of scanning
    # every historical block and masking the old ones away.
    block_idx = start_block
    while block_idx < end_block:
        block_start = block_idx * BLOCK_SIZE
        block_valid_tokens = tl.maximum(tl.minimum(context_len - block_start, BLOCK_SIZE), 0)
        physical_block_id = tl.load(
            block_tables_ptr + seq_id * BLOCK_TABLE_STRIDE + block_idx,
        )

        for token_start in range(0, BLOCK_SIZE, BLOCK_TOKENS):
            token_offsets = token_start + tl.arange(0, BLOCK_TOKENS)
            token_positions = block_start + token_offsets
            token_mask = (token_offsets < block_valid_tokens) & (token_positions >= window_start)
            token_slots = physical_block_id * BLOCK_SIZE + token_offsets

            scale_offsets = token_slots * NUM_KV_HEADS + kv_head_id
            k_scale = tl.load(k_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
            v_scale = tl.load(v_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)

            kv_offsets = (
                token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
                + kv_head_id * HEAD_DIM
                + q_dim_offsets[None, :]
            )
            kv_mask = token_mask[:, None] & q_dim_mask[None, :]
            k_q = tl.load(k_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)
            v_q = tl.load(v_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)

            # Dequantize only the tile that this program is about to use.
            # k_scale/v_scale are scalar scales for each token and KV head.
            k_fp = k_q * k_scale[:, None]
            scores = tl.sum(q[None, :] * k_fp, axis=1) * SOFTMAX_SCALE
            scores = tl.where(token_mask, scores, -float("inf"))

            # Online softmax update:
            # m/l/acc hold the stable softmax state for all previous tiles.
            # new_m rescales the old accumulator before adding this tile, which
            # avoids storing all attention scores for the full context.
            tile_max = tl.max(scores, axis=0)
            new_m = tl.maximum(m, tile_max)
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m)

            v_fp = v_q * v_scale[:, None]
            acc = acc * alpha + tl.sum(p[:, None] * v_fp, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = new_m
        block_idx += 1

    out = acc / tl.maximum(l, 1.0e-20)
    o_offsets = seq_id * O_SEQ_STRIDE + q_head_id * O_HEAD_STRIDE + q_dim_offsets
    tl.store(o_ptr + o_offsets, out, mask=q_dim_mask)


@triton.jit
def _fused_int8_decode_attention_latev_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    o_ptr,
    Q_SEQ_STRIDE,
    Q_HEAD_STRIDE,
    O_SEQ_STRIDE,
    O_HEAD_STRIDE,
    BLOCK_TABLE_STRIDE,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    Q_HEADS_PER_KV_HEAD: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    kv_head_id = q_head_id // Q_HEADS_PER_KV_HEAD

    context_len = tl.load(context_lens_ptr + seq_id)
    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    q_offsets = seq_id * Q_SEQ_STRIDE + q_head_id * Q_HEAD_STRIDE + dim_offsets
    q = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # Same math as _fused_int8_decode_attention_kernel, but V is loaded only
    # after the softmax weights are known. This reduces the live range of V and
    # v_scale, which may improve occupancy on some tile/warp settings.
    m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)
    if WINDOW_SIZE > 0:
        window_start = tl.maximum(context_len - WINDOW_SIZE, 0)
    else:
        window_start = 0
    start_block = window_start // BLOCK_SIZE
    end_block = tl.minimum((context_len + BLOCK_SIZE - 1) // BLOCK_SIZE, MAX_BLOCKS)

    block_idx = start_block
    while block_idx < end_block:
        block_start = block_idx * BLOCK_SIZE
        block_valid_tokens = tl.maximum(tl.minimum(context_len - block_start, BLOCK_SIZE), 0)
        physical_block_id = tl.load(
            block_tables_ptr + seq_id * BLOCK_TABLE_STRIDE + block_idx,
        )

        for token_start in range(0, BLOCK_SIZE, BLOCK_TOKENS):
            token_offsets = token_start + tl.arange(0, BLOCK_TOKENS)
            token_positions = block_start + token_offsets
            token_mask = (token_offsets < block_valid_tokens) & (token_positions >= window_start)
            token_slots = physical_block_id * BLOCK_SIZE + token_offsets

            scale_offsets = token_slots * NUM_KV_HEADS + kv_head_id
            k_scale = tl.load(k_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)

            kv_offsets = (
                token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
                + kv_head_id * HEAD_DIM
                + dim_offsets[None, :]
            )
            kv_mask = token_mask[:, None] & dim_mask[None, :]
            k_q = tl.load(k_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)
            k_fp = k_q * k_scale[:, None]
            scores = tl.sum(q[None, :] * k_fp, axis=1) * SOFTMAX_SCALE
            scores = tl.where(token_mask, scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            new_m = tl.maximum(m, tile_max)
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m)

            v_scale = tl.load(v_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
            v_q = tl.load(v_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)
            v_fp = v_q * v_scale[:, None]
            acc = acc * alpha + tl.sum(p[:, None] * v_fp, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = new_m
        block_idx += 1

    out = acc / tl.maximum(l, 1.0e-20)
    o_offsets = seq_id * O_SEQ_STRIDE + q_head_id * O_HEAD_STRIDE + dim_offsets
    tl.store(o_ptr + o_offsets, out, mask=dim_mask)


@triton.jit
def _fused_int8_decode_attention_v3_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    o_ptr,
    Q_SEQ_STRIDE,
    Q_HEAD_STRIDE,
    O_SEQ_STRIDE,
    O_HEAD_STRIDE,
    BLOCK_TABLE_STRIDE,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    Q_HEADS_PER_KV_HEAD: tl.constexpr,
    SOFTMAX_SCALE_LOG2: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    kv_head_id = q_head_id // Q_HEADS_PER_KV_HEAD

    context_len = tl.load(context_lens_ptr + seq_id)
    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    q_offsets = seq_id * Q_SEQ_STRIDE + q_head_id * Q_HEAD_STRIDE + dim_offsets
    q = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # V3 keeps the same one-program-per-[request, q_head] structure as the
    # fastest V1 kernel, but changes the inner math:
    #   score = dot(q, int8_k) * k_scale
    #   acc   = sum((softmax_prob * v_scale) * int8_v)
    # Because scales are per-token/per-KV-head scalars, this is exactly
    # equivalent to dequantizing K/V elementwise, while avoiding two large
    # [BLOCK_TOKENS, HEAD_DIM] scale multiplies per tile.
    m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)
    if WINDOW_SIZE > 0:
        window_start = tl.maximum(context_len - WINDOW_SIZE, 0)
    else:
        window_start = 0
    start_block = window_start // BLOCK_SIZE
    end_block = tl.minimum((context_len + BLOCK_SIZE - 1) // BLOCK_SIZE, MAX_BLOCKS)

    block_idx = start_block
    while block_idx < end_block:
        block_start = block_idx * BLOCK_SIZE
        block_valid_tokens = tl.maximum(tl.minimum(context_len - block_start, BLOCK_SIZE), 0)
        physical_block_id = tl.load(
            block_tables_ptr + seq_id * BLOCK_TABLE_STRIDE + block_idx,
        )

        for token_start in range(0, BLOCK_SIZE, BLOCK_TOKENS):
            token_offsets = token_start + tl.arange(0, BLOCK_TOKENS)
            token_positions = block_start + token_offsets
            token_mask = (token_offsets < block_valid_tokens) & (token_positions >= window_start)
            token_slots = physical_block_id * BLOCK_SIZE + token_offsets

            scale_offsets = token_slots * NUM_KV_HEADS + kv_head_id
            k_scale = tl.load(k_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)

            kv_offsets = (
                token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
                + kv_head_id * HEAD_DIM
                + dim_offsets[None, :]
            )
            kv_mask = token_mask[:, None] & dim_mask[None, :]
            k_q = tl.load(k_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)

            raw_scores = tl.sum(q[None, :] * k_q, axis=1)
            scores = raw_scores * k_scale * SOFTMAX_SCALE_LOG2
            scores = tl.where(token_mask, scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            new_m = tl.maximum(m, tile_max)
            alpha = tl.exp2(m - new_m)
            p = tl.exp2(scores - new_m)

            # Load V only after the softmax weights are known. This shortens
            # the live range of the K tile and avoids materializing V_fp.
            v_scale = tl.load(v_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
            v_q = tl.load(v_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)
            weight = p * v_scale
            acc = acc * alpha + tl.sum(weight[:, None] * v_q, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = new_m
        block_idx += 1

    out = acc / tl.maximum(l, 1.0e-20)
    o_offsets = seq_id * O_SEQ_STRIDE + q_head_id * O_HEAD_STRIDE + dim_offsets
    tl.store(o_ptr + o_offsets, out, mask=dim_mask)


def fused_int8_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    sliding_window_size: int | None = None,
    block_tokens: int = 256,
    num_warps: int | None = None,
    num_stages: int = 2,
):
    """Compute decode attention directly from INT8 paged KV cache.

    This kernel fuses INT8 dequantization with online attention so the
    intermediate FP16/BF16 KV tiles never need to be written back to HBM. When
    sliding_window_size is set, each decode query only attends to the newest
    window tokens, matching flash-attn's left-window decode semantics.
    """
    assert q.ndim == 3
    assert k_cache.ndim == 4 and v_cache.ndim == 4
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert block_tables.ndim == 2
    assert context_lens.ndim == 1
    assert q.size(0) == context_lens.size(0) == block_tables.size(0)
    assert q.size(2) == k_cache.size(3) == v_cache.size(3)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert k_scale.shape == k_cache.shape[:3]
    assert v_scale.shape == v_cache.shape[:3]
    assert sliding_window_size is None or sliding_window_size > 0
    assert block_tokens > 0 and (block_tokens & (block_tokens - 1)) == 0

    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    q_heads_per_kv_head = num_heads // num_kv_heads
    assert q_heads_per_kv_head * num_kv_heads == num_heads
    assert block_tokens <= block_size
    assert block_size % block_tokens == 0

    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256

    flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
    flat_k_scale = k_scale.view(-1, num_kv_heads)
    flat_v_scale = v_scale.view(-1, num_kv_heads)
    o = torch.empty_like(q)
    # On RTX 3090 with Qwen3-0.6B-shaped decode (head_dim=128), the focused
    # sweep shows 8 warps is consistently faster than 4 warps for the default
    # bt256 fused path. Keep this overridable for benchmark ablations.
    launch_num_warps = num_warps or (4 if head_dim < 128 else 8)

    _fused_int8_decode_attention_kernel[(num_seqs, num_heads)](
        q,
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        block_tables,
        context_lens,
        o,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        block_tables.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS=block_tables.size(1),
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        SOFTMAX_SCALE=softmax_scale,
        WINDOW_SIZE=0 if sliding_window_size is None else sliding_window_size,
        num_warps=launch_num_warps,
        num_stages=num_stages,
    )
    return o


def fused_int8_decode_attention_latev(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    sliding_window_size: int | None = None,
    block_tokens: int = 256,
    num_warps: int | None = None,
    num_stages: int = 2,
):
    """V1 math with delayed V load for register-pressure experiments."""
    assert q.ndim == 3
    assert k_cache.ndim == 4 and v_cache.ndim == 4
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert block_tables.ndim == 2
    assert context_lens.ndim == 1
    assert q.size(0) == context_lens.size(0) == block_tables.size(0)
    assert q.size(2) == k_cache.size(3) == v_cache.size(3)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert k_scale.shape == k_cache.shape[:3]
    assert v_scale.shape == v_cache.shape[:3]
    assert sliding_window_size is None or sliding_window_size > 0
    assert block_tokens > 0 and (block_tokens & (block_tokens - 1)) == 0

    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    q_heads_per_kv_head = num_heads // num_kv_heads
    assert q_heads_per_kv_head * num_kv_heads == num_heads
    assert block_tokens <= block_size
    assert block_size % block_tokens == 0

    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256

    flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
    flat_k_scale = k_scale.view(-1, num_kv_heads)
    flat_v_scale = v_scale.view(-1, num_kv_heads)
    o = torch.empty_like(q)
    launch_num_warps = num_warps or (4 if head_dim < 128 else 8)

    _fused_int8_decode_attention_latev_kernel[(num_seqs, num_heads)](
        q,
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        block_tables,
        context_lens,
        o,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        block_tables.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS=block_tables.size(1),
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        SOFTMAX_SCALE=softmax_scale,
        WINDOW_SIZE=0 if sliding_window_size is None else sliding_window_size,
        num_warps=launch_num_warps,
        num_stages=num_stages,
    )
    return o


def fused_int8_decode_attention_v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    sliding_window_size: int | None = None,
    block_tokens: int = 256,
    num_warps: int | None = None,
    num_stages: int = 2,
):
    """V3 fused INT8 decode attention.

    This variant uses scale-hoisting and exp2 online softmax. It is kept
    separate from fused_int8_decode_attention so benchmarks can compare V1 and
    V3 directly before the faster path becomes the production default.
    """
    assert q.ndim == 3
    assert k_cache.ndim == 4 and v_cache.ndim == 4
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert block_tables.ndim == 2
    assert context_lens.ndim == 1
    assert q.size(0) == context_lens.size(0) == block_tables.size(0)
    assert q.size(2) == k_cache.size(3) == v_cache.size(3)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert k_scale.shape == k_cache.shape[:3]
    assert v_scale.shape == v_cache.shape[:3]
    assert sliding_window_size is None or sliding_window_size > 0
    assert block_tokens > 0 and (block_tokens & (block_tokens - 1)) == 0

    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    q_heads_per_kv_head = num_heads // num_kv_heads
    assert q_heads_per_kv_head * num_kv_heads == num_heads
    assert block_tokens <= block_size
    assert block_size % block_tokens == 0

    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256

    flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
    flat_k_scale = k_scale.view(-1, num_kv_heads)
    flat_v_scale = v_scale.view(-1, num_kv_heads)
    o = torch.empty_like(q)
    launch_num_warps = num_warps or (4 if head_dim < 128 else 8)

    _fused_int8_decode_attention_v3_kernel[(num_seqs, num_heads)](
        q,
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        block_tables,
        context_lens,
        o,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        block_tables.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS=block_tables.size(1),
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        SOFTMAX_SCALE_LOG2=softmax_scale * 1.4426950408889634,
        WINDOW_SIZE=0 if sliding_window_size is None else sliding_window_size,
        num_warps=launch_num_warps,
        num_stages=num_stages,
    )
    return o


@triton.jit
def _partitioned_int8_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    Q_SEQ_STRIDE,
    Q_HEAD_STRIDE,
    BLOCK_TABLE_STRIDE,
    PARTIAL_SEQ_STRIDE,
    PARTIAL_HEAD_STRIDE,
    PARTIAL_PART_STRIDE,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    Q_HEADS_PER_KV_HEAD: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    partition_id = tl.program_id(2)
    kv_head_id = q_head_id // Q_HEADS_PER_KV_HEAD

    context_len = tl.load(context_lens_ptr + seq_id)
    if WINDOW_SIZE > 0:
        visible_start = tl.maximum(context_len - WINDOW_SIZE, 0)
    else:
        visible_start = 0
    partition_start = visible_start + partition_id * PARTITION_SIZE
    partition_end = tl.minimum(partition_start + PARTITION_SIZE, context_len)
    has_tokens = partition_start < partition_end

    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    q_offsets = seq_id * Q_SEQ_STRIDE + q_head_id * Q_HEAD_STRIDE + dim_offsets
    q = tl.load(q_ptr + q_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    # First phase: each program computes a local online-softmax state for one
    # [request, q_head, context_partition]. The second kernel combines these
    # states with the same online-softmax algebra, so no partition writes full
    # attention probabilities or temporary FP16 KV tiles to HBM.
    m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    token_base = partition_start
    while token_base < partition_end:
        logical_block_id = token_base // BLOCK_SIZE
        block_offset = token_base - logical_block_id * BLOCK_SIZE
        physical_block_id = tl.load(
            block_tables_ptr + seq_id * BLOCK_TABLE_STRIDE + logical_block_id,
        )
        tile_remaining_in_block = BLOCK_SIZE - block_offset
        tile_remaining_in_partition = partition_end - token_base
        tile_tokens = tl.minimum(tile_remaining_in_block, tile_remaining_in_partition)

        tile_inner_start = 0
        while tile_inner_start < tile_tokens:
            offsets = tile_inner_start + tl.arange(0, BLOCK_TOKENS)
            token_offsets = block_offset + offsets
            token_positions = token_base + offsets
            token_mask = offsets < tile_tokens
            token_slots = physical_block_id * BLOCK_SIZE + token_offsets

            scale_offsets = token_slots * NUM_KV_HEADS + kv_head_id
            k_scale = tl.load(k_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)
            v_scale = tl.load(v_scale_ptr + scale_offsets, mask=token_mask, other=0.0).to(tl.float32)

            kv_offsets = (
                token_slots[:, None] * NUM_KV_HEADS * HEAD_DIM
                + kv_head_id * HEAD_DIM
                + dim_offsets[None, :]
            )
            kv_mask = token_mask[:, None] & dim_mask[None, :]
            k_q = tl.load(k_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)
            v_q = tl.load(v_cache_ptr + kv_offsets, mask=kv_mask, other=0).to(tl.float32)

            k_fp = k_q * k_scale[:, None]
            scores = tl.sum(q[None, :] * k_fp, axis=1) * SOFTMAX_SCALE
            scores = tl.where(token_mask & (token_positions < partition_end), scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            new_m = tl.maximum(m, tile_max)
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m)

            v_fp = v_q * v_scale[:, None]
            acc = acc * alpha + tl.sum(p[:, None] * v_fp, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = new_m

            tile_inner_start += BLOCK_TOKENS
        token_base += tile_tokens

    if not has_tokens:
        m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    partial_offsets = (
        seq_id * PARTIAL_SEQ_STRIDE
        + q_head_id * PARTIAL_HEAD_STRIDE
        + partition_id * PARTIAL_PART_STRIDE
        + dim_offsets
    )
    tl.store(partial_acc_ptr + partial_offsets, acc, mask=dim_mask)
    scalar_offset = (seq_id * tl.num_programs(1) + q_head_id) * NUM_PARTITIONS + partition_id
    tl.store(partial_m_ptr + scalar_offset, m)
    tl.store(partial_l_ptr + scalar_offset, l)


@triton.jit
def _partitioned_int8_decode_attention_reduce_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    o_ptr,
    context_lens_ptr,
    PARTIAL_SEQ_STRIDE,
    PARTIAL_HEAD_STRIDE,
    PARTIAL_PART_STRIDE,
    O_SEQ_STRIDE,
    O_HEAD_STRIDE,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    dim_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM

    # Second phase: merge partition-local softmax states. If each partition has
    # state (m_i, l_i, acc_i), the global state is:
    #   m = max_i(m_i)
    #   l = sum_i(exp(m_i - m) * l_i)
    #   acc = sum_i(exp(m_i - m) * acc_i)
    # output = acc / l
    m = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    partition_id = 0
    while partition_id < NUM_PARTITIONS:
        scalar_offset = (seq_id * NUM_HEADS + q_head_id) * NUM_PARTITIONS + partition_id
        part_m = tl.load(partial_m_ptr + scalar_offset).to(tl.float32)
        part_l = tl.load(partial_l_ptr + scalar_offset).to(tl.float32)
        part_offsets = (
            seq_id * PARTIAL_SEQ_STRIDE
            + q_head_id * PARTIAL_HEAD_STRIDE
            + partition_id * PARTIAL_PART_STRIDE
            + dim_offsets
        )
        part_acc = tl.load(partial_acc_ptr + part_offsets, mask=dim_mask, other=0.0).to(tl.float32)

        new_m = tl.maximum(m, part_m)
        old_scale = tl.exp(m - new_m)
        part_scale = tl.exp(part_m - new_m)
        acc = acc * old_scale + part_acc * part_scale
        l = l * old_scale + part_l * part_scale
        m = new_m
        partition_id += 1

    out = acc / tl.maximum(l, 1.0e-20)
    o_offsets = seq_id * O_SEQ_STRIDE + q_head_id * O_HEAD_STRIDE + dim_offsets
    tl.store(o_ptr + o_offsets, out, mask=dim_mask)


def allocate_partitioned_workspace(
    q: torch.Tensor,
    num_partitions: int,
    block_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_partitions <= 0 or block_head_dim <= 0:
        raise ValueError("partitioned workspace dimensions must be positive")
    partial_shape = (q.size(0), q.size(1), num_partitions)
    partial_items = q.size(0) * q.size(1) * num_partitions
    workspace = torch.empty(
        partial_items * (block_head_dim + 2),
        dtype=torch.float32,
        device=q.device,
    )
    partial_acc = workspace[: partial_items * block_head_dim].view(
        *partial_shape,
        block_head_dim,
    )
    partial_m = workspace[
        partial_items * block_head_dim : partial_items * (block_head_dim + 1)
    ].view(partial_shape)
    partial_l = workspace[
        partial_items * (block_head_dim + 1) :
    ].view(partial_shape)
    return partial_acc, partial_m, partial_l


class PartitionedDecodeBufferPool:
    """Reusable single-stream buffers for partitioned INT8 decode.

    One pool can be shared by every attention layer in a model runner because
    layers enqueue their attention and output projection serially on the same
    CUDA stream.  The pool is intentionally not re-entrant; callers using
    concurrent streams must own one pool per stream.
    """

    def __init__(self) -> None:
        self.workspace_storage: torch.Tensor | None = None
        self.output_storage: torch.Tensor | None = None

    def reserve(
        self,
        *,
        num_seqs: int,
        num_heads: int,
        num_partitions: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate the maximum configured workspace before KV-cache sizing."""

        if min(num_seqs, num_heads, num_partitions, head_dim) <= 0:
            raise ValueError("partitioned buffer reservation dimensions must be positive")
        block_head_dim = _next_power_of_two(head_dim)
        if block_head_dim > 256:
            raise ValueError("partitioned buffer head dimension exceeds kernel limit")
        partial_items = num_seqs * num_heads * num_partitions
        workspace_items = partial_items * (block_head_dim + 2)
        output_items = num_seqs * num_heads * head_dim
        if self._needs_storage(
            self.workspace_storage,
            size=workspace_items,
            dtype=torch.float32,
            device=device,
        ):
            self.workspace_storage = torch.empty(
                workspace_items,
                dtype=torch.float32,
                device=device,
            )
        if self._needs_storage(
            self.output_storage,
            size=output_items,
            dtype=dtype,
            device=device,
        ):
            self.output_storage = torch.empty(
                output_items,
                dtype=dtype,
                device=device,
            )

    @staticmethod
    def _needs_storage(
        storage: torch.Tensor | None,
        *,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            storage is None
            or storage.numel() < size
            or storage.dtype != dtype
            or storage.device != device
        )

    def acquire(
        self,
        q: torch.Tensor,
        num_partitions: int,
        block_head_dim: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        if num_partitions <= 0 or block_head_dim <= 0:
            raise ValueError("partitioned workspace dimensions must be positive")
        partial_shape = (q.size(0), q.size(1), num_partitions)
        partial_items = q.size(0) * q.size(1) * num_partitions
        workspace_items = partial_items * (block_head_dim + 2)
        if self._needs_storage(
            self.workspace_storage,
            size=workspace_items,
            dtype=torch.float32,
            device=q.device,
        ):
            self.workspace_storage = torch.empty(
                workspace_items,
                dtype=torch.float32,
                device=q.device,
            )
        if self._needs_storage(
            self.output_storage,
            size=q.numel(),
            dtype=q.dtype,
            device=q.device,
        ):
            self.output_storage = torch.empty(
                q.numel(),
                dtype=q.dtype,
                device=q.device,
            )
        assert self.workspace_storage is not None
        assert self.output_storage is not None
        storage = self.workspace_storage[:workspace_items]
        partial_acc = storage[: partial_items * block_head_dim].view(
            *partial_shape,
            block_head_dim,
        )
        partial_m = storage[
            partial_items * block_head_dim : partial_items * (block_head_dim + 1)
        ].view(partial_shape)
        partial_l = storage[
            partial_items * (block_head_dim + 1) :
        ].view(partial_shape)
        output = self.output_storage[: q.numel()].view_as(q)
        return (partial_acc, partial_m, partial_l), output

    def storage_stats(self) -> dict[str, int]:
        workspace_bytes = (
            0
            if self.workspace_storage is None
            else self.workspace_storage.numel()
            * self.workspace_storage.element_size()
        )
        output_bytes = (
            0
            if self.output_storage is None
            else self.output_storage.numel() * self.output_storage.element_size()
        )
        return {
            "workspace_bytes": workspace_bytes,
            "output_bytes": output_bytes,
            "total_bytes": workspace_bytes + output_bytes,
        }


def validate_partitioned_workspace(
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    q: torch.Tensor,
    num_partitions: int,
    block_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    partial_shape = (q.size(0), q.size(1), num_partitions)
    expected_shapes = (
        (*partial_shape, block_head_dim),
        partial_shape,
        partial_shape,
    )
    if len(workspace) != 3:
        raise ValueError("partitioned workspace must contain acc, m, and l tensors")
    ranges = []
    for name, tensor, shape in zip(("acc", "m", "l"), workspace, expected_shapes):
        if tensor.shape != shape:
            raise ValueError(
                f"partitioned workspace {name} has shape {tuple(tensor.shape)}; "
                f"expected {shape}"
            )
        if tensor.dtype != torch.float32 or tensor.device != q.device:
            raise ValueError(
                f"partitioned workspace {name} must be float32 on {q.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"partitioned workspace {name} must be contiguous")
        start = tensor.data_ptr()
        ranges.append((name, start, start + tensor.numel() * tensor.element_size()))
    for index, (name, start, end) in enumerate(ranges):
        for other_name, other_start, other_end in ranges[index + 1 :]:
            if start < other_end and other_start < end:
                raise ValueError(
                    f"partitioned workspace {name} overlaps {other_name}"
                )
    return workspace


def validate_partitioned_output(
    output: torch.Tensor,
    q: torch.Tensor,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if (
        output.shape != q.shape
        or output.dtype != q.dtype
        or output.device != q.device
        or not output.is_contiguous()
    ):
        raise ValueError("partitioned attention output must match q layout")
    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    q_start = q.data_ptr()
    q_end = q_start + q.numel() * q.element_size()
    if output_start < q_end and q_start < output_end:
        raise ValueError("partitioned attention output must not alias q")
    for name, tensor in zip(("acc", "m", "l"), workspace):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if output_start < end and start < output_end:
            raise ValueError(
                f"partitioned attention output overlaps workspace {name}"
            )
    return output


def partitioned_fused_int8_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    sliding_window_size: int | None = None,
    block_tokens: int = 256,
    partition_size: int = 256,
    *,
    max_context_len: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output: torch.Tensor | None = None,
    buffer_pool: PartitionedDecodeBufferPool | None = None,
):
    """Partitioned INT8 decode attention for long-context eager decode.

    v1 assigns one Triton program to one [request, q_head] pair and scans the
    whole context serially. This v2 path splits the visible context into
    partitions, computes local online-softmax states in parallel, then reduces
    those states into the final attention output. The caller supplies the
    actual maximum context length so padded block-table width does not create
    unnecessary partitions. This path is eager-only and is not captured by the
    current CUDA Graph implementation; performance benefit is shape-dependent.
    """
    assert q.ndim == 3
    assert k_cache.ndim == 4 and v_cache.ndim == 4
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert block_tables.ndim == 2
    assert context_lens.ndim == 1
    assert q.size(0) == context_lens.size(0) == block_tables.size(0)
    assert q.size(2) == k_cache.size(3) == v_cache.size(3)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert k_scale.shape == k_cache.shape[:3]
    assert v_scale.shape == v_cache.shape[:3]
    assert sliding_window_size is None or sliding_window_size > 0
    assert block_tokens > 0 and (block_tokens & (block_tokens - 1)) == 0
    assert partition_size > 0
    assert max_context_len >= 0

    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    q_heads_per_kv_head = num_heads // num_kv_heads
    assert q_heads_per_kv_head * num_kv_heads == num_heads
    assert block_tokens <= block_size
    assert block_size % block_tokens == 0

    block_head_dim = _next_power_of_two(head_dim)
    assert block_head_dim <= 256

    num_partitions = partition_count(
        max_context_len=max_context_len,
        partition_size=partition_size,
        sliding_window_size=sliding_window_size,
    )

    if buffer_pool is not None:
        if workspace is not None or output is not None:
            raise ValueError(
                "buffer_pool cannot be combined with explicit workspace or output"
            )
        workspace, output = buffer_pool.acquire(
            q,
            num_partitions,
            block_head_dim,
        )

    flat_k_cache = k_cache.view(-1, num_kv_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_kv_heads, head_dim)
    flat_k_scale = k_scale.view(-1, num_kv_heads)
    flat_v_scale = v_scale.view(-1, num_kv_heads)
    if workspace is None:
        partial_acc, partial_m, partial_l = allocate_partitioned_workspace(
            q,
            num_partitions,
            block_head_dim,
        )
    else:
        partial_acc, partial_m, partial_l = validate_partitioned_workspace(
            workspace,
            q,
            num_partitions,
            block_head_dim,
        )
    validated_workspace = (partial_acc, partial_m, partial_l)
    if output is None:
        o = torch.empty_like(q)
    else:
        o = validate_partitioned_output(output, q, validated_workspace)

    _partitioned_int8_decode_attention_kernel[(num_seqs, num_heads, num_partitions)](
        q,
        flat_k_cache,
        flat_v_cache,
        flat_k_scale,
        flat_v_scale,
        block_tables,
        context_lens,
        partial_acc,
        partial_m,
        partial_l,
        q.stride(0),
        q.stride(1),
        block_tables.stride(0),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS=block_tables.size(1),
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        SOFTMAX_SCALE=softmax_scale,
        WINDOW_SIZE=0 if sliding_window_size is None else sliding_window_size,
        PARTITION_SIZE=partition_size,
        NUM_PARTITIONS=num_partitions,
        num_warps=4 if head_dim <= 128 else 8,
        num_stages=2,
    )
    _partitioned_int8_decode_attention_reduce_kernel[(num_seqs, num_heads)](
        partial_acc,
        partial_m,
        partial_l,
        o,
        context_lens,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        o.stride(0),
        o.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_HEAD_DIM=block_head_dim,
        NUM_HEADS=num_heads,
        NUM_PARTITIONS=num_partitions,
        num_warps=4 if head_dim <= 128 else 8,
        num_stages=2,
    )
    return o
