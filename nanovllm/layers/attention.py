import torch
from torch import nn

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.engine.execution import select_int8_decode_attention_path
from nanovllm.layers.kv_cache_quant import (
    dequant_packed_kvcache,
    store_kvcache,
    store_kvcache_int8,
    store_kvcache_int8_range,
    store_kvcache_range,
)
from nanovllm.layers.int8_fused_attention import fused_int8_decode_attention
from nanovllm.layers.int8_fused_attention import partitioned_fused_int8_decode_attention
from nanovllm.utils.context import get_context
from nanovllm.shape_trace import active_trace


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])
        self.kv_cache_dtype = "auto"
        self.kv_dequant_backend = "fused"
        self.int8_partitioned_decode_threshold = 8192
        self.int8_partitioned_decode_partition_size = 512

    def _window_size(self):
        context = get_context()
        if context.sliding_window_size is None:
            return None
        # During autoregressive decode, the current query token is part of
        # cache_seqlens. A left window of W - 1 plus the current token gives
        # exactly W visible KV tokens.
        return (context.sliding_window_size - 1, 0)

    def _flash_attn_with_kvcache(self, q, k_cache, v_cache, block_table, context_lens=None):
        context = get_context()
        if context_lens is None:
            context_lens = context.context_lens
        window_size = self._window_size()
        if window_size is None:
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                block_table=block_table,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                block_table=block_table,
                softmax_scale=self.scale,
                causal=True,
                window_size=window_size,
            )
        # flash_attn_with_kvcache normally returns [batch, 1, heads, head_dim]
        # when q is [batch, 1, heads, head_dim]. The rest of nano-vLLM expects
        # [tokens, heads, head_dim], which also lets mixed decode/prefill
        # concatenate the two paths before the output projection. Keep this
        # tolerant of versions that already return the squeezed rank.
        if output.dim() == 4:
            return output.squeeze(1)
        return output

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_k: int,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor | None,
    ):
        kwargs = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=block_table,
        )
        window_size = self._window_size()
        if window_size is not None:
            kwargs["window_size"] = window_size
        return flash_attn_varlen_func(q, k, v, **kwargs)

    def _dequant_kvcache(
        self,
        dtype: torch.dtype,
        dequant_block_ids: torch.Tensor | None = None,
        dequant_block_tables: torch.Tensor | None = None,
    ):
        context = get_context()
        if dequant_block_ids is None:
            dequant_block_ids = context.dequant_block_ids
        if dequant_block_tables is None:
            dequant_block_tables = context.dequant_block_tables
        assert dequant_block_ids is not None
        assert dequant_block_tables is not None
        num_selected_blocks = dequant_block_ids.numel()
        if num_selected_blocks == 0:
            return self.k_cache, self.v_cache

        if self.kv_dequant_backend == "torch":
            block_ids = dequant_block_ids.to(device=self.k_cache.device, dtype=torch.long)
            k_cache = self.k_cache.index_select(0, block_ids).to(dtype)
            v_cache = self.v_cache.index_select(0, block_ids).to(dtype)
            k_scale = self.k_scale.index_select(0, block_ids).unsqueeze(-1).to(dtype)
            v_scale = self.v_scale.index_select(0, block_ids).unsqueeze(-1).to(dtype)
            return k_cache * k_scale, v_cache * v_scale

        packed_shape = (
            num_selected_blocks,
            self.k_cache.size(1),
            self.k_cache.size(2),
            self.k_cache.size(3),
        )
        packed_k_cache = torch.empty(packed_shape, dtype=dtype, device=self.k_cache.device)
        packed_v_cache = torch.empty_like(packed_k_cache)
        dequant_packed_kvcache(
            self.k_cache,
            self.v_cache,
            self.k_scale,
            self.v_scale,
            dequant_block_ids,
            packed_k_cache,
            packed_v_cache,
        )
        return packed_k_cache, packed_v_cache

    def _int8_decode_attention(
        self,
        q: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ):
        # For short decode, one program per [sequence, q_head] has the lowest
        # overhead. For long decode, each program serially scans too many KV
        # tokens; split-KV/Flash-Decoding-style partitioning exposes more
        # parallelism and the reduce kernel becomes worthwhile.
        decode_path = select_int8_decode_attention_path(
            kv_dequant_backend=self.kv_dequant_backend,
            max_context_len=max_context_len,
            partition_threshold=self.int8_partitioned_decode_threshold,
            sliding_window_size=get_context().sliding_window_size,
        )
        if decode_path == "int8_partitioned_decode":
            return partitioned_fused_int8_decode_attention(
                q,
                self.k_cache,
                self.v_cache,
                self.k_scale,
                self.v_scale,
                block_tables,
                context_lens,
                self.scale,
                sliding_window_size=None,
                block_tokens=256,
                partition_size=self.int8_partitioned_decode_partition_size,
                max_context_len=max_context_len,
            )
        return fused_int8_decode_attention(
            q,
            self.k_cache,
            self.v_cache,
            self.k_scale,
            self.v_scale,
            block_tables,
            context_lens,
            self.scale,
            get_context().sliding_window_size,
        )

    def _forward_mixed(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        decode_n = context.decode_token_count
        prefill_n = context.prefill_token_count
        assert decode_n > 0 and prefill_n > 0
        outputs: list[torch.Tensor] = []

        q_decode = q[:decode_n]
        q_prefill = q[decode_n:decode_n + prefill_n]
        k_prefill = k[decode_n:decode_n + prefill_n]
        v_prefill = v[decode_n:decode_n + prefill_n]

        if self.kv_cache_dtype == "int8":
            # Store decode tokens first and compute decode attention before
            # writing prefill chunks. Otherwise decode requests in this mixed
            # batch could see K/V from unrelated prefill requests scheduled in
            # the same engine step.
            store_kvcache_int8_range(k, v, self.k_cache, self.v_cache, self.k_scale, self.v_scale, context.slot_mapping, 0, decode_n)
            if self.kv_dequant_backend == "fused":
                outputs.append(
                    self._int8_decode_attention(
                        q_decode,
                        context.decode_block_tables,
                        context.decode_context_lens,
                        context.decode_max_context_len,
                    )
                )
            else:
                decode_k_cache, decode_v_cache = self._dequant_kvcache(
                    q.dtype,
                    context.decode_dequant_block_ids,
                    context.decode_dequant_block_tables,
                )
                outputs.append(
                    self._flash_attn_with_kvcache(
                        q_decode,
                        decode_k_cache,
                        decode_v_cache,
                        context.decode_dequant_block_tables,
                        context.decode_context_lens,
                    )
                )
            store_kvcache_int8_range(k, v, self.k_cache, self.v_cache, self.k_scale, self.v_scale, context.slot_mapping, decode_n, decode_n + prefill_n)
            if context.prefill_block_tables is not None:
                prefill_k_cache, prefill_v_cache = self._dequant_kvcache(
                    q.dtype,
                    context.prefill_dequant_block_ids,
                    context.prefill_dequant_block_tables,
                )
                k_prefill, v_prefill = prefill_k_cache, prefill_v_cache
                prefill_block_tables = context.prefill_dequant_block_tables
            else:
                prefill_block_tables = None
        else:
            store_kvcache_range(k, v, self.k_cache, self.v_cache, context.slot_mapping, 0, decode_n)
            outputs.append(
                self._flash_attn_with_kvcache(
                    q_decode,
                    self.k_cache,
                    self.v_cache,
                    context.decode_block_tables,
                    context.decode_context_lens,
                )
            )
            store_kvcache_range(k, v, self.k_cache, self.v_cache, context.slot_mapping, decode_n, decode_n + prefill_n)
            if context.prefill_block_tables is not None:
                k_prefill, v_prefill = self.k_cache, self.v_cache
            prefill_block_tables = context.prefill_block_tables

        outputs.append(
            self._flash_attn_varlen(
                q_prefill,
                k_prefill,
                v_prefill,
                context.prefill_max_seqlen_q,
                context.prefill_cu_seqlens_q,
                context.prefill_max_seqlen_k,
                context.prefill_cu_seqlens_k,
                prefill_block_tables,
            )
        )
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        trace = active_trace()
        if trace is not None:
            trace.record_attention(
                layer_id=getattr(self, "layer_id", None),
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                k_scale=self.k_scale if self.k_scale.numel() else None,
                v_scale=self.v_scale if self.v_scale.numel() else None,
                context=context,
            )
        if context.is_mixed:
            return self._forward_mixed(q, k, v)
        if self.kv_cache_dtype == "int8":
            if k_cache.numel() and v_cache.numel():
                store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None:    # prefix cache
                    k_cache, v_cache = self._dequant_kvcache(q.dtype)
                    k, v = k_cache, v_cache
                    block_table = context.dequant_block_tables
                else:
                    block_table = None
                o = self._flash_attn_varlen(
                    q,
                    k,
                    v,
                    context.max_seqlen_q,
                    context.cu_seqlens_q,
                    context.max_seqlen_k,
                    context.cu_seqlens_k,
                    block_table,
                )
            else:    # decode
                if self.kv_dequant_backend == "fused":
                    o = self._int8_decode_attention(
                        q,
                        context.block_tables,
                        context.context_lens,
                        context.max_context_len,
                    )
                else:
                    k_cache, v_cache = self._dequant_kvcache(q.dtype)
                    o = self._flash_attn_with_kvcache(q, k_cache, v_cache, context.dequant_block_tables)
            return o
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = self._flash_attn_varlen(
                q,
                k,
                v,
                context.max_seqlen_q,
                context.cu_seqlens_q,
                context.max_seqlen_k,
                context.cu_seqlens_k,
                context.block_tables,
            )
        else:    # decode
            o = self._flash_attn_with_kvcache(q, k_cache, v_cache, context.block_tables)
        return o
