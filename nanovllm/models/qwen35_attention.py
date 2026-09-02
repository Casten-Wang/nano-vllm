"""Qwen3.5 full-attention layer for the text-only runtime."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen35_moe import Qwen35RMSNorm


class Qwen35Attention(nn.Module):
    """Full attention with Qwen3.5 query gate and partial text RoPE."""

    def __init__(self, config) -> None:
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % self.tp_size:
            raise ValueError("query heads must divide tensor parallel size")
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // self.total_num_heads,
            )
        )
        attention_bias = bool(getattr(config, "attention_bias", False))
        self.qkv_proj = QKVParallelLinear(
            int(config.hidden_size),
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            q_head_size=2 * self.head_dim,
        )
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            int(config.hidden_size),
            bias=attention_bias,
        )
        self.q_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=float(config.rms_norm_eps),
        )
        self.k_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=float(config.rms_norm_eps),
        )

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rotary_factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = int(self.head_dim * rotary_factor)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=int(
                getattr(
                    config,
                    "nanovllm_max_model_len",
                    config.max_position_embeddings,
                )
            ),
            base=float(rope_parameters.get("rope_theta", 10_000.0)),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        query_and_gate, key, value = self.qkv_proj(hidden_states).split(
            (
                self.num_heads * 2 * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        query_and_gate = query_and_gate.view(
            -1,
            self.num_heads,
            2 * self.head_dim,
        )
        query, gate = query_and_gate.chunk(2, dim=-1)
        # RotaryEmbedding is compiled separately. Give its key input distinct
        # storage so Dynamo does not have to synthesize a base for non-overlap
        # views of the packed QKV result. This copies only the small local KV
        # projection (256 values per token for the official TP4/TP8 model).
        key = key.view(
            -1,
            self.num_kv_heads,
            self.head_dim,
        ).clone()
        value = value.view_as(key)
        query = self.q_norm(query, inplace_output=True)
        key = self.k_norm(key, inplace_output=True)
        query, key = self.rotary_emb(
            positions,
            query,
            key,
            inplace_output=True,
        )
        output = self.attn(query, key, value)
        if output.requires_grad or gate.requires_grad:
            output = output * torch.sigmoid(gate)
        else:
            gate.sigmoid_()
            output.mul_(gate)
        return self.o_proj(output.flatten(1, -1))
