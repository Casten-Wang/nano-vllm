"""Text-only Qwen3.5 MoE runtime."""

from __future__ import annotations

import torch
from torch import nn

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.models.qwen35_attention import Qwen35Attention
from nanovllm.models.qwen35_gated_delta import Qwen35GatedDeltaNet
from nanovllm.models.qwen35_moe import Qwen35RMSNorm, Qwen35SparseMoeBlock


def _add_residual(
    branch_output: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    if torch.is_grad_enabled():
        return residual + branch_output
    return branch_output.add_(residual)


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen35SparseMoeBlock(config)
        self.input_layernorm = Qwen35RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = _add_residual(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return _add_residual(self.mlp(hidden_states), residual)


class Qwen35Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size), int(config.hidden_size)
        )
        self.layers = nn.ModuleList(
            Qwen35DecoderLayer(config, layer_idx)
            for layer_idx in range(int(config.num_hidden_layers))
        )
        self.norm = Qwen35RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5MoeForCausalLM(nn.Module):
    strict_weight_loading = True
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "in_proj_z": ("in_proj_zba", 0),
        "in_proj_b": ("in_proj_zba", 1),
        "in_proj_a": ("in_proj_zba", 2),
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config) -> None:
        super().__init__()
        text_config = getattr(config, "text_config", None) or config
        self.model = Qwen35Model(text_config)
        self.lm_head = ParallelLMHead(
            int(text_config.vocab_size), int(text_config.hidden_size)
        )
        if bool(getattr(text_config, "tie_word_embeddings", False)):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    @staticmethod
    def map_weight_name(weight_name: str) -> str | None:
        if weight_name.startswith("model.visual.") or weight_name.startswith("mtp."):
            return None
        prefix = "model.language_model."
        if weight_name.startswith(prefix):
            return "model." + weight_name[len(prefix) :]
        return weight_name

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


Qwen3_5MoeForConditionalGeneration = Qwen3_5MoeForCausalLM
