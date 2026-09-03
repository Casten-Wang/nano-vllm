"""Text-only Qwen3.6-compatible MoE runtime."""

from __future__ import annotations

import torch
from torch import nn

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.models.qwen35_attention import Qwen35Attention, Qwen35KeyBufferPool
from nanovllm.models.qwen35_gated_delta import Qwen35GatedDeltaNet
from nanovllm.models.qwen35_gptq import (
    GPTQExpertWorkspacePool,
    Qwen35GPTQExperts,
)
from nanovllm.models.moe_dispatch import BatchedExpertWeightBufferPool
from nanovllm.models.qwen35_moe import (
    ResidentFP8WeightBufferPool,
    Qwen35Experts,
    Qwen35RMSNorm,
    Qwen35SparseMoeBlock,
)


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
            raise ValueError(
                f"unsupported Qwen3.6-compatible layer type: {self.layer_type}"
            )
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
        self.max_num_batched_tokens = int(
            getattr(config, "nanovllm_max_num_batched_tokens", 0)
        )
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size), int(config.hidden_size)
        )
        self.layers = nn.ModuleList(
            Qwen35DecoderLayer(config, layer_idx)
            for layer_idx in range(int(config.num_hidden_layers))
        )
        self.full_attention_key_buffer_pool = Qwen35KeyBufferPool()
        self.moe_decode_weight_buffer_pool = BatchedExpertWeightBufferPool()
        self.resident_fp8_weight_buffer_pool = ResidentFP8WeightBufferPool()
        self.gptq_expert_workspace_pool = GPTQExpertWorkspacePool()
        for layer in self.layers:
            if isinstance(getattr(layer, "self_attn", None), Qwen35Attention):
                layer.self_attn.key_buffer_pool = (
                    self.full_attention_key_buffer_pool
                )
            if isinstance(layer.mlp.experts, Qwen35Experts):
                layer.mlp.experts.decode_weight_buffer_pool = (
                    self.moe_decode_weight_buffer_pool
                )
                layer.mlp.experts.resident_weight_buffer_pool = (
                    self.resident_fp8_weight_buffer_pool
                )
            elif isinstance(layer.mlp.experts, Qwen35GPTQExperts):
                layer.mlp.experts.gptq_workspace_pool = (
                    self.gptq_expert_workspace_pool
                )
        self.norm = Qwen35RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )

    def reserve_runtime_buffers(self, max_decode_tokens: int) -> None:
        """Reserve predictable model scratch before the KV cache consumes VRAM."""

        if max_decode_tokens <= 0:
            raise ValueError("max_decode_tokens must be positive")
        attention = next(
            (
                layer.self_attn
                for layer in self.layers
                if isinstance(getattr(layer, "self_attn", None), Qwen35Attention)
            ),
            None,
        )
        if attention is not None:
            max_key_tokens = (
                self.max_num_batched_tokens
                if self.max_num_batched_tokens > 0
                else max_decode_tokens
            )
            self.full_attention_key_buffer_pool.reserve(
                elements=(
                    max_key_tokens * attention.num_kv_heads * attention.head_dim
                ),
                dtype=self.embed_tokens.weight.dtype,
                device=self.embed_tokens.weight.device,
            )
        block = next(
            (
                layer.mlp
                for layer in self.layers
                if isinstance(layer.mlp.experts, Qwen35Experts)
                and layer.mlp.experts.decode_backend == "batched"
            ),
            None,
        )
        if block is not None:
            experts = block.experts
            chunk_tokens = min(max_decode_tokens, experts.decode_chunk_size)
            route_count = chunk_tokens * block.gate.top_k
            weight_elements = route_count * max(
                experts.gate_up_proj[0].numel(),
                experts.down_proj[0].numel(),
            )
            workspace_elements = route_count * (
                experts.gate_up_proj.shape[1] + experts.down_proj.shape[1]
            )
            self.moe_decode_weight_buffer_pool.reserve(
                weight_elements=weight_elements,
                workspace_elements=workspace_elements,
                weight_dtype=experts.gate_up_proj.dtype,
                activation_dtype=experts.gate_up_proj.dtype,
                device=experts.gate_up_proj.device,
            )

        gptq_block = next(
            (
                layer.mlp
                for layer in self.layers
                if isinstance(layer.mlp.experts, Qwen35GPTQExperts)
                and layer.mlp.experts.backend == "triton"
            ),
            None,
        )
        if gptq_block is not None:
            experts = gptq_block.experts
            self.gptq_expert_workspace_pool.reserve(
                max_decode_tokens,
                experts.local_intermediate_size,
                experts.hidden_size,
                dtype=self.embed_tokens.weight.dtype,
                device=self.embed_tokens.weight.device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        # The final hidden state has no residual consumer. Reuse its storage
        # for inference instead of allocating another [tokens, hidden_size]
        # output; Qwen35RMSNorm keeps the autograd path out of place.
        return self.norm(hidden_states, inplace_output=True)


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
        self.checkpoint_quantization_spec = getattr(
            text_config,
            "nanovllm_quantization_spec",
            None,
        )
        self.model = Qwen35Model(text_config)
        self.lm_head = ParallelLMHead(
            int(text_config.vocab_size),
            int(text_config.hidden_size),
        )
        self.lm_head.max_top_k_reduction_width = min(
            int(
                getattr(
                    text_config,
                    "nanovllm_tp_top_k_reduction_max_width",
                    256,
                )
            ),
            int(text_config.vocab_size) - 1,
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

    def resolve_checkpoint_parameter(self, weight_name: str):
        from nanovllm.models.qwen35_fp8 import resolve_fp8_expert_parameter
        from nanovllm.models.qwen35_gptq import resolve_gptq_expert_parameter

        checkpoint_format = getattr(
            self.checkpoint_quantization_spec,
            "format",
            "bf16",
        )
        if checkpoint_format == "gptq_int4":
            resolved = resolve_gptq_expert_parameter(weight_name)
        elif checkpoint_format == "fp8_block":
            resolved = resolve_fp8_expert_parameter(weight_name)
        else:
            resolved = None
        if resolved is None:
            return None
        target, expert_id = resolved
        try:
            self.get_parameter(target)
        except AttributeError:
            return None
        return target, expert_id

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        greedy: bool = False,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if greedy:
            return self.lm_head(hidden_states, greedy=True)
        if top_k is not None:
            return self.lm_head(hidden_states, top_k=top_k)
        return self.lm_head(hidden_states)


Qwen3_5MoeForConditionalGeneration = Qwen3_5MoeForCausalLM
