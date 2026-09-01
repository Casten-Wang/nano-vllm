"""Tiny end-to-end parity check against the official Transformers model."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch


transformers = pytest.importorskip("transformers", minversion="5.2.0")
safetensors_torch = pytest.importorskip("safetensors.torch")
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as TransformersQwen35,
)


ROOT = Path(__file__).parents[1]
CURRENT_CONTEXT = {}


def _load(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_runtime():
    names = (
        "nanovllm",
        "nanovllm.layers",
        "nanovllm.models",
        "nanovllm.utils",
        "nanovllm.utils.context",
        "nanovllm.layers.activation",
        "nanovllm.layers.linear",
        "nanovllm.layers.embed_head",
        "nanovllm.layers.rotary_embedding",
        "nanovllm.layers.attention",
        "nanovllm.models.qwen35_moe",
        "nanovllm.models.qwen35_gated_delta",
        "nanovllm.models.qwen35_attention",
        "nanovllm.models.qwen35",
        "nanovllm.utils.loader",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:4]:
            sys.modules[name] = types.ModuleType(name)
        context_module = types.ModuleType("nanovllm.utils.context")
        context_module.get_context = lambda: CURRENT_CONTEXT["value"]
        sys.modules[context_module.__name__] = context_module
        _load("nanovllm.layers.activation", "nanovllm/layers/activation.py")
        linear = _load("nanovllm.layers.linear", "nanovllm/layers/linear.py")
        _load("nanovllm.layers.embed_head", "nanovllm/layers/embed_head.py")
        _load(
            "nanovllm.layers.rotary_embedding",
            "nanovllm/layers/rotary_embedding.py",
        )
        _load("nanovllm.models.qwen35_moe", "nanovllm/models/qwen35_moe.py")
        _load(
            "nanovllm.models.qwen35_gated_delta",
            "nanovllm/models/qwen35_gated_delta.py",
        )

        attention_module = types.ModuleType("nanovllm.layers.attention")

        class EagerAttention(torch.nn.Module):
            def __init__(self, num_heads, head_dim, scale, num_kv_heads):
                super().__init__()
                self.num_heads = num_heads
                self.num_kv_heads = num_kv_heads
                self.scale = scale
                self.k_cache = self.v_cache = torch.tensor([])

            def forward(self, query, key, value):
                repeats = self.num_heads // self.num_kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
                output = torch.empty_like(query)
                for start, end in CURRENT_CONTEXT["value"].state_token_ranges:
                    scores = torch.einsum(
                        "thd,shd->hts",
                        query[start:end],
                        key[start:end],
                    ) * self.scale
                    mask = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
                    scores = scores.masked_fill(mask, float("-inf"))
                    output[start:end] = torch.einsum(
                        "hts,shd->thd",
                        scores.softmax(dim=-1),
                        value[start:end],
                    )
                return output

        attention_module.Attention = EagerAttention
        sys.modules[attention_module.__name__] = attention_module
        _load(
            "nanovllm.models.qwen35_attention",
            "nanovllm/models/qwen35_attention.py",
        )
        local_model = _load(
            "nanovllm.models.qwen35",
            "nanovllm/models/qwen35.py",
        )
        loader = _load("nanovllm.utils.loader", "nanovllm/utils/loader.py")
        return local_model, loader, linear, context_module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


LOCAL, LOADER, LINEAR, CONTEXT_MODULE = _load_local_runtime()


def tiny_config():
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=False,
        attention_bias=False,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def test_tiny_text_model_matches_transformers_end_to_end(tmp_path):
    config = tiny_config()
    torch.manual_seed(29)
    reference = TransformersQwen35(config).eval()
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=1),
        patch.object(LINEAR.dist, "get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        local = LOCAL.Qwen3_5MoeForCausalLM(config).eval()
        for module in local.modules():
            allocate = getattr(module, "allocate_state_cache", None)
            if allocate is not None:
                allocate(2, "cpu")
        safetensors_torch.save_file(
            {
                name: value.detach().contiguous()
                for name, value in reference.state_dict().items()
            },
            str(tmp_path / "model.safetensors"),
        )
        LOADER.load_model(local, str(tmp_path))

    tokens = torch.tensor([[1, 5, 7, 2, 9]])
    positions = torch.arange(tokens.shape[1])
    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        is_prefill=True,
        is_mixed=False,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=((0, tokens.shape[1]),),
        cu_seqlens_q=torch.tensor([0, tokens.shape[1]], dtype=torch.int32),
    )
    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        expected = reference.model(
            input_ids=tokens,
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        actual = local(tokens.squeeze(0), positions)
        expected_logits = reference.lm_head(expected)
        actual_logits = torch.nn.functional.linear(actual, local.lm_head.weight)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=2e-4,
        atol=2e-4,
    )
