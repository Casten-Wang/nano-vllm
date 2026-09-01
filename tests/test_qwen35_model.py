from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).parents[1]


class FakeEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, token_ids):
        return self.weight[token_ids]


class FakeHead(FakeEmbedding):
    def forward(self, hidden_states):
        return hidden_states @ self.weight.T


class FakeMixer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

    def forward(self, *args):
        return torch.zeros_like(args[-1])


class FakeNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return x


class FakeMoe(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


def load_qwen35_module():
    modules = {
        "nanovllm": types.ModuleType("nanovllm"),
        "nanovllm.layers": types.ModuleType("nanovllm.layers"),
        "nanovllm.models": types.ModuleType("nanovllm.models"),
        "nanovllm.layers.embed_head": types.ModuleType("nanovllm.layers.embed_head"),
        "nanovllm.models.qwen35_attention": types.ModuleType("nanovllm.models.qwen35_attention"),
        "nanovllm.models.qwen35_gated_delta": types.ModuleType("nanovllm.models.qwen35_gated_delta"),
        "nanovllm.models.qwen35_moe": types.ModuleType("nanovllm.models.qwen35_moe"),
    }
    modules["nanovllm.layers.embed_head"].ParallelLMHead = FakeHead
    modules["nanovllm.layers.embed_head"].VocabParallelEmbedding = FakeEmbedding
    modules["nanovllm.models.qwen35_attention"].Qwen35Attention = FakeMixer
    modules["nanovllm.models.qwen35_gated_delta"].Qwen35GatedDeltaNet = FakeMixer
    modules["nanovllm.models.qwen35_moe"].Qwen35RMSNorm = FakeNorm
    modules["nanovllm.models.qwen35_moe"].Qwen35SparseMoeBlock = FakeMoe
    saved = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        path = ROOT / "nanovllm" / "models" / "qwen35.py"
        spec = spec_from_file_location("qwen35_model_under_test", path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


QWEN35 = load_qwen35_module()


def tiny_outer_config():
    text_config = SimpleNamespace(
        vocab_size=17,
        hidden_size=4,
        num_hidden_layers=4,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    return SimpleNamespace(text_config=text_config)


def test_conditional_checkpoint_names_map_only_text_weights():
    mapper = QWEN35.Qwen3_5MoeForCausalLM.map_weight_name

    assert mapper("model.language_model.layers.3.self_attn.q_proj.weight") == (
        "model.layers.3.self_attn.q_proj.weight"
    )
    assert mapper("lm_head.weight") == "lm_head.weight"
    assert mapper("model.visual.blocks.0.weight") is None
    assert mapper("mtp.layers.0.weight") is None


def test_text_model_uses_declared_hybrid_layer_pattern():
    model = QWEN35.Qwen3_5MoeForConditionalGeneration(tiny_outer_config())

    assert [layer.layer_type for layer in model.model.layers] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    hidden = model(torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert hidden.shape == (2, 4)
    assert model.compute_logits(hidden).shape == (2, 17)
