from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nanovllm = sys.modules.setdefault("nanovllm", types.ModuleType("nanovllm"))
layers = sys.modules.setdefault("nanovllm.layers", types.ModuleType("nanovllm.layers"))
models = sys.modules.setdefault("nanovllm.models", types.ModuleType("nanovllm.models"))
linear = load_module("nanovllm.layers.linear", "nanovllm/layers/linear.py")
rotary = load_module(
    "nanovllm.layers.rotary_embedding",
    "nanovllm/layers/rotary_embedding.py",
)


class FakeAttention(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, query, key, value):
        return query


attention_module = types.ModuleType("nanovllm.layers.attention")
attention_module.Attention = FakeAttention
sys.modules["nanovllm.layers.attention"] = attention_module

moe_module = types.ModuleType("nanovllm.models.qwen35_moe")


class DeltaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x.float().mul(torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).mul(
            1 + self.weight.float()
        ).to(x.dtype)


moe_module.Qwen35RMSNorm = DeltaRMSNorm
sys.modules["nanovllm.models.qwen35_moe"] = moe_module
attention = load_module(
    "qwen35_attention_under_test",
    "nanovllm/models/qwen35_attention.py",
)


def config():
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_parameters={
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000,
        },
    )


def make_attention(rank=0, world_size=1):
    rotary.get_rope.cache_clear()
    with (
        patch.object(attention.dist, "get_world_size", return_value=world_size),
        patch.object(attention.dist, "get_rank", return_value=rank),
        patch.object(linear.dist, "get_world_size", return_value=world_size),
        patch.object(linear.dist, "get_rank", return_value=rank),
    ):
        return attention.Qwen35Attention(config())


def test_attention_shapes_include_local_query_gate_and_replicated_kv():
    layer = make_attention(rank=3, world_size=4)

    assert layer.q_proj.weight.shape == (8, 8)
    assert layer.k_proj.weight.shape == (4, 8)
    assert layer.v_proj.weight.shape == (4, 8)
    assert layer.k_proj.num_kv_head_replicas == 4
    assert layer.rotary_emb.head_size == 4
    assert layer.rotary_emb.cos_sin_cache.shape[-1] == 2


def test_query_gate_is_applied_after_attention():
    layer = make_attention()
    hidden = torch.randn(3, 8)
    positions = torch.arange(3)
    with torch.no_grad():
        layer.q_proj.weight.zero_()
        # Per head: first head_dim rows are query, second head_dim are gate.
        for head in range(4):
            start = head * 8
            layer.q_proj.weight[start : start + 4, :4] = torch.eye(4)
        layer.k_proj.weight.zero_()
        layer.v_proj.weight.zero_()
        layer.o_proj.weight.fill_(0.1)

    output_closed = layer(positions, hidden)
    with torch.no_grad():
        for head in range(4):
            start = head * 8
            layer.q_proj.weight[start + 4 : start + 8].fill_(20)
    output_open = layer(positions, hidden)

    assert not torch.allclose(output_closed, output_open)
