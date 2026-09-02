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
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def forward(self, query, key, value):
        repeats = self.num_heads // self.num_kv_heads
        return (
            query
            + key.repeat_interleave(repeats, dim=1)
            + value.repeat_interleave(repeats, dim=1)
        )


attention_module = types.ModuleType("nanovllm.layers.attention")
attention_module.Attention = FakeAttention
sys.modules["nanovllm.layers.attention"] = attention_module

moe_module = types.ModuleType("nanovllm.models.qwen35_moe")


class DeltaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x, *, inplace_output=False):
        output = x.float().mul(torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).mul(
            1 + self.weight.float()
        ).to(x.dtype)
        if inplace_output and not torch.is_grad_enabled():
            x.copy_(output)
            return x
        return output


moe_module.Qwen35RMSNorm = DeltaRMSNorm
sys.modules["nanovllm.models.qwen35_moe"] = moe_module
attention = load_module(
    "qwen35_attention_under_test",
    "nanovllm/models/qwen35_attention.py",
)


def config(
    *,
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=4,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        rope_parameters={
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000,
        },
    )


def make_attention(rank=0, world_size=1, layer_config=None):
    rotary.get_rope.cache_clear()
    with (
        patch.object(attention.dist, "get_world_size", return_value=world_size),
        patch.object(attention.dist, "get_rank", return_value=rank),
        patch.object(linear.dist, "get_world_size", return_value=world_size),
        patch.object(linear.dist, "get_rank", return_value=rank),
    ):
        return attention.Qwen35Attention(layer_config or config())


def test_attention_shapes_include_local_query_gate_and_replicated_kv():
    layer = make_attention(rank=3, world_size=4)

    assert layer.qkv_proj.weight.shape == (16, 8)
    assert layer.qkv_proj.q_head_size == 8
    assert layer.qkv_proj.num_kv_heads == 1
    assert layer.qkv_proj.num_kv_head_replicas == 4
    assert layer.rotary_emb.head_size == 4
    assert layer.rotary_emb.cos_sin_cache.shape[-1] == 2


def test_attention_bounds_rope_cache_to_runtime_model_length():
    layer_config = config()
    layer_config.max_position_embeddings = 262_144
    layer_config.nanovllm_max_model_len = 4096

    layer = make_attention(layer_config=layer_config)

    assert layer.rotary_emb.cos_sin_cache.shape[0] == 4096


def test_query_gate_is_applied_after_attention():
    layer = make_attention()
    hidden = torch.randn(3, 8)
    positions = torch.arange(3)
    with torch.no_grad():
        layer.qkv_proj.weight.zero_()
        # Per head: first head_dim rows are query, second head_dim are gate.
        for head in range(4):
            start = head * 8
            layer.qkv_proj.weight[start : start + 4, :4] = torch.eye(4)
        layer.o_proj.weight.fill_(0.1)

    output_closed = layer(positions, hidden)
    with torch.no_grad():
        for head in range(4):
            start = head * 8
            layer.qkv_proj.weight[start + 4 : start + 8].fill_(20)
    output_open = layer(positions, hidden)

    assert not torch.allclose(output_closed, output_open)


def test_attention_uses_one_packed_qkv_projection():
    layer = make_attention()
    calls = []
    hook = layer.qkv_proj.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(True)
    )
    try:
        layer(torch.arange(3), torch.randn(3, 8))
    finally:
        hook.remove()

    assert calls == [True]


@torch.no_grad()
def test_query_gate_reuses_attention_output_storage_in_inference():
    layer = make_attention()
    storage = {}
    attention_hook = layer.attn.register_forward_hook(
        lambda _module, _inputs, output: storage.update(attention=output.data_ptr())
    )
    projection_hook = layer.o_proj.register_forward_pre_hook(
        lambda _module, inputs: storage.update(gated=inputs[0].data_ptr())
    )

    try:
        layer(torch.arange(3), torch.randn(3, 8))
    finally:
        attention_hook.remove()
        projection_hook.remove()

    assert storage["gated"] == storage["attention"]


@torch.no_grad()
def test_attention_norms_reuse_query_and_key_projection_storage():
    layer = make_attention()
    storage = {}
    query_hook = layer.q_norm.register_forward_hook(
        lambda _module, inputs, output: storage.update(
            query_input=inputs[0].data_ptr(),
            query_output=output.data_ptr(),
        )
    )
    key_hook = layer.k_norm.register_forward_hook(
        lambda _module, inputs, output: storage.update(
            key_input=inputs[0].data_ptr(),
            key_output=output.data_ptr(),
        )
    )

    try:
        layer(torch.arange(3), torch.randn(3, 8))
    finally:
        query_hook.remove()
        key_hook.remove()

    assert storage["query_output"] == storage["query_input"]
    assert storage["key_output"] == storage["key_input"]


def test_query_gate_keeps_separate_autograd_output():
    layer = make_attention()
    storage = {}
    attention_hook = layer.attn.register_forward_hook(
        lambda _module, _inputs, output: storage.update(attention=output.data_ptr())
    )
    projection_hook = layer.o_proj.register_forward_pre_hook(
        lambda _module, inputs: storage.update(gated=inputs[0].data_ptr())
    )
    hidden = torch.randn(3, 8, requires_grad=True)

    try:
        output = layer(torch.arange(3), hidden)
        output.sum().backward()
    finally:
        attention_hook.remove()
        projection_hook.remove()

    assert storage["gated"] != storage["attention"]
    assert hidden.grad is not None


def test_tensor_parallel_attention_sums_to_single_rank_reference():
    torch.manual_seed(9)
    full = make_attention(world_size=1)
    ranks = [make_attention(rank=rank, world_size=2) for rank in range(2)]
    sources = {
        "q_proj.weight": torch.randn(32, 8),
        "k_proj.weight": torch.randn(4, 8),
        "v_proj.weight": torch.randn(4, 8),
        "o_proj.weight": torch.randn(8, 16),
        "q_norm.weight": torch.randn(4),
        "k_norm.weight": torch.randn(4),
    }

    def load(layer):
        for name, source in sources.items():
            packed_shard = {
                "q_proj.weight": "q",
                "k_proj.weight": "k",
                "v_proj.weight": "v",
            }.get(name)
            if packed_shard is not None:
                layer.qkv_proj.weight.weight_loader(
                    layer.qkv_proj.weight,
                    source,
                    packed_shard,
                )
                continue
            parameter = layer.get_parameter(name)
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.data.copy_(source)
            else:
                loader(parameter, source)

    load(full)
    for rank_layer in ranks:
        load(rank_layer)
    hidden = torch.randn(3, 8)
    positions = torch.arange(3)
    with patch.object(linear.dist, "all_reduce", return_value=None):
        expected = full(positions, hidden)
        actual = sum(layer(positions, hidden) for layer in ranks)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@torch.no_grad()
def test_official_head_layout_matches_across_tp4_and_tp8():
    torch.manual_seed(29)
    layer_config = config(
        hidden_size=8,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=4,
    )
    full = make_attention(world_size=1, layer_config=layer_config)
    sources = {
        "q_proj.weight": torch.randn(128, 8),
        "k_proj.weight": torch.randn(8, 8),
        "v_proj.weight": torch.randn(8, 8),
        "o_proj.weight": torch.randn(8, 64),
        "q_norm.weight": torch.randn(4),
        "k_norm.weight": torch.randn(4),
    }

    def load(layer):
        for name, source in sources.items():
            packed_shard = {
                "q_proj.weight": "q",
                "k_proj.weight": "k",
                "v_proj.weight": "v",
            }.get(name)
            if packed_shard is not None:
                layer.qkv_proj.weight.weight_loader(
                    layer.qkv_proj.weight,
                    source,
                    packed_shard,
                )
                continue
            parameter = layer.get_parameter(name)
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.copy_(source)
            else:
                loader(parameter, source)

    load(full)
    hidden = torch.randn(5, 8)
    positions = torch.arange(5)
    expected = full(positions, hidden)

    for tp_size in (4, 8):
        ranks = [
            make_attention(rank=rank, world_size=tp_size, layer_config=layer_config)
            for rank in range(tp_size)
        ]
        for rank_layer in ranks:
            load(rank_layer)
        with patch.object(linear.dist, "all_reduce", return_value=None):
            actual = sum(rank_layer(positions, hidden) for rank_layer in ranks)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
