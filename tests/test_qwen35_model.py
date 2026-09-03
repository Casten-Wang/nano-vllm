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
        self.num_kv_heads = 1
        self.head_dim = 2

    def forward(self, *args):
        return torch.zeros_like(args[-1])


class FakeNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.last_inplace_output = None

    def forward(self, x, *, inplace_output=False):
        self.last_inplace_output = inplace_output
        return x


class FakeMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = (
            FakeGPTQExperts()
            if getattr(config, "use_fake_gptq", False)
            else FakeExperts()
        )

    def forward(self, x):
        return torch.zeros_like(x)


class FakeExperts(nn.Module):
    pass


class FakeGPTQExperts(nn.Module):
    backend = "triton"
    local_intermediate_size = 3
    hidden_size = 4


class FakeGPTQWorkspacePool:
    def __init__(self):
        self.reservations = []

    def reserve(self, *args, **kwargs):
        self.reservations.append((args, kwargs))


class FakeWeightBufferPool:
    def __init__(self):
        self.reservations = []

    def reserve(self, **kwargs):
        self.reservations.append(kwargs)


def load_qwen35_module():
    modules = {
        "nanovllm": types.ModuleType("nanovllm"),
        "nanovllm.layers": types.ModuleType("nanovllm.layers"),
        "nanovllm.models": types.ModuleType("nanovllm.models"),
        "nanovllm.models.moe_dispatch": types.ModuleType(
            "nanovllm.models.moe_dispatch"
        ),
        "nanovllm.layers.embed_head": types.ModuleType("nanovllm.layers.embed_head"),
        "nanovllm.models.qwen35_attention": types.ModuleType("nanovllm.models.qwen35_attention"),
        "nanovllm.models.qwen35_gated_delta": types.ModuleType("nanovllm.models.qwen35_gated_delta"),
        "nanovllm.models.qwen35_gptq": types.ModuleType("nanovllm.models.qwen35_gptq"),
        "nanovllm.models.qwen35_moe": types.ModuleType("nanovllm.models.qwen35_moe"),
    }
    modules["nanovllm.layers.embed_head"].ParallelLMHead = FakeHead
    modules["nanovllm.layers.embed_head"].VocabParallelEmbedding = FakeEmbedding
    modules["nanovllm.models.qwen35_attention"].Qwen35Attention = FakeMixer
    modules[
        "nanovllm.models.qwen35_attention"
    ].Qwen35KeyBufferPool = FakeWeightBufferPool
    modules["nanovllm.models.qwen35_gated_delta"].Qwen35GatedDeltaNet = FakeMixer
    modules["nanovllm.models.qwen35_gptq"].GPTQExpertWorkspacePool = (
        FakeGPTQWorkspacePool
    )
    modules["nanovllm.models.qwen35_gptq"].Qwen35GPTQExperts = FakeGPTQExperts
    modules["nanovllm.models.qwen35_moe"].Qwen35RMSNorm = FakeNorm
    modules["nanovllm.models.qwen35_moe"].Qwen35SparseMoeBlock = FakeMoe
    modules["nanovllm.models.qwen35_moe"].Qwen35Experts = FakeExperts
    modules[
        "nanovllm.models.qwen35_moe"
    ].ResidentFP8WeightBufferPool = FakeWeightBufferPool
    modules[
        "nanovllm.models.moe_dispatch"
    ].BatchedExpertWeightBufferPool = FakeWeightBufferPool
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
        nanovllm_max_num_batched_tokens=11,
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
    assert model.model.norm.last_inplace_output is True
    pool = model.model.moe_decode_weight_buffer_pool
    assert isinstance(pool, FakeWeightBufferPool)
    assert all(
        layer.mlp.experts.decode_weight_buffer_pool is pool
        for layer in model.model.layers
    )
    key_pool = model.model.full_attention_key_buffer_pool
    assert isinstance(key_pool, FakeWeightBufferPool)
    assert model.model.layers[-1].self_attn.key_buffer_pool is key_pool


def test_text_model_reserves_batched_moe_buffers_for_max_decode_chunk():
    model = QWEN35.Qwen3_5MoeForConditionalGeneration(tiny_outer_config())
    block = model.model.layers[0].mlp
    block.gate = SimpleNamespace(top_k=2)
    block.experts.decode_backend = "batched"
    block.experts.decode_chunk_size = 3
    block.experts.gate_up_proj = torch.empty(4, 6, 4, dtype=torch.bfloat16)
    block.experts.down_proj = torch.empty(4, 4, 3, dtype=torch.bfloat16)

    model.model.reserve_runtime_buffers(max_decode_tokens=8)

    assert model.model.full_attention_key_buffer_pool.reservations == [
        {
            "elements": 11 * 1 * 2,
            "dtype": torch.float32,
            "device": torch.device("cpu"),
        }
    ]
    assert model.model.moe_decode_weight_buffer_pool.reservations == [
        {
            "weight_elements": 3 * 2 * 6 * 4,
            "workspace_elements": 3 * 2 * (6 + 4),
            "weight_dtype": torch.bfloat16,
            "activation_dtype": torch.bfloat16,
            "device": torch.device("cpu"),
        }
    ]


def test_text_model_shares_and_reserves_one_gptq_workspace():
    config = tiny_outer_config()
    config.text_config.use_fake_gptq = True
    model = QWEN35.Qwen3_5MoeForConditionalGeneration(config)

    pool = model.model.gptq_expert_workspace_pool
    assert all(
        layer.mlp.experts.gptq_workspace_pool is pool
        for layer in model.model.layers
    )

    model.model.reserve_runtime_buffers(max_decode_tokens=8)

    assert pool.reservations == [
        (
            (8, 3, 4),
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
            },
        )
    ]


def test_residual_merge_reuses_branch_output_during_inference():
    residual = torch.randn(3, 4)
    branch = torch.randn(3, 4)
    expected = residual + branch
    branch_storage = branch.data_ptr()

    with torch.inference_mode():
        output = QWEN35._add_residual(branch, residual)

    assert output.data_ptr() == branch_storage
    torch.testing.assert_close(output, expected)


def test_residual_merge_preserves_autograd_inputs():
    residual = torch.randn(3, 4, requires_grad=True)
    branch = torch.randn(3, 4, requires_grad=True)
    residual_before = residual.detach().clone()
    branch_before = branch.detach().clone()

    output = QWEN35._add_residual(branch, residual)
    output.square().mean().backward()

    assert output.data_ptr() != branch.data_ptr()
    assert torch.equal(residual, residual_before)
    assert torch.equal(branch, branch_before)
    assert residual.grad is not None
    assert branch.grad is not None
