from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import types
from unittest.mock import patch

import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "audit_checkpoint_mapping",
    ROOT / "scripts" / "audit_checkpoint_mapping.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parameter_storage_bytes_uses_parameter_dtype_and_shape():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=True, dtype=torch.bfloat16),
        torch.nn.Linear(4, 2, bias=False, dtype=torch.float32),
    )

    expected = (3 * 4 + 4) * 2 + (4 * 2) * 4

    assert MODULE.parameter_storage_bytes(model) == expected


def test_cache_storage_metadata_uses_tp_and_model_dtype():
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        max_position_embeddings=262_144,
    )
    model_spec = MODULE.resolve_model_spec(
        SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=SimpleNamespace(
                **vars(config),
                layer_types=["linear_attention", "full_attention"],
            ),
        )
    )

    result = MODULE.cache_storage_metadata(model_spec, 2, 2)

    assert result == {
        "model_max_position_embeddings": 262_144,
        "rotary_cache_bytes_per_position": 32,
        "kv_bytes_per_token": 32,
        "kv_bytes_per_token_by_dtype": {
            "auto": 32,
            "int8": 20,
        },
        "state_bytes_per_sequence": {
            "float32": 384,
            "model": 256,
            "convolution": 128,
        },
    }


def test_meta_model_falls_back_when_only_flash_attention_is_missing(monkeypatch):
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    spec = SimpleNamespace(architecture="Demo", text_config=config)
    calls = []

    monkeypatch.setattr(MODULE.AutoConfig, "from_pretrained", lambda _: config)
    monkeypatch.setattr(MODULE, "resolve_model_spec", lambda _: spec)
    monkeypatch.setattr(MODULE, "find_spec", lambda _: None)

    def fake_create_model(_architecture, _config):
        calls.append(MODULE.sys.modules.get("nanovllm.layers.attention"))
        module = calls[0]
        assert isinstance(module, types.ModuleType)
        return module.Attention(1, 1, 1.0, 1)

    monkeypatch.setattr(MODULE, "create_model", fake_create_model)

    model = MODULE.instantiate_meta_model("/model", 4)

    assert isinstance(model, torch.nn.Module)
    assert len(calls) == 1
