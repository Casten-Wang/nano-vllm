from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
from types import SimpleNamespace
import types

import torch
from safetensors.torch import save_file


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
        "kv_data_bytes_per_token_by_dtype": {
            "auto": 32,
            "int8": 16,
        },
        "kv_scale_bytes_per_token_by_dtype": {
            "auto": 0,
            "int8": 4,
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
        assert _config.nanovllm_weight_quant_backend == "resident"
        calls.append(MODULE.sys.modules.get("nanovllm.layers.attention"))
        module = calls[0]
        assert isinstance(module, types.ModuleType)
        return module.Attention(1, 1, 1.0, 1)

    monkeypatch.setattr(MODULE, "create_model", fake_create_model)

    model = MODULE.instantiate_meta_model(
        "/model",
        4,
        weight_quant_backend="resident",
    )

    assert isinstance(model, torch.nn.Module)
    assert len(calls) == 1


def test_mapping_audit_classifies_intentionally_skipped_weights(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))

        @staticmethod
        def map_weight_name(name):
            if name.startswith("model.visual.") or name.startswith("mtp."):
                return None
            return name

    index = {
        "weight_map": {
            "weight": "model.safetensors",
            "model.visual.patch.weight": "model.safetensors",
            "mtp.layers.0.weight": "model.safetensors",
            "mtp.norm.weight": "model.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    result = MODULE.audit_checkpoint_mapping(Model(), tmp_path)

    assert result["valid"]
    assert result["skipped_tensor_count"] == 3
    assert result["skipped_tensor_groups"] == {"model.visual": 1, "mtp": 2}


def test_mapping_audit_validates_fp8_scale_and_expert_resolution(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Module()
            parameter = torch.nn.Parameter(torch.empty(2, 8, 4, device="meta"))
            parameter.weight_loader = lambda _param, _source, _shard: None
            parameter.required_checkpoint_shards = frozenset(
                (expert_id, projection)
                for expert_id in range(2)
                for projection in ("gate", "up")
            )
            self.experts.register_parameter("gate_up_proj", parameter)
            self.checkpoint_quantization_spec = SimpleNamespace(
                format="fp8_block",
                weight_block_size=(2, 2),
            )

        @staticmethod
        def resolve_checkpoint_parameter(name):
            prefix = "experts."
            if not name.startswith(prefix) or not name.endswith("_proj.weight"):
                return None
            expert_id = int(name.split(".")[1])
            projection = name.split(".")[2].removesuffix("_proj")
            return "experts.gate_up_proj", (expert_id, projection)

    tensors = {}
    for expert_id in range(2):
        for projection in ("gate", "up"):
            name = f"experts.{expert_id}.{projection}_proj.weight"
            tensors[name] = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
            tensors[f"{name}_scale_inv"] = torch.ones(2, 2)
    save_file(tensors, tmp_path / "model.safetensors")

    result = MODULE.audit_checkpoint_mapping(Model(), tmp_path)

    assert result["valid"]
    assert result["source_tensor_count"] == 8
    assert result["mapped_parameter_count"] == 1
    assert result["incomplete_checkpoint_shards"] == []

    del tensors["experts.1.up_proj.weight"]
    del tensors["experts.1.up_proj.weight_scale_inv"]
    save_file(tensors, tmp_path / "model.safetensors")

    incomplete = MODULE.audit_checkpoint_mapping(Model(), tmp_path)

    assert not incomplete["valid"]
    assert incomplete["incomplete_checkpoint_shards"][0]["parameter"] == (
        "experts.gate_up_proj"
    )


def test_mapping_audit_rejects_invalid_fp8_scale_shape(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(
                4,
                4,
                bias=False,
                device="meta",
            )
            self.checkpoint_quantization_spec = SimpleNamespace(
                format="fp8_block",
                weight_block_size=(2, 2),
            )

    save_file(
        {
            "linear.weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn),
            "linear.weight_scale_inv": torch.ones(1, 1),
        },
        tmp_path / "model.safetensors",
    )

    result = MODULE.audit_checkpoint_mapping(Model(), tmp_path)

    assert not result["valid"]
    assert "invalid FP8 scale shape" in result["shape_errors"][0]["error"]
