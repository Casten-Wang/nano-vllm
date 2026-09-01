from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

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
        "kv_bytes_per_token": 32,
        "state_bytes_per_sequence": {
            "float32": 384,
            "model": 256,
            "convolution": 128,
        },
    }
