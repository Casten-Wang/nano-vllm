from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODEL_SPEC_PATH = (
    Path(__file__).parents[1] / "nanovllm" / "models" / "model_spec.py"
)
SPEC = spec_from_file_location("model_spec", MODEL_SPEC_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL_SPEC = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL_SPEC
SPEC.loader.exec_module(MODEL_SPEC)
resolve_model_spec = MODEL_SPEC.resolve_model_spec


def test_dense_qwen_uses_every_layer_for_kv_cache():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=4,
    )

    spec = resolve_model_spec(config)

    assert spec.text_config is config
    assert spec.full_attention_layers == (0, 1, 2, 3)
    assert spec.linear_attention_layers == ()
    assert spec.num_kv_cache_layers == 4
    assert not spec.is_hybrid


def test_qwen35_conditional_generation_uses_nested_text_config():
    text_config = SimpleNamespace(
        num_hidden_layers=8,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
        * 2,
    )
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text_config,
    )

    spec = resolve_model_spec(config)

    assert spec.text_config is text_config
    assert spec.full_attention_layers == (3, 7)
    assert spec.linear_attention_layers == (0, 1, 2, 4, 5, 6)
    assert spec.num_kv_cache_layers == 2
    assert spec.is_hybrid


def test_qwen35_rejects_inconsistent_layer_metadata():
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=("linear_attention",),
    )
    config = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text_config,
    )

    with pytest.raises(ValueError, match="layer_types length"):
        resolve_model_spec(config)


def test_unknown_architecture_is_rejected_early():
    config = SimpleNamespace(
        architectures=["UnknownForCausalLM"],
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="unsupported model architecture"):
        resolve_model_spec(config)
