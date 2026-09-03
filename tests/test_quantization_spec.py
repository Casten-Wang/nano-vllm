from types import SimpleNamespace

import pytest

from nanovllm.models.quantization_spec import resolve_quantization_spec


def resolve(config=None):
    return resolve_quantization_spec(SimpleNamespace(quantization_config=config))


def test_unquantized_checkpoint_resolves_to_bf16():
    spec = resolve()

    assert spec.format == "bf16"
    assert spec.weight_bits == 16
    assert not spec.is_quantized
    spec.require_runtime_support()


def test_official_block_fp8_config_is_normalized():
    spec = resolve(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head", "model.layers.1.mlp.gate"],
        }
    )

    assert spec.format == "fp8_block"
    assert spec.weight_bits == 8
    assert spec.weight_block_size == (128, 128)
    assert spec.activation_scheme == "dynamic"
    assert spec.ignores_module("lm_head")
    assert spec.ignores_module("model.layers.1.mlp.gate.weight")
    assert not spec.ignores_module("model.layers.10.mlp.gate")


def test_qwen36_block_fp8_defaults_are_normalized():
    spec = resolve(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
    )

    assert spec.format == "fp8_block"
    assert spec.weight_block_size == (128, 128)


@pytest.mark.parametrize("field", ("weight_per_tensor", "act_per_tensor"))
def test_fp8_explicit_per_tensor_mode_is_rejected(field):
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        field: True,
    }
    with pytest.raises(ValueError, match=field):
        resolve(config)


def test_unverified_fp8_format_is_rejected():
    with pytest.raises(ValueError, match="fmt"):
        resolve(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e5m2",
                "weight_block_size": [128, 128],
            }
        )


def test_official_gptq_int4_config_is_normalized():
    spec = resolve(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "dynamic": {"-:.*attn.*": {}, "lm_head": {}},
            "modules_to_not_convert": ["model.embed_tokens"],
        }
    )

    assert spec.format == "gptq_int4"
    assert spec.weight_bits == 4
    assert spec.group_size == 128
    assert spec.symmetric
    assert spec.desc_act is False
    assert spec.ignores_module("model.layers.3.self_attn.q_proj")
    assert spec.ignores_module("model.embed_tokens.weight")
    assert not spec.ignores_module("lm_head")


@pytest.mark.parametrize("block", ([128], [128, 0], [128, True], [64, 128]))
def test_malformed_or_unverified_fp8_block_size_is_rejected(block):
    with pytest.raises(ValueError, match="weight_block_size"):
        resolve(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_per_tensor": False,
                "act_per_tensor": False,
                "weight_block_size": block,
            }
        )


@pytest.mark.parametrize("group_size", (0, True, 64))
def test_malformed_or_unverified_gptq_group_size_is_rejected(group_size):
    with pytest.raises(ValueError, match="group_size"):
        resolve(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": group_size,
                "sym": True,
                "desc_act": False,
            }
        )


def test_unsupported_quantization_method_is_rejected():
    with pytest.raises(ValueError, match="unsupported checkpoint quantization"):
        resolve({"quant_method": "awq"})


def test_quantized_format_is_not_mistaken_for_executable_support():
    spec = resolve(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
    )

    with pytest.raises(NotImplementedError, match="recognized but not executable"):
        spec.require_runtime_support()


def test_invalid_gptq_exclusion_regex_is_rejected():
    with pytest.raises(ValueError, match="invalid GPTQ exclusion pattern"):
        resolve(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": False,
                "dynamic": {"-:[": {}},
            }
        )
