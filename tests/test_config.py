from types import SimpleNamespace
import json

import pytest

from nanovllm import config as config_module
from nanovllm.models.quantization_spec import (
    BF16_QUANTIZATION_SPEC,
    QuantizationSpec,
)


def make_config(monkeypatch, tmp_path, **kwargs):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_config=text_config,
            quantization=BF16_QUANTIZATION_SPEC,
        ),
    )
    return config_module.Config(str(tmp_path), **kwargs), text_config


def test_qwen35_moe_decode_backend_is_forwarded_to_text_config(
    monkeypatch,
    tmp_path,
):
    config, text_config = make_config(
        monkeypatch,
        tmp_path,
        qwen35_moe_decode_backend="batched",
        qwen35_moe_decode_chunk_size=4,
    )

    assert config.qwen35_moe_decode_backend == "batched"
    assert text_config.qwen35_moe_decode_backend == "batched"
    assert config.qwen35_moe_decode_chunk_size == 4
    assert text_config.qwen35_moe_decode_chunk_size == 4


def test_qwen35_decode_conv_backend_is_forwarded_to_text_config(
    monkeypatch,
    tmp_path,
):
    config, text_config = make_config(
        monkeypatch,
        tmp_path,
        qwen35_decode_conv_backend="channel_accumulate",
    )

    assert config.qwen35_decode_conv_backend == "channel_accumulate"
    assert text_config.qwen35_decode_conv_backend == "channel_accumulate"


def test_runtime_model_length_is_forwarded_as_allocation_bound(
    monkeypatch,
    tmp_path,
):
    config, text_config = make_config(
        monkeypatch,
        tmp_path,
        max_model_len=4096,
    )

    assert config.max_model_len == 4096
    assert text_config.max_position_embeddings == 32768
    assert text_config.nanovllm_max_model_len == 4096


def test_runtime_batch_token_bound_is_forwarded_for_buffer_reservations(
    monkeypatch,
    tmp_path,
):
    config, text_config = make_config(
        monkeypatch,
        tmp_path,
        max_num_batched_tokens=2048,
    )

    assert config.max_num_batched_tokens == 2048
    assert text_config.nanovllm_max_num_batched_tokens == 2048


def test_invalid_qwen35_moe_decode_backend_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_moe_decode_backend"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_moe_decode_backend="unknown",
        )


def test_invalid_qwen35_decode_conv_backend_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_decode_conv_backend"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_decode_conv_backend="unknown",
        )


def test_invalid_qwen35_moe_decode_chunk_size_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_moe_decode_chunk_size"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_moe_decode_chunk_size=0,
        )


def test_invalid_kv_block_override_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="num_kvcache_blocks_override"):
        make_config(monkeypatch, tmp_path, num_kvcache_blocks_override=0)


def test_invalid_preemption_policy_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="preemption_policy"):
        make_config(monkeypatch, tmp_path, preemption_policy="smallest")


def test_decode_kv_reservation_requires_boolean(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="enable_decode_kv_reservation"):
        make_config(monkeypatch, tmp_path, enable_decode_kv_reservation=1)


def test_negative_prefill_starvation_threshold_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="prefill_starvation_threshold"):
        make_config(
            monkeypatch,
            tmp_path,
            prefill_starvation_threshold=-1,
        )


def test_non_positive_prefill_starvation_token_budget_is_rejected(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(ValueError, match="prefill_starvation_token_budget"):
        make_config(
            monkeypatch,
            tmp_path,
            prefill_starvation_token_budget=0,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_remote_prefill_transfer_limit_is_rejected(
    monkeypatch,
    tmp_path,
    value,
):
    with pytest.raises(ValueError, match="max_remote_prefill_transfers"):
        make_config(
            monkeypatch,
            tmp_path,
            max_remote_prefill_transfers=value,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_remote_prefill_staging_limit_is_rejected(
    monkeypatch,
    tmp_path,
    value,
):
    with pytest.raises(ValueError, match="max_remote_prefill_staging_bytes"):
        make_config(
            monkeypatch,
            tmp_path,
            max_remote_prefill_staging_bytes=value,
        )


def test_gptq_auto_selects_triton_backend(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_config=text_config,
            quantization=QuantizationSpec(format="gptq_int4", weight_bits=4),
        ),
    )

    config = config_module.Config(str(tmp_path))

    assert config.weight_quant_backend == "triton"
    assert text_config.nanovllm_weight_quant_backend == "triton"


def test_fp8_checkpoint_selects_reference_dequantization_backend(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_config=text_config,
            quantization=QuantizationSpec(
                format="fp8_block",
                weight_bits=8,
                weight_block_size=(128, 128),
            ),
        ),
    )

    config = config_module.Config(str(tmp_path))

    assert config.weight_quant_backend == "reference"
    assert text_config.nanovllm_weight_quant_backend == "reference"
    assert text_config.nanovllm_quantization_spec.weight_block_size == (128, 128)


def test_fp8_checkpoint_rejects_unimplemented_native_backend(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_config=text_config,
            quantization=QuantizationSpec(
                format="fp8_block",
                weight_bits=8,
                weight_block_size=(128, 128),
            ),
        ),
    )

    with pytest.raises(ValueError, match="require.*reference.*resident"):
        config_module.Config(str(tmp_path), weight_quant_backend="triton")


def test_fp8_checkpoint_accepts_resident_reference_backend(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_config=text_config,
            quantization=QuantizationSpec(
                format="fp8_block",
                weight_bits=8,
                weight_block_size=(128, 128),
            ),
        ),
    )

    config = config_module.Config(
        str(tmp_path),
        weight_quant_backend="resident",
    )

    assert config.weight_quant_backend == "resident"
    assert text_config.nanovllm_weight_quant_backend == "resident"


def test_fp8_reference_loader_is_scoped_to_qwen35(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            architecture="Qwen3ForCausalLM",
            text_config=text_config,
            quantization=QuantizationSpec(
                format="fp8_block",
                weight_bits=8,
                weight_block_size=(128, 128),
            ),
        ),
    )

    with pytest.raises(NotImplementedError, match="only for Qwen3.6-compatible MoE"):
        config_module.Config(str(tmp_path))


def test_gptq_reference_backend_is_explicitly_admitted(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    quantization = QuantizationSpec(
        format="gptq_int4",
        weight_bits=4,
        group_size=128,
        symmetric=True,
        desc_act=False,
    )
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            text_config=text_config,
            quantization=quantization,
        ),
    )

    config = config_module.Config(
        str(tmp_path),
        weight_quant_backend="reference",
    )

    assert config.model_config.nanovllm_quantization_spec is quantization


def test_bf16_rejects_gptq_backend(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires a GPTQ-Int4 checkpoint"):
        make_config(monkeypatch, tmp_path, weight_quant_backend="triton")


def test_gptq_triton_backend_is_forwarded(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    quantization = QuantizationSpec(
        format="gptq_int4",
        weight_bits=4,
        group_size=128,
    )
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            text_config=text_config,
            quantization=quantization,
        ),
    )

    config_module.Config(str(tmp_path), weight_quant_backend="triton")

    assert text_config.nanovllm_weight_quant_backend == "triton"


def test_gptq_reference_rejects_incompatible_moe_backend(monkeypatch, tmp_path):
    text_config = SimpleNamespace(max_position_embeddings=32768)
    quantization = QuantizationSpec(
        format="gptq_int4",
        weight_bits=4,
        group_size=128,
    )
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(),
    )
    monkeypatch.setattr(
        config_module,
        "resolve_model_spec",
        lambda _config: SimpleNamespace(
            text_config=text_config,
            quantization=quantization,
        ),
    )

    with pytest.raises(ValueError, match="requires.*sorted"):
        config_module.Config(
            str(tmp_path),
            weight_quant_backend="reference",
            qwen35_moe_decode_backend="batched",
        )


def test_generation_config_resolves_all_eos_tokens(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [248046, 248044, 248046]}),
        encoding="utf-8",
    )

    assert config_module.resolve_eos_token_ids(str(tmp_path), 7) == (
        248046,
        248044,
    )


def test_generation_config_falls_back_to_tokenizer_eos(tmp_path):
    assert config_module.resolve_eos_token_ids(str(tmp_path), 7) == (7,)


def test_sampling_chunk_size_must_be_positive(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="sampling_chunk_size must be positive"):
        make_config(monkeypatch, tmp_path, sampling_chunk_size=0)
