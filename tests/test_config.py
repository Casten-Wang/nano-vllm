from types import SimpleNamespace
import json

import pytest

from nanovllm import config as config_module


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
        lambda _config: SimpleNamespace(text_config=text_config),
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


def test_invalid_qwen35_moe_decode_backend_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_moe_decode_backend"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_moe_decode_backend="unknown",
        )


def test_invalid_qwen35_moe_decode_chunk_size_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_moe_decode_chunk_size"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_moe_decode_chunk_size=0,
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
