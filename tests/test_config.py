from types import SimpleNamespace

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
    )

    assert config.qwen35_moe_decode_backend == "batched"
    assert text_config.qwen35_moe_decode_backend == "batched"


def test_invalid_qwen35_moe_decode_backend_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen35_moe_decode_backend"):
        make_config(
            monkeypatch,
            tmp_path,
            qwen35_moe_decode_backend="unknown",
        )
