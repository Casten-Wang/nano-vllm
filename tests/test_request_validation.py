import sys
import types
from types import SimpleNamespace

import pytest

flash_attn = types.ModuleType("flash_attn")
flash_attn.flash_attn_varlen_func = object
flash_attn.flash_attn_with_kvcache = object
sys.modules.setdefault("flash_attn", flash_attn)

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_sampling_params_reject_invalid_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="positive integer"):
        SamplingParams(max_tokens=max_tokens)


def test_request_length_rejects_total_over_limit():
    with pytest.raises(ValueError, match="9 tokens.*max_model_len=8"):
        SamplingParams(max_tokens=4).validate_request_length(5, 8)


def test_request_length_rejects_empty_prompt():
    with pytest.raises(ValueError, match="at least one token"):
        SamplingParams(max_tokens=1).validate_request_length(0, 8)


def test_request_length_accepts_total_at_limit():
    SamplingParams(max_tokens=4).validate_request_length(4, 8)


def test_engine_validates_string_prompt_after_tokenization():
    engine = object.__new__(LLMEngine)
    engine.max_model_len = 3
    engine.tokenizer = SimpleNamespace(encode=lambda _: [1, 2, 3])
    engine.scheduler = SimpleNamespace(add=lambda _: pytest.fail("scheduled"))

    with pytest.raises(ValueError, match="4 tokens.*max_model_len=3"):
        engine.add_request("prompt", SamplingParams(max_tokens=1))


@pytest.mark.parametrize("prompt", [[], ""])
def test_engine_rejects_empty_tokenized_prompt(prompt):
    engine = object.__new__(LLMEngine)
    engine.max_model_len = 8
    engine.tokenizer = SimpleNamespace(encode=lambda _: [])
    engine.scheduler = SimpleNamespace(add=lambda _: pytest.fail("scheduled"))

    with pytest.raises(ValueError, match="at least one token"):
        engine.add_request(prompt, SamplingParams(max_tokens=1))
