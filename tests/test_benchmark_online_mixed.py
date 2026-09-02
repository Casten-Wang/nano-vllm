from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_online_mixed",
    ROOT / "scripts" / "benchmark_online_mixed.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_explicit_prompt_lengths_are_preserved():
    assert MODULE.resolve_prompt_lengths(2, 128, (256, 1024), "initial") == (
        256,
        1024,
    )


def test_prompt_length_count_must_match_request_count():
    with pytest.raises(ValueError, match="exactly 2 values"):
        MODULE.resolve_prompt_lengths(2, 128, (256,), "initial")


def test_prompt_generation_supports_heterogeneous_lengths():
    prompts = MODULE.build_prompts_for_lengths((2, 5), vocab_size=17, seed=3)

    assert [len(prompt) for prompt in prompts] == [2, 5]
    assert all(0 <= token < 17 for prompt in prompts for token in prompt)


def test_workload_rejects_context_overflow_before_model_initialization():
    args = SimpleNamespace(
        initial_seqs=2,
        injected_seqs=1,
        initial_input_len=128,
        injected_input_len=128,
        initial_input_lens=(256, 1024),
        injected_input_lens=None,
        output_len=16,
        inject_after_decode_steps=1,
        max_model_len=1039,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        vocab_size=1000,
    )

    with pytest.raises(ValueError, match="exceeds max_model_len"):
        MODULE.validate_workload(args)
