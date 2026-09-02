from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_qwen35_quality_matrix",
    ROOT / "scripts" / "benchmark_qwen35_quality_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "model": "/models/qwen35",
        "prompt_lengths": "128,1024",
        "cases_per_length": 2,
        "continuation_len": 16,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "trace_max_events": 128,
        "trace_max_index_values": 32,
        "result_dir": "benchmark_results/quality",
        "cases_file": None,
        "run_id": "rental-a",
        "tp_sizes": (4, 8),
    }
    values.update(overrides)
    return Namespace(**values)


def summary(
    ppl_bf16,
    ppl_int8,
    *,
    top1=7,
    logprob=-2.0,
    top1_agreement=0.99,
    js_divergence=0.001,
):
    return {
        "case_token_digest": "cases",
        "batches": [
            {
                "decode_trajectories": {
                    "bf16_top1_token_ids": [[top1]],
                    "int8_top1_token_ids": [[top1]],
                    "bf16_target_logprobs": [[logprob]],
                    "int8_target_logprobs": [[logprob - 0.01]],
                }
            }
        ],
        "summary": {
            "decode_ppl": {
                "bf16": ppl_bf16,
                "int8": ppl_int8,
                "relative_change": ppl_int8 / ppl_bf16 - 1.0,
            },
            "decode_aggregate": {
                "top1_agreement": top1_agreement,
                "js_divergence": js_divergence,
            },
            "kv_sensitive_token_rows_compared": 100,
        }
    }


def test_quality_matrix_covers_tp_and_recurrent_state_modes():
    cases = MODULE.build_cases((4, 8))

    assert len(cases) == 4
    assert {case.tensor_parallel_size for case in cases} == {4, 8}
    assert {case.recurrent_state_dtype for case in cases} == {"float32", "model"}


def test_quality_case_command_forwards_all_quality_dimensions():
    case = MODULE.QualityCase(8, "model")
    command = MODULE.command_for_case(args(), case, "run-1")

    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--recurrent-state-dtype") + 1] == "model"
    assert command[command.index("--continuation-len") + 1] == "16"
    assert command[command.index("--name") + 1] == "run-1_qwen35_tp8_state-model"


def test_matrix_summary_separates_state_and_kv_quality_effects():
    cases = {
        MODULE.QualityCase(4, "float32"): summary(10.0, 10.5),
        MODULE.QualityCase(4, "model"): summary(10.2, 10.8),
    }

    result = MODULE.summarize_results(cases)
    comparison = result["comparisons_by_tp"]["tp4"]

    assert comparison["model_state_bf16_kv_relative_change"] == pytest.approx(0.02)
    assert comparison["float32_state_int8_kv_relative_change"] == pytest.approx(0.05)
    assert comparison["model_state_int8_kv_relative_change"] == pytest.approx(0.08)


def test_matrix_summary_requires_cross_tp_logit_parity():
    cases = {
        MODULE.QualityCase(4, "float32"): summary(10.0, 10.5),
        MODULE.QualityCase(4, "model"): summary(10.2, 10.8),
        MODULE.QualityCase(8, "float32"): summary(10.0, 10.5),
        MODULE.QualityCase(8, "model"): summary(
            10.2,
            10.8,
            logprob=-2.06,
        ),
    }

    result = MODULE.summarize_results(cases)

    assert not result["cross_tp"]["all_passed"]
    model_comparison = result["cross_tp"]["comparisons"][1]
    assert model_comparison["recurrent_state_dtype"] == "model"
    assert not model_comparison["modes"]["bf16"]["passed"]
    assert model_comparison["modes"]["bf16"][
        "max_target_logprob_diff"
    ] == pytest.approx(0.06)


def test_matrix_summary_rejects_int8_quality_regression():
    cases = {
        MODULE.QualityCase(4, "float32"): summary(
            10.0,
            10.5,
            top1_agreement=0.97,
        ),
        MODULE.QualityCase(4, "model"): summary(10.2, 10.8),
    }

    result = MODULE.summarize_results(cases)

    assert not result["quality_gates"]["all_passed"]
    assert not result["quality_gates"]["per_case"][0]["int8_top1"]
    assert result["quality_gates"]["thresholds"][
        "min_int8_top1_agreement"
    ] == 0.98


def test_cases_file_is_forwarded():
    command = MODULE.command_for_case(
        args(cases_file="quality_cases.json"),
        MODULE.QualityCase(4, "float32"),
        "run-1",
    )

    assert command[command.index("--cases-file") + 1] == "quality_cases.json"


def test_quality_matrix_requires_complete_checkpoint_audit():
    command = MODULE.checkpoint_audit_command(args(), "run-1")

    assert command[command.index("--tp-sizes") + 1] == "4,8"
    assert "--require-shards" in command
    assert command[command.index("--output") + 1].endswith(
        "run-1_checkpoint_audit.json"
    )
