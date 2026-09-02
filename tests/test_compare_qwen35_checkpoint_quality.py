from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "compare_qwen35_checkpoint_quality",
    ROOT / "scripts" / "compare_qwen35_checkpoint_quality.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(tp_size, state_dtype, digest, top1, logprobs):
    return {
        "commit": "abc",
        "case_token_digest": "same-cases",
        "checkpoint_manifest": {"digest": digest},
        "configuration": {
            "tensor_parallel_size": tp_size,
            "recurrent_state_dtype": state_dtype,
        },
        "batches": [
            {
                "decode_trajectories": {
                    "bf16_top1_token_ids": [top1],
                    "bf16_target_logprobs": [logprobs],
                }
            }
        ],
    }


def write_case(root, run_id, tp_size, state_dtype, value):
    path = MODULE.case_path(root, run_id, tp_size, state_dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_checkpoint_quality_compares_each_tp_and_state_case(tmp_path):
    baseline_dir = tmp_path / "bf16"
    candidate_dir = tmp_path / "gptq"
    for tp_size in (4, 8):
        for state_dtype in ("float32", "model"):
            write_case(
                baseline_dir,
                "base",
                tp_size,
                state_dtype,
                result(tp_size, state_dtype, "bf16", [1, 2], [-1.0, -2.0]),
            )
            write_case(
                candidate_dir,
                "quant",
                tp_size,
                state_dtype,
                result(tp_size, state_dtype, "gptq", [1, 2], [-1.1, -2.1]),
            )

    report = MODULE.compare_quality_runs(
        baseline_dir,
        "base",
        candidate_dir,
        "quant",
        (4, 8),
        min_top1_agreement=0.8,
        max_mean_logprob_diff=0.5,
        max_ppl_relative_change=0.15,
    )

    assert report["valid"]
    assert report["identity_valid"]
    assert set(report["cases"]) == {
        "tp4_state-float32",
        "tp4_state-model",
        "tp8_state-float32",
        "tp8_state-model",
    }
    assert report["cases"]["tp4_state-float32"]["top1_agreement"] == 1.0
    assert report["cases"]["tp4_state-float32"][
        "mean_abs_target_logprob_diff"
    ] == pytest.approx(0.1)


def test_checkpoint_quality_rejects_degraded_candidate():
    baseline = result(4, "float32", "bf16", [1, 2], [-1.0, -2.0])
    candidate = result(4, "float32", "gptq", [3, 4], [-2.0, -3.0])

    comparison = MODULE.compare_case(
        baseline,
        candidate,
        min_top1_agreement=0.8,
        max_mean_logprob_diff=0.5,
        max_ppl_relative_change=0.15,
    )

    assert not comparison["valid"]
    assert comparison["top1_agreement"] == 0.0
    assert comparison["mean_abs_target_logprob_diff"] == 1.0


def test_checkpoint_quality_rejects_misaligned_cases():
    baseline = result(4, "float32", "bf16", [1], [-1.0])
    candidate = result(4, "float32", "gptq", [1], [-1.0])
    candidate["case_token_digest"] = "different"

    with pytest.raises(ValueError, match="case_token_digest"):
        MODULE.compare_case(
            baseline,
            candidate,
            min_top1_agreement=0.8,
            max_mean_logprob_diff=0.5,
            max_ppl_relative_change=0.15,
        )
