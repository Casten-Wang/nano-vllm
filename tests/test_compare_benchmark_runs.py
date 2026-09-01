from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "compare_benchmark_runs",
    ROOT / "scripts" / "compare_benchmark_runs.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(
    *,
    throughput=100.0,
    memory=1000.0,
    digest="same",
    input_len=128,
    checkpoint_digest="checkpoint",
):
    return {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "checkpoint_manifest": {"digest": checkpoint_digest},
        "num_seqs": 16,
        "input_len": input_len,
        "output_len": 32,
        "seed": 7,
        "vocab_size": 10000,
        "max_model_len": 4096,
        "tensor_parallel_size": 4,
        "recurrent_state_dtype": "float32",
        "output_throughput_tok_s": throughput,
        "peak_torch_allocated_mib": memory,
        "generated_token_ids": {"digest": digest},
        "execution_validation": {"valid": True},
        "metrics": {
            "avg_ttft_s": 2.0,
            "avg_tpot_s": 0.05,
        },
    }


def test_comparison_reports_relative_metrics_and_output_parity():
    baseline = result()
    candidate = result(throughput=125.0, memory=750.0)

    comparison = MODULE.compare_results(
        [baseline, candidate],
        ["baseline", "candidate"],
    )

    candidate_row = comparison["runs"][1]
    assert candidate_row["throughput_vs_baseline"] == 1.25
    assert candidate_row["peak_memory_vs_baseline"] == 0.75
    assert comparison["all_output_digests_match"]
    assert comparison["all_execution_paths_valid"]


def test_comparison_rejects_different_workloads():
    with pytest.raises(ValueError, match="candidate.input_len"):
        MODULE.compare_results(
            [result(), result(input_len=256)],
            ["baseline", "candidate"],
        )


def test_comparison_surfaces_generation_drift():
    comparison = MODULE.compare_results(
        [result(), result(digest="changed")],
        ["baseline", "candidate"],
    )

    assert not comparison["all_output_digests_match"]
    assert not comparison["runs"][1]["output_digest_matches_baseline"]


def test_comparison_rejects_different_checkpoint_manifest():
    with pytest.raises(ValueError, match="checkpoint_manifest.digest"):
        MODULE.compare_results(
            [result(), result(checkpoint_digest="different")],
            ["baseline", "candidate"],
        )


def test_load_result_rejects_invalid_checkpoint_manifest(tmp_path):
    path = tmp_path / "result.json"
    value = result()
    value["checkpoint_manifest"] = {}
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="invalid checkpoint manifest"):
        MODULE.load_result(path)
