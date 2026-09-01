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
    device="NVIDIA H100 80GB HBM3",
    torch_version="2.8.0",
):
    return {
        "model": "Qwen/Qwen3.5-35B-A3B",
        "commit": "0123456789abcdef",
        "git_dirty": False,
        "checkpoint_manifest": {
            "digest": checkpoint_digest,
            "strength": "content-addressed",
        },
        "num_seqs": 16,
        "input_len": input_len,
        "output_len": 32,
        "seed": 7,
        "vocab_size": 10000,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.9,
        "warmup": True,
        "device": device,
        "device_capability": [9, 0],
        "cuda_device_count": 8,
        "torch_version": torch_version,
        "cuda_version": "12.8",
        "transformers_version": "4.56.0",
        "triton_version": "3.4.0",
        "flash_attn_version": "2.8.3",
        "nvidia_smi_gpus": [
            f"{rank}, NVIDIA H100 80GB HBM3, 570.00, 81559"
            for rank in range(8)
        ],
        "tensor_parallel_size": 4,
        "recurrent_state_dtype": "float32",
        "kv_cache_dtype": "auto",
        "kv_dequant_backend": "fused",
        "int8_partitioned_decode_threshold": 8192,
        "int8_partitioned_decode_partition_size": 512,
        "sliding_window_size": None,
        "enable_dynamic_chunked_prefill": False,
        "enforce_eager": True,
        "output_throughput_tok_s": throughput,
        "peak_torch_allocated_mib": memory,
        "generated_token_ids": {"digest": digest},
        "execution_validation": {"valid": True},
        "generation_validation": {"valid": True},
        "metrics": {
            "avg_ttft_s": 2.0,
            "avg_tpot_s": 0.05,
            "avg_request_latency_s": 3.0,
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
    assert comparison["all_generation_valid"]
    assert comparison["commits"] == [
        "0123456789abcdef",
        "0123456789abcdef",
    ]
    assert comparison["checkpoint_identity_strength"] == "content-addressed"


def test_comparison_allows_explicit_optimization_variables_to_change():
    candidate = result()
    candidate["tensor_parallel_size"] = 8
    candidate["recurrent_state_dtype"] = "model"
    candidate["kv_cache_dtype"] = "int8"

    comparison = MODULE.compare_results(
        [result(), candidate],
        ["baseline", "candidate"],
    )

    assert comparison["runs"][1]["tensor_parallel_size"] == 8
    assert comparison["runs"][1]["recurrent_state_dtype"] == "model"


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


def test_comparison_surfaces_incomplete_generation():
    candidate = result()
    candidate["generation_validation"] = {"valid": False}

    comparison = MODULE.compare_results(
        [result(), candidate],
        ["baseline", "candidate"],
    )

    assert not comparison["all_generation_valid"]
    assert not comparison["runs"][1]["generation_valid"]


def test_comparison_rejects_different_checkpoint_manifest():
    with pytest.raises(ValueError, match="checkpoint_manifest.digest"):
        MODULE.compare_results(
            [result(), result(checkpoint_digest="different")],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"device": "NVIDIA H200"}, "device"),
        ({"torch_version": "2.9.0"}, "torch_version"),
    ],
)
def test_comparison_rejects_different_environments(override, field):
    with pytest.raises(ValueError, match=rf"candidate\.{field}"):
        MODULE.compare_results(
            [result(), result(**override)],
            ["baseline", "candidate"],
        )


def test_comparison_rejects_different_scheduler_capacity():
    candidate = result()
    candidate["max_num_batched_tokens"] = 4096

    with pytest.raises(ValueError, match="candidate.max_num_batched_tokens"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


def test_repeat_summary_reports_distribution_and_stability():
    runs = [result(throughput=value) for value in (90.0, 100.0, 110.0)]

    summary = MODULE.summarize_repeats(runs, ["r1", "r2", "r3"])

    throughput = summary["statistics"]["output_throughput_tok_s"]
    assert throughput["count"] == 3
    assert throughput["median"] == 100.0
    assert throughput["mean"] == 100.0
    assert throughput["population_stdev"] == pytest.approx(8.1649658)
    assert throughput["coefficient_of_variation"] == pytest.approx(0.081649658)
    assert summary["all_output_digests_match"]
    assert summary["generated_token_ids_digest"] == "same"


def test_matrix_summary_compares_configuration_medians_and_quality():
    first = MODULE.summarize_repeats(
        [result(throughput=90.0), result(throughput=110.0)],
        ["a1", "a2"],
    )
    candidate_results = [
        result(throughput=120.0, memory=750.0, digest="changed"),
        result(throughput=130.0, memory=750.0, digest="changed"),
    ]
    for item in candidate_results:
        item["kv_cache_dtype"] = "int8"
    second = MODULE.summarize_repeats(candidate_results, ["b1", "b2"])

    comparison = MODULE.compare_repeat_summaries(
        [first, second],
        ["float", "int8"],
    )

    assert comparison["runs"][0]["median"]["output_throughput_tok_s"] == 100.0
    assert comparison["runs"][1]["median"]["peak_torch_allocated_mib"] == 750.0
    assert comparison["all_repeat_output_digests_match"]
    assert not comparison["all_output_digests_match"]
    assert comparison["all_execution_paths_valid"]
    assert comparison["all_generation_valid"]


def test_matrix_summary_rejects_different_workloads():
    first = MODULE.summarize_repeats([result(), result()], ["a1", "a2"])
    second = MODULE.summarize_repeats(
        [result(input_len=256), result(input_len=256)],
        ["b1", "b2"],
    )

    with pytest.raises(ValueError, match="candidate.workload"):
        MODULE.compare_repeat_summaries(
            [first, second],
            ["baseline", "candidate"],
        )


def test_repeat_summary_rejects_different_optimization_configuration():
    candidate = result()
    candidate["kv_cache_dtype"] = "int8"

    with pytest.raises(ValueError, match="candidate.kv_cache_dtype"):
        MODULE.summarize_repeats(
            [result(), candidate],
            ["baseline", "candidate"],
        )


def test_load_result_rejects_invalid_checkpoint_manifest(tmp_path):
    path = tmp_path / "result.json"
    value = result()
    value["checkpoint_manifest"] = {}
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="invalid checkpoint manifest"):
        MODULE.load_result(path)


def test_load_result_rejects_dirty_worktree(tmp_path):
    path = tmp_path / "result.json"
    value = result()
    value["git_dirty"] = True
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="dirty worktree"):
        MODULE.load_result(path)


def test_load_result_rejects_unknown_commit(tmp_path):
    path = tmp_path / "result.json"
    value = result()
    value["commit"] = "unknown"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="Git commit identity"):
        MODULE.load_result(path)


def test_comparison_rejects_dirty_worktree():
    candidate = result()
    candidate["git_dirty"] = True

    with pytest.raises(ValueError, match="candidate"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )
