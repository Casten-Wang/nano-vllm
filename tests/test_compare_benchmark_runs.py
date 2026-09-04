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
        "model": "Qwen/Qwen3.6-35B-A3B",
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
        "cuda_devices": [
            {"index": rank, "name": device, "capability": [9, 0]}
            for rank in range(8)
        ],
        "torch_version": torch_version,
        "cuda_version": "12.8",
        "nccl_version": [2, 27, 3],
        "transformers_version": "4.56.0",
        "triton_version": "3.4.0",
        "flash_attn_version": "2.8.3",
        "nvidia_smi_gpus": [
            f"{rank}, NVIDIA H100 80GB HBM3, 570.00, 81559"
            for rank in range(8)
        ],
        "nvidia_smi_topology": "GPU0 GPU1 NV18",
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "cuda_device_order": "PCI_BUS_ID",
        "nccl_environment": {"NCCL_ALGO": "Ring"},
        "tensor_parallel_size": 4,
        "recurrent_state_dtype": "float32",
        "qwen35_decode_conv_backend": "weighted",
        "qwen35_moe_decode_backend": "sorted",
        "qwen35_moe_decode_chunk_size": 8,
        "sampling_chunk_size": 32,
        "quantization_format": "bf16",
        "requested_weight_quant_backend": "auto",
        "weight_quant_backend": "auto",
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
        "execution_validation": {
            "valid": True,
            "required_paths": ["decode_contiguous_view"],
            "observed_paths": [
                "decode_contiguous_view",
                "decode_eager",
                "float_flash_decode",
            ],
        },
        "generation_validation": {"valid": True},
        "execution_stats": {
            "input_preparation_stats": {
                "prefill": {
                    "call_count": 2,
                    "total_time_s": 0.4,
                    "max_time_s": 0.25,
                },
                "decode": {
                    "call_count": 4,
                    "total_time_s": 0.2,
                    "max_time_s": 0.08,
                },
            },
        },
        "metrics": {
            "avg_ttft_s": 2.0,
            "p50_ttft_s": 1.8,
            "p95_ttft_s": 3.0,
            "p99_ttft_s": 3.5,
            "avg_tpot_s": 0.05,
            "p50_tpot_s": 0.04,
            "p95_tpot_s": 0.08,
            "p99_tpot_s": 0.1,
            "avg_request_latency_s": 3.0,
            "p50_request_latency_s": 2.8,
            "p95_request_latency_s": 4.0,
            "p99_request_latency_s": 4.5,
        },
        "model_parameter_storage": {
            "total_bytes_local_rank": 4096,
            "by_dtype": {"torch.bfloat16": {"storage_count": 2, "bytes": 4096}},
        },
        "model_parameter_storage_by_rank": [
            {"rank": rank, "total_bytes_local_rank": 4096}
            for rank in range(4)
        ],
        "model_parameter_total_all_ranks_bytes": 16384,
        "kv_cache_storage": {"total_bytes": 1024},
        "kv_cache_storage_by_rank": [
            {"rank": rank, "total_bytes": 1024} for rank in range(4)
        ],
        "num_kvcache_blocks": 1,
        "recurrent_state_storage": {
            "total_bytes_local_rank": 2048,
            "rotary_cache_bytes_local_rank": 512,
            "total_model_state_bytes_local_rank": 2560,
        },
        "recurrent_state_storage_by_rank": [
            {
                "rank": rank,
                "total_bytes_local_rank": 2048,
                "rotary_cache_bytes_local_rank": 512,
            }
            for rank in range(4)
        ],
        "recurrent_state_total_all_ranks_bytes": 8192,
        "runtime_buffer_storage": {
            "int8_partitioned_decode_pool_count": 1,
            "int8_partitioned_workspace_bytes": 4096,
            "int8_partitioned_output_bytes": 512,
            "total_bytes_local_rank": 4608,
        },
        "runtime_buffer_storage_by_rank": [
            {"rank": rank, "total_bytes_local_rank": 4608}
            for rank in range(4)
        ],
        "runtime_buffer_total_all_ranks_bytes": 18432,
    }


def test_comparison_reports_relative_metrics_and_output_parity():
    baseline = result()
    candidate = result(throughput=125.0, memory=750.0)
    candidate["execution_stats"]["input_preparation_stats"]["decode"][
        "total_time_s"
    ] = 0.1

    comparison = MODULE.compare_results(
        [baseline, candidate],
        ["baseline", "candidate"],
    )

    candidate_row = comparison["runs"][1]
    assert candidate_row["throughput_vs_baseline"] == 1.25
    assert candidate_row["peak_memory_vs_baseline"] == 0.75
    assert candidate_row["p99_ttft_s"] == 3.5
    assert candidate_row["latency_vs_baseline"]["p95_tpot_s"] == 1.0
    assert comparison["all_output_digests_match"]
    assert comparison["all_execution_paths_valid"]
    assert comparison["all_generation_valid"]
    assert comparison["commits"] == [
        "0123456789abcdef",
        "0123456789abcdef",
    ]
    assert comparison["checkpoint_identity_strength"] == "content-addressed"
    assert candidate_row["input_preparation"]["prefill"]["average_time_s"] == 0.2
    assert candidate_row["input_preparation_vs_baseline"] == {
        "prefill": 1.0,
        "decode": 0.5,
    }


def test_comparison_allows_explicit_optimization_variables_to_change():
    candidate = result()
    candidate["tensor_parallel_size"] = 8
    candidate["model_parameter_storage_by_rank"] = [
        {"rank": rank, "total_bytes_local_rank": 4096}
        for rank in range(8)
    ]
    candidate["model_parameter_total_all_ranks_bytes"] = 32768
    candidate["kv_cache_storage_by_rank"] = [
        {"rank": rank, "total_bytes": 1024} for rank in range(8)
    ]
    candidate["recurrent_state_storage_by_rank"] = [
        {
            "rank": rank,
            "total_bytes_local_rank": 2048,
            "rotary_cache_bytes_local_rank": 512,
        }
        for rank in range(8)
    ]
    candidate["recurrent_state_total_all_ranks_bytes"] = 16384
    candidate["runtime_buffer_storage_by_rank"] = [
        {"rank": rank, "total_bytes_local_rank": 4608}
        for rank in range(8)
    ]
    candidate["runtime_buffer_total_all_ranks_bytes"] = 36864
    candidate["recurrent_state_dtype"] = "model"
    candidate["qwen35_moe_decode_backend"] = "batched"
    candidate["quantization_format"] = "gptq_int4"
    candidate["weight_quant_backend"] = "triton"
    candidate["kv_cache_dtype"] = "int8"

    comparison = MODULE.compare_results(
        [result(), candidate],
        ["baseline", "candidate"],
    )

    assert comparison["runs"][1]["tensor_parallel_size"] == 8
    assert comparison["runs"][1]["recurrent_state_dtype"] == "model"
    assert comparison["runs"][1]["qwen35_moe_decode_backend"] == "batched"
    assert comparison["runs"][1]["quantization_format"] == "gptq_int4"
    assert comparison["runs"][1]["weight_quant_backend"] == "triton"


def test_comparison_rejects_incomplete_ranked_storage():
    candidate = result()
    candidate["kv_cache_storage_by_rank"].pop()

    with pytest.raises(ValueError, match="kv_cache_storage_by_rank.*missing ranks"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


def test_comparison_rejects_inconsistent_ranked_storage_total():
    candidate = result()
    candidate["runtime_buffer_total_all_ranks_bytes"] += 1

    with pytest.raises(ValueError, match="does not match per-rank storage"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantization_format", "gptq_int4"),
        ("weight_quant_backend", "triton"),
    ],
)
def test_repeat_summary_rejects_mixed_weight_quantization(field, value):
    candidate = result()
    candidate[field] = value

    with pytest.raises(ValueError, match=rf"r2\.{field}"):
        MODULE.summarize_repeats([result(), candidate], ["r1", "r2"])


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
    assert summary["statistics"]["p99_request_latency_s"]["median"] == 4.5
    assert summary["statistics"]["host_decode_preparation_average_time_s"][
        "median"
    ] == 0.05
    assert summary["all_output_digests_match"]
    assert summary["generated_token_ids_digest"] == "same"
    assert summary["execution_paths"] == {
        "required": ["decode_contiguous_view"],
        "observed_in_all_repeats": [
            "decode_contiguous_view",
            "decode_eager",
            "float_flash_decode",
        ],
    }
    assert summary["storage"]["recurrent_state_storage"][
        "rotary_cache_bytes_local_rank"
    ] == 512
    assert summary["storage"]["runtime_buffer_storage"][
        "total_bytes_local_rank"
    ] == 4608


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
    assert comparison["runs"][0]["storage"]["kv_cache_storage"][
        "total_bytes"
    ] == 1024
    assert comparison["baseline"] == "float"
    assert comparison["runs"][0]["vs_baseline"]["output_throughput"] == 1.0
    assert comparison["runs"][0]["execution_paths"]["required"] == [
        "decode_contiguous_view"
    ]
    assert comparison["runs"][1]["vs_baseline"]["output_throughput"] == 1.25
    assert comparison["runs"][1]["vs_baseline"]["peak_memory"] == 0.75
    assert comparison["runs"][1]["vs_baseline"]["input_preparation"][
        "host_decode_preparation_average_time_s"
    ] == 1.0
    assert comparison["runs"][1]["vs_baseline"]["latency"]["p99_ttft_s"] == 1.0
    assert comparison["all_repeat_output_digests_match"]
    assert not comparison["all_output_digests_match"]
    assert comparison["all_execution_paths_valid"]
    assert comparison["all_generation_valid"]


def test_comparison_supports_legacy_results_without_preparation_stats():
    baseline = result()
    candidate = result()
    del baseline["execution_stats"]
    del candidate["execution_stats"]

    comparison = MODULE.compare_results(
        [baseline, candidate],
        ["baseline", "candidate"],
    )

    assert comparison["runs"][0]["input_preparation"] == {}
    assert comparison["runs"][1]["input_preparation_vs_baseline"] == {}


def test_comparison_uses_slowest_tensor_parallel_rank_preparation_time():
    baseline = result()
    candidate = result()
    for item, rank1_total in ((baseline, 0.4), (candidate, 0.2)):
        rank0 = {
            "rank": 0,
            **item["execution_stats"],
        }
        rank1 = {
            "rank": 1,
            "input_preparation_stats": {
                "prefill": {
                    "call_count": 2,
                    "total_time_s": rank1_total,
                    "max_time_s": rank1_total / 2,
                },
                "decode": {
                    "call_count": 4,
                    "total_time_s": rank1_total,
                    "max_time_s": rank1_total / 2,
                },
            },
        }
        item["tensor_parallel_size"] = 2
        for field in (
            "model_parameter_storage_by_rank",
            "kv_cache_storage_by_rank",
            "recurrent_state_storage_by_rank",
            "runtime_buffer_storage_by_rank",
        ):
            item[field] = item[field][:2]
        item["model_parameter_total_all_ranks_bytes"] = 8192
        item["recurrent_state_total_all_ranks_bytes"] = 4096
        item["runtime_buffer_total_all_ranks_bytes"] = 9216
        item["execution_stats_by_rank"] = [rank0, rank1]

    comparison = MODULE.compare_results(
        [baseline, candidate],
        ["baseline", "candidate"],
    )

    candidate_decode = comparison["runs"][1]["input_preparation"]["decode"]
    assert candidate_decode["average_time_s"] == 0.05
    assert candidate_decode["slowest_rank"] == 0
    assert candidate_decode["rank_skew"] == 1.0
    assert [item["rank"] for item in candidate_decode["by_rank"]] == [0, 1]
    assert comparison["runs"][1]["input_preparation_vs_baseline"]["decode"] == 0.5
    summary = MODULE.summarize_repeats(
        [baseline, candidate],
        ["r1", "r2"],
    )
    assert summary["statistics"]["host_decode_preparation_rank_skew"][
        "median"
    ] == 1.5


def test_comparison_rejects_incomplete_tensor_parallel_preparation_stats():
    candidate = result()
    candidate["execution_stats_by_rank"] = [
        {"rank": 0, **candidate["execution_stats"]}
    ]

    with pytest.raises(ValueError, match="execution_stats_by_rank is incomplete"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("call_count", 0, "must be a positive integer"),
        ("call_count", True, "must be a positive integer"),
        ("total_time_s", float("nan"), "finite non-negative number"),
        ("max_time_s", 0.5, "cannot exceed total_time_s"),
    ],
)
def test_comparison_rejects_invalid_preparation_stats(field, value, message):
    candidate = result()
    candidate["execution_stats"]["input_preparation_stats"]["decode"][
        field
    ] = value

    with pytest.raises(ValueError, match=message):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


def test_comparison_rejects_unknown_preparation_step():
    candidate = result()
    candidate["execution_stats"]["input_preparation_stats"]["prefilll"] = {
        "call_count": 1,
        "total_time_s": 0.1,
        "max_time_s": 0.1,
    }

    with pytest.raises(ValueError, match="unsupported steps: prefilll"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


def test_comparison_rejects_missing_tail_latency_metrics():
    candidate = result()
    del candidate["metrics"]["p99_ttft_s"]

    with pytest.raises(ValueError, match="candidate.metrics.p99_ttft_s"):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1.0])
def test_comparison_rejects_invalid_latency_measurements(value):
    candidate = result()
    candidate["metrics"]["p99_ttft_s"] = value

    with pytest.raises(
        ValueError,
        match=r"candidate\.metrics\.p99_ttft_s must be a finite non-negative number",
    ):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), True, -1.0])
def test_comparison_rejects_invalid_throughput_measurements(value):
    candidate = result(throughput=value)

    with pytest.raises(
        ValueError,
        match=r"candidate\.output_throughput_tok_s must be a finite positive number",
    ):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1.0])
def test_comparison_rejects_invalid_memory_measurements(value):
    candidate = result(memory=value)

    with pytest.raises(
        ValueError,
        match=r"candidate\.peak_torch_allocated_mib must be a finite non-negative number",
    ):
        MODULE.compare_results(
            [result(), candidate],
            ["baseline", "candidate"],
        )


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


def test_repeat_summary_rejects_different_execution_contracts():
    candidate = result()
    candidate["execution_validation"]["required_paths"] = [
        "decode_graph_indexed"
    ]

    with pytest.raises(ValueError, match="different execution paths"):
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
