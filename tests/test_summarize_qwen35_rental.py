from importlib.util import module_from_spec, spec_from_file_location
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "summarize_qwen35_rental",
    ROOT / "scripts" / "summarize_qwen35_rental.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE_COMMIT = "a" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bind_manifest_artifacts(root, run_id):
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    write(
        root / "manifest.json",
        {
            "run_id": run_id,
            "source_commit": SOURCE_COMMIT,
            "stages": [
                {"name": "fixture"},
                {"name": "final-summary"},
            ],
            "completed_stages": ["fixture"],
            "completed_stage_artifacts": {"fixture": records},
        },
    )


def test_run_manifest_is_required(tmp_path):
    with pytest.raises(ValueError, match="manifest.json"):
        MODULE.load_run_manifest(tmp_path, "rental-a")


@pytest.mark.parametrize(
    "manifest, message",
    (
        ({"run_id": "other", "source_commit": SOURCE_COMMIT}, "run_id"),
        ({"run_id": "rental-a", "source_commit": "abc"}, "40-character"),
        ({"run_id": "rental-a", "source_commit": "A" * 40}, "lowercase"),
    ),
)
def test_run_manifest_rejects_invalid_identity(tmp_path, manifest, message):
    write(tmp_path / "manifest.json", manifest)

    with pytest.raises(ValueError, match=message):
        MODULE.load_run_manifest(tmp_path, "rental-a")


def test_manifest_artifact_integrity_detects_tampering(tmp_path):
    artifact = tmp_path / "performance" / "result.json"
    write(artifact, {"valid": True})
    bind_manifest_artifacts(tmp_path, "rental-a")
    manifest = MODULE.load_run_manifest(tmp_path, "rental-a")

    assert MODULE.manifest_artifacts_match(tmp_path, manifest)

    write(artifact, {"valid": False})

    assert not MODULE.manifest_artifacts_match(tmp_path, manifest)


def test_manifest_artifact_integrity_requires_complete_stage_prefix(tmp_path):
    artifact = tmp_path / "result.json"
    write(artifact, {"valid": True})
    bind_manifest_artifacts(tmp_path, "rental-a")
    manifest = MODULE.load_run_manifest(tmp_path, "rental-a")
    manifest["completed_stages"] = []

    assert not MODULE.manifest_artifacts_match(tmp_path, manifest)


def write_gptq_summary_inputs(root, run_id, *, backend="triton"):
    gptq_run_id = f"{run_id}-gptq"
    shards = [
        {"name": "model-00001.safetensors", "size_bytes": 123, "sha256": "a" * 64}
    ]
    write(
        root / "gptq/official_checkpoint_header_audit.json",
        {
            "valid": True,
            "repo": MODULE.OFFICIAL_GPTQ_CHECKPOINT_REPO,
            "resolved_revision": MODULE.OFFICIAL_GPTQ_CHECKPOINT_REVISION,
            "semantic_contract": MODULE.expected_checkpoint_semantic_contract(
                "gptq_int4"
            ),
            "config_sha256": "b" * 64,
            "index_sha256": "c" * 64,
            "shard_count": 1,
            "checkpoint_shards": shards,
            "results": {"tp4": {"valid": True}, "tp8": {"valid": True}},
        },
    )
    write(
        root / "gptq/preflight/checkpoint_mapping_audit.json",
        {
            "valid": True,
            "complete": True,
            "results": {"tp4": {"valid": True}, "tp8": {"valid": True}},
            "checkpoint_manifest": {
                "config_sha256": "b" * 64,
                "index_sha256": "c" * 64,
                "shard_count": 1,
                "present_shard_count": 1,
                "missing_shards": [],
                "files": [
                    {
                        "name": shards[0]["name"],
                        "size_bytes": shards[0]["size_bytes"],
                        "content_sha256": shards[0]["sha256"],
                        "present": True,
                    }
                ],
            },
        },
    )
    write(
        root / "gptq/preflight/memory_preflight.json",
        {"valid": True, "results": {"tp4": {}, "tp8": {}}},
    )
    rows = [
        {
            "tensor_parallel_size": tp,
            "requested_weight_quant_backend": "auto",
            "weight_quant_backend": backend,
            "quantization_format": "gptq_int4",
            "qwen35_moe_decode_backend": "sorted",
            "enforce_eager": True,
            "storage": {
                "runtime_buffer_storage_by_rank": [
                    {
                        "rank": rank,
                        "gptq_expert_workspace_pool_count": 1,
                        "gptq_expert_workspace_bytes": (
                            64 * (2 * (512 // tp) + 2048) * 2
                        ),
                        "gptq_expert_workspace_allocation_count": 1,
                        "gptq_expert_workspace_reuse_count": 40,
                    }
                    for rank in range(tp)
                ]
            },
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {
                "output_throughput_tok_s": throughput,
                "peak_torch_allocated_mib": memory,
            },
        }
        for tp, throughput, memory in ((4, 100.0, 20_000.0), (8, 180.0, 12_000.0))
    ]
    write(
        root / f"gptq/performance/{gptq_run_id}_matrix_summary.json",
        {
            "commits": [SOURCE_COMMIT],
            "workload": {"max_num_seqs": 64},
            "all_execution_paths_valid": True,
            "all_generation_valid": True,
            "all_repeat_output_digests_match": True,
            "runs": rows,
        },
    )
    write(
        root / f"gptq/quality/{gptq_run_id}_summary.json",
        {
            "commit": SOURCE_COMMIT,
            "cases": [
                {
                    "tensor_parallel_size": tp,
                    "requested_weight_quant_backend": "auto",
                    "weight_quant_backend": backend,
                    "qwen35_moe_decode_backend": "sorted",
                }
                for tp in (4, 8)
            ],
            "quality_gates": {"all_passed": True},
            "cross_tp": {"all_passed": True},
        },
    )
    write(
        root / "gptq/quality/bf16_vs_gptq.json",
        {
            "valid": True,
            "commit": SOURCE_COMMIT,
            "baseline_run_id": run_id,
            "candidate_run_id": gptq_run_id,
            "tensor_parallel_sizes": [4, 8],
            "cases": {},
        },
    )


def write_fp8_summary_inputs(
    root,
    run_id,
    *,
    backend="reference",
    requested_backend="reference",
    throughput_by_tp=None,
    memory_by_tp=None,
):
    fp8_run_id = f"{run_id}-fp8"
    shards = [
        {"name": "model-00001.safetensors", "size_bytes": 123, "sha256": "d" * 64}
    ]
    write(
        root / "fp8/official_checkpoint_header_audit.json",
        {
            "valid": True,
            "repo": MODULE.OFFICIAL_FP8_CHECKPOINT_REPO,
            "resolved_revision": MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION,
            "config_sha256": MODULE.OFFICIAL_FP8_CONFIG_SHA256,
            "index_sha256": MODULE.OFFICIAL_FP8_INDEX_SHA256,
            "headers_sha256": MODULE.OFFICIAL_FP8_HEADERS_SHA256,
            "semantic_contract": MODULE.expected_checkpoint_semantic_contract(
                "fp8_block"
            ),
            "fp8_runtime_backend": requested_backend,
            "shard_count": 1,
            "checkpoint_shards": shards,
            "quantization": {"format": "fp8_block", "valid": True},
            "results": {
                "tp4": {
                    "valid": True,
                    "skipped_by_prefix": MODULE.OFFICIAL_SKIPPED_WEIGHT_PREFIXES,
                    "unclassified_skipped_weights": [],
                    "local_parameter_bytes": 1000,
                    "local_parameter_and_resident_runtime_bytes": (
                        76_536 if backend == "resident" else 1000
                    ),
                    "resident_fp8_expert_storage": {
                        "layer_count": 40 if backend == "resident" else 0,
                        "weight_bytes": 1_000_000 if backend == "resident" else 0,
                        "scale_bytes": 10_000 if backend == "resident" else 0,
                        "total_bytes": 1_010_000 if backend == "resident" else 0,
                        "dequant_workspace_pool_count": (
                            1 if backend == "resident" else 0
                        ),
                        "dequant_workspace_bytes": (
                            65_536 if backend == "resident" else 0
                        ),
                        "total_runtime_bytes": (
                            1_075_536 if backend == "resident" else 0
                        ),
                    },
                    "quantized_tp_layout": {
                        "valid": True,
                        "requires_partial_unit_loader": False,
                        "partial_quantization_unit_count": 0,
                    },
                },
                "tp8": {
                    "valid": True,
                    "skipped_by_prefix": MODULE.OFFICIAL_SKIPPED_WEIGHT_PREFIXES,
                    "unclassified_skipped_weights": [],
                    "local_parameter_bytes": 500,
                    "local_parameter_and_resident_runtime_bytes": (
                        76_036 if backend == "resident" else 500
                    ),
                    "resident_fp8_expert_storage": {
                        "layer_count": 40 if backend == "resident" else 0,
                        "weight_bytes": 1_000_000 if backend == "resident" else 0,
                        "scale_bytes": 10_000 if backend == "resident" else 0,
                        "total_bytes": 1_010_000 if backend == "resident" else 0,
                        "dequant_workspace_pool_count": (
                            1 if backend == "resident" else 0
                        ),
                        "dequant_workspace_bytes": (
                            65_536 if backend == "resident" else 0
                        ),
                        "total_runtime_bytes": (
                            1_075_536 if backend == "resident" else 0
                        ),
                    },
                    "quantized_tp_layout": {
                        "valid": True,
                        "requires_partial_unit_loader": True,
                        "partial_quantization_unit_count": 2,
                    },
                },
            },
        },
    )
    write(
        root / "fp8/preflight/checkpoint_mapping_audit.json",
        {
            "valid": True,
            "complete": True,
            "fp8_runtime_backend": requested_backend,
            "results": {
                "tp4": {
                    "valid": True,
                    "local_parameter_and_resident_runtime_bytes": (
                        76_536 if backend == "resident" else 1000
                    ),
                },
                "tp8": {
                    "valid": True,
                    "local_parameter_and_resident_runtime_bytes": (
                        76_036 if backend == "resident" else 500
                    ),
                },
            },
            "checkpoint_manifest": {
                "config_sha256": MODULE.OFFICIAL_FP8_CONFIG_SHA256,
                "index_sha256": MODULE.OFFICIAL_FP8_INDEX_SHA256,
                "shard_count": 1,
                "present_shard_count": 1,
                "missing_shards": [],
                "files": [
                    {
                        "name": shards[0]["name"],
                        "size_bytes": shards[0]["size_bytes"],
                        "content_sha256": shards[0]["sha256"],
                        "present": True,
                    }
                ],
            },
        },
    )
    write(
        root / "fp8/preflight/memory_preflight.json",
        {"valid": True, "results": {"tp4": {}, "tp8": {}}},
    )
    throughput_by_tp = throughput_by_tp or {4: 90.0, 8: 160.0}
    memory_by_tp = memory_by_tp or {4: 15_000.0, 8: 9_000.0}
    rows = [
        {
            "tensor_parallel_size": tp,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "requested_weight_quant_backend": requested_backend,
            "weight_quant_backend": backend,
            "quantization_format": "fp8_block",
            "qwen35_moe_decode_backend": "sorted",
            "enforce_eager": True,
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "storage": {
                "runtime_buffer_storage_by_rank": [
                    {
                        "rank": rank,
                        "resident_fp8_expert_layer_count": (
                            40 if backend == "resident" else 0
                        ),
                        "resident_fp8_expert_weight_bytes": (
                            1_000_000 if backend == "resident" else 0
                        ),
                        "resident_fp8_expert_scale_bytes": (
                            10_000 if backend == "resident" else 0
                        ),
                        "resident_fp8_weight_pool_count": (
                            1 if backend == "resident" else 0
                        ),
                        "resident_fp8_dequant_workspace_bytes": (
                            65_536 if backend == "resident" else 0
                        ),
                        "resident_fp8_dequant_workspace_allocation_count": (
                            1 if backend == "resident" else 0
                        ),
                        "resident_fp8_dequant_workspace_reuse_count": (
                            10 if backend == "resident" else 0
                        ),
                    }
                    for rank in range(tp)
                ]
            },
            "median": {
                "output_throughput_tok_s": throughput,
                "peak_torch_allocated_mib": memory,
            },
        }
        for tp, throughput, memory in (
            (4, throughput_by_tp[4], memory_by_tp[4]),
            (8, throughput_by_tp[8], memory_by_tp[8]),
        )
    ]
    write(
        root / f"fp8/performance/{fp8_run_id}_matrix_summary.json",
        {
            "commits": [SOURCE_COMMIT],
            "all_execution_paths_valid": True,
            "all_generation_valid": True,
            "all_repeat_output_digests_match": True,
            "runs": rows,
        },
    )
    write(
        root / f"fp8/quality/{fp8_run_id}_summary.json",
        {
            "commit": SOURCE_COMMIT,
            "cases": [
                {
                    "tensor_parallel_size": tp,
                    "requested_weight_quant_backend": requested_backend,
                    "weight_quant_backend": backend,
                    "qwen35_moe_decode_backend": "sorted",
                }
                for tp in (4, 8)
            ],
            "quality_gates": {"all_passed": True},
            "cross_tp": {"all_passed": True},
        },
    )
    write(
        root / "fp8/quality/bf16_vs_fp8.json",
        {
            "valid": True,
            "commit": SOURCE_COMMIT,
            "baseline_run_id": run_id,
            "candidate_run_id": fp8_run_id,
            "tensor_parallel_sizes": [4, 8],
            "cases": {},
        },
    )


def fp8_baseline_rows():
    return [
        {
            "tensor_parallel_size": tp,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "qwen35_moe_decode_backend": "sorted",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "generation_valid": True,
            "median": {
                "output_throughput_tok_s": throughput,
                "peak_torch_allocated_mib": memory,
            },
        }
        for tp, throughput, memory in (
            (4, 100.0, 20_000.0),
            (8, 180.0, 12_000.0),
        )
    ]


def test_optional_gptq_summary_is_disabled_when_directory_is_absent(tmp_path):
    assert MODULE.summarize_optional_gptq(tmp_path, "run") == {
        "enabled": False,
        "valid": True,
    }


@pytest.mark.parametrize(
    "quantization_format", ("bf16", "gptq_int4", "fp8_block")
)
def test_checkpoint_semantic_contract_rejects_official_metadata_drift(
    quantization_format,
):
    contract = MODULE.expected_checkpoint_semantic_contract(quantization_format)
    assert MODULE.checkpoint_semantic_contract_matches(
        {"semantic_contract": contract}, quantization_format
    )

    contract["num_experts_per_tok"] = 4
    assert not MODULE.checkpoint_semantic_contract_matches(
        {"semantic_contract": contract}, quantization_format
    )


def test_optional_fp8_audit_is_disabled_without_changing_run_validity(tmp_path):
    assert MODULE.summarize_optional_fp8_audit(tmp_path) == {
        "enabled": False,
        "valid": True,
        "executable": False,
    }


def test_optional_fp8_audit_reports_tp_alignment_without_execution_claim(
    tmp_path,
):
    write(
        tmp_path / "fp8/official_checkpoint_header_audit.json",
        {
            "valid": True,
            "repo": MODULE.OFFICIAL_FP8_CHECKPOINT_REPO,
            "resolved_revision": MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION,
            "config_sha256": MODULE.OFFICIAL_FP8_CONFIG_SHA256,
            "index_sha256": MODULE.OFFICIAL_FP8_INDEX_SHA256,
            "headers_sha256": MODULE.OFFICIAL_FP8_HEADERS_SHA256,
            "semantic_contract": MODULE.expected_checkpoint_semantic_contract(
                "fp8_block"
            ),
            "quantization": {"format": "fp8_block", "valid": True},
            "results": {
                "tp4": {
                    "valid": True,
                    "skipped_by_prefix": MODULE.OFFICIAL_SKIPPED_WEIGHT_PREFIXES,
                    "unclassified_skipped_weights": [],
                    "local_parameter_bytes": 17_372_983_712,
                    "quantized_tp_layout": {
                        "valid": True,
                        "requires_partial_unit_loader": False,
                        "partial_quantization_unit_count": 0,
                    },
                },
                "tp8": {
                    "valid": True,
                    "skipped_by_prefix": MODULE.OFFICIAL_SKIPPED_WEIGHT_PREFIXES,
                    "unclassified_skipped_weights": [],
                    "local_parameter_bytes": 8_718_380_752,
                    "quantized_tp_layout": {
                        "valid": True,
                        "requires_partial_unit_loader": True,
                        "partial_quantization_unit_count": 30_860,
                    },
                },
            },
        },
    )

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )

    assert report["valid"]
    assert not report["executable"]
    assert not report["tensor_parallel"]["tp4"]["requires_partial_unit_loader"]
    assert report["tensor_parallel"]["tp8"]["requires_partial_unit_loader"]
    assert (
        report["tensor_parallel"]["tp8"]["partial_quantization_unit_count"]
        == 30_860
    )


@pytest.mark.parametrize(
    "field",
    ("config_sha256", "index_sha256", "headers_sha256"),
)
def test_optional_fp8_audit_rejects_checkpoint_identity_drift(tmp_path, field):
    write_fp8_summary_inputs(tmp_path, "run")
    path = tmp_path / "fp8/official_checkpoint_header_audit.json"
    audit = json.loads(path.read_text())
    audit[field] = "0" * 64
    write(path, audit)

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )

    assert not report["valid"]
    assert report["executable"]
    assert not report["execution_validated"]


def test_optional_fp8_audit_rejects_unexpected_text_only_skips(tmp_path):
    write_fp8_summary_inputs(tmp_path, "run")
    path = tmp_path / "fp8/official_checkpoint_header_audit.json"
    audit = json.loads(path.read_text())
    audit["results"]["tp4"]["unclassified_skipped_weights"] = ["model.unknown"]
    write(path, audit)

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )

    assert not report["valid"]
    assert report["executable"]
    assert not report["execution_validated"]


def test_optional_fp8_summary_requires_reference_execution_and_quality(tmp_path):
    write_fp8_summary_inputs(tmp_path, "run")

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path,
        "run",
        fp8_baseline_rows(),
        expected_source_commit=SOURCE_COMMIT,
    )

    assert report["valid"]
    assert report["executable"]
    assert report["execution_validated"]
    assert not report["native_fp8"]
    assert report["local_checkpoint_matches_official"]
    assert report["bf16_vs_fp8_quality_valid"]
    assert report["runtime_storage_valid"]
    assert report["source_commit_valid"]
    assert report["best_throughput"]["tensor_parallel_size"] == 8

    write_fp8_summary_inputs(tmp_path, "run", backend="triton")
    invalid = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )
    assert not invalid["valid"]
    assert not invalid["executable"]


def test_optional_fp8_summary_rejects_mixed_source_evidence(tmp_path):
    write_fp8_summary_inputs(tmp_path, "run")
    quality_path = tmp_path / "fp8/quality/run-fp8_summary.json"
    quality = json.loads(quality_path.read_text())
    quality["commit"] = "b" * 40
    write(quality_path, quality)

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path,
        "run",
        fp8_baseline_rows(),
        expected_source_commit=SOURCE_COMMIT,
    )

    assert not report["source_commit_valid"]
    assert not report["execution_validated"]
    assert not report["valid"]


def test_optional_fp8_summary_accepts_resident_execution_without_native_claim(tmp_path):
    write_fp8_summary_inputs(
        tmp_path,
        "run",
        backend="resident",
        requested_backend="resident",
    )

    report = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )

    assert report["valid"]
    assert report["runtime_backend"] == "resident"
    assert report["runtime_storage_valid"]
    assert report["runtime_storage_by_tp"]["tp8"]["valid"]
    assert report["runtime_storage_by_tp"]["tp8"]["matches_header_audit"]
    assert not report["native_fp8"]
    assert "on-demand" in report["scope"]
    assert report["performance_comparison_valid"]
    assert report["performance_comparisons"]["tp4"][
        "peak_memory_reduction_mib"
    ] == 5_000.0
    assert report["performance_comparisons"]["tp8"][
        "throughput_ratio"
    ] == pytest.approx(160.0 / 180.0)
    assert report["performance_comparisons"]["tp4"][
        "resident_expert_storage_bytes_by_rank"
    ] == [1_010_000] * 4
    assert report["performance_comparisons"]["tp4"][
        "resident_dequant_workspace_bytes_by_rank"
    ] == [65_536] * 4

    performance_path = (
        tmp_path / "fp8/performance/run-fp8_matrix_summary.json"
    )
    performance = json.loads(performance_path.read_text())
    performance["runs"][0]["storage"]["runtime_buffer_storage_by_rank"][0][
        "resident_fp8_dequant_workspace_reuse_count"
    ] = 0
    write(performance_path, performance)
    invalid = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )
    assert not invalid["runtime_storage_valid"]
    assert not invalid["performance_valid"]
    assert not invalid["valid"]

    write_fp8_summary_inputs(
        tmp_path,
        "run",
        backend="resident",
        requested_backend="resident",
    )
    audit_path = tmp_path / "fp8/official_checkpoint_header_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["results"]["tp4"]["resident_fp8_expert_storage"][
        "dequant_workspace_bytes"
    ] += 1
    write(audit_path, audit)
    mismatch = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )
    assert not mismatch["runtime_storage_by_tp"]["tp4"][
        "matches_header_audit"
    ]
    assert not mismatch["runtime_storage_valid"]
    assert not mismatch["valid"]

    write_fp8_summary_inputs(
        tmp_path,
        "run",
        backend="resident",
        requested_backend="resident",
    )
    local_audit_path = tmp_path / "fp8/preflight/checkpoint_mapping_audit.json"
    local_audit = json.loads(local_audit_path.read_text())
    local_audit["fp8_runtime_backend"] = "reference"
    write(local_audit_path, local_audit)
    wrong_preflight = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )
    assert not wrong_preflight["local_checkpoint_matches_official"]
    assert not wrong_preflight["valid"]


def test_resident_fp8_summary_rejects_missing_or_regressed_bf16_comparison(tmp_path):
    write_fp8_summary_inputs(
        tmp_path,
        "run",
        backend="resident",
        requested_backend="resident",
    )

    missing = MODULE.summarize_optional_fp8_audit(tmp_path, "run")
    assert not missing["performance_comparison_valid"]
    assert not missing["performance_valid"]
    assert not missing["valid"]

    no_memory_win = MODULE.summarize_optional_fp8_audit(
        tmp_path,
        "run",
        [
            {
                **row,
                "median": {
                    **row["median"],
                    "peak_torch_allocated_mib": 10_000.0
                    if row["tensor_parallel_size"] == 4
                    else row["median"]["peak_torch_allocated_mib"],
                },
            }
            for row in fp8_baseline_rows()
        ],
    )
    assert not no_memory_win["performance_comparisons"]["tp4"]["valid"]
    assert not no_memory_win["valid"]

    write_fp8_summary_inputs(
        tmp_path,
        "run",
        backend="resident",
        requested_backend="resident",
        throughput_by_tp={4: 79.0, 8: 160.0},
    )
    too_slow = MODULE.summarize_optional_fp8_audit(
        tmp_path, "run", fp8_baseline_rows()
    )
    assert (
        too_slow["performance_comparisons"]["tp4"]["throughput_ratio"]
        == 0.79
    )
    assert not too_slow["performance_comparisons"]["tp4"]["valid"]
    assert not too_slow["valid"]


def test_optional_gptq_summary_requires_actual_triton_execution(tmp_path):
    write_gptq_summary_inputs(tmp_path, "run")

    report = MODULE.summarize_optional_gptq(
        tmp_path,
        "run",
        expected_source_commit=SOURCE_COMMIT,
    )

    assert report["valid"]
    assert report["local_checkpoint_matches_official"]
    assert report["memory_preflight_valid"]
    assert report["bf16_vs_gptq_quality_valid"]
    assert report["tensor_parallel_sizes"] == [4, 8]
    assert report["best_throughput"]["tensor_parallel_size"] == 8
    assert report["lowest_peak_memory"]["tensor_parallel_size"] == 8
    assert report["workspace"]["valid"]
    assert report["source_commit_valid"]
    assert report["workspace"]["by_tp"]["tp4"][
        "expected_bytes_per_rank"
    ] == 64 * (2 * 128 + 2048) * 2

    write_gptq_summary_inputs(tmp_path, "run", backend="reference")
    invalid = MODULE.summarize_optional_gptq(tmp_path, "run")
    assert not invalid["valid"]
    assert not invalid["performance_valid"]
    assert not invalid["quality_valid"]


def test_optional_gptq_summary_rejects_mixed_source_evidence(tmp_path):
    write_gptq_summary_inputs(tmp_path, "run")
    comparison_path = tmp_path / "gptq/quality/bf16_vs_gptq.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["commit"] = "b" * 40
    write(comparison_path, comparison)

    report = MODULE.summarize_optional_gptq(
        tmp_path,
        "run",
        expected_source_commit=SOURCE_COMMIT,
    )

    assert not report["source_commit_valid"]
    assert not report["valid"]


def test_optional_gptq_summary_rejects_workspace_without_runtime_reuse(tmp_path):
    write_gptq_summary_inputs(tmp_path, "run")
    path = tmp_path / "gptq/performance/run-gptq_matrix_summary.json"
    performance = json.loads(path.read_text())
    performance["runs"][0]["storage"]["runtime_buffer_storage_by_rank"][0][
        "gptq_expert_workspace_reuse_count"
    ] = 0
    write(path, performance)

    report = MODULE.summarize_optional_gptq(tmp_path, "run")

    assert not report["workspace"]["by_tp"]["tp4"]["valid"]
    assert not report["performance_valid"]
    assert not report["valid"]


def test_gptq_workspace_rejects_duplicate_rank_evidence():
    rank = {
        "rank": 0,
        "gptq_expert_workspace_pool_count": 1,
        "gptq_expert_workspace_bytes": 5120,
        "gptq_expert_workspace_allocation_count": 1,
        "gptq_expert_workspace_reuse_count": 1,
    }
    summary = MODULE.summarize_gptq_workspace(
        [
            {
                "tensor_parallel_size": 2,
                "storage": {
                    "runtime_buffer_storage_by_rank": [rank, dict(rank)]
                },
            }
        ],
        max_rows=1,
    )

    assert not summary["valid"]
    assert not summary["by_tp"]["tp2"]["valid"]


def write_attention_case(
    root,
    name,
    context_len,
    *,
    partitioned,
    batch_size=4,
    max_abs_diff=0.01,
):
    results = {
        "flash_reference": {
            "status": "ok",
            "median_ms": 2.0,
        },
        "int8_v3_bt256_w8_s2": {
            "status": "ok",
            "median_ms": 1.0,
            "max_abs_diff_vs_flash_reference": max_abs_diff,
            "peak_extra_mib": 2.0,
        },
    }
    if partitioned:
        results["int8_partitioned_ps256"] = {
            "status": "ok",
            "median_ms": 0.8,
            "max_abs_diff_vs_flash_reference": 0.02,
            "peak_extra_mib": 4.0,
        }
        results["int8_partitioned_ps512"] = {
            "status": "ok",
            "median_ms": 0.9,
            "max_abs_diff_vs_flash_reference": 0.015,
            "peak_extra_mib": 3.0,
        }
        results["int8_partitioned_ps512_workspace_reuse"] = {
            "status": "ok",
            "median_ms": 0.85,
            "max_abs_diff_vs_flash_reference": 0.015,
            "peak_extra_mib": 0.0,
            "speedup_vs_allocating": 0.9 / 0.85,
        }
    write(
        root / f"attention/tp4/{name}.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "cuda_available": True,
            "batch_size": batch_size,
            "context_len": context_len,
            "num_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 256,
            "results": results,
            "shape_manifest": {
                "workspace": {
                    "partitioned": {
                        str(MODULE.PRODUCTION_PARTITION_SIZE): {
                            "allocation_count": 1,
                            "shared_storage": True,
                        }
                    }
                    if partitioned
                    else {}
                }
            },
        },
    )


def write_cudagraph_case(
    root,
    context_name,
    attention_path,
    batch_sizes,
    *,
    decode_conv_backend="weighted",
):
    base = (
        "cudagraph"
        if decode_conv_backend == "weighted"
        else f"cudagraph_conv/{decode_conv_backend}"
    )
    write(
        root / f"{base}/tp4/{context_name}/run_1/summary.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "cuda_available": True,
            "kv_cache_dtype": "int8",
            "passed": True,
            "hybrid_graph_captured": True,
            "qwen35_decode_conv_backend": decode_conv_backend,
            "scenarios": [
                {
                    "batch_size": batch_size,
                    "comparison": {
                        "expected_attention_path": attention_path,
                        "expected_eager_attention_path": True,
                        "expected_graph_attention_path": True,
                        "scratch_primed_across_bucket": True,
                    },
                }
                for batch_size in batch_sizes
            ],
        },
    )


def write_long_prefill_case(root, *, max_abs_error=0.01):
    measurement = {
        "reference": None,
        "candidate": {"median_ms": 4.0, "peak_extra_mib": 128.0},
        "speedup": None,
        "errors": [{"max_abs_error": max_abs_error}],
        "reused_fp32_output_buffer_mib": 32.0,
        "reused_fp32_pairwise_buffer_mib": 16.0,
    }
    convolution = {
        **deepcopy(measurement),
        "compact_state_storage_mib": 0.015625,
        "reused_prefill_state_mib": 0.015625,
        "released_history_storage_mib": 31.996,
        "next_state_reuses_input_storage": True,
    }
    write(
        root / "kernels_long/tp4.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "cuda_available": True,
            "configuration": {
                "prefill_only": True,
                "prefill_batch": 1,
                "prefill_tokens": 8192,
                "tp_size": 4,
                "resolved_device": "cuda",
                "delta_prefill_chunk_sizes": [32, 64, 128],
            },
            "results": {
                "vectorized_prefill_convolution": convolution,
                "grouped_delta_prefill": deepcopy(measurement),
                "grouped_delta_prefill_chunk_sweep": {
                    "baseline_chunk_size": 64,
                    "candidates": {
                        str(chunk_size): {
                            **deepcopy(measurement),
                            "chunk_size": chunk_size,
                            "errors_vs_chunk64": deepcopy(
                                measurement["errors"]
                            ),
                            "candidate": {
                                "median_ms": latency,
                                "peak_extra_mib": memory,
                            },
                        }
                        for chunk_size, latency, memory in (
                            (32, 5.0, 64.0),
                            (64, 4.0, 96.0),
                            (128, 6.0, 160.0),
                        )
                    }
                },
            },
        },
    )


def write_mixed_case(root):
    for repeat, p95 in enumerate((0.019, 0.020, 0.021), start=1):
        write(
            root / f"mixed/tp4/r{repeat}.json",
            {
                "commit": SOURCE_COMMIT,
                "git_dirty": False,
                "cuda_available": True,
                "tensor_parallel_size": 4,
                "qwen35_moe_decode_backend": "batched",
                "enable_dynamic_chunked_prefill": True,
                "injected": True,
                "initial_p95_decode_gap_s": p95,
                "peak_torch_allocated_mib": 12_000.0,
                "generated_token_ids": {"digest": "mixed-tokens"},
                "execution_stats": {
                    "model_path_counts": {"mixed_eager": 3},
                },
                "execution_validation": {"valid": True},
                "generation_validation": {"valid": True},
            },
        )


def write_pressure_case(root):
    cases = (
        ("fcfs", "fcfs", False, 1536, 1024, 6, (9.8, 10.0, 10.2)),
        (
            "min_recompute",
            "min_recompute",
            False,
            256,
            256,
            1,
            (7.8, 8.0, 8.2),
        ),
        (
            "min_recompute_reserved",
            "min_recompute",
            True,
            0,
            0,
            0,
            (7.7, 7.9, 8.1),
        ),
    )
    for (
        case_name,
        policy,
        reservation,
        token_progress,
        max_progress,
        reclaimed,
        elapsed_samples,
    ) in cases:
        for repeat, elapsed in enumerate(elapsed_samples, start=1):
            write(
                root / f"pressure/tp4/{case_name}/r{repeat}.json",
                {
                    "commit": SOURCE_COMMIT,
                    "checkpoint_manifest": {"digest": "weights"},
                    "git_dirty": False,
                    "cuda_available": True,
                    "tensor_parallel_size": 4,
                    "initial_seqs": MODULE.PRESSURE_INITIAL_SEQUENCES,
                    "injected_seqs": MODULE.PRESSURE_INJECTED_SEQUENCES,
                    "initial_input_lens": MODULE.PRESSURE_INITIAL_LENGTHS,
                    "injected_input_lens": MODULE.PRESSURE_INJECTED_LENGTHS,
                    "output_len": 16,
                    "num_kvcache_blocks_override": MODULE.PRESSURE_KV_BLOCKS,
                    "num_kvcache_blocks": MODULE.PRESSURE_KV_BLOCKS,
                    "enable_dynamic_chunked_prefill": True,
                    "enable_decode_kv_reservation": reservation,
                    "preemption_policy": policy,
                    "injected": True,
                    "expected_requests": 4,
                    "finished_requests": 4,
                    "total_time_s": elapsed,
                    "step_count": 63 if policy == "fcfs" else 48,
                    "peak_torch_allocated_mib": 12_000.0,
                    "generated_token_ids": {"digest": "pressure-tokens"},
                    "metrics": {
                        "preemption_count": (
                            2 if policy == "fcfs" else (0 if reservation else 1)
                        ),
                        "waiting_prefill_preemptions": (
                            1 if policy == "min_recompute" and not reservation else 0
                        ),
                        "preempted_token_progress": token_progress,
                        "max_preempted_token_progress": max_progress,
                        "reclaimed_kv_blocks": reclaimed,
                        "prefill_stopped_by_decode_kv_reservation": (
                            2 if reservation else 0
                        ),
                        "avg_ttft_s": 1.0,
                        "p50_ttft_s": 0.8,
                        "p95_ttft_s": 1.8 if policy == "fcfs" else 1.5,
                        "p99_ttft_s": 1.95 if policy == "fcfs" else 1.7,
                        "max_ttft_s": 2.0,
                        "avg_tpot_s": 0.2,
                        "p50_tpot_s": 0.18,
                        "p95_tpot_s": 0.3 if policy == "fcfs" else 0.25,
                        "p99_tpot_s": 0.35 if policy == "fcfs" else 0.3,
                        "max_tpot_s": 0.4,
                        "avg_request_latency_s": 5.0,
                        "p50_request_latency_s": 4.0,
                        "p95_request_latency_s": (
                            9.0 if policy == "fcfs" else 8.0
                        ),
                        "p99_request_latency_s": (
                            9.8 if policy == "fcfs" else 9.0
                        ),
                        "max_request_latency_s": 10.0,
                    },
                    "execution_validation": {"valid": True},
                    "generation_validation": {"valid": True},
                },
            )


def write_fairness_case(root):
    for mode, threshold, starvation, ttft, decode_gap, tpot, throughput in (
        ("disabled", 0, 40, 4.0, 0.020, 0.012, 100.0),
        ("enabled", MODULE.FAIRNESS_THRESHOLD, 4, 1.5, 0.023, 0.013, 98.0),
    ):
        for repeat in range(1, 4):
            write(
                root / f"fairness/{mode}/tp4/r{repeat}.json",
                {
                    "commit": SOURCE_COMMIT,
                    "checkpoint_manifest": {"digest": "weights"},
                    "git_dirty": False,
                    "cuda_available": True,
                    "tensor_parallel_size": 4,
                    "initial_seqs": MODULE.FAIRNESS_INITIAL_SEQUENCES,
                    "injected_seqs": MODULE.FAIRNESS_INJECTED_SEQUENCES,
                    "initial_input_len": MODULE.FAIRNESS_INITIAL_INPUT_LENGTH,
                    "injected_input_len": MODULE.FAIRNESS_INJECTED_INPUT_LENGTH,
                    "output_len": MODULE.FAIRNESS_OUTPUT_LENGTH,
                    "inject_after_decode_steps": (
                        MODULE.FAIRNESS_INJECT_AFTER_DECODE_STEPS
                    ),
                    "max_num_batched_tokens": (
                        MODULE.FAIRNESS_MAX_BATCHED_TOKENS
                    ),
                    "max_num_seqs": (
                        MODULE.FAIRNESS_INITIAL_SEQUENCES
                        + MODULE.FAIRNESS_INJECTED_SEQUENCES
                    ),
                    "prefill_starvation_threshold": threshold,
                    "prefill_starvation_token_budget": (
                        MODULE.FAIRNESS_TOKEN_BUDGET
                    ),
                    "enable_dynamic_chunked_prefill": True,
                    "qwen35_moe_decode_backend": "batched",
                    "injected_ttft_count": MODULE.FAIRNESS_INJECTED_SEQUENCES,
                    "injected_p95_ttft_s": ttft + (repeat - 2) * 0.02,
                    "initial_p95_decode_gap_s": decode_gap,
                    "initial_max_decode_gap_s": decode_gap * 1.5,
                    "output_throughput_tok_s": throughput,
                    "generated_token_ids": {"digest": "fairness-tokens"},
                    "metrics": {
                        "prefill_starved_steps": starvation,
                        "max_prefill_starvation_steps": starvation,
                        "p95_tpot_s": tpot,
                    },
                    "execution_stats": {
                        "model_path_counts": (
                            {
                                "prefill_eager": 3,
                                "decode_cuda_graph": 60,
                            }
                            if mode == "disabled"
                            else {"mixed_eager": 3}
                        ),
                    },
                    "execution_validation": {"valid": True},
                    "generation_validation": {"valid": True},
                },
            )


def scheduler_trace_result(mode, repeat=1):
    optimized = mode == "optimized"
    classes = ("decode-heavy", "prefill-heavy", "short")
    request_samples = [
        {
            "request_id": f"{workload_class}-{index}",
            "seq_id": class_index * 8 + index,
            "workload_class": workload_class,
            "arrival_step": index,
            "first_schedule_step": index + (1 if optimized else 2),
            "completion_step": index + 4,
            "scheduler_wait_steps": 1 if optimized else 2,
            "input_tokens": 128,
            "requested_output_tokens": 4,
            "output_tokens": 4,
            "preemption_count": (
                1 if not optimized and class_index == 0 and index < 2 else 0
            ),
            "preempted_token_progress": (
                64 if not optimized and class_index == 0 and index < 2 else 0
            ),
            "recomputed_tokens": (
                16 if not optimized and class_index == 0 and index < 2 else 0
            ),
            "time_to_first_schedule_s": (
                0.03 if optimized else 0.05
            ) + repeat * 0.0005,
            "first_token_service_s": (
                0.05 if optimized else 0.07
            ) + repeat * 0.0005,
            "ttft_s": (0.08 if optimized else 0.12) + repeat * 0.001,
            "tpot_s": 0.018 if optimized else 0.02,
            "latency_s": (0.18 if optimized else 0.22) + repeat * 0.001,
        }
        for class_index, workload_class in enumerate(classes)
        for index in range(8)
    ]
    return {
        "commit": SOURCE_COMMIT,
        "checkpoint_manifest": {"digest": "weights"},
        "git_dirty": False,
        "cuda_available": True,
        "tensor_parallel_size": 4,
        "qwen35_moe_decode_backend": "batched",
        "temperature": 0.0,
        "enable_dynamic_chunked_prefill": optimized,
        "enable_decode_kv_reservation": optimized,
        "prefill_starvation_threshold": (
            MODULE.FAIRNESS_THRESHOLD if optimized else 0
        ),
        "preemption_policy": "min_recompute" if optimized else "fcfs",
        "total_time_s": 8.0 if optimized else 10.0,
        "output_throughput_tok_s": 12.0 if optimized else 9.6,
        "peak_torch_allocated_mib": 11_900.0 if optimized else 12_000.0,
        "replay": {
            "workload": {
                "name": "mixed",
                "digest": "mixed-workload",
                "request_count": 24,
                "requested_output_tokens": 96,
                "requests_by_class": {
                    "decode-heavy": 8,
                    "prefill-heavy": 8,
                    "short": 8,
                },
            },
            "engine_steps": 11,
            "request_samples": request_samples,
            "output_token_ids": {
                "digest": "greedy-output",
                "request_count": 24,
                "token_count": 96,
            },
            "latency": {
                "all": {
                    "request_count": 24,
                    "p95_scheduler_wait_steps": 1 if optimized else 2,
                    "p95_ttft_s": (
                        0.08 if optimized else 0.12
                    ) + repeat * 0.001,
                    "p95_time_to_first_schedule_s": (
                        0.03 if optimized else 0.05
                    ) + repeat * 0.0005,
                    "p95_first_token_service_s": (
                        0.05 if optimized else 0.07
                    ) + repeat * 0.0005,
                    "p95_tpot_s": 0.018 if optimized else 0.02,
                    "p95_latency_s": (
                        0.18 if optimized else 0.22
                    ) + repeat * 0.001,
                },
                "by_class": {
                    workload_class: {
                        "request_count": 8,
                        "p95_scheduler_wait_steps": 1 if optimized else 2,
                        "p95_time_to_first_schedule_s": (
                            0.03 if optimized else 0.05
                        ) + repeat * 0.0005,
                        "p95_first_token_service_s": (
                            0.05 if optimized else 0.07
                        ) + repeat * 0.0005,
                        "p95_ttft_s": (
                            0.08 if optimized else 0.12
                        ) + repeat * 0.001,
                        "p95_tpot_s": 0.018 if optimized else 0.02,
                        "p95_latency_s": (
                            0.18 if optimized else 0.22
                        ) + repeat * 0.001,
                    }
                    for workload_class in classes
                },
            },
            "preemption": {
                "all": {
                    "request_count": 24,
                    "preempted_request_count": 0 if optimized else 2,
                    "preempted_request_rate": 0.0 if optimized else 2 / 24,
                    "total_preemption_count": 0 if optimized else 2,
                    "max_preemptions_per_request": 0 if optimized else 1,
                    "total_preempted_token_progress": 0 if optimized else 128,
                    "p95_preempted_token_progress": (
                        54.4 if not optimized else 0.0
                    ),
                    "max_preempted_token_progress": 0 if optimized else 64,
                    "total_recomputed_tokens": 0 if optimized else 32,
                    "p95_recomputed_tokens": 13.6 if not optimized else 0.0,
                    "max_recomputed_tokens": 0 if optimized else 16,
                },
                "by_class": {
                    workload_class: {
                        "request_count": 8,
                        "preempted_request_count": (
                            2
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "preempted_request_rate": (
                            0.25
                            if not optimized and workload_class == "decode-heavy"
                            else 0.0
                        ),
                        "total_preemption_count": (
                            2
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "max_preemptions_per_request": (
                            1
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "total_preempted_token_progress": (
                            128
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "p95_preempted_token_progress": (
                            64.0
                            if not optimized and workload_class == "decode-heavy"
                            else 0.0
                        ),
                        "max_preempted_token_progress": (
                            64
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "total_recomputed_tokens": (
                            32
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                        "p95_recomputed_tokens": (
                            16.0
                            if not optimized and workload_class == "decode-heavy"
                            else 0.0
                        ),
                        "max_recomputed_tokens": (
                            16
                            if not optimized and workload_class == "decode-heavy"
                            else 0
                        ),
                    }
                    for workload_class in classes
                },
            },
            "step_samples": [
                {
                    "logical_step": step,
                    "elapsed_s": 0.01,
                    "scheduled_request_ids": [
                        item["request_id"]
                        for item in request_samples
                        if item["first_schedule_step"] == step
                    ],
                    "capacity": {
                        "sequence_slots_total": 24,
                        "sequence_slots_used": min(step + 1, 24),
                        "sequence_slots_free": 24 - min(step + 1, 24),
                        "sequence_slots_waiting_owned": 0,
                        "sequence_slots_local_running": min(step + 1, 24),
                        "sequence_slots_remote_destination": 0,
                        "sequence_slots_remote_source": 0,
                        "state_slots_total": 24,
                        "state_slots_used": min(step + 1, 24),
                        "state_slots_free": 24 - min(step + 1, 24),
                        "kv_blocks_total": 16,
                        "kv_blocks_used": step,
                        "kv_blocks_free": 16 - step,
                        "kv_block_usage": step / 16,
                    },
                }
                for step in range(11)
            ],
            "engine_metrics": {
                "num_finished_requests": 24,
                "preemption_count": 0 if optimized else 2,
                "max_prefill_starvation_steps": 4 if optimized else 12,
            },
        },
        "execution_stats": {
            "model_path_counts": (
                {"mixed_eager": 2} if optimized else {"prefill_eager": 2}
            )
        },
        "execution_validation": {"valid": True},
    }


def write_scheduler_trace_case(root):
    for mode in ("baseline", "optimized"):
        for repeat in range(1, 4):
            write(
                root / f"scheduler/{mode}/tp4/r{repeat}.json",
                scheduler_trace_result(mode, repeat),
            )


def test_scheduler_trace_comparison_requires_greedy_output_parity():
    baseline_results = [
        scheduler_trace_result("baseline", repeat) for repeat in (1, 2)
    ]
    optimized_results = [
        scheduler_trace_result("optimized", repeat) for repeat in (1, 2)
    ]
    optimized_results[1]["replay"]["output_token_ids"]["digest"] = "changed"

    baseline = MODULE.summarize_scheduler_trace_repeats(
        baseline_results,
        expected_tp_size=4,
        mode="baseline",
    )
    optimized = MODULE.summarize_scheduler_trace_repeats(
        optimized_results,
        expected_tp_size=4,
        mode="optimized",
    )
    comparison = MODULE.compare_scheduler_trace_modes(baseline, optimized)

    assert baseline["valid"]
    assert not optimized["valid"]
    assert not comparison["valid"]


def test_scheduler_trace_compares_tail_latency_for_every_workload_class():
    baseline = MODULE.summarize_scheduler_trace_repeats(
        [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)],
        expected_tp_size=4,
        mode="baseline",
    )
    optimized = MODULE.summarize_scheduler_trace_repeats(
        [scheduler_trace_result("optimized", repeat) for repeat in (1, 2)],
        expected_tp_size=4,
        mode="optimized",
    )

    comparison = MODULE.compare_scheduler_trace_modes(baseline, optimized)

    assert comparison["valid"]
    assert optimized["capacity_peaks"]["sequence_slots_used"] == 11
    assert optimized["capacity_peaks"]["state_slots_used"] == 11
    assert optimized["capacity_peaks"]["kv_blocks_used"] == 10
    assert comparison["capacity_comparable"]
    assert all(
        delta == 0
        for delta in comparison[
            "capacity_peak_delta_optimized_minus_baseline"
        ].values()
    )
    assert comparison["class_latency_comparable"]
    assert set(comparison["ratios_optimized_over_baseline_by_class"]) == {
        "decode-heavy",
        "prefill-heavy",
        "short",
    }
    assert comparison["ratios_optimized_over_baseline_by_class"]["short"][
        "p95_ttft_s"
    ] == pytest.approx(0.0815 / 0.1215)
    assert comparison["class_tail_regressions"] == []


def test_scheduler_trace_rejects_per_class_scheduler_wait_regression():
    baseline = MODULE.summarize_scheduler_trace_repeats(
        [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)],
        expected_tp_size=4,
        mode="baseline",
    )
    optimized_results = [
        scheduler_trace_result("optimized", repeat) for repeat in (1, 2)
    ]
    for result in optimized_results:
        samples = result["replay"]["request_samples"]
        for sample in samples:
            if sample["workload_class"] == "short":
                sample["scheduler_wait_steps"] = 3
                sample["first_schedule_step"] = sample["arrival_step"] + 3
        result["replay"]["latency"]["all"]["p95_scheduler_wait_steps"] = 3
        result["replay"]["latency"]["by_class"]["short"][
            "p95_scheduler_wait_steps"
        ] = 3
        for step in result["replay"]["step_samples"]:
            step["scheduled_request_ids"] = [
                sample["request_id"]
                for sample in samples
                if sample["first_schedule_step"] == step["logical_step"]
            ]
    optimized = MODULE.summarize_scheduler_trace_repeats(
        optimized_results,
        expected_tp_size=4,
        mode="optimized",
    )

    comparison = MODULE.compare_scheduler_trace_modes(baseline, optimized)

    assert comparison["valid"]
    assert {
        "workload_class": "short",
        "metric": "p95_scheduler_wait_steps",
        "ratio": 1.5,
    } in comparison["class_tail_regressions"]


def test_scheduler_trace_rejects_inconsistent_first_schedule_trace():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["step_samples"][2]["scheduled_request_ids"] = []

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["schedule_trace_contract_valid"]


def test_scheduler_trace_rejects_capacity_ownership_drift():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["step_samples"][0]["capacity"][
        "sequence_slots_remote_source"
    ] = 1

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["step_contract_valid"]


def test_scheduler_trace_rejects_missing_class_latency_evidence():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["latency"]["by_class"].pop("short")

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["class_latency_contract_valid"]


def test_scheduler_trace_rejects_unbalanced_classes_and_missing_metrics():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["request_samples"][0]["workload_class"] = "short"
    results[1]["replay"]["engine_metrics"].pop("preemption_count")

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["sample_contract_valid"]
    assert not summary["runs"][1]["scheduler_metrics_valid"]
    assert summary["preemption_count"] is None


def test_scheduler_trace_rejects_inconsistent_request_preemption_summary():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["preemption"]["all"][
        "total_preemption_count"
    ] = 99
    results[1]["replay"]["request_samples"][0]["preemption_count"] = 2

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["preemption_contract_valid"]
    assert not summary["runs"][1]["scheduler_metrics_valid"]


def test_scheduler_trace_rejects_invalid_ttft_decomposition():
    results = [scheduler_trace_result("baseline", repeat) for repeat in (1, 2)]
    results[0]["replay"]["request_samples"][0][
        "first_token_service_s"
    ] += 1.0

    summary = MODULE.summarize_scheduler_trace_repeats(
        results,
        expected_tp_size=4,
        mode="baseline",
    )

    assert not summary["valid"]
    assert not summary["runs"][0]["sample_contract_valid"]


def load_fairness_repeats(root, mode):
    return [
        json.loads(path.read_text())
        for path in sorted((root / f"fairness/{mode}/tp4").glob("r*.json"))
    ]


def test_scheduler_fairness_comparison_rejects_ttft_regression(tmp_path):
    write_fairness_case(tmp_path)
    disabled = MODULE.summarize_fairness_repeats(
        load_fairness_repeats(tmp_path, "disabled"),
        expected_tp_size=4,
        mode="disabled",
    )
    enabled_results = load_fairness_repeats(tmp_path, "enabled")
    for result in enabled_results:
        result["injected_p95_ttft_s"] = 5.0
    enabled = MODULE.summarize_fairness_repeats(
        enabled_results,
        expected_tp_size=4,
        mode="enabled",
    )

    comparison = MODULE.compare_fairness_modes(disabled, enabled)

    assert not comparison["valid"]
    assert comparison["starvation_improved"]
    assert not comparison["injected_ttft_improved"]


def test_summary_selects_valid_performance_and_preserves_evidence(tmp_path):
    run_id = "rental-a"
    write(
        tmp_path / "manifest.json",
        {"run_id": run_id, "source_commit": SOURCE_COMMIT},
    )
    write(
        tmp_path / "preflight/official_checkpoint_header_audit.json",
        {
            "valid": True,
            "repo": MODULE.OFFICIAL_CHECKPOINT_REPO,
            "resolved_revision": MODULE.OFFICIAL_CHECKPOINT_REVISION,
            "config_sha256": MODULE.OFFICIAL_CONFIG_SHA256,
            "index_sha256": MODULE.OFFICIAL_INDEX_SHA256,
            "headers_sha256": MODULE.OFFICIAL_HEADERS_SHA256,
            "semantic_contract": MODULE.expected_checkpoint_semantic_contract(
                "bf16"
            ),
                "source_tensor_count": 1045,
            "shard_count": 14,
            "checkpoint_shards": [
                {
                    "name": f"model-{index:05d}-of-00014.safetensors",
                    "size_bytes": index,
                    "sha256": f"{index:064x}",
                }
                for index in range(1, 15)
            ],
            "results": {
                "tp4": {
                    "valid": True,
                    "skipped_by_prefix": {
                        "model.visual.": 333,
                            "mtp.": 19,
                    },
                    "unclassified_skipped_weights": [],
                }
            },
        },
    )
    write(
        tmp_path / "preflight/checkpoint_mapping_audit.json",
        {
            "valid": True,
            "complete": True,
            "results": {
                "tp4": {
                    "skipped_tensor_groups": {
                        "model.visual": 333,
                            "mtp": 19,
                    }
                }
            },
            "checkpoint_manifest": {
                "digest": "weights",
                "strength": "metadata-only",
                "config_sha256": MODULE.OFFICIAL_CONFIG_SHA256,
                "index_sha256": MODULE.OFFICIAL_INDEX_SHA256,
                "shard_count": 14,
                "present_shard_count": 14,
                "missing_shards": [],
                    "total_size_bytes": 71_903_645_408,
                "files": [
                    {
                        "name": f"model-{index:05d}-of-00014.safetensors",
                        "size_bytes": index,
                        "content_id": f"{index:064x}",
                        "content_sha256": f"{index:064x}",
                        "present": True,
                    }
                    for index in range(1, 15)
                ],
            },
        },
    )
    write(
        tmp_path / "preflight/memory_preflight.json",
        {
            "valid": True,
            "results": {
                "tp4": {
                    "local_parameter_bytes": 16_000,
                    "model_max_position_embeddings": 262_144,
                    "configured_max_model_len": 16_384,
                    "rotary_cache_bytes_per_rank": 4096,
                    "max_state_bytes_per_rank": 1_000,
                    "state_bytes_per_rank_by_dtype": {
                        "float32": 1_000,
                        "model": 500,
                    },
                    "minimum_workload_kv_bytes_per_rank": 2_000,
                    "kv_bytes_per_token_by_dtype": {
                        "auto": 10_240,
                        "int8": 5_160,
                    },
                    "pd_transfer_bytes_per_sequence_by_dtype": {
                        "auto": {"float32": 10_000, "model": 9_000},
                        "int8": {"float32": 6_000, "model": 5_000},
                    },
                    "pd_transfer_bytes_all_tp_ranks_by_dtype": {
                        "auto": {"float32": 40_000, "model": 36_000},
                        "int8": {"float32": 24_000, "model": 20_000},
                    },
                    "pd_transfer_components_per_sequence_by_dtype": {
                        "auto": {
                            "float32": {
                                "kv": 8_000,
                                "kv_scales": 0,
                                "recurrent": 1_500,
                                "convolution": 500,
                                "total": 10_000,
                            },
                            "model": {
                                "kv": 8_000,
                                "kv_scales": 0,
                                "recurrent": 500,
                                "convolution": 500,
                                "total": 9_000,
                            },
                        },
                        "int8": {
                            "float32": {
                                "kv": 4_000,
                                "kv_scales": 500,
                                "recurrent": 1_000,
                                "convolution": 500,
                                "total": 6_000,
                            },
                            "model": {
                                "kv": 4_000,
                                "kv_scales": 500,
                                "recurrent": 0,
                                "convolution": 500,
                                "total": 5_000,
                            },
                        },
                    },
                    "pd_transfer_allocated_tokens": 512,
                    "pd_transfer_context_tokens": 511,
                    "kv_capacity_by_dtype": {
                        "auto": {
                            "memory_limited_total_token_slots": 100_000,
                            "memory_limited_context_tokens_per_sequence": 1_280,
                            "effective_context_tokens_per_sequence": 1_280,
                        },
                        "int8": {
                            "memory_limited_total_token_slots": 200_000,
                            "memory_limited_context_tokens_per_sequence": 3_072,
                            "effective_context_tokens_per_sequence": 3_072,
                        },
                    },
                    "capacity_concurrent_sequences": 64,
                    "required_free_bytes_per_rank": 20_000,
                    "available_budget_bytes_by_rank": [25_000] * 4,
                }
            },
        },
    )

    def kv_rank(rank, block_bytes):
        allocatable_bytes = 4_200_000
        shared_num_blocks = 1
        return {
            "rank": rank,
            "total_bytes": block_bytes,
            "free_bytes_before_kv": 6_000_000,
            "total_device_bytes": 8_000_000,
            "used_bytes_before_kv": 2_000_000,
            "requested_memory_bytes": 7_200_000,
            "peak_allocated_bytes": 3_000_000,
            "current_allocated_bytes": 2_000_000,
            "transient_peak_bytes": 1_000_000,
            "allocatable_bytes_before_sync": allocatable_bytes,
            "block_bytes": block_bytes,
            "local_num_blocks_before_sync": allocatable_bytes // block_bytes,
            "shared_num_blocks": shared_num_blocks,
            "unused_capacity_bytes_after_sync": (
                allocatable_bytes - shared_num_blocks * block_bytes
            ),
        }

    rows = [
        {
            "label": "sorted",
            "commit": SOURCE_COMMIT,
            "tensor_parallel_size": 4,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "qwen35_moe_decode_backend": "sorted",
            "generated_token_ids_digest": "tokens",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "execution_paths": {
                "required": [
                    "prefill_contiguous_view",
                    "decode_contiguous_view",
                ],
                "observed_in_all_repeats": [
                    "prefill_contiguous_view",
                    "decode_contiguous_view",
                ],
            },
            "generation_valid": True,
            "storage": {
                "num_kvcache_blocks": 1,
                "kv_cache_storage": {"total_bytes": 256 * 10_240},
                "kv_cache_storage_by_rank": [
                    kv_rank(rank, 256 * 10_240)
                    for rank in range(4)
                ],
                "recurrent_state_storage": {
                    "total_bytes_local_rank": 500,
                    "rotary_cache_bytes_local_rank": 4096,
                },
                "recurrent_state_storage_by_rank": [
                    {
                        "rank": rank,
                        "total_bytes_local_rank": 500,
                        "rotary_cache_bytes_local_rank": 4096,
                    }
                    for rank in range(4)
                ],
                "runtime_buffer_storage_by_rank": [
                    {
                        "rank": rank,
                        "moe_decode_host_route_sync_count": 40,
                        "moe_prefill_host_route_sync_count": 40,
                        "moe_batched_dispatch_count": 0,
                    }
                    for rank in range(4)
                ],
            },
            "median": {
                "output_throughput_tok_s": 10,
                "avg_tpot_s": 0.2,
                "peak_torch_allocated_mib": 8,
            },
            "coefficient_of_variation": {
                "output_throughput_tok_s": 0.01,
                "avg_tpot_s": 0.02,
            },
        },
        {
            "label": "batched",
            "commit": SOURCE_COMMIT,
            "tensor_parallel_size": 4,
            "recurrent_state_dtype": "model",
            "kv_cache_dtype": "auto",
            "qwen35_moe_decode_backend": "batched",
            "generated_token_ids_digest": "tokens",
            "repeat_output_digests_match": True,
            "execution_paths_valid": True,
            "execution_paths": {
                "required": [
                    "prefill_contiguous_view",
                    "decode_graph_indexed",
                ],
                "observed_in_all_repeats": [
                    "prefill_contiguous_view",
                    "decode_graph_indexed",
                ],
            },
            "generation_valid": True,
            "storage": {
                "num_kvcache_blocks": 1,
                "kv_cache_storage": {"total_bytes": 256 * 10_240},
                "kv_cache_storage_by_rank": [
                    kv_rank(rank, 256 * 10_240)
                    for rank in range(4)
                ],
                "recurrent_state_storage": {
                    "total_bytes_local_rank": 500,
                    "rotary_cache_bytes_local_rank": 4096,
                },
                "recurrent_state_storage_by_rank": [
                    {
                        "rank": rank,
                        "total_bytes_local_rank": 500,
                        "rotary_cache_bytes_local_rank": 4096,
                    }
                    for rank in range(4)
                ],
                "runtime_buffer_storage_by_rank": [
                    {
                        "rank": rank,
                        "moe_decode_host_route_sync_count": 0,
                        "moe_prefill_host_route_sync_count": 40,
                        "moe_batched_dispatch_count": 40,
                    }
                    for rank in range(4)
                ],
            },
            "median": {
                "output_throughput_tok_s": 20,
                "avg_tpot_s": 0.1,
                "peak_torch_allocated_mib": 12,
            },
            "coefficient_of_variation": {
                "output_throughput_tok_s": 0.02,
                "avg_tpot_s": 0.01,
            },
        },
    ]
    int8_baseline = deepcopy(rows[0])
    int8_baseline.update(label="sorted-int8", kv_cache_dtype="int8")
    int8_baseline["storage"]["kv_cache_storage"]["total_bytes"] = 256 * 5_160
    int8_baseline["storage"]["kv_cache_storage_by_rank"] = [
        kv_rank(rank, 256 * 5_160) for rank in range(4)
    ]
    int8_candidate = deepcopy(rows[1])
    int8_candidate.update(label="batched-int8", kv_cache_dtype="int8")
    int8_candidate["storage"]["kv_cache_storage"]["total_bytes"] = 256 * 5_160
    int8_candidate["storage"]["kv_cache_storage_by_rank"] = [
        kv_rank(rank, 256 * 5_160) for rank in range(4)
    ]
    conv_candidate = deepcopy(rows[1])
    conv_candidate.update(
        label="batched-conv-channel-accumulate",
        qwen35_decode_conv_backend="channel_accumulate",
    )
    conv_candidate["median"]["peak_torch_allocated_mib"] = 11
    int8_conv_candidate = deepcopy(int8_candidate)
    int8_conv_candidate.update(
        label="batched-int8-conv-channel-accumulate",
        qwen35_decode_conv_backend="channel_accumulate",
    )
    int8_conv_candidate["median"]["peak_torch_allocated_mib"] = 11
    rows.extend(
        (
            int8_baseline,
            int8_candidate,
            conv_candidate,
            int8_conv_candidate,
        )
    )
    write(
        tmp_path / f"performance/{run_id}_matrix_summary.json",
        {
            "commits": [SOURCE_COMMIT],
            "workload": {
                "checkpoint_manifest_digest": "weights",
                "max_num_seqs": 64,
            },
            "all_execution_paths_valid": True,
            "all_generation_valid": True,
            "all_output_digests_match": True,
            "runs": rows,
        },
    )
    write(
        tmp_path / f"quality/{run_id}_summary.json",
        {
            "model": "/model",
            "quality_scope": "decode",
            "cases": [
                {
                    "kv_sensitive_token_rows": 2,
                    "int8_partitioned_decode_observed": True,
                }
            ],
            "comparisons_by_tp": {"tp4": {"baseline_decode_ppl": 3.0}},
            "cross_tp": {"all_passed": True, "comparisons": []},
            "quality_gates": {"all_passed": True, "thresholds": {}},
        },
    )
    write(
        tmp_path
        / "quality_conv/channel_accumulate"
        / f"{run_id}-conv-channel_accumulate_summary.json",
        {
            "model": "/model",
            "tensor_parallel_sizes": None,
            "case_token_digest": None,
            "cases": [
                {"qwen35_decode_conv_backend": "channel_accumulate"}
            ],
            "cross_tp": {"all_passed": True, "comparisons": []},
            "quality_gates": {"all_passed": True, "thresholds": {}},
        },
    )
    quality_dir = tmp_path / f"quality/{run_id}_qwen35_tp4"
    write(
        quality_dir / f"{quality_dir.name}.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "checkpoint_manifest": {"digest": "weights"},
        },
    )
    write(quality_dir / "batch0_len128_cases.json", [{"prompt_ids": [1, 2]}])
    conv_quality_dir = (
        tmp_path
        / "quality_conv/channel_accumulate"
        / f"{run_id}-conv-channel_accumulate_qwen35_tp4"
    )
    write(
        conv_quality_dir / f"{conv_quality_dir.name}.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "checkpoint_manifest": {"digest": "weights"},
        },
    )
    write(
        tmp_path / "kernels/tp4.json",
        {
            "commit": SOURCE_COMMIT,
            "git_dirty": False,
            "cuda_available": True,
            "results": {
                "sampling_filter_fast_paths": {
                    name: {
                        "reference": {"peak_extra_mib": 128.0},
                        "candidate": {"peak_extra_mib": 32.0},
                        "speedup": 2.0,
                        "errors": [{"max_abs_error": 0.0}],
                        "avoided_full_sort_workspace_mib": 128.0,
                        "uses_host_sampling_metadata": True,
                        **(
                            {
                                "avoided_top_p_shift_clone_mib": 1.0,
                                "eliminated_top_p_mask_clones_per_step": 1,
                            }
                            if name == "top_k_top_p"
                            else {}
                        ),
                    }
                    for name in ("unfiltered", "top_k", "top_k_top_p")
                }
                | {
                    "top_p": {
                        "reference": {"peak_extra_mib": 128.0},
                        "candidate": {"peak_extra_mib": 32.0},
                        "speedup": 1.1,
                        "errors": [{"max_abs_error": 0.0}],
                        "avoided_top_k_mask_workspace_mib": 64.0,
                        "avoided_top_p_shift_clone_mib": 32.0,
                        "eliminated_top_p_mask_clones_per_step": 1,
                        "uses_host_sampling_metadata": True,
                    }
                },
                "greedy_sampler_precision_fast_path": {
                    "reference": {"peak_extra_mib": 128.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 2.0,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_fp32_logits_mib": 64.0,
                    "uses_host_sampling_metadata": True,
                },
                "sampling_input_buffer_reuse": {
                    "reference": {"peak_extra_mib": 1.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}] * 3,
                    "eliminated_tensor_allocations_per_step": 6,
                    "persistent_sampling_input_mib": 0.01,
                    "candidate_reuses_host_device_storage": True,
                },
                "sampling_noise_buffer_reuse": {
                    "reference": {"peak_extra_mib": 1.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "eliminated_tensor_allocations_per_sampling_step": 1,
                    "persistent_sampling_noise_mib": 0.0,
                    "reused_filtered_logits_mib": 0.01,
                    "candidate_reuses_filtered_logits_storage": True,
                },
                "packed_block_metadata_buffer_reuse": {
                    "reference": {"peak_extra_mib": 1.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}] * 2,
                    "eliminated_tensor_allocations_per_update": 4,
                    "persistent_metadata_buffers_mib": 0.01,
                    "candidate_reuses_two_isolated_buffer_banks": True,
                },
                "compact_top_k_sampling": {
                    "reference": {"peak_extra_mib": 32.0},
                    "candidate": {"peak_extra_mib": 2.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "full_fp32_logits_mib": 32.0,
                    "compact_fp32_logits_mib": 1.0,
                    "avoided_fp32_logits_mib": 31.0,
                },
                "sampling_filter_output_reuse": {
                    "reference": {"peak_extra_mib": 64.0},
                    "candidate": {"peak_extra_mib": 32.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_fp32_logits_mib": 64.0,
                    "eliminated_tensor_allocations_per_sampling_step": 2,
                    "candidate_reuses_temperature_and_filter_storage": True,
                },
                "gated_delta_packed_projection": {
                    "reference": {"peak_extra_mib": 2.0},
                    "candidate": {"peak_extra_mib": 2.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "reference_gemm_launches": 3,
                    "candidate_gemm_launches": 1,
                    "avoided_gemm_launches": 2,
                },
                "attention_packed_qkv": {
                    "reference": {"peak_extra_mib": 2.0},
                    "candidate": {"peak_extra_mib": 2.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "reference_gemm_launches": 3,
                    "candidate_gemm_launches": 1,
                    "avoided_gemm_launches": 2,
                    "key_alias_break_copy_mib": 0.01,
                },
                "contiguous_decode_state": {
                    "reference": {"peak_extra_mib": 16.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_state_gather_mib": 15.0,
                    "avoided_state_scatter_mib": 15.0,
                    "candidate_uses_cache_views": True,
                },
                "decode_convolution_state_reuse": {
                    "reference": {"peak_extra_mib": 16.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.5,
                    "errors": [
                        {"max_abs_error": 0.0},
                        {"max_abs_error": 0.0},
                    ],
                    "reused_convolution_state_mib": 8.0,
                    "eliminated_weighted_state_temporary_mib": 32.0,
                    "candidate_uses_inplace_channel_accumulation": True,
                },
                "router_topk_first": {
                    "reference": {"peak_extra_mib": 2.0},
                    "candidate": {"peak_extra_mib": 1.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_selected_logits_mib": 0.1,
                },
                "expert_dispatch_torch": {
                    batch: {
                        "graph_safe_batched_candidate": {
                            "promotion": {"promote_to_runtime": True},
                            "median_ms": 1.0,
                            "speedup_vs_current": 1.2,
                            "peak_extra_mib": 4.0,
                            "errors_vs_current": {"max_abs_error": 0.01},
                            "reused_weighted_route_mib": 0.25,
                            "weight_buffer_reuse": {
                                "reference": {"peak_extra_mib": 8.0},
                                "speedup": 1.1,
                                "peak_extra_mib_delta": -4.0,
                                "errors": {"max_abs_error": 0.0},
                                "persistent_expert_weight_buffer_mib": 4.0,
                                "eliminated_weight_allocations_per_chunk": 2,
                                "candidate_reuses_expert_weight_storage": True,
                                "measured_on_cuda": True,
                            },
                            "broadcast_route_input": {
                                "valid": True,
                                "measured_on_cuda": True,
                                "speedup_vs_repeated_input": 1.05,
                                "peak_extra_mib_delta": -0.25,
                                "errors": {"max_abs_error": 0.0},
                                "reference": {"median_ms": 1.05},
                            },
                        },
                        **(
                            {
                                "device_scalar_candidate": {
                                    "promotion": {"promote_to_runtime": True},
                                    "median_ms": 0.9,
                                    "speedup_vs_current": 1.1,
                                    "peak_extra_mib": 3.0,
                                    "errors_vs_current": {"max_abs_error": 0.0},
                                    "avoids_host_route_sync": True,
                                    "estimated_selected_weight_mib": 2.0,
                                }
                            }
                            if batch == "1"
                            else {}
                        ),
                    }
                    for batch in ("1", "64")
                },
                "mixed_expert_dispatch": {
                    "decode32_prefill512": {
                        "reference": {"peak_extra_mib": 8.0},
                        "candidate": {"peak_extra_mib": 12.0},
                        "errors": [{"max_abs_error": 0.01}],
                        "decode_tokens": 32,
                        "prefill_tokens": 512,
                        "avoided_route_hidden_allocation_mib_per_step": 1.0,
                        "speedup_vs_grouped": 1.1,
                        "measured_on_cuda": True,
                    }
                },
                "rmsnorm_fp32_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_fp32_copy_mib": 4.0,
                    "eliminated_per_call_gain_materialization_mib": 0.01,
                    "persistent_gain_storage_mib": 0.01,
                    "persistent_storage_delta_mib": 0.005,
                    "candidate_reuses_fp32_workspace": True,
                    "candidate_uses_precomputed_gain": True,
                },
                "delta_l2_normalization_reuse": {
                    "reference": {"peak_extra_mib": 12.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_query_key_fp32_mib": 4.0,
                },
                "delta_causal_mask_cache": {
                    "cache_max_entries": 32,
                    "maximum_cached_chunk_size": 1024,
                    "candidates": {
                        chunk_size: {
                            "reference": {"peak_extra_mib": 1.0},
                            "candidate": {"peak_extra_mib": 0.0},
                            "speedup": 1.2,
                            "errors": [{"max_abs_error": 0.0}],
                            "persistent_mask_mib": 0.01,
                            "cache_reuses_storage": True,
                            "eliminated_allocations_per_additional_layer": 1,
                        }
                        for chunk_size in ("32", "64", "128")
                    },
                },
                "gated_rmsnorm_fp32_reuse": {
                    "reference": {"peak_extra_mib": 12.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_hidden_fp32_workspace_mib": 4.0,
                    "reused_gate_fp32_workspace_mib": 4.0,
                    "candidate_reuses_fp32_workspaces": True,
                },
                "gated_delta_beta_buffer_reuse": {
                    "reference": {"peak_extra_mib": 1.0},
                    "candidate": {"peak_extra_mib": 0.0},
                    "speedup": 1.01,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_beta_projection_mib": 0.01,
                },
                "gated_delta_decay_rate_precompute": {
                    "reference": {"peak_extra_mib": 1.0},
                    "candidate": {"peak_extra_mib": 1.0},
                    "speedup": 1.01,
                    "errors": [{"max_abs_error": 0.0}],
                    "precomputed_decay_rate_mib": 0.001,
                    "reused_decay_projection_fp32_mib": 0.01,
                    "reused_softplus_output_mib": 0.01,
                },
                "moe_output_buffer_reuse": {
                    "reference": {"peak_extra_mib": 12.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_routed_output_mib": 4.0,
                    "reused_shared_output_mib": 4.0,
                    "reused_gate_mib": 0.01,
                },
                "sorted_route_weighting_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_weighted_expert_output_mib": 4.0,
                },
                "batched_route_sum_output_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_route_sum_output_mib": 4.0,
                    "candidate_reuses_dispatch_output": True,
                },
                "residual_output_buffer_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_branch_output_mib_per_merge": 4.0,
                    "residual_merges_per_decoder_layer": 2,
                },
                "attention_norm_output_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_projection_output_mib": 4.0,
                },
                "rotary_output_reuse": {
                    "reference": {"peak_extra_mib": 8.0},
                    "candidate": {"peak_extra_mib": 4.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_query_key_output_mib": 4.0,
                },
                "vocab_gather_layout": {
                    "reference": {"peak_extra_mib": 16.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_full_vocab_copy_mib": 8.0,
                    "candidate_returns_transpose_view": True,
                },
                "torch_kv_dequant_buffer_reuse": {
                    "reference": {"peak_extra_mib": 32.0},
                    "candidate": {"peak_extra_mib": 16.0},
                    "speedup": 1.1,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_output_workspace_mib": 16.0,
                    "avoided_block_id_cast_mib": 0.01,
                },
                "delta_prefill_state_reuse": {
                    "reference": {"peak_extra_mib": 16.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.05,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_recurrent_state_mib": 8.0,
                    "avoided_state_reallocations": 127,
                },
                "specialized_delta_decode": {
                    "reference": {"peak_extra_mib": 24.0},
                    "candidate": {"peak_extra_mib": 8.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "reused_recurrent_state_mib": 8.0,
                    "reused_prediction_workspace_mib": 0.25,
                    "reused_decay_exp_mib": 0.01,
                    "avoided_full_state_intermediates": 2,
                },
                "delta_state_contraction": {
                    "reference": {"peak_extra_mib": 48.0},
                    "candidate": {"peak_extra_mib": 16.0},
                    "speedup": 1.2,
                    "errors": [{"max_abs_error": 0.0}],
                    "avoided_state_product_mib_per_contraction": 32.0,
                    "state_contractions_per_decode": 2,
                },
            },
        },
    )
    write_cudagraph_case(
        tmp_path,
        "short",
        "int8_fused_decode",
        [3, 9, 64],
    )
    write_cudagraph_case(
        tmp_path,
        "long",
        "int8_partitioned_decode",
        [3],
    )
    write_cudagraph_case(
        tmp_path,
        "short",
        "int8_fused_decode",
        [3, 9, 64],
        decode_conv_backend="channel_accumulate",
    )
    write_cudagraph_case(
        tmp_path,
        "long",
        "int8_partitioned_decode",
        [3],
        decode_conv_backend="channel_accumulate",
    )
    write_attention_case(tmp_path, "short", 4096, partitioned=False)
    write_attention_case(tmp_path, "long", 16385, partitioned=True)
    write_attention_case(
        tmp_path,
        "max",
        262143,
        partitioned=True,
        batch_size=1,
    )
    write_long_prefill_case(tmp_path)
    write_mixed_case(tmp_path)
    write_pressure_case(tmp_path)
    write_fairness_case(tmp_path)
    write_scheduler_trace_case(tmp_path)
    for profile_name, kv_dtype, state_dtype, components in (
        (
            "auto-float32",
            "auto",
            "float32",
            {
                "kv": 8_000,
                "kv_scales": 0,
                "recurrent": 1_500,
                "convolution": 500,
                "total": 10_000,
            },
        ),
        (
            "int8-model",
            "int8",
            "model",
            {
                "kv": 4_000,
                "kv_scales": 500,
                "recurrent": 0,
                "convolution": 500,
                "total": 5_000,
            },
        ),
    ):
        write(
            tmp_path / f"pd_transfer/tp4/{profile_name}.json",
            {
                "schema_version": 1,
                "scope": "single-rank synchronous TCP loopback correctness baseline",
                "profile": {
                    "memory_preflight": "memory_preflight.json",
                    "tp_size": 4,
                    "kv_dtype": kv_dtype,
                    "state_dtype": state_dtype,
                },
                "workload": {
                    "warmup": 2,
                    "repeats": 10,
                    "components_bytes": components,
                    "payload_frame_bytes_sent": components["total"] + 400,
                    "receiver_ack_bytes": 1,
                    "payload_tensor_count": 4,
                },
                "results": {
                    "latency_ms_samples": [1.0] * 10,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": 1.0,
                    "effective_payload_gib_s_p50": 1.0,
                    "receiver_storage_count": 1,
                    "receiver_storage_coalesced": True,
                    "receiver_host_staging_pool": {
                        "storage_bytes": components["total"],
                        "allocation_count": 1,
                        "reuse_count": 11,
                        "expected_reuse_count": 11,
                        "transient_allocation_count": 0,
                        "leased": 0,
                        "valid": True,
                    },
                },
                "cuda_install": {
                    "enabled": True,
                    "valid": True,
                    "measured_on_cuda": True,
                    "avoids_full_payload_device_conversion": True,
                    "reference_full_payload_staging": {
                        "latency_ms_samples": [1.0] * 10,
                    },
                    "candidate_direct_block_install": {
                        "latency_ms_samples": [1.1] * 10,
                    },
                    "peak_device_bytes_reduction": components["total"],
                    "latency_ratio_vs_reference": 1.1,
                },
                "limitations": ["loopback TCP is not cross-node network evidence"],
            },
        )
        write(
            tmp_path / f"pd_export/tp4/{profile_name}.json",
            {
                "schema_version": 1,
                "scope": "single-rank Qwen3.6 GPU-to-host cache export",
                "environment": {"device": "NVIDIA H20"},
                "correctness": {"candidate_matches_reference": True},
                "profile": {
                    "tp_size": 4,
                    "kv_dtype": kv_dtype,
                    "state_dtype": state_dtype,
                    "components": components,
                    "allocated_tokens": 512,
                    "cached_tokens": 511,
                    "warmup": 2,
                    "repeats": 10,
                },
                "reference_gpu_gather_then_host_copy": {
                    "latency_ms_samples": [2.0] * 10,
                    "latency_ms_p50": 2.0,
                    "peak_extra_device_bytes_samples": [components["total"]] * 10,
                    "peak_extra_device_bytes_max": components["total"],
                },
                "candidate_direct_host_staging": {
                    "latency_ms_samples": [2.5] * 10,
                    "latency_ms_p50": 2.5,
                    "peak_extra_device_bytes_samples": [1024] * 10,
                    "peak_extra_device_bytes_max": 1024,
                    "host_layout": {
                        "tensor_count": 62 if kv_dtype == "int8" else 61,
                        "storage_count": 1,
                        "all_cpu": True,
                        "all_pinned": True,
                    },
                    "host_staging_pool": {
                        "storage_bytes": components["total"],
                        "allocation_count": 1,
                        "reuse_count": 11,
                        "expected_reuse_count": 11,
                        "transient_allocation_count": 0,
                        "leased": 0,
                        "valid": True,
                    },
                },
                "limitations": ["synthetic export benchmark"],
            },
        )
        bounded_export = json.loads(
            (
                tmp_path / f"pd_export/tp4/{profile_name}.json"
            ).read_text()
        )
        bounded_export["profile"]["max_cached_bytes"] = 0
        bounded_pool = bounded_export["candidate_direct_host_staging"][
            "host_staging_pool"
        ]
        bounded_pool.update({
            "max_cached_bytes": 0,
            "storage_bytes": 0,
            "allocation_count": 0,
            "reuse_count": 0,
            "expected_reuse_count": 0,
            "transient_allocation_count": 12,
        })
        write(
            tmp_path / f"pd_export_bounded/tp4/{profile_name}.json",
            bounded_export,
        )

    bind_manifest_artifacts(tmp_path, run_id)
    report = MODULE.summarize(tmp_path, run_id)

    assert report["valid"], {
        key: value for key, value in report["evidence"].items() if not value
    }
    assert report["manifest_source_commit"] == SOURCE_COMMIT
    assert report["evidence"]["manifest_artifact_integrity"]
    assert report["evidence"]["artifacts_match_manifest_commit"]
    assert report["evidence"]["official_checkpoint_headers_valid"]
    assert report["evidence"]["local_checkpoint_matches_official"]
    assert (
        report["local_checkpoint_manifest"]["index_sha256"]
        == MODULE.OFFICIAL_INDEX_SHA256
    )
    assert report["performance"]["best_throughput"]["label"] == "batched"
    assert report["performance"]["lowest_peak_memory"]["label"] == "sorted"
    assert report["performance"]["host_input_preparation"] == {
        "available": False,
        "valid": False,
        "runs": [],
    }
    assert report["evidence"]["host_input_preparation_valid"]
    state_access = report["performance"]["recurrent_state_access"]
    assert state_access["all_configurations_valid"]
    assert len(state_access["by_configuration"]["tp4"]) == 6
    assert report["decode_convolution"]["promote_to_default"]
    assert report["evidence"]["decode_conv_runtime_evidence"]
    assert report["evidence"]["decode_conv_cudagraph_parity"]
    assert report["evidence"]["decode_conv_quality"]
    assert report["graph_safe_moe"]["all_tp_promoted"]
    assert report["hybrid_cudagraph"]["all_tp_passed"]
    assert report["hybrid_cudagraph"]["by_tp"]["tp4"]["short"][
        "scratch_isolation_observed"
    ]
    assert report["evidence"]["long_prefill_kernel_evidence"]
    assert report["evidence"]["mixed_workload_evidence"]
    assert report["evidence"]["mixed_moe_dispatch_evidence"]
    assert report["evidence"]["moe_route_input_broadcast_evidence"]
    assert report["evidence"]["normalization_workspace_evidence"]
    assert report["evidence"]["buffer_reuse_evidence"]
    assert report["evidence"]["rotary_storage_matches_preflight"]
    assert report["evidence"]["recurrent_storage_matches_preflight"]
    assert report["evidence"]["kv_storage_matches_preflight"]
    assert report["evidence"]["pd_transfer_baseline_valid"]
    assert report["evidence"]["pd_export_memory_evidence"]
    assert report["evidence"]["pd_export_retention_bound_evidence"]
    assert report["pd_export"]["by_tp"]["tp4"]["int8-model"]["valid"]
    assert report["pd_export"]["by_tp"]["tp4"]["int8-model"][
        "candidate_host_layout"
    ]["storage_count"] == 1
    assert report["pd_export"]["by_tp"]["tp4"]["int8-model"][
        "host_staging_pool"
    ]["valid"]
    assert report["pd_export"]["bounded_by_tp"]["tp4"]["int8-model"][
        "host_staging_pool"
    ]["valid"]
    assert report["pd_transfer"]["by_tp"]["tp4"]["int8-model"]["valid"]
    assert report["pd_transfer"]["by_tp"]["tp4"]["int8-model"][
        "receiver_host_staging_pool"
    ]["valid"]
    install = report["pd_transfer"]["by_tp"]["tp4"]["int8-model"][
        "cuda_install"
    ]
    assert install["valid"]
    assert install["peak_device_bytes_reduction"] == 5_000
    assert install["latency_ratio_vs_reference"] == 1.1
    write(
        tmp_path / "manifest.json",
        {"run_id": run_id, "source_commit": "b" * 40},
    )
    wrong_commit_report = MODULE.summarize(tmp_path, run_id)
    assert wrong_commit_report["evidence"]["single_commit"]
    assert not wrong_commit_report["evidence"][
        "artifacts_match_manifest_commit"
    ]
    assert not wrong_commit_report["valid"]
    write(
        tmp_path / "manifest.json",
        {"run_id": run_id, "source_commit": SOURCE_COMMIT},
    )
    transfer_path = tmp_path / "pd_transfer/tp4/int8-model.json"
    transfer_result = json.loads(transfer_path.read_text())
    transfer_result["cuda_install"]["latency_ratio_vs_reference"] = 1.3
    write(transfer_path, transfer_result)
    slow_install_report = MODULE.summarize(tmp_path, run_id)
    assert not slow_install_report["evidence"]["pd_transfer_baseline_valid"]
    assert not slow_install_report["valid"]
    transfer_result["cuda_install"]["latency_ratio_vs_reference"] = 1.1
    write(transfer_path, transfer_result)
    transfer_result["workload"]["components_bytes"]["kv"] += 1
    write(transfer_path, transfer_result)
    invalid_transfer_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_transfer_report["evidence"]["pd_transfer_baseline_valid"]
    assert not invalid_transfer_report["valid"]
    transfer_result["workload"]["components_bytes"]["kv"] -= 1
    write(transfer_path, transfer_result)
    transfer_result["results"]["receiver_host_staging_pool"][
        "reuse_count"
    ] = 10
    write(transfer_path, transfer_result)
    invalid_receive_reuse_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_receive_reuse_report["evidence"][
        "pd_transfer_baseline_valid"
    ]
    assert not invalid_receive_reuse_report["valid"]
    transfer_result["results"]["receiver_host_staging_pool"][
        "reuse_count"
    ] = 11
    write(transfer_path, transfer_result)
    export_path = tmp_path / "pd_export/tp4/int8-model.json"
    export_result = json.loads(export_path.read_text())
    export_result["candidate_direct_host_staging"][
        "peak_extra_device_bytes_samples"
    ] = [components["total"]] * 10
    export_result["candidate_direct_host_staging"][
        "peak_extra_device_bytes_max"
    ] = components["total"]
    write(export_path, export_result)
    invalid_export_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_export_report["evidence"]["pd_export_memory_evidence"]
    assert not invalid_export_report["valid"]
    export_result["candidate_direct_host_staging"][
        "peak_extra_device_bytes_samples"
    ] = [1024] * 10
    export_result["candidate_direct_host_staging"][
        "peak_extra_device_bytes_max"
    ] = 1024
    export_result["candidate_direct_host_staging"]["host_staging_pool"][
        "reuse_count"
    ] = 10
    write(export_path, export_result)
    invalid_reuse_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_reuse_report["evidence"]["pd_export_memory_evidence"]
    assert not invalid_reuse_report["pd_export"]["by_tp"]["tp4"][
        "int8-model"
    ]["host_staging_pool"]["valid"]
    assert not invalid_reuse_report["valid"]
    assert report["long_prefill"]["by_tp"]["tp4"]["valid"]
    chunk_sweep = report["long_prefill"]["by_tp"]["tp4"]["chunk_sweep"]
    assert chunk_sweep["fastest_chunk_size"] == 64
    assert chunk_sweep["lowest_memory_chunk_size"] == 32
    mixed = report["mixed_workload"]["by_tp"]["tp4"]
    assert mixed["repeat_count"] == 3
    assert mixed["median_mixed_steps"] == 3
    assert mixed["initial_p95_decode_gap_cv"] < 0.1
    assert report["mixed_workload"]["cross_tp_output_parity"]
    fairness = report["scheduler_fairness"]
    assert report["evidence"]["scheduler_fairness_evidence"]
    assert fairness["by_tp"]["tp4"]["disabled"]["repeat_count"] == 3
    assert fairness["comparisons"]["tp4"]["starvation_improved"]
    assert fairness["comparisons"]["tp4"]["injected_ttft_improved"]
    assert fairness["comparisons"]["tp4"]["throughput_ratio"] == 0.98
    scheduler_traces = report["scheduler_traces"]
    assert report["evidence"]["scheduler_trace_evidence"]
    assert scheduler_traces["by_tp"]["tp4"]["baseline"]["repeat_count"] == 3
    assert scheduler_traces["comparisons"]["tp4"]["output_parity"]
    assert scheduler_traces["comparisons"]["tp4"]["throughput_ratio"] == 1.25
    assert report["normalization"]["by_tp"]["tp4"]["rmsnorm"][
        "peak_extra_mib_delta"
    ] == -4.0
    assert report["normalization"]["by_tp"]["tp4"]["gated_rmsnorm"][
        "workspace"
    ]["reused_gate_fp32_workspace_mib"] == 4.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["torch_kv_dequant"][
        "workspace"
    ]["avoided_output_workspace_mib"] == 16.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["sampling_compact_top_k"][
        "workspace"
    ]["avoided_fp32_logits_mib"] == 31.0
    sampling_filter_output = report["buffer_reuse"]["by_tp"]["tp4"][
        "sampling_filter_output"
    ]
    assert sampling_filter_output["workspace"]["avoided_fp32_logits_mib"] == 64.0
    assert sampling_filter_output["metadata"][
        "candidate_reuses_temperature_and_filter_storage"
    ]
    assert report["buffer_reuse"]["by_tp"]["tp4"][
        "gated_delta_packed_projection"
    ]["workspace"]["avoided_gemm_launches"] == 2
    assert report["buffer_reuse"]["by_tp"]["tp4"]["attention_packed_qkv"][
        "workspace"
    ]["avoided_gemm_launches"] == 2
    assert report["buffer_reuse"]["by_tp"]["tp4"]["contiguous_decode_state"][
        "workspace"
    ]["avoided_state_gather_mib"] == 15.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["attention_norm_output"][
        "workspace"
    ]["reused_projection_output_mib"] == 4.0
    assert report["buffer_reuse"]["by_tp"]["tp4"][
        "delta_l2_normalization"
    ]["workspace"]["reused_query_key_fp32_mib"] == 4.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["rotary_output"][
        "workspace"
    ]["reused_query_key_output_mib"] == 4.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["vocab_gather"][
        "workspace"
    ]["avoided_full_vocab_copy_mib"] == 8.0
    assert report["buffer_reuse"]["by_tp"]["tp4"]["recurrent_decode"][
        "metadata"
    ]["avoided_full_state_intermediates"] == 2
    assert report["buffer_reuse"]["by_tp"]["tp4"]["recurrent_decode"][
        "workspace"
    ]["reused_prediction_workspace_mib"] == 0.25
    assert report["buffer_reuse"]["by_tp"]["tp4"]["recurrent_decode"][
        "workspace"
    ]["reused_decay_exp_mib"] == 0.01
    contraction = report["buffer_reuse"]["by_tp"]["tp4"][
        "recurrent_state_contraction"
    ]
    assert contraction["workspace"][
        "avoided_state_product_mib_per_contraction"
    ] == 32.0
    assert contraction["metadata"]["state_contractions_per_decode"] == 2
    assert report["buffer_reuse"]["by_tp"]["tp4"]["recurrent_prefill"][
        "workspace"
    ]["reused_recurrent_state_mib"] == 8.0
    assert report["buffer_reuse"]["by_tp"]["tp4"][
        "sorted_route_weighting"
    ]["workspace"]["avoided_weighted_expert_output_mib"] == 4.0
    sampling_top_p = report["buffer_reuse"]["by_tp"]["tp4"][
        "sampling_top_p"
    ]
    assert sampling_top_p["workspace"]["avoided_top_k_mask_workspace_mib"] == 64.0
    assert sampling_top_p["workspace"]["avoided_top_p_shift_clone_mib"] == 32.0
    assert sampling_top_p["metadata"][
        "eliminated_top_p_mask_clones_per_step"
    ] == 1
    sampling_top_k_top_p = report["buffer_reuse"]["by_tp"]["tp4"][
        "sampling_top_k_top_p"
    ]
    assert sampling_top_k_top_p["workspace"][
        "avoided_top_p_shift_clone_mib"
    ] == 1.0
    sampling_inputs = report["buffer_reuse"]["by_tp"]["tp4"][
        "sampling_inputs"
    ]
    assert sampling_inputs["workspace"][
        "eliminated_tensor_allocations_per_step"
    ] == 6
    assert sampling_inputs["metadata"][
        "candidate_reuses_host_device_storage"
    ]
    sampling_noise = report["buffer_reuse"]["by_tp"]["tp4"][
        "sampling_noise"
    ]
    assert sampling_noise["workspace"]["reused_filtered_logits_mib"] == 0.01
    assert sampling_noise["metadata"]["persistent_sampling_noise_mib"] == 0
    assert sampling_noise["metadata"][
        "candidate_reuses_filtered_logits_storage"
    ]
    assert report["buffer_reuse"]["by_tp"]["tp4"][
        "batched_route_sum_output"
    ]["workspace"]["avoided_route_sum_output_mib"] == 4.0
    delta_mask = report["buffer_reuse"]["by_tp"]["tp4"][
        "delta_causal_mask"
    ]
    assert delta_mask["available"]
    assert delta_mask["measured_on_cuda"]
    assert delta_mask["valid"]
    assert delta_mask["all_cuda_beneficial"]
    assert set(delta_mask["by_chunk_size"]) == {"32", "64", "128"}
    assert report["graph_safe_moe"]["by_tp"]["tp4"]["promotion"][
        "selected_decode_batches"
    ] == [1, 64]
    mixed_dispatch = report["graph_safe_moe"]["mixed_dispatch_by_tp"]["tp4"]
    assert mixed_dispatch["valid"]
    assert mixed_dispatch["minimum_speedup_vs_grouped"] == 1.1
    assert mixed_dispatch["maximum_peak_extra_mib_delta"] == 4.0
    assert mixed_dispatch["case_count"] == 1
    weight_reuse = report["graph_safe_moe"]["weight_buffer_reuse_by_tp"][
        "tp4"
    ]
    assert report["evidence"]["moe_weight_buffer_reuse_evidence"]
    assert weight_reuse["valid"]
    assert weight_reuse["peak_extra_mib_delta"] == -4.0
    route_input = report["graph_safe_moe"][
        "route_input_broadcast_by_tp"
    ]["tp4"]["1"]
    assert route_input["valid"]
    assert route_input["speedup_vs_repeated_input"] == 1.05
    device_scalar = report["graph_safe_moe"]["device_scalar_by_tp"]["tp4"]
    assert report["graph_safe_moe"]["device_scalar_all_tp_promoted"]
    assert device_scalar["available"]
    assert device_scalar["avoids_host_route_sync"]
    assert device_scalar["speedup_vs_current"] == 1.1
    assert device_scalar["estimated_selected_weight_mib"] == 2.0
    runtime = report["graph_safe_moe"]["runtime_by_tp"]["tp4"]["auto"]
    assert runtime["output_digest_matches"]
    assert runtime["throughput_speedup"] == 2.0
    assert runtime["tpot_speedup"] == 2.0
    assert runtime["peak_memory_delta_mib"] == 4
    assert runtime["decode_host_sync_eliminated"]
    assert runtime["baseline_decode_host_syncs_by_rank"] == {
        rank: 40 for rank in range(4)
    }
    assert runtime["candidate_decode_host_syncs_by_rank"] == {
        rank: 0 for rank in range(4)
    }
    assert runtime["promotion"]["promote_to_default"]
    assert report["graph_safe_moe"]["runtime_by_tp"]["tp4"]["int8"][
        "promotion"
    ]["promote_to_default"]
    attention = report["int8_attention"]["by_tp"]["tp4"]
    assert attention["short"]["best_fused"]["speedup_vs_flash_reference"] == 2.0
    assert attention["long"]["best_partitioned"]["backend"] == (
        "int8_partitioned_ps256"
    )
    assert attention["long"]["production_partitioned"]["backend"] == (
        "int8_partitioned_ps512"
    )
    assert attention["long"]["partitioned_workspace_valid"]
    assert attention["long"]["production_workspace_reuse"][
        "promote_to_runtime"
    ]
    pressure = report["kv_pressure"]["by_tp"]["tp4"]
    assert pressure["fcfs"]["valid"]
    assert pressure["min_recompute"]["valid"]
    assert pressure["min_recompute_reserved"]["valid"]
    assert pressure["min_recompute_reserved"][
        "decode_kv_reservation_observed"
    ]
    assert pressure["fcfs"]["preemption_count"] == 2
    assert pressure["min_recompute"]["waiting_prefill_preemptions"] == 1
    assert pressure["min_recompute"]["preempted_token_progress"] == 256
    comparison = report["kv_pressure"]["comparisons"]["tp4"]
    assert comparison["valid"]
    assert comparison["recomputed_token_reduction"] == 1280
    assert comparison["elapsed_speedup"] == 1.25
    assert comparison["tail_latency_non_regressing"]
    reservation = comparison["decode_kv_reservation"]
    assert reservation["valid"]
    assert reservation["output_parity"]
    assert reservation["implementation_parity"]
    assert reservation["checkpoint_parity"]
    assert reservation["recomputed_token_reduction"] == 256
    assert reservation["reservation_stop_count"] == 2
    assert comparison["candidate_latency_vs_fcfs"]["p95_ttft_s"] == (
        1.5 / 1.8
    )
    candidate_pressure_path = tmp_path / "pressure/tp4/min_recompute/r1.json"
    candidate_pressure = json.loads(candidate_pressure_path.read_text())
    candidate_pressure["generated_token_ids"]["digest"] = "different"
    write(candidate_pressure_path, candidate_pressure)
    parity_failure = MODULE.summarize(tmp_path, run_id)
    assert not parity_failure["evidence"]["kv_pressure_evidence"]
    assert not parity_failure["kv_pressure"]["comparisons"]["tp4"][
        "output_parity"
    ]
    candidate_pressure["generated_token_ids"]["digest"] = "pressure-tokens"
    write(candidate_pressure_path, candidate_pressure)
    reserved_pressure_path = (
        tmp_path / "pressure/tp4/min_recompute_reserved/r1.json"
    )
    reserved_pressure = json.loads(reserved_pressure_path.read_text())
    reserved_pressure["metrics"][
        "prefill_stopped_by_decode_kv_reservation"
    ] = 0
    write(reserved_pressure_path, reserved_pressure)
    reservation_failure = MODULE.summarize(tmp_path, run_id)
    assert not reservation_failure["evidence"]["kv_pressure_evidence"]
    assert not reservation_failure["kv_pressure"]["comparisons"]["tp4"][
        "decode_kv_reservation"
    ]["valid"]
    reserved_pressure["metrics"][
        "prefill_stopped_by_decode_kv_reservation"
    ] = 2
    write(reserved_pressure_path, reserved_pressure)
    candidate_pressure = json.loads(candidate_pressure_path.read_text())
    waiting_prefill_preemptions = candidate_pressure["metrics"].pop(
        "waiting_prefill_preemptions"
    )
    write(candidate_pressure_path, candidate_pressure)
    missing_waiting_prefill_metric = MODULE.summarize(tmp_path, run_id)
    assert not missing_waiting_prefill_metric["evidence"][
        "kv_pressure_evidence"
    ]
    assert not missing_waiting_prefill_metric["kv_pressure"]["by_tp"]["tp4"][
        "min_recompute"
    ]["waiting_prefill_metric_valid"]
    candidate_pressure["metrics"]["waiting_prefill_preemptions"] = (
        waiting_prefill_preemptions
    )
    write(candidate_pressure_path, candidate_pressure)
    memory = report["memory"]["by_tp"]["tp4"]
    assert memory["int8_kv_reduction_ratio"] == 0.49609375
    assert memory["minimum_budget_margin_bytes"] == 5_000
    assert memory["kv_capacity_by_dtype"]["int8"][
        "memory_limited_context_tokens_per_sequence"
    ] == 3_072

    performance_path = tmp_path / f"performance/{run_id}_matrix_summary.json"
    performance_result = json.loads(performance_path.read_text())
    performance_result["runs"][0]["execution_paths"][
        "observed_in_all_repeats"
    ] = ["prefill_contiguous_view", "decode_indexed_copy"]
    write(performance_path, performance_result)
    invalid_state_path_report = MODULE.summarize(tmp_path, run_id)
    state_access = invalid_state_path_report["performance"][
        "recurrent_state_access"
    ]
    assert not state_access["all_configurations_valid"]
    assert not invalid_state_path_report["evidence"]["performance_paths_valid"]
    assert not invalid_state_path_report["valid"]
    performance_result["runs"][0]["execution_paths"][
        "observed_in_all_repeats"
    ] = ["prefill_contiguous_view", "decode_contiguous_view"]
    performance_result["runs"][0]["storage"]["recurrent_state_storage_by_rank"][1][
        "rotary_cache_bytes_local_rank"
    ] = 8192
    write(performance_path, performance_result)
    mismatched_storage_report = MODULE.summarize(tmp_path, run_id)
    assert not mismatched_storage_report["evidence"][
        "rotary_storage_matches_preflight"
    ]
    assert not mismatched_storage_report["valid"]
    performance_result["runs"][0]["storage"]["recurrent_state_storage_by_rank"][1][
        "rotary_cache_bytes_local_rank"
    ] = 4096
    performance_result["runs"][0]["storage"]["recurrent_state_storage_by_rank"][2][
        "total_bytes_local_rank"
    ] = 501
    write(performance_path, performance_result)
    mismatched_state_report = MODULE.summarize(tmp_path, run_id)
    assert not mismatched_state_report["evidence"][
        "recurrent_storage_matches_preflight"
    ]
    assert not mismatched_state_report["valid"]
    performance_result["runs"][0]["storage"]["recurrent_state_storage_by_rank"][2][
        "total_bytes_local_rank"
    ] = 500
    performance_result["runs"][0]["storage"]["kv_cache_storage_by_rank"][3][
        "total_bytes"
    ] += 1
    write(performance_path, performance_result)
    mismatched_kv_report = MODULE.summarize(tmp_path, run_id)
    assert not mismatched_kv_report["evidence"]["kv_storage_matches_preflight"]
    assert not mismatched_kv_report["valid"]
    performance_result["runs"][0]["storage"]["kv_cache_storage_by_rank"][3][
        "total_bytes"
    ] -= 1
    performance_result["runs"][0]["storage"]["kv_cache_storage_by_rank"][3][
        "unused_capacity_bytes_after_sync"
    ] += 1
    write(performance_path, performance_result)
    mismatched_capacity_report = MODULE.summarize(tmp_path, run_id)
    assert not mismatched_capacity_report["evidence"][
        "kv_capacity_accounting_valid"
    ]
    assert not mismatched_capacity_report["valid"]
    performance_result["runs"][0]["storage"]["kv_cache_storage_by_rank"][3][
        "unused_capacity_bytes_after_sync"
    ] -= 1
    write(performance_path, performance_result)

    local_audit_path = tmp_path / "preflight/checkpoint_mapping_audit.json"
    local_audit = json.loads(local_audit_path.read_text())
    local_audit["checkpoint_manifest"]["index_sha256"] = "different"
    write(local_audit_path, local_audit)
    mismatched_checkpoint_report = MODULE.summarize(tmp_path, run_id)
    assert not mismatched_checkpoint_report["evidence"][
        "local_checkpoint_matches_official"
    ]
    assert not mismatched_checkpoint_report["valid"]
    local_audit["checkpoint_manifest"][
        "index_sha256"
    ] = MODULE.OFFICIAL_INDEX_SHA256
    write(local_audit_path, local_audit)

    local_audit["results"]["tp4"]["skipped_tensor_groups"]["other"] = 1
    write(local_audit_path, local_audit)
    unexpected_skip_report = MODULE.summarize(tmp_path, run_id)
    assert not unexpected_skip_report["evidence"]["checkpoint_mapping_valid"]
    assert not unexpected_skip_report["valid"]
    del local_audit["results"]["tp4"]["skipped_tensor_groups"]["other"]
    write(local_audit_path, local_audit)

    mixed_path = tmp_path / "mixed/tp4/r1.json"
    mixed_result = json.loads(mixed_path.read_text())
    mixed_result["initial_p95_decode_gap_s"] = 0.2
    write(mixed_path, mixed_result)
    unstable_report = MODULE.summarize(tmp_path, run_id)
    unstable_mixed = unstable_report["mixed_workload"]["by_tp"]["tp4"]
    assert unstable_mixed["initial_p95_decode_gap_cv"] > 0.1
    assert not unstable_mixed["valid"]
    assert not unstable_report["evidence"]["mixed_workload_evidence"]

    mixed_result["initial_p95_decode_gap_s"] = 0.019
    mixed_result.pop("generated_token_ids")
    write(mixed_path, mixed_result)
    invalid_report = MODULE.summarize(tmp_path, run_id)
    assert not invalid_report["evidence"]["mixed_workload_evidence"]
    assert not invalid_report["valid"]


def test_summary_rejects_inaccurate_attention_kernel(tmp_path):
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 0.5,
                "max_abs_diff_vs_flash_reference": 0.051,
                "peak_extra_mib": 1.0,
            },
        },
        "context_len": 4096,
        "batch_size": 4,
    }

    summary = MODULE.summarize_attention_case(result, partitioned=False)

    assert not summary["fused_correctness_valid"]
    assert summary["best_fused"] is None


def test_partitioned_attention_requires_shared_workspace_evidence():
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 1.0,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 2.0,
            },
            "int8_partitioned_ps512": {
                "status": "ok",
                "median_ms": 0.9,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 3.0,
            },
        },
        "context_len": MODULE.ATTENTION_LONG_CONTEXT,
        "batch_size": 4,
    }

    summary = MODULE.summarize_attention_case(result, partitioned=True)

    assert summary["partitioned_correctness_valid"]
    assert not summary["partitioned_workspace_valid"]
    assert summary["max_allowed_abs_error"] == 0.05


def test_partitioned_attention_reports_workspace_reuse_regression():
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 1.0,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 2.0,
            },
            "int8_partitioned_ps512": {
                "status": "ok",
                "median_ms": 0.9,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 3.0,
            },
            "int8_partitioned_ps512_workspace_reuse": {
                "status": "ok",
                "median_ms": 1.0,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 0.0,
                "speedup_vs_allocating": 0.9,
            },
        },
        "context_len": MODULE.ATTENTION_LONG_CONTEXT,
        "batch_size": 4,
        "shape_manifest": {
            "workspace": {
                "partitioned": {
                    "512": {"allocation_count": 1, "shared_storage": True}
                }
            }
        },
    }

    summary = MODULE.summarize_attention_case(result, partitioned=True)

    assert summary["workspace_reuse_measurement_valid"]
    assert not summary["production_workspace_reuse"]["promote_to_runtime"]


def test_long_context_requires_production_partition_size():
    result = {
        "results": {
            "flash_reference": {"status": "ok", "median_ms": 2.0},
            "int8_v3_bt256_w8_s2": {
                "status": "ok",
                "median_ms": 1.0,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 1.0,
            },
            "int8_partitioned_ps256": {
                "status": "ok",
                "median_ms": 0.8,
                "max_abs_diff_vs_flash_reference": 0.01,
                "peak_extra_mib": 2.0,
            },
        },
        "context_len": 16_384,
        "batch_size": 4,
    }

    summary = MODULE.summarize_attention_case(result, partitioned=True)

    assert not summary["partitioned_correctness_valid"]
    assert summary["production_partitioned"] is None


def test_runtime_promotion_rejects_unstable_or_regressing_candidate():
    result = MODULE.evaluate_moe_runtime_candidate(
        output_digest_matches=True,
        throughput_speedup=0.98,
        tpot_speedup=1.10,
        peak_memory_delta_mib=4.0,
        max_coefficient_of_variation=0.06,
        baseline_decode_host_sync_observed=True,
        candidate_decode_host_sync_eliminated=True,
        candidate_batched_dispatch_observed=True,
    )

    assert not result["promote_to_default"]
    assert not result["checks"]["throughput_non_regression"]
    assert not result["checks"]["stable_repeats"]


def test_runtime_promotion_requires_measured_decode_dispatch_evidence():
    result = MODULE.evaluate_moe_runtime_candidate(
        output_digest_matches=True,
        throughput_speedup=1.1,
        tpot_speedup=1.1,
        peak_memory_delta_mib=0.0,
        max_coefficient_of_variation=0.01,
        baseline_decode_host_sync_observed=True,
        candidate_decode_host_sync_eliminated=False,
        candidate_batched_dispatch_observed=True,
    )

    assert not result["promote_to_default"]
    assert not result["checks"]["candidate_decode_host_sync_eliminated"]


def test_long_prefill_rejects_inaccurate_kernel():
    result = {
        "configuration": {
            "prefill_only": True,
            "prefill_batch": 1,
            "prefill_tokens": 8192,
            "tp_size": 4,
            "resolved_device": "cuda:0",
            "delta_prefill_chunk_sizes": [32, 64],
        },
        "results": {
            name: {
                "candidate": {"median_ms": 1.0, "peak_extra_mib": 2.0},
                "errors": [{"max_abs_error": error}],
            }
            for name, error in (
                ("vectorized_prefill_convolution", 0.01),
                ("grouped_delta_prefill", 0.051),
            )
        } | {
            "grouped_delta_prefill_chunk_sweep": {
                "baseline_chunk_size": 64,
                "candidates": {
                    str(chunk_size): {
                        "chunk_size": chunk_size,
                        "candidate": {
                            "median_ms": 1.0,
                            "peak_extra_mib": 2.0,
                        },
                        "errors_vs_chunk64": [{"max_abs_error": 0.01}],
                    }
                    for chunk_size in (32, 64)
                }
            }
        },
    }

    summary = MODULE.summarize_long_prefill(result, expected_tp_size=4)

    assert not summary["valid"]
    assert not summary["cases"]["grouped_delta_prefill"]["valid"]


def test_memory_summary_rejects_non_saving_int8_cache():
    report = {
        "results": {
            "tp4": {
                "kv_bytes_per_token_by_dtype": {
                    "auto": 100,
                    "int8": 100,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="does not reduce memory"):
        MODULE.summarize_memory_preflight(report)


def test_buffer_reuse_summary_rejects_missing_workspace_metric():
    result = {
        "reference": {"peak_extra_mib": 2.0},
        "candidate": {"peak_extra_mib": 1.0},
        "speedup": 1.1,
        "errors": [{"max_abs_error": 0.0}],
    }

    with pytest.raises(ValueError, match="missing workspace metrics"):
        MODULE.summarize_buffer_reuse_candidate(
            result,
            ("reused_workspace_mib",),
        )


def test_buffer_reuse_summary_preserves_failed_accuracy_evidence():
    result = {
        "reference": {"peak_extra_mib": 2.0},
        "candidate": {"peak_extra_mib": 1.0},
        "speedup": 1.1,
        "errors": [{"max_abs_error": 0.051}],
        "reused_workspace_mib": 1.0,
        "expected_count": 2,
    }

    summary = MODULE.summarize_buffer_reuse_candidate(
        result,
        ("reused_workspace_mib",),
        {"expected_count": 2},
    )

    assert not summary["valid"]
    assert summary["max_abs_error"] == 0.051
    assert summary["peak_extra_mib_delta"] == -1.0


def test_host_input_preparation_summary_preserves_all_configurations():
    runs = []
    for label, tp_size, average_s in (
        ("tp4", 4, 0.002),
        ("tp8", 8, 0.001),
    ):
        runs.append(
            {
                "label": label,
                "tensor_parallel_size": tp_size,
                "weight_quant_backend": "auto",
                "kv_cache_dtype": "auto",
                "median": {
                    "host_decode_preparation_call_count": 32,
                    "host_decode_preparation_total_time_s": average_s * 32,
                    "host_decode_preparation_max_time_s": average_s * 2,
                    "host_decode_preparation_average_time_s": average_s,
                    "host_decode_preparation_rank_skew": 1.25,
                },
                "coefficient_of_variation": {
                    "host_decode_preparation_average_time_s": 0.05,
                    "host_decode_preparation_rank_skew": 0.02,
                },
            }
        )

    summary = MODULE.summarize_host_input_preparation(runs)

    assert summary["available"]
    assert summary["valid"]
    assert [row["label"] for row in summary["runs"]] == ["tp4", "tp8"]
    assert summary["runs"][1]["steps"]["decode"] == {
        "call_count": 32,
        "total_time_s": 0.032,
        "max_time_s": 0.002,
        "average_time_s": 0.001,
        "average_time_cv": 0.05,
        "rank_skew": 1.25,
        "rank_skew_cv": 0.02,
        "valid": True,
    }


def test_host_input_preparation_summary_is_optional_but_rejects_partial_data():
    assert MODULE.summarize_host_input_preparation([{"median": {}}]) == {
        "available": False,
        "valid": False,
        "runs": [],
    }

    summary = MODULE.summarize_host_input_preparation(
        [
            {
                "label": "broken",
                "median": {
                    "host_prefill_preparation_call_count": 1,
                    "host_prefill_preparation_total_time_s": 0.1,
                },
            }
        ]
    )

    assert summary["available"]
    assert not summary["valid"]
    assert not summary["runs"][0]["steps"]["prefill"]["valid"]


def test_buffer_reuse_summary_rejects_peak_memory_regression():
    result = {
        "reference": {"peak_extra_mib": 1.0},
        "candidate": {"peak_extra_mib": 1.5},
        "speedup": 1.2,
        "errors": [{"max_abs_error": 0.0}],
        "reused_workspace_mib": 1.0,
    }

    summary = MODULE.summarize_buffer_reuse_candidate(
        result,
        ("reused_workspace_mib",),
    )

    assert summary["measurement_valid"]
    assert not summary["memory_non_regression"]
    assert not summary["valid"]


def test_delta_causal_mask_summary_requires_cuda_for_benefit_claim():
    result = {
        "cache_max_entries": 32,
        "maximum_cached_chunk_size": 1024,
        "candidates": {
            "64": {
                "reference": {"peak_extra_mib": 1.0},
                "candidate": {"peak_extra_mib": 0.0},
                "speedup": 1.5,
                "errors": [{"max_abs_error": 0.0}],
                "persistent_mask_mib": 0.01,
                "cache_reuses_storage": True,
                "eliminated_allocations_per_additional_layer": 1,
            }
        },
    }

    summary = MODULE.summarize_delta_causal_mask_cache(
        result,
        measured_on_cuda=False,
    )

    assert summary["valid"]
    assert not summary["all_cuda_beneficial"]


def test_mixed_moe_summary_requires_cuda_and_non_regressing_speed():
    result = {
        "reference": {"peak_extra_mib": 8.0},
        "candidate": {"peak_extra_mib": 9.0},
        "errors": [{"max_abs_error": 0.01}],
        "decode_tokens": 32,
        "prefill_tokens": 512,
        "avoided_route_hidden_allocation_mib_per_step": 1.0,
        "speedup_vs_grouped": 0.99,
        "measured_on_cuda": False,
    }

    summary = MODULE.summarize_mixed_moe_dispatch(result)

    assert not summary["valid"]
    assert not summary["checks"]["cuda_measurement"]
    assert not summary["checks"]["speed"]
    assert summary["peak_extra_mib_delta"] == 1.0


def test_mixed_moe_sweep_rejects_one_regressing_shape():
    valid = {
        "reference": {"peak_extra_mib": 8.0},
        "candidate": {"peak_extra_mib": 9.0},
        "errors": [{"max_abs_error": 0.01}],
        "decode_tokens": 8,
        "prefill_tokens": 128,
        "avoided_route_hidden_allocation_mib_per_step": 0.25,
        "speedup_vs_grouped": 1.1,
        "measured_on_cuda": True,
    }
    regressing = deepcopy(valid)
    regressing.update(
        decode_tokens=64,
        prefill_tokens=2048,
        speedup_vs_grouped=0.98,
    )

    summary = MODULE.summarize_mixed_moe_dispatch_sweep(
        {"latency": valid, "throughput": regressing}
    )

    assert not summary["valid"]
    assert summary["case_count"] == 2
    assert summary["minimum_speedup_vs_grouped"] == 0.98
    assert not summary["cases"]["throughput"]["checks"]["speed"]


def test_normalization_summary_rejects_invalid_measurements():
    result = {
        "reference": {"peak_extra_mib": 1.0},
        "candidate": {"peak_extra_mib": float("nan")},
        "speedup": float("nan"),
        "errors": [{"max_abs_error": 0.0}],
        "reuses_workspace": True,
    }

    summary = MODULE.summarize_normalization_candidate(
        result,
        "reuses_workspace",
    )

    assert not summary["measurement_valid"]
    assert not summary["valid"]


def load_pressure_repeats(root, policy):
    return [
        json.loads(path.read_text())
        for path in sorted((root / f"pressure/tp4/{policy}").glob("r*.json"))
    ]


def test_kv_pressure_repeats_use_median_and_require_stable_evidence(tmp_path):
    write_pressure_case(tmp_path)
    results = load_pressure_repeats(tmp_path, "fcfs")

    summary = MODULE.summarize_kv_pressure_repeats(
        results,
        expected_tp_size=4,
        expected_policy="fcfs",
    )

    assert summary["valid"]
    assert summary["repeat_count"] == 3
    assert summary["total_time_s"] == 10.0
    assert summary["peak_torch_allocated_mib"] == 12_000.0
    assert summary["scheduler_counters_stable"]
    assert summary["output_parity"]
    assert summary["checkpoint_stable"]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda result: result["generated_token_ids"].update(
                digest="different"
            ),
            "output_parity",
        ),
        (
            lambda result: result.update(step_count=999),
            "scheduler_counters_stable",
        ),
        (
            lambda result: result["checkpoint_manifest"].update(
                digest="different"
            ),
            "checkpoint_stable",
        ),
    ),
)
def test_kv_pressure_repeats_reject_drift(tmp_path, mutation, failed_check):
    write_pressure_case(tmp_path)
    results = load_pressure_repeats(tmp_path, "fcfs")
    mutation(results[-1])

    summary = MODULE.summarize_kv_pressure_repeats(
        results,
        expected_tp_size=4,
        expected_policy="fcfs",
    )

    assert not summary["valid"]
    assert not summary[failed_check]


def test_kv_pressure_repeats_reject_high_runtime_variance(tmp_path):
    write_pressure_case(tmp_path)
    results = load_pressure_repeats(tmp_path, "fcfs")
    results[-1]["total_time_s"] = 20.0

    summary = MODULE.summarize_kv_pressure_repeats(
        results,
        expected_tp_size=4,
        expected_policy="fcfs",
    )

    assert not summary["valid"]
    assert summary["total_time_cv"] > summary["max_allowed_cv"]
