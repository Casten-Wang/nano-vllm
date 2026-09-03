"""Summarize one complete Qwen3.6 rental validation run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


MOE_RUNTIME_MIN_THROUGHPUT_RATIO = 0.99
MOE_RUNTIME_MIN_TPOT_SPEEDUP = 1.02
MOE_RUNTIME_MAX_PEAK_EXTRA_MIB = 64.0
MOE_RUNTIME_MAX_CV = 0.05
QWEN35_TOTAL_QUERY_HEADS = 16
QWEN35_KV_HEADS_PER_RANK = 1
QWEN35_HEAD_DIM = 256
ATTENTION_SHORT_CONTEXT = 4096
ATTENTION_LONG_CONTEXT = 16385
ATTENTION_MAX_CONTEXT = 262143
ATTENTION_MAX_ABS_ERROR = 0.05
PRODUCTION_PARTITION_SIZE = 512
ATTENTION_WORKSPACE_REUSE_MIN_SPEEDUP = 1.0
PRESSURE_KV_BLOCKS = 5
PRESSURE_INITIAL_SEQUENCES = 2
PRESSURE_INJECTED_SEQUENCES = 2
PRESSURE_INITIAL_LENGTHS = [256, 1024]
PRESSURE_INJECTED_LENGTHS = [512, 512]
LONG_PREFILL_TOKENS = 8192
LONG_PREFILL_MAX_ABS_ERROR = 0.05
MIXED_MAX_COEFFICIENT_OF_VARIATION = 0.10
PRESSURE_MAX_COEFFICIENT_OF_VARIATION = 0.10
FAIRNESS_INITIAL_SEQUENCES = 32
FAIRNESS_INJECTED_SEQUENCES = 8
FAIRNESS_INITIAL_INPUT_LENGTH = 128
FAIRNESS_INJECTED_INPUT_LENGTH = 1024
FAIRNESS_OUTPUT_LENGTH = 64
FAIRNESS_INJECT_AFTER_DECODE_STEPS = 4
FAIRNESS_MAX_BATCHED_TOKENS = 32
FAIRNESS_THRESHOLD = 4
FAIRNESS_TOKEN_BUDGET = 256
NORMALIZATION_MAX_ABS_ERROR = 0.05
BUFFER_REUSE_MAX_ABS_ERROR = 0.05
MIXED_MOE_MIN_SPEEDUP = 1.0
MIXED_MOE_MAX_PEAK_EXTRA_MIB = 64.0
MIXED_MOE_MAX_ABS_ERROR = 0.05
PD_INSTALL_MAX_LATENCY_RATIO = 1.25
RESIDENT_FP8_MIN_THROUGHPUT_RATIO = 0.80
KVCACHE_BLOCK_SIZE = 256
OFFICIAL_CHECKPOINT_REPO = "Qwen/Qwen3.6-35B-A3B"
OFFICIAL_SKIPPED_WEIGHT_GROUPS = {"model.visual": 333, "mtp": 19}
OFFICIAL_SKIPPED_WEIGHT_PREFIXES = {"model.visual.": 333, "mtp.": 19}
OFFICIAL_CHECKPOINT_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
OFFICIAL_GPTQ_CHECKPOINT_REPO = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
OFFICIAL_GPTQ_CHECKPOINT_REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"
OFFICIAL_FP8_CHECKPOINT_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
OFFICIAL_FP8_CHECKPOINT_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
OFFICIAL_FP8_CONFIG_SHA256 = (
    "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
)
OFFICIAL_FP8_INDEX_SHA256 = (
    "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
)
OFFICIAL_FP8_HEADERS_SHA256 = (
    "eee3953b23b7dd1df38209d1fd39b1b9eaa875a28b724628d4183cef46f8e78f"
)
OFFICIAL_CONFIG_SHA256 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
OFFICIAL_INDEX_SHA256 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
OFFICIAL_HEADERS_SHA256 = (
    "49fdf93cdd3e482e01e4891a65a3d714b557b227643543dbbcc3ba96b2db27c1"
)
OFFICIAL_INDEX_CONTRACTS = {
    "bf16": (1045, 71_903_645_408),
    "gptq_int4": (124_611, 24_403_162_208),
    "fp8_block": (64_196, 37_454_789_472),
}


def expected_checkpoint_semantic_contract(quantization_format: str) -> dict:
    tensor_count, total_size = OFFICIAL_INDEX_CONTRACTS[quantization_format]
    quantization = {
        "format": quantization_format,
        "weight_bits": 16,
        "activation_scheme": None,
        "weight_block_size": None,
        "group_size": None,
        "symmetric": None,
        "desc_act": None,
        "ignored_module_count": 0,
        "ignored_patterns": [],
    }
    if quantization_format == "gptq_int4":
        quantization.update(
            weight_bits=4,
            group_size=128,
            symmetric=True,
            desc_act=False,
            ignored_patterns=[
                ".*attn.*",
                ".*shared_expert.*",
                ".*mtp.*",
                ".*visual.*",
            ],
        )
    elif quantization_format == "fp8_block":
        quantization.update(
            weight_bits=8,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            ignored_module_count=648,
        )
    return {
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "outer_model_type": "qwen3_5_moe",
        "text_model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "full_attention_layers": list(range(3, 40, 4)),
        "linear_attention_layers": [
            layer for layer in range(40) if layer % 4 != 3
        ],
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "eos_token_ids": [248046, 248044],
        "index_tensor_count": tensor_count,
        "index_total_size_bytes": total_size,
        "quantization": quantization,
    }


def checkpoint_semantic_contract_matches(
    audit: dict, quantization_format: str
) -> bool:
    return audit.get("semantic_contract") == expected_checkpoint_semantic_contract(
        quantization_format
    )


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required validation artifact is missing: {path}")
    return json.loads(path.read_text())


def checkpoint_manifest_matches_remote(local: dict, remote: dict) -> bool:
    remote_shards = {
        item.get("name"): item
        for item in remote.get("checkpoint_shards", ())
        if isinstance(item, dict)
    }
    local_shards = {
        item.get("name"): item
        for item in local.get("files", ())
        if isinstance(item, dict)
    }
    shards_match = (
        bool(remote_shards)
        and set(local_shards) == set(remote_shards)
        and all(
            local_shards[name].get("present") is True
            and local_shards[name].get("size_bytes")
            == remote_shards[name].get("size_bytes")
            and (
                local_shards[name].get("content_sha256")
                or local_shards[name].get("content_id")
            )
            == remote_shards[name].get("sha256")
            for name in remote_shards
        )
    )
    return (
        local.get("config_sha256") == remote.get("config_sha256")
        and local.get("index_sha256") == remote.get("index_sha256")
        and local.get("shard_count") == remote.get("shard_count")
        and local.get("present_shard_count") == remote.get("shard_count")
        and not local.get("missing_shards")
        and shards_match
    )


def summarize_gptq_workspace(
    performance_runs: list[dict],
    max_rows: int,
) -> dict:
    """Require the shared GPTQ decode workspace on every measured rank."""

    if (
        not isinstance(max_rows, int)
        or isinstance(max_rows, bool)
        or max_rows <= 0
    ):
        return {"valid": False, "by_tp": {}}
    by_tp = {}
    for row in performance_runs:
        tp_size = row.get("tensor_parallel_size")
        if not isinstance(tp_size, int) or tp_size <= 0 or 512 % tp_size:
            continue
        expected_bytes = max_rows * (2 * (512 // tp_size) + 2048) * 2
        ranks = row.get("storage", {}).get(
            "runtime_buffer_storage_by_rank", []
        )
        valid = (
            len(ranks) == tp_size
            and all(
                item.get("gptq_expert_workspace_pool_count") == 1
                and item.get("gptq_expert_workspace_bytes") == expected_bytes
                and item.get("gptq_expert_workspace_allocation_count") == 1
                and item.get("gptq_expert_workspace_reuse_count", 0) > 0
                for item in ranks
            )
        )
        entry = by_tp.setdefault(
            f"tp{tp_size}",
            {
                "valid": True,
                "expected_bytes_per_rank": expected_bytes,
                "run_count": 0,
                "bytes_by_rank_by_run": [],
                "allocation_count_by_rank_by_run": [],
                "reuse_count_by_rank_by_run": [],
            },
        )
        entry["valid"] = entry["valid"] and valid
        entry["run_count"] += 1
        entry["bytes_by_rank_by_run"].append(
            [item.get("gptq_expert_workspace_bytes") for item in ranks]
        )
        entry["allocation_count_by_rank_by_run"].append(
            [
                item.get("gptq_expert_workspace_allocation_count")
                for item in ranks
            ]
        )
        entry["reuse_count_by_rank_by_run"].append(
            [item.get("gptq_expert_workspace_reuse_count") for item in ranks]
        )
    return {
        "valid": bool(by_tp) and all(item["valid"] for item in by_tp.values()),
        "by_tp": by_tp,
    }


def summarize_optional_gptq(run_dir: Path, run_id: str) -> dict:
    """Validate and summarize the optional GPTQ rental stages."""

    root = run_dir / "gptq"
    if not root.exists():
        return {"enabled": False, "valid": True}
    gptq_run_id = f"{run_id}-gptq"
    audit = load_json(root / "official_checkpoint_header_audit.json")
    local_audit = load_json(root / "preflight" / "checkpoint_mapping_audit.json")
    memory = load_json(root / "preflight" / "memory_preflight.json")
    performance = load_json(
        root / "performance" / f"{gptq_run_id}_matrix_summary.json"
    )
    quality = load_json(root / "quality" / f"{gptq_run_id}_summary.json")
    checkpoint_quality = load_json(root / "quality" / "bf16_vs_gptq.json")
    performance_runs = performance.get("runs", [])
    quality_cases = quality.get("cases", [])
    tp_names = {
        f"tp{row.get('tensor_parallel_size')}" for row in performance_runs
    }
    gptq_workspace = summarize_gptq_workspace(
        performance_runs,
        performance.get("workload", {}).get("max_num_seqs", 0),
    )
    audit_valid = (
        audit.get("valid") is True
        and audit.get("repo") == OFFICIAL_GPTQ_CHECKPOINT_REPO
        and audit.get("resolved_revision") == OFFICIAL_GPTQ_CHECKPOINT_REVISION
        and checkpoint_semantic_contract_matches(audit, "gptq_int4")
        and set(audit.get("results", {})) == tp_names
        and all(
            result.get("valid") is True
            for result in audit.get("results", {}).values()
        )
    )
    local_checkpoint_valid = (
        local_audit.get("valid") is True
        and local_audit.get("complete") is True
        and set(local_audit.get("results", {})) == tp_names
        and checkpoint_manifest_matches_remote(
            local_audit.get("checkpoint_manifest", {}),
            audit,
        )
    )
    memory_valid = (
        memory.get("valid") is True
        and set(memory.get("results", {})) == tp_names
    )
    performance_valid = (
        bool(performance_runs)
        and performance.get("all_execution_paths_valid") is True
        and performance.get("all_generation_valid") is True
        and performance.get("all_repeat_output_digests_match") is True
        and gptq_workspace["valid"]
        and all(
            row.get("requested_weight_quant_backend") == "auto"
            and row.get("weight_quant_backend") == "triton"
            and row.get("quantization_format") == "gptq_int4"
            and row.get("qwen35_moe_decode_backend") == "sorted"
            and row.get("enforce_eager") is True
            for row in performance_runs
        )
    )
    quality_tp_names = {
        f"tp{row.get('tensor_parallel_size')}" for row in quality_cases
    }
    quality_valid = (
        bool(quality_cases)
        and quality_tp_names == tp_names
        and quality.get("quality_gates", {}).get("all_passed") is True
        and quality.get("cross_tp", {}).get("all_passed") is True
        and all(
            row.get("requested_weight_quant_backend") == "auto"
            and row.get("weight_quant_backend") == "triton"
            and row.get("qwen35_moe_decode_backend") == "sorted"
            for row in quality_cases
        )
    )
    checkpoint_quality_valid = (
        checkpoint_quality.get("valid") is True
        and checkpoint_quality.get("baseline_run_id") == run_id
        and checkpoint_quality.get("candidate_run_id") == gptq_run_id
        and checkpoint_quality.get("tensor_parallel_sizes")
        == sorted(int(name.removeprefix("tp")) for name in tp_names)
    )
    valid_runs = [
        row
        for row in performance_runs
        if row.get("repeat_output_digests_match")
        and row.get("execution_paths_valid")
        and row.get("generation_valid")
    ]
    return {
        "enabled": True,
        "valid": (
            audit_valid
            and local_checkpoint_valid
            and memory_valid
            and performance_valid
            and quality_valid
            and checkpoint_quality_valid
        ),
        "audit_valid": audit_valid,
        "local_checkpoint_matches_official": local_checkpoint_valid,
        "memory_preflight_valid": memory_valid,
        "performance_valid": performance_valid,
        "workspace": gptq_workspace,
        "quality_valid": quality_valid,
        "bf16_vs_gptq_quality_valid": checkpoint_quality_valid,
        "official_checkpoint": {
            "repo": audit.get("repo"),
            "resolved_revision": audit.get("resolved_revision"),
        },
        "tensor_parallel_sizes": sorted(
            {row["tensor_parallel_size"] for row in performance_runs}
        ),
        "best_throughput": (
            max(
                valid_runs,
                key=lambda row: row["median"]["output_throughput_tok_s"],
            )
            if valid_runs
            else None
        ),
        "lowest_peak_memory": (
            min(
                valid_runs,
                key=lambda row: row["median"]["peak_torch_allocated_mib"],
            )
            if valid_runs
            else None
        ),
        "quality_gates": quality.get("quality_gates"),
        "cross_tp": quality.get("cross_tp"),
        "bf16_vs_gptq": checkpoint_quality,
    }


def summarize_optional_fp8_audit(
    run_dir: Path,
    run_id: str | None = None,
    baseline_rows: list[dict] | None = None,
) -> dict:
    """Summarize optional FP8 layout and reference-execution evidence."""

    path = run_dir / "fp8" / "official_checkpoint_header_audit.json"
    if not path.is_file():
        return {"enabled": False, "valid": True, "executable": False}
    audit = load_json(path)
    results = audit.get("results", {})
    layouts = {
        name: result.get("quantized_tp_layout", {})
        for name, result in results.items()
    }
    audit_valid = (
        audit.get("valid") is True
        and audit.get("repo") == OFFICIAL_FP8_CHECKPOINT_REPO
        and audit.get("resolved_revision") == OFFICIAL_FP8_CHECKPOINT_REVISION
        and audit.get("config_sha256") == OFFICIAL_FP8_CONFIG_SHA256
        and audit.get("index_sha256") == OFFICIAL_FP8_INDEX_SHA256
        and audit.get("headers_sha256") == OFFICIAL_FP8_HEADERS_SHA256
        and checkpoint_semantic_contract_matches(audit, "fp8_block")
        and audit.get("quantization", {}).get("format") == "fp8_block"
        and bool(results)
        and all(
            result.get("valid") is True
            and result.get("skipped_by_prefix")
            == OFFICIAL_SKIPPED_WEIGHT_PREFIXES
            and not result.get("unclassified_skipped_weights")
            for result in results.values()
        )
    )
    report = {
        "enabled": True,
        "valid": audit_valid,
        "executable": False,
        "scope": (
            "remote checkpoint headers and TP quantization-block alignment only; "
            "FP8 payload loading, kernels, correctness, quality, and performance "
            "are not validated"
        ),
        "official_checkpoint": {
            "repo": audit.get("repo"),
            "resolved_revision": audit.get("resolved_revision"),
        },
        "tensor_parallel": {
            name: {
                "valid": results[name].get("valid") is True,
                "local_parameter_bytes": results[name].get(
                    "local_parameter_bytes"
                ),
                "requires_partial_unit_loader": layouts[name].get(
                    "requires_partial_unit_loader"
                ),
                "partial_quantization_unit_count": layouts[name].get(
                    "partial_quantization_unit_count"
                ),
            }
            for name in sorted(results)
        },
    }
    preflight_dir = run_dir / "fp8" / "preflight"
    if not preflight_dir.exists():
        return report
    if run_id is None:
        raise ValueError("run_id is required for FP8 execution evidence")

    fp8_run_id = f"{run_id}-fp8"
    local_audit = load_json(preflight_dir / "checkpoint_mapping_audit.json")
    memory = load_json(preflight_dir / "memory_preflight.json")
    performance = load_json(
        run_dir / "fp8" / "performance" / f"{fp8_run_id}_matrix_summary.json"
    )
    quality = load_json(
        run_dir / "fp8" / "quality" / f"{fp8_run_id}_summary.json"
    )
    checkpoint_quality = load_json(
        run_dir / "fp8" / "quality" / "bf16_vs_fp8.json"
    )
    performance_runs = performance.get("runs", [])
    quality_cases = quality.get("cases", [])
    runtime_backend = audit.get("fp8_runtime_backend", "reference")
    tp_names = {
        f"tp{row.get('tensor_parallel_size')}" for row in performance_runs
    }
    audit_valid = audit_valid and set(results) == tp_names
    local_checkpoint_valid = (
        local_audit.get("valid") is True
        and local_audit.get("complete") is True
        and local_audit.get("fp8_runtime_backend") == runtime_backend
        and set(local_audit.get("results", {})) == tp_names
        and checkpoint_manifest_matches_remote(
            local_audit.get("checkpoint_manifest", {}),
            audit,
        )
        and all(
            local_audit["results"][name].get(
                "local_parameter_and_resident_runtime_bytes"
            )
            == results[name].get("local_parameter_and_resident_runtime_bytes")
            for name in tp_names
        )
    )
    memory_valid = (
        memory.get("valid") is True
        and set(memory.get("results", {})) == tp_names
    )
    runtime_storage_by_tp = {}
    for row in performance_runs:
        tp_size = row.get("tensor_parallel_size")
        tp_key = f"tp{tp_size}"
        expected_storage = results.get(tp_key, {}).get(
            "resident_fp8_expert_storage",
            {},
        )
        rank_stats = row.get("storage", {}).get(
            "runtime_buffer_storage_by_rank",
            [],
        )
        ranks_complete = (
            isinstance(tp_size, int)
            and {item.get("rank") for item in rank_stats}
            == set(range(tp_size))
        )
        if runtime_backend == "resident":
            expected_storage_valid = all(
                isinstance(expected_storage.get(key), int)
                and expected_storage[key] > 0
                for key in (
                    "layer_count",
                    "weight_bytes",
                    "scale_bytes",
                    "dequant_workspace_pool_count",
                    "dequant_workspace_bytes",
                )
            )
            matches_audit = expected_storage_valid and all(
                item.get("resident_fp8_expert_layer_count")
                == expected_storage["layer_count"]
                and item.get("resident_fp8_expert_weight_bytes")
                == expected_storage["weight_bytes"]
                and item.get("resident_fp8_expert_scale_bytes")
                == expected_storage["scale_bytes"]
                and item.get("resident_fp8_weight_pool_count")
                == expected_storage["dequant_workspace_pool_count"]
                and item.get("resident_fp8_dequant_workspace_bytes")
                == expected_storage["dequant_workspace_bytes"]
                for item in rank_stats
            )
            valid = ranks_complete and matches_audit and all(
                item.get("resident_fp8_expert_layer_count", 0) > 0
                and item.get("resident_fp8_expert_weight_bytes", 0) > 0
                and item.get("resident_fp8_expert_scale_bytes", 0) > 0
                and item.get("resident_fp8_weight_pool_count", 0) > 0
                and item.get("resident_fp8_dequant_workspace_bytes", 0) > 0
                and item.get(
                    "resident_fp8_dequant_workspace_allocation_count", 0
                )
                > 0
                and item.get("resident_fp8_dequant_workspace_reuse_count", 0)
                > 0
                for item in rank_stats
            )
        else:
            expected_storage_valid = all(
                expected_storage.get(key) == 0
                for key in (
                    "layer_count",
                    "weight_bytes",
                    "scale_bytes",
                    "dequant_workspace_pool_count",
                    "dequant_workspace_bytes",
                )
            )
            matches_audit = expected_storage_valid and all(
                item.get("resident_fp8_expert_layer_count") == 0
                and item.get("resident_fp8_expert_weight_bytes") == 0
                and item.get("resident_fp8_expert_scale_bytes") == 0
                and item.get("resident_fp8_weight_pool_count") == 0
                and item.get("resident_fp8_dequant_workspace_bytes") == 0
                for item in rank_stats
            )
            valid = ranks_complete and matches_audit
        summary = runtime_storage_by_tp.setdefault(
            tp_key,
            {
                "valid": True,
                "matches_header_audit": True,
                "configurations": [],
            },
        )
        summary["valid"] = summary["valid"] and valid
        summary["matches_header_audit"] = (
            summary["matches_header_audit"] and matches_audit
        )
        summary["configurations"].append(
            {
                "valid": valid,
                "matches_header_audit": matches_audit,
                "recurrent_state_dtype": row.get("recurrent_state_dtype"),
                "kv_cache_dtype": row.get("kv_cache_dtype"),
                "qwen35_moe_decode_backend": row.get(
                    "qwen35_moe_decode_backend"
                ),
                "ranks": rank_stats,
            }
        )
        if (
            row.get("recurrent_state_dtype") == "model"
            and row.get("kv_cache_dtype") == "auto"
            and row.get("qwen35_moe_decode_backend") == "sorted"
        ):
            summary["ranks"] = rank_stats
    runtime_storage_valid = (
        set(runtime_storage_by_tp) == tp_names
        and all(item["valid"] for item in runtime_storage_by_tp.values())
    )
    canonical_fp8_rows = [
        row
        for row in performance_runs
        if row.get("recurrent_state_dtype") == "model"
        and row.get("kv_cache_dtype") == "auto"
        and row.get("qwen35_moe_decode_backend") == "sorted"
        and row.get("repeat_output_digests_match") is True
        and row.get("execution_paths_valid") is True
        and row.get("generation_valid") is True
    ]
    canonical_baseline_rows = [
        row
        for row in (baseline_rows or [])
        if row.get("recurrent_state_dtype") == "model"
        and row.get("kv_cache_dtype") == "auto"
        and row.get("qwen35_moe_decode_backend") == "sorted"
        and row.get("repeat_output_digests_match") is True
        and row.get("execution_paths_valid") is True
        and row.get("generation_valid") is True
    ]
    fp8_rows_by_tp: dict[int, list[dict]] = {}
    baseline_rows_by_tp: dict[int, list[dict]] = {}
    for row in canonical_fp8_rows:
        fp8_rows_by_tp.setdefault(row.get("tensor_parallel_size"), []).append(row)
    for row in canonical_baseline_rows:
        baseline_rows_by_tp.setdefault(row.get("tensor_parallel_size"), []).append(
            row
        )
    performance_comparisons = {}
    comparison_tp_sizes = sorted(
        int(name.removeprefix("tp")) for name in tp_names
    )
    for tp_size in comparison_tp_sizes:
        candidates = fp8_rows_by_tp.get(tp_size, [])
        baselines = baseline_rows_by_tp.get(tp_size, [])
        comparison_valid = len(candidates) == 1 and len(baselines) == 1
        comparison = {
            "valid": False,
            "tensor_parallel_size": tp_size,
            "configuration": {
                "recurrent_state_dtype": "model",
                "kv_cache_dtype": "auto",
                "qwen35_moe_decode_backend": "sorted",
            },
            "candidate_count": len(candidates),
            "baseline_count": len(baselines),
        }
        if comparison_valid:
            candidate = candidates[0]
            baseline = baselines[0]
            candidate_median = candidate.get("median", {})
            baseline_median = baseline.get("median", {})
            fp8_peak = candidate_median.get("peak_torch_allocated_mib")
            baseline_peak = baseline_median.get("peak_torch_allocated_mib")
            fp8_throughput = candidate_median.get("output_throughput_tok_s")
            baseline_throughput = baseline_median.get("output_throughput_tok_s")
            metrics_valid = all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 0
                for value in (
                    fp8_peak,
                    baseline_peak,
                    fp8_throughput,
                    baseline_throughput,
                )
            )
            rank_stats = candidate.get("storage", {}).get(
                "runtime_buffer_storage_by_rank", []
            )
            resident_expert_bytes = [
                item.get("resident_fp8_expert_weight_bytes", 0)
                + item.get("resident_fp8_expert_scale_bytes", 0)
                for item in rank_stats
            ]
            workspace_bytes = [
                item.get("resident_fp8_dequant_workspace_bytes", 0)
                for item in rank_stats
            ]
            if metrics_valid:
                peak_reduction = baseline_peak - fp8_peak
                peak_ratio = fp8_peak / baseline_peak
                throughput_ratio = fp8_throughput / baseline_throughput
                memory_gate = peak_reduction > 0
                throughput_gate = (
                    throughput_ratio >= RESIDENT_FP8_MIN_THROUGHPUT_RATIO
                )
            else:
                peak_reduction = None
                peak_ratio = None
                throughput_ratio = None
                memory_gate = False
                throughput_gate = False
            benefit_required = runtime_backend == "resident"
            comparison.update(
                {
                    "valid": metrics_valid
                    and (not benefit_required or (memory_gate and throughput_gate)),
                    "baseline": {
                        "output_throughput_tok_s": baseline_throughput,
                        "peak_torch_allocated_mib": baseline_peak,
                    },
                    "candidate": {
                        "output_throughput_tok_s": fp8_throughput,
                        "peak_torch_allocated_mib": fp8_peak,
                    },
                    "peak_memory_reduction_mib": peak_reduction,
                    "peak_memory_ratio": peak_ratio,
                    "throughput_ratio": throughput_ratio,
                    "resident_expert_storage_bytes_by_rank": resident_expert_bytes,
                    "resident_dequant_workspace_bytes_by_rank": workspace_bytes,
                    "gates": {
                        "benefit_required": benefit_required,
                        "peak_memory_reduced": memory_gate,
                        "min_throughput_ratio": (
                            RESIDENT_FP8_MIN_THROUGHPUT_RATIO
                            if benefit_required
                            else None
                        ),
                        "throughput_preserved": throughput_gate,
                    },
                }
            )
        performance_comparisons[f"tp{tp_size}"] = comparison
    performance_comparison_valid = (
        bool(performance_comparisons)
        and all(item["valid"] for item in performance_comparisons.values())
    )
    performance_valid = (
        bool(performance_runs)
        and performance.get("all_execution_paths_valid") is True
        and performance.get("all_generation_valid") is True
        and performance.get("all_repeat_output_digests_match") is True
        and all(
            row.get("requested_weight_quant_backend") == runtime_backend
            and row.get("weight_quant_backend") == runtime_backend
            and row.get("quantization_format") == "fp8_block"
            and row.get("qwen35_moe_decode_backend") == "sorted"
            and row.get("enforce_eager") is True
            for row in performance_runs
        )
        and runtime_storage_valid
        and performance_comparison_valid
    )
    quality_valid = (
        bool(quality_cases)
        and {f"tp{row.get('tensor_parallel_size')}" for row in quality_cases}
        == tp_names
        and quality.get("quality_gates", {}).get("all_passed") is True
        and quality.get("cross_tp", {}).get("all_passed") is True
        and all(
            row.get("requested_weight_quant_backend") == runtime_backend
            and row.get("weight_quant_backend") == runtime_backend
            and row.get("qwen35_moe_decode_backend") == "sorted"
            for row in quality_cases
        )
    )
    checkpoint_quality_valid = (
        checkpoint_quality.get("valid") is True
        and checkpoint_quality.get("baseline_run_id") == run_id
        and checkpoint_quality.get("candidate_run_id") == fp8_run_id
        and checkpoint_quality.get("tensor_parallel_sizes")
        == sorted(int(name.removeprefix("tp")) for name in tp_names)
    )
    valid_runs = [
        row
        for row in performance_runs
        if row.get("repeat_output_digests_match")
        and row.get("execution_paths_valid")
        and row.get("generation_valid")
    ]
    execution_valid = (
        audit_valid
        and local_checkpoint_valid
        and memory_valid
        and performance_valid
        and quality_valid
        and checkpoint_quality_valid
    )
    report.update(
        {
            "valid": execution_valid,
            "executable": performance_valid,
            "execution_validated": execution_valid,
            "runtime_backend": runtime_backend,
            "native_fp8": False,
            "scope": (
                "official FP8 checkpoint execution with "
                + (
                    "resident expert FP8 storage and on-demand model-dtype "
                    "dequantization"
                    if runtime_backend == "resident"
                    else "model-dtype dequantization at load"
                )
                + "; this does not validate native FP8 kernels"
            ),
            "local_checkpoint_matches_official": local_checkpoint_valid,
            "memory_preflight_valid": memory_valid,
            "performance_valid": performance_valid,
            "performance_comparison_valid": performance_comparison_valid,
            "performance_comparisons": performance_comparisons,
            "runtime_storage_valid": runtime_storage_valid,
            "runtime_storage_by_tp": runtime_storage_by_tp,
            "quality_valid": quality_valid,
            "bf16_vs_fp8_quality_valid": checkpoint_quality_valid,
            "tensor_parallel_sizes": sorted(
                {row["tensor_parallel_size"] for row in performance_runs}
            ),
            "best_throughput": (
                max(
                    valid_runs,
                    key=lambda row: row["median"]["output_throughput_tok_s"],
                )
                if valid_runs
                else None
            ),
            "lowest_peak_memory": (
                min(
                    valid_runs,
                    key=lambda row: row["median"]["peak_torch_allocated_mib"],
                )
                if valid_runs
                else None
            ),
            "quality_gates": quality.get("quality_gates"),
            "cross_tp": quality.get("cross_tp"),
            "bf16_vs_fp8": checkpoint_quality,
        }
    )
    return report


def summarize_normalization_candidate(
    result: dict,
    reuse_key: str,
    *,
    required_flags: tuple[str, ...] = (),
) -> dict:
    max_abs_error = max(item["max_abs_error"] for item in result["errors"])
    reference = result["reference"]
    candidate = result["candidate"]
    speedup = result["speedup"]
    reference_peak = reference["peak_extra_mib"]
    candidate_peak = candidate["peak_extra_mib"]
    measurement_valid = (
        math.isfinite(speedup)
        and speedup > 0
        and math.isfinite(reference_peak)
        and reference_peak >= 0
        and math.isfinite(candidate_peak)
        and candidate_peak >= 0
    )
    memory_non_regression = measurement_valid and candidate_peak <= reference_peak
    return {
        "valid": (
            result[reuse_key]
            and all(result.get(flag) is True for flag in required_flags)
            and max_abs_error <= NORMALIZATION_MAX_ABS_ERROR
            and memory_non_regression
        ),
        "speedup": speedup,
        "measurement_valid": measurement_valid,
        "memory_non_regression": memory_non_regression,
        "reference_peak_extra_mib": reference_peak,
        "candidate_peak_extra_mib": candidate_peak,
        "peak_extra_mib_delta": candidate_peak - reference_peak,
        "max_abs_error": max_abs_error,
        "max_allowed_abs_error": NORMALIZATION_MAX_ABS_ERROR,
        "workspace": {
            key: value
            for key, value in result.items()
            if key.endswith("_mib")
        },
        "required_optimizations": {
            flag: result.get(flag) is True for flag in required_flags
        },
    }


def summarize_buffer_reuse_candidate(
    result: dict,
    workspace_keys: tuple[str, ...],
    required_metadata: dict[str, int] | None = None,
) -> dict:
    missing = [key for key in workspace_keys if key not in result]
    if missing:
        raise ValueError(
            "buffer-reuse benchmark is missing workspace metrics: "
            + ", ".join(missing)
        )
    errors = result.get("errors", [])
    if not errors:
        raise ValueError("buffer-reuse benchmark has no correctness metrics")
    max_abs_error = max(item["max_abs_error"] for item in errors)
    reference = result["reference"]
    candidate = result["candidate"]
    metadata = {key: result.get(key) for key in (required_metadata or {})}
    metadata_valid = all(
        metadata[key] == expected
        for key, expected in (required_metadata or {}).items()
    )
    workspace = {key: result[key] for key in workspace_keys}
    speedup = result["speedup"]
    reference_peak = reference["peak_extra_mib"]
    candidate_peak = candidate["peak_extra_mib"]
    measurement_valid = (
        math.isfinite(speedup)
        and speedup > 0
        and math.isfinite(reference_peak)
        and reference_peak >= 0
        and math.isfinite(candidate_peak)
        and candidate_peak >= 0
    )
    memory_non_regression = measurement_valid and candidate_peak <= reference_peak
    return {
        "valid": (
            all(value > 0 for value in workspace.values())
            and metadata_valid
            and max_abs_error <= BUFFER_REUSE_MAX_ABS_ERROR
            and memory_non_regression
        ),
        "speedup": speedup,
        "measurement_valid": measurement_valid,
        "memory_non_regression": memory_non_regression,
        "reference_peak_extra_mib": reference_peak,
        "candidate_peak_extra_mib": candidate_peak,
        "peak_extra_mib_delta": candidate_peak - reference_peak,
        "max_abs_error": max_abs_error,
        "max_allowed_abs_error": BUFFER_REUSE_MAX_ABS_ERROR,
        "workspace": workspace,
        "metadata": metadata,
    }


def summarize_delta_causal_mask_cache(
    result: dict | None,
    *,
    measured_on_cuda: bool,
) -> dict:
    if result is None:
        return {"available": False, "measured_on_cuda": measured_on_cuda}
    candidates = result.get("candidates", {})
    if not candidates:
        raise ValueError("DeltaNet causal-mask benchmark has no candidates")
    summaries = {
        chunk_size: summarize_buffer_reuse_candidate(
            candidate,
            ("persistent_mask_mib",),
            {
                "cache_reuses_storage": True,
                "eliminated_allocations_per_additional_layer": 1,
            },
        )
        for chunk_size, candidate in candidates.items()
    }
    return {
        "available": True,
        "measured_on_cuda": measured_on_cuda,
        "valid": all(item["valid"] for item in summaries.values()),
        "all_cuda_beneficial": (
            measured_on_cuda
            and all(
                item["valid"] and item["speedup"] >= 1.0
                for item in summaries.values()
            )
        ),
        "cache_max_entries": result["cache_max_entries"],
        "maximum_cached_chunk_size": result["maximum_cached_chunk_size"],
        "by_chunk_size": summaries,
    }


def summarize_mixed_moe_dispatch(result: dict) -> dict:
    errors = result.get("errors", [])
    max_abs_error = max(
        (item.get("max_abs_error", math.inf) for item in errors),
        default=math.inf,
    )
    reference_peak = result.get("reference", {}).get(
        "peak_extra_mib",
        math.nan,
    )
    candidate_peak = result.get("candidate", {}).get(
        "peak_extra_mib",
        math.nan,
    )
    speedup = result.get("speedup_vs_grouped", math.nan)
    avoided_route_hidden = result.get(
        "avoided_route_hidden_allocation_mib_per_step",
        math.nan,
    )
    avoided_output_zero = result.get(
        "avoided_redundant_output_zero_mib_per_step",
        math.nan,
    )
    peak_delta = candidate_peak - reference_peak
    checks = {
        "cuda_measurement": result.get("measured_on_cuda") is True,
        "mixed_shape": (
            isinstance(result.get("decode_tokens"), int)
            and result["decode_tokens"] > 0
            and isinstance(result.get("prefill_tokens"), int)
            and result["prefill_tokens"] > 0
        ),
        "accuracy": (
            bool(errors)
            and math.isfinite(max_abs_error)
            and max_abs_error <= MIXED_MOE_MAX_ABS_ERROR
        ),
        "speed": math.isfinite(speedup) and speedup >= MIXED_MOE_MIN_SPEEDUP,
        "peak_memory": (
            math.isfinite(peak_delta)
            and peak_delta <= MIXED_MOE_MAX_PEAK_EXTRA_MIB
        ),
        "route_hidden_eliminated": (
            math.isfinite(avoided_route_hidden) and avoided_route_hidden > 0
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "decode_tokens": result.get("decode_tokens"),
        "prefill_tokens": result.get("prefill_tokens"),
        "speedup_vs_grouped": speedup,
        "peak_extra_mib_delta": peak_delta,
        "avoided_route_hidden_allocation_mib_per_step": avoided_route_hidden,
        "avoided_redundant_output_zero_mib_per_step": avoided_output_zero,
        "max_abs_error": max_abs_error,
        "thresholds": {
            "min_speedup": MIXED_MOE_MIN_SPEEDUP,
            "max_peak_extra_mib": MIXED_MOE_MAX_PEAK_EXTRA_MIB,
            "max_abs_error": MIXED_MOE_MAX_ABS_ERROR,
        },
    }


def summarize_moe_weight_buffer_reuse(result: dict) -> dict:
    errors = result.get("errors", {})
    max_abs_error = errors.get("max_abs_error", math.inf)
    speedup = result.get("speedup", math.nan)
    peak_delta = result.get("peak_extra_mib_delta", math.nan)
    persistent_mib = result.get(
        "persistent_expert_weight_buffer_mib",
        math.nan,
    )
    checks = {
        "cuda_measurement": result.get("measured_on_cuda") is True,
        "accuracy": (
            math.isfinite(max_abs_error)
            and max_abs_error <= MIXED_MOE_MAX_ABS_ERROR
        ),
        "speed": math.isfinite(speedup) and speedup >= 1.0,
        "peak_memory": math.isfinite(peak_delta) and peak_delta <= 0.0,
        "persistent_storage": (
            math.isfinite(persistent_mib) and persistent_mib > 0
        ),
        "allocation_elimination": (
            result.get("eliminated_weight_allocations_per_chunk") == 2
            and result.get("candidate_reuses_expert_weight_storage") is True
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "speedup": speedup,
        "peak_extra_mib_delta": peak_delta,
        "persistent_expert_weight_buffer_mib": persistent_mib,
        "max_abs_error": max_abs_error,
    }


def summarize_mixed_moe_dispatch_sweep(results: dict[str, dict]) -> dict:
    if not results:
        raise ValueError("mixed MoE dispatch sweep has no cases")
    cases = {
        name: summarize_mixed_moe_dispatch(result)
        for name, result in results.items()
    }
    return {
        "valid": all(case["valid"] for case in cases.values()),
        "case_count": len(cases),
        "minimum_speedup_vs_grouped": min(
            case["speedup_vs_grouped"] for case in cases.values()
        ),
        "maximum_peak_extra_mib_delta": max(
            case["peak_extra_mib_delta"] for case in cases.values()
        ),
        "maximum_abs_error": max(
            case["max_abs_error"] for case in cases.values()
        ),
        "cases": cases,
    }


def evaluate_moe_runtime_candidate(
    *,
    output_digest_matches: bool,
    throughput_speedup: float,
    tpot_speedup: float,
    peak_memory_delta_mib: float,
    max_coefficient_of_variation: float,
    baseline_decode_host_sync_observed: bool,
    candidate_decode_host_sync_eliminated: bool,
    candidate_batched_dispatch_observed: bool,
) -> dict:
    checks = {
        "output_parity": output_digest_matches,
        "stable_repeats": (
            max_coefficient_of_variation <= MOE_RUNTIME_MAX_CV
        ),
        "throughput_non_regression": (
            throughput_speedup >= MOE_RUNTIME_MIN_THROUGHPUT_RATIO
        ),
        "tpot_speedup": tpot_speedup >= MOE_RUNTIME_MIN_TPOT_SPEEDUP,
        "peak_memory": (
            peak_memory_delta_mib <= MOE_RUNTIME_MAX_PEAK_EXTRA_MIB
        ),
        "baseline_decode_host_sync_observed": (
            baseline_decode_host_sync_observed
        ),
        "candidate_decode_host_sync_eliminated": (
            candidate_decode_host_sync_eliminated
        ),
        "candidate_batched_dispatch_observed": (
            candidate_batched_dispatch_observed
        ),
    }
    return {
        "promote_to_default": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_throughput_ratio": MOE_RUNTIME_MIN_THROUGHPUT_RATIO,
            "min_tpot_speedup": MOE_RUNTIME_MIN_TPOT_SPEEDUP,
            "max_peak_extra_mib": MOE_RUNTIME_MAX_PEAK_EXTRA_MIB,
            "max_coefficient_of_variation": MOE_RUNTIME_MAX_CV,
        },
    }


def summarize_moe_runtime(rows: list[dict]) -> dict[str, dict]:
    baselines = {
        (
            row["tensor_parallel_size"],
            row["recurrent_state_dtype"],
            row["kv_cache_dtype"],
        ): row
        for row in rows
        if row.get("qwen35_moe_decode_backend") == "sorted"
    }
    comparisons = {}
    candidates = [
        row
        for row in rows
        if row.get("qwen35_moe_decode_backend") == "batched"
    ]
    if not candidates:
        raise ValueError("performance matrix contains no batched MoE candidate")
    expected_candidate_keys = {
        key for key in baselines if key[1] == "model"
    }
    candidate_keys = {
        (
            row["tensor_parallel_size"],
            row["recurrent_state_dtype"],
            row["kv_cache_dtype"],
        )
        for row in candidates
    }
    if candidate_keys != expected_candidate_keys:
        raise ValueError(
            "batched MoE candidates do not cover every model-state KV mode"
        )
    for candidate in candidates:
        key = (
            candidate["tensor_parallel_size"],
            candidate["recurrent_state_dtype"],
            candidate["kv_cache_dtype"],
        )
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(
                "batched MoE candidate has no matching sorted baseline: "
                f"TP={key[0]}, state={key[1]}, KV={key[2]}"
            )
        baseline_median = baseline["median"]
        candidate_median = candidate["median"]
        stability_metrics = ("output_throughput_tok_s", "avg_tpot_s")
        max_cv = max(
            baseline["coefficient_of_variation"][metric]
            for metric in stability_metrics
        )
        max_cv = max(
            max_cv,
            *(
                candidate["coefficient_of_variation"][metric]
                for metric in stability_metrics
            ),
        )
        tp_name = f"tp{key[0]}"
        output_digest_matches = (
            baseline["generated_token_ids_digest"]
            == candidate["generated_token_ids_digest"]
        )
        throughput_speedup = (
            candidate_median["output_throughput_tok_s"]
            / baseline_median["output_throughput_tok_s"]
        )
        tpot_speedup = (
            baseline_median["avg_tpot_s"]
            / candidate_median["avg_tpot_s"]
        )
        peak_memory_delta_mib = (
            candidate_median["peak_torch_allocated_mib"]
            - baseline_median["peak_torch_allocated_mib"]
        )
        baseline_rank_stats = baseline.get("storage", {}).get(
            "runtime_buffer_storage_by_rank", []
        )
        candidate_rank_stats = candidate.get("storage", {}).get(
            "runtime_buffer_storage_by_rank", []
        )
        expected_ranks = set(range(key[0]))
        baseline_decode_syncs = {
            item.get("rank"): item.get("moe_decode_host_route_sync_count")
            for item in baseline_rank_stats
        }
        candidate_decode_syncs = {
            item.get("rank"): item.get("moe_decode_host_route_sync_count")
            for item in candidate_rank_stats
        }
        candidate_batched_dispatches = {
            item.get("rank"): item.get("moe_batched_dispatch_count")
            for item in candidate_rank_stats
        }
        baseline_sync_observed = (
            set(baseline_decode_syncs) == expected_ranks
            and all(
                value is not None and value > 0
                for value in baseline_decode_syncs.values()
            )
        )
        candidate_sync_eliminated = (
            set(candidate_decode_syncs) == expected_ranks
            and all(value == 0 for value in candidate_decode_syncs.values())
        )
        candidate_batched_observed = (
            set(candidate_batched_dispatches) == expected_ranks
            and all(
                value is not None and value > 0
                for value in candidate_batched_dispatches.values()
            )
        )
        by_kv = comparisons.setdefault(tp_name, {})
        if key[2] in by_kv:
            raise ValueError(
                f"duplicate batched MoE candidate for TP={key[0]}, KV={key[2]}"
            )
        by_kv[key[2]] = {
            "configuration": {
                "recurrent_state_dtype": key[1],
                "kv_cache_dtype": key[2],
            },
            "baseline_label": baseline["label"],
            "candidate_label": candidate["label"],
            "output_digest_matches": output_digest_matches,
            "throughput_speedup": throughput_speedup,
            "tpot_speedup": tpot_speedup,
            "peak_memory_delta_mib": peak_memory_delta_mib,
            "max_coefficient_of_variation": max_cv,
            "baseline_decode_host_syncs_by_rank": baseline_decode_syncs,
            "candidate_decode_host_syncs_by_rank": candidate_decode_syncs,
            "candidate_batched_dispatches_by_rank": candidate_batched_dispatches,
            "decode_host_sync_eliminated": (
                baseline_sync_observed and candidate_sync_eliminated
            ),
            "promotion": evaluate_moe_runtime_candidate(
                output_digest_matches=output_digest_matches,
                throughput_speedup=throughput_speedup,
                tpot_speedup=tpot_speedup,
                peak_memory_delta_mib=peak_memory_delta_mib,
                max_coefficient_of_variation=max_cv,
                baseline_decode_host_sync_observed=baseline_sync_observed,
                candidate_decode_host_sync_eliminated=candidate_sync_eliminated,
                candidate_batched_dispatch_observed=candidate_batched_observed,
            ),
        }
    return comparisons


def summarize_recurrent_state_access(rows: list[dict]) -> dict:
    by_configuration = {}
    for row in rows:
        backend = row.get("qwen35_moe_decode_backend")
        expected = {
            "sorted": {"prefill_contiguous_view", "decode_contiguous_view"},
            "batched": {"prefill_contiguous_view", "decode_graph_indexed"},
        }.get(backend)
        if expected is None:
            raise ValueError(
                "performance row has unsupported MoE decode backend: "
                f"{backend!r}"
            )
        paths = row.get("execution_paths", {})
        required = paths.get("required", [])
        observed = paths.get("observed_in_all_repeats", [])
        required_set = set(required) if isinstance(required, list) else set()
        observed_set = set(observed) if isinstance(observed, list) else set()
        missing_required = sorted(expected - required_set)
        missing_observed = sorted(expected - observed_set)
        label = row.get("label")
        tp_name = f"tp{row['tensor_parallel_size']}"
        configuration_name = (
            f"{label}:state={row['recurrent_state_dtype']}:"
            f"kv={row['kv_cache_dtype']}:moe={backend}"
        )
        by_tp = by_configuration.setdefault(tp_name, {})
        if configuration_name in by_tp:
            raise ValueError(
                "duplicate performance configuration for recurrent-state "
                f"evidence: {tp_name}/{configuration_name}"
            )
        by_tp[configuration_name] = {
            "valid": not missing_required and not missing_observed,
            "expected": sorted(expected),
            "required": sorted(required_set),
            "observed_in_all_repeats": sorted(observed_set),
            "missing_from_required": missing_required,
            "missing_from_observed": missing_observed,
        }
    return {
        "all_configurations_valid": (
            bool(by_configuration)
            and all(
                item["valid"]
                for by_tp in by_configuration.values()
                for item in by_tp.values()
            )
        ),
        "by_configuration": by_configuration,
    }


def summarize_attention_case(result: dict, *, partitioned: bool) -> dict:
    reference = result["results"]["flash_reference"]
    fused = [
        (name, item)
        for name, item in result["results"].items()
        if name.startswith("int8_")
        and not name.startswith("int8_partitioned_")
        and item.get("status") == "ok"
    ]
    partitioned_results = [
        (name, item)
        for name, item in result["results"].items()
        if name.startswith("int8_partitioned_") and item.get("status") == "ok"
    ]
    if not fused:
        raise ValueError("attention benchmark has no successful fused INT8 kernel")
    if partitioned and not partitioned_results:
        raise ValueError(
            "long-context attention benchmark has no successful partitioned kernel"
        )

    def best_accurate(items: list[tuple[str, dict]]) -> dict | None:
        accurate = [
            pair
            for pair in items
            if pair[1]["max_abs_diff_vs_flash_reference"]
            <= ATTENTION_MAX_ABS_ERROR
        ]
        if not accurate:
            return None
        name, item = min(accurate, key=lambda pair: pair[1]["median_ms"])
        return {
            "backend": name,
            "median_ms": item["median_ms"],
            "speedup_vs_flash_reference": (
                reference["median_ms"] / item["median_ms"]
            ),
            "max_abs_diff_vs_flash_reference": item[
                "max_abs_diff_vs_flash_reference"
            ],
            "peak_extra_mib": item["peak_extra_mib"],
        }

    production_name = f"int8_partitioned_ps{PRODUCTION_PARTITION_SIZE}"
    production_reuse_name = f"{production_name}_workspace_reuse"
    production_item = result["results"].get(production_name)
    production_reuse_item = result["results"].get(production_reuse_name)
    production_accurate = (
        isinstance(production_item, dict)
        and production_item.get("status") == "ok"
        and production_item["max_abs_diff_vs_flash_reference"]
        <= ATTENTION_MAX_ABS_ERROR
    )
    production_summary = (
        best_accurate([(production_name, production_item)])
        if production_accurate
        else None
    )
    production_workspace = (
        result.get("shape_manifest", {})
        .get("workspace", {})
        .get("partitioned", {})
        .get(str(PRODUCTION_PARTITION_SIZE), {})
    )
    partitioned_workspace_valid = (
        not partitioned
        or (
            production_workspace.get("allocation_count") == 1
            and production_workspace.get("shared_storage") is True
        )
    )
    workspace_reuse_measurement_valid = (
        not partitioned
        or (
            isinstance(production_reuse_item, dict)
            and production_reuse_item.get("status") == "ok"
            and production_reuse_item.get(
                "max_abs_diff_vs_flash_reference", math.inf
            )
            <= ATTENTION_MAX_ABS_ERROR
            and math.isfinite(
                production_reuse_item.get("speedup_vs_allocating", math.nan)
            )
            and production_reuse_item.get("speedup_vs_allocating", 0) > 0
            and math.isfinite(
                production_reuse_item.get("peak_extra_mib", math.nan)
            )
            and production_reuse_item.get("peak_extra_mib", -1) >= 0
        )
    )
    workspace_reuse = None
    if isinstance(production_reuse_item, dict):
        speedup = production_reuse_item.get("speedup_vs_allocating", math.nan)
        baseline_peak = (
            production_item.get("peak_extra_mib", math.nan)
            if isinstance(production_item, dict)
            else math.nan
        )
        reuse_peak = production_reuse_item.get("peak_extra_mib", math.nan)
        workspace_reuse = {
            "backend": production_reuse_name,
            "measurement_valid": workspace_reuse_measurement_valid,
            "speedup_vs_allocating": speedup,
            "allocating_peak_extra_mib": baseline_peak,
            "reuse_peak_extra_mib": reuse_peak,
            "avoided_peak_extra_mib": (
                baseline_peak - reuse_peak
                if math.isfinite(baseline_peak) and math.isfinite(reuse_peak)
                else math.nan
            ),
            "promote_to_runtime": (
                workspace_reuse_measurement_valid
                and speedup >= ATTENTION_WORKSPACE_REUSE_MIN_SPEEDUP
                and reuse_peak <= baseline_peak
            ),
            "minimum_speedup": ATTENTION_WORKSPACE_REUSE_MIN_SPEEDUP,
        }
    return {
        "context_len": result["context_len"],
        "batch_size": result["batch_size"],
        "max_allowed_abs_error": ATTENTION_MAX_ABS_ERROR,
        "fused_correctness_valid": best_accurate(fused) is not None,
        "partitioned_correctness_valid": (
            not partitioned
            or (
                best_accurate(partitioned_results) is not None
                and production_accurate
            )
        ),
        "partitioned_workspace_valid": partitioned_workspace_valid,
        "workspace_reuse_measurement_valid": workspace_reuse_measurement_valid,
        "best_fused": best_accurate(fused),
        "best_partitioned": best_accurate(partitioned_results),
        "production_partition_size": PRODUCTION_PARTITION_SIZE,
        "production_partitioned": production_summary,
        "production_workspace_reuse": workspace_reuse,
    }


def summarize_kv_pressure_case(
    result: dict,
    *,
    expected_tp_size: int,
    expected_policy: str,
    expected_decode_reservation: bool = False,
) -> dict:
    metrics = result.get("metrics", {})
    expected_requests = PRESSURE_INITIAL_SEQUENCES + PRESSURE_INJECTED_SEQUENCES
    configuration_valid = (
        result.get("tensor_parallel_size") == expected_tp_size
        and result.get("initial_seqs") == PRESSURE_INITIAL_SEQUENCES
        and result.get("injected_seqs") == PRESSURE_INJECTED_SEQUENCES
        and result.get("initial_input_lens") == PRESSURE_INITIAL_LENGTHS
        and result.get("injected_input_lens") == PRESSURE_INJECTED_LENGTHS
        and result.get("output_len") == 16
        and result.get("num_kvcache_blocks_override") == PRESSURE_KV_BLOCKS
        and result.get("num_kvcache_blocks") == PRESSURE_KV_BLOCKS
        and result.get("enable_dynamic_chunked_prefill") is True
        and result.get("preemption_policy") == expected_policy
        and result.get("enable_decode_kv_reservation")
        is expected_decode_reservation
        and result.get("cuda_available") is True
    )
    preemption_observed = (
        metrics.get("preemption_count", 0) > 0
        and metrics.get("preempted_token_progress", 0) > 0
        and metrics.get("max_preempted_token_progress", 0) > 0
        and metrics.get("reclaimed_kv_blocks", 0) > 0
    )
    reservation_stops = metrics.get(
        "prefill_stopped_by_decode_kv_reservation", 0
    )
    reservation_observed = (
        isinstance(reservation_stops, int)
        and not isinstance(reservation_stops, bool)
        and reservation_stops > 0
    )
    completion_valid = (
        result.get("injected") is True
        and result.get("expected_requests") == expected_requests
        and result.get("finished_requests") == expected_requests
        and result.get("execution_validation", {}).get("valid") is True
        and result.get("generation_validation", {}).get("valid") is True
    )
    latency_metric_names = (
        "p50_ttft_s",
        "p95_ttft_s",
        "p99_ttft_s",
        "p50_tpot_s",
        "p95_tpot_s",
        "p99_tpot_s",
        "p50_request_latency_s",
        "p95_request_latency_s",
        "p99_request_latency_s",
    )
    latency_metrics_valid = all(
        isinstance(metrics.get(name), (int, float)) and metrics[name] >= 0
        for name in latency_metric_names
    )
    return {
        "valid": (
            configuration_valid
            and (preemption_observed or reservation_observed)
            and completion_valid
            and latency_metrics_valid
        ),
        "configuration_valid": configuration_valid,
        "preemption_observed": preemption_observed,
        "decode_kv_reservation_enabled": expected_decode_reservation,
        "decode_kv_reservation_observed": reservation_observed,
        "prefill_stopped_by_decode_kv_reservation": reservation_stops,
        "completion_valid": completion_valid,
        "latency_metrics_valid": latency_metrics_valid,
        "preemption_count": metrics.get("preemption_count"),
        "waiting_prefill_preemptions": metrics.get(
            "waiting_prefill_preemptions"
        ),
        "preempted_token_progress": metrics.get("preempted_token_progress"),
        "max_preempted_token_progress": metrics.get(
            "max_preempted_token_progress"
        ),
        "reclaimed_kv_blocks": metrics.get("reclaimed_kv_blocks"),
        "total_time_s": result.get("total_time_s"),
        "peak_torch_allocated_mib": result.get("peak_torch_allocated_mib"),
        "step_count": result.get("step_count"),
        "generated_token_ids_digest": result.get("generated_token_ids", {}).get(
            "digest"
        ),
        "avg_ttft_s": metrics.get("avg_ttft_s"),
        "p50_ttft_s": metrics.get("p50_ttft_s"),
        "p95_ttft_s": metrics.get("p95_ttft_s"),
        "p99_ttft_s": metrics.get("p99_ttft_s"),
        "max_ttft_s": metrics.get("max_ttft_s"),
        "avg_tpot_s": metrics.get("avg_tpot_s"),
        "p50_tpot_s": metrics.get("p50_tpot_s"),
        "p95_tpot_s": metrics.get("p95_tpot_s"),
        "p99_tpot_s": metrics.get("p99_tpot_s"),
        "max_tpot_s": metrics.get("max_tpot_s"),
        "avg_request_latency_s": metrics.get("avg_request_latency_s"),
        "p50_request_latency_s": metrics.get("p50_request_latency_s"),
        "p95_request_latency_s": metrics.get("p95_request_latency_s"),
        "p99_request_latency_s": metrics.get("p99_request_latency_s"),
        "max_request_latency_s": metrics.get("max_request_latency_s"),
    }


def summarize_kv_pressure_repeats(
    results: list[dict],
    *,
    expected_tp_size: int,
    expected_policy: str,
    expected_decode_reservation: bool = False,
) -> dict:
    if not results:
        raise ValueError("KV-pressure benchmark has no repeat results")
    rows = [
        summarize_kv_pressure_case(
            result,
            expected_tp_size=expected_tp_size,
            expected_policy=expected_policy,
            expected_decode_reservation=expected_decode_reservation,
        )
        for result in results
    ]
    stable_names = (
        "preemption_count",
        "preempted_token_progress",
        "max_preempted_token_progress",
        "reclaimed_kv_blocks",
        "step_count",
        "prefill_stopped_by_decode_kv_reservation",
    )
    counters_stable = all(
        len({row[name] for row in rows}) == 1 for name in stable_names
    )
    digests = {row["generated_token_ids_digest"] for row in rows}
    output_parity = None not in digests and len(digests) == 1
    commits = {result.get("commit") for result in results}
    checkpoint_digests = {
        result.get("checkpoint_manifest", {}).get("digest")
        for result in results
    }
    implementation_stable = None not in commits and len(commits) == 1
    checkpoint_stable = (
        None not in checkpoint_digests and len(checkpoint_digests) == 1
    )
    elapsed_samples = [row["total_time_s"] for row in rows]
    memory_samples = [row["peak_torch_allocated_mib"] for row in rows]
    measurements_valid = all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in (*elapsed_samples, *memory_samples)
    )
    elapsed_mean = (
        statistics.mean(elapsed_samples) if measurements_valid else math.nan
    )
    elapsed_cv = (
        statistics.pstdev(elapsed_samples) / elapsed_mean
        if elapsed_mean > 0
        else math.inf
    )
    latency_names = (
        "avg_ttft_s",
        "p50_ttft_s",
        "p95_ttft_s",
        "p99_ttft_s",
        "max_ttft_s",
        "avg_tpot_s",
        "p50_tpot_s",
        "p95_tpot_s",
        "p99_tpot_s",
        "max_tpot_s",
        "avg_request_latency_s",
        "p50_request_latency_s",
        "p95_request_latency_s",
        "p99_request_latency_s",
        "max_request_latency_s",
    )
    return {
        "valid": (
            len(rows) >= 2
            and all(row["valid"] for row in rows)
            and counters_stable
            and output_parity
            and implementation_stable
            and checkpoint_stable
            and measurements_valid
            and elapsed_cv <= PRESSURE_MAX_COEFFICIENT_OF_VARIATION
        ),
        "repeat_count": len(rows),
        "configuration_valid": all(
            row["configuration_valid"] for row in rows
        ),
        "preemption_observed": all(
            row["preemption_observed"] for row in rows
        ),
        "decode_kv_reservation_enabled": expected_decode_reservation,
        "decode_kv_reservation_observed": all(
            row["decode_kv_reservation_observed"] for row in rows
        ),
        "completion_valid": all(row["completion_valid"] for row in rows),
        "latency_metrics_valid": all(
            row["latency_metrics_valid"] for row in rows
        ),
        "scheduler_counters_stable": counters_stable,
        "output_parity": output_parity,
        "implementation_stable": implementation_stable,
        "checkpoint_stable": checkpoint_stable,
        "measurements_valid": measurements_valid,
        "commit": next(iter(commits)) if implementation_stable else None,
        "checkpoint_digest": (
            next(iter(checkpoint_digests)) if checkpoint_stable else None
        ),
        **{
            name: rows[0][name] if counters_stable else None
            for name in stable_names
        },
        "total_time_s": (
            statistics.median(elapsed_samples) if measurements_valid else None
        ),
        "total_time_cv": elapsed_cv,
        "max_allowed_cv": PRESSURE_MAX_COEFFICIENT_OF_VARIATION,
        "peak_torch_allocated_mib": (
            max(memory_samples) if measurements_valid else None
        ),
        "generated_token_ids_digest": (
            next(iter(digests)) if output_parity else None
        ),
        **{
            name: statistics.median(row[name] for row in rows)
            for name in latency_names
        },
        "runs": rows,
    }


def summarize_fairness_repeats(
    results: list[dict],
    *,
    expected_tp_size: int,
    mode: str,
) -> dict:
    """Validate one scheduler-fairness mode and aggregate repeated runs."""

    if not results:
        raise ValueError("scheduler-fairness benchmark has no repeat results")
    expected_threshold = 0 if mode == "disabled" else FAIRNESS_THRESHOLD
    numeric_fields = (
        "injected_p95_ttft_s",
        "initial_p95_decode_gap_s",
        "initial_max_decode_gap_s",
        "output_throughput_tok_s",
    )
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        values = {key: result.get(key) for key in numeric_fields}
        values.update(
            prefill_starved_steps=metrics.get("prefill_starved_steps"),
            max_prefill_starvation_steps=metrics.get(
                "max_prefill_starvation_steps"
            ),
            p95_tpot_s=metrics.get("p95_tpot_s"),
        )
        measurements_valid = all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in values.values()
        )
        configuration_valid = (
            result.get("tensor_parallel_size") == expected_tp_size
            and result.get("initial_seqs") == FAIRNESS_INITIAL_SEQUENCES
            and result.get("injected_seqs") == FAIRNESS_INJECTED_SEQUENCES
            and result.get("initial_input_len") == FAIRNESS_INITIAL_INPUT_LENGTH
            and result.get("injected_input_len") == FAIRNESS_INJECTED_INPUT_LENGTH
            and result.get("output_len") == FAIRNESS_OUTPUT_LENGTH
            and result.get("inject_after_decode_steps")
            == FAIRNESS_INJECT_AFTER_DECODE_STEPS
            and result.get("max_num_batched_tokens")
            == FAIRNESS_MAX_BATCHED_TOKENS
            and result.get("max_num_seqs")
            == FAIRNESS_INITIAL_SEQUENCES + FAIRNESS_INJECTED_SEQUENCES
            and result.get("prefill_starvation_threshold")
            == expected_threshold
            and result.get("prefill_starvation_token_budget")
            == FAIRNESS_TOKEN_BUDGET
            and result.get("enable_dynamic_chunked_prefill") is True
            and result.get("qwen35_moe_decode_backend") == "batched"
            and result.get("injected_ttft_count")
            == FAIRNESS_INJECTED_SEQUENCES
        )
        model_paths = result.get("execution_stats", {}).get(
            "model_path_counts", {}
        )
        mode_path_valid = (
            model_paths.get("mixed_eager", 0) == 0
            and model_paths.get("prefill_eager", 0) > 0
            and (
                model_paths.get("decode_eager", 0) > 0
                or model_paths.get("decode_cuda_graph", 0) > 0
            )
            if mode == "disabled"
            else model_paths.get("mixed_eager", 0) > 0
        )
        execution_valid = (
            result.get("execution_validation", {}).get("valid") is True
            and result.get("generation_validation", {}).get("valid") is True
            and mode_path_valid
            and result.get("cuda_available") is True
        )
        rows.append(
            {
                "valid": configuration_valid and execution_valid and measurements_valid,
                "configuration_valid": configuration_valid,
                "execution_valid": execution_valid,
                "measurements_valid": measurements_valid,
                "generated_token_ids_digest": result.get(
                    "generated_token_ids", {}
                ).get("digest"),
                **values,
            }
        )
    digests = {row["generated_token_ids_digest"] for row in rows}
    output_parity = None not in digests and len(digests) == 1
    commits = {result.get("commit") for result in results}
    checkpoint_digests = {
        result.get("checkpoint_manifest", {}).get("digest")
        for result in results
    }
    implementation_stable = None not in commits and len(commits) == 1
    checkpoint_stable = (
        None not in checkpoint_digests and len(checkpoint_digests) == 1
    )
    summary = {
        "valid": (
            len(rows) >= 2
            and all(row["valid"] for row in rows)
            and output_parity
            and implementation_stable
            and checkpoint_stable
        ),
        "repeat_count": len(rows),
        "output_parity": output_parity,
        "implementation_stable": implementation_stable,
        "checkpoint_stable": checkpoint_stable,
        "commit": next(iter(commits)) if implementation_stable else None,
        "checkpoint_digest": (
            next(iter(checkpoint_digests)) if checkpoint_stable else None
        ),
        "generated_token_ids_digest": next(iter(digests)) if output_parity else None,
        "runs": rows,
    }
    for key in (
        *numeric_fields,
        "prefill_starved_steps",
        "max_prefill_starvation_steps",
        "p95_tpot_s",
    ):
        summary[f"median_{key}"] = statistics.median(row[key] for row in rows)
    return summary


def compare_fairness_modes(disabled: dict, enabled: dict) -> dict:
    """Gate causal fairness evidence while reporting decode cost separately."""

    output_parity = (
        disabled.get("generated_token_ids_digest") is not None
        and disabled.get("generated_token_ids_digest")
        == enabled.get("generated_token_ids_digest")
    )
    same_implementation = (
        disabled.get("commit") is not None
        and disabled.get("commit") == enabled.get("commit")
    )
    same_checkpoint = (
        disabled.get("checkpoint_digest") is not None
        and disabled.get("checkpoint_digest")
        == enabled.get("checkpoint_digest")
    )
    starvation_improved = (
        enabled["median_max_prefill_starvation_steps"]
        < disabled["median_max_prefill_starvation_steps"]
    )
    injected_ttft_improved = (
        enabled["median_injected_p95_ttft_s"]
        < disabled["median_injected_p95_ttft_s"]
    )

    def ratio(candidate: float, baseline: float) -> float | None:
        return candidate / baseline if baseline > 0 else None

    return {
        "valid": (
            disabled["valid"]
            and enabled["valid"]
            and disabled["repeat_count"] == enabled["repeat_count"]
            and output_parity
            and same_implementation
            and same_checkpoint
            and starvation_improved
            and injected_ttft_improved
        ),
        "output_parity": output_parity,
        "same_implementation": same_implementation,
        "same_checkpoint": same_checkpoint,
        "starvation_improved": starvation_improved,
        "injected_ttft_improved": injected_ttft_improved,
        "injected_p95_ttft_ratio": ratio(
            enabled["median_injected_p95_ttft_s"],
            disabled["median_injected_p95_ttft_s"],
        ),
        "initial_p95_decode_gap_ratio": ratio(
            enabled["median_initial_p95_decode_gap_s"],
            disabled["median_initial_p95_decode_gap_s"],
        ),
        "initial_max_decode_gap_ratio": ratio(
            enabled["median_initial_max_decode_gap_s"],
            disabled["median_initial_max_decode_gap_s"],
        ),
        "p95_tpot_ratio": ratio(
            enabled["median_p95_tpot_s"],
            disabled["median_p95_tpot_s"],
        ),
        "throughput_ratio": ratio(
            enabled["median_output_throughput_tok_s"],
            disabled["median_output_throughput_tok_s"],
        ),
        "performance_ratios_are_observations_not_acceptance_gates": True,
    }


def summarize_memory_preflight(report: dict) -> dict[str, dict]:
    summaries = {}
    for tp_name, item in sorted(report.get("results", {}).items()):
        kv_sizes = item.get("kv_bytes_per_token_by_dtype", {})
        if set(kv_sizes) != {"auto", "int8"} or not all(
            isinstance(value, int) and value > 0
            for value in kv_sizes.values()
        ):
            raise ValueError(
                f"memory preflight has no complete KV dtype sizes for {tp_name}"
            )
        auto_bytes = kv_sizes["auto"]
        int8_bytes = kv_sizes["int8"]
        if int8_bytes >= auto_bytes:
            raise ValueError(f"INT8 KV cache does not reduce memory for {tp_name}")
        capacities = item.get("kv_capacity_by_dtype", {})
        capacity_fields = (
            "memory_limited_total_token_slots",
            "memory_limited_context_tokens_per_sequence",
            "effective_context_tokens_per_sequence",
        )
        if set(capacities) != set(kv_sizes) or not all(
            isinstance(capacity.get(field), int) and capacity[field] >= 0
            for capacity in capacities.values()
            for field in capacity_fields
        ):
            raise ValueError(f"memory preflight has invalid KV capacity for {tp_name}")
        if any(
            capacities["int8"][field] < capacities["auto"][field]
            for field in capacity_fields
        ):
            raise ValueError(f"INT8 KV capacity is smaller than auto for {tp_name}")
        concurrent_sequences = item.get("capacity_concurrent_sequences")
        if not isinstance(concurrent_sequences, int) or concurrent_sequences <= 0:
            raise ValueError(
                f"memory preflight has invalid capacity concurrency for {tp_name}"
            )
        transfer_per_rank = item.get(
            "pd_transfer_bytes_per_sequence_by_dtype",
            {},
        )
        transfer_all_ranks = item.get(
            "pd_transfer_bytes_all_tp_ranks_by_dtype",
            {},
        )
        state_dtypes = {"float32", "model"}
        if (
            set(transfer_per_rank) != set(kv_sizes)
            or set(transfer_all_ranks) != set(kv_sizes)
            or any(
                set(by_state_dtype) != state_dtypes
                or not all(
                    isinstance(value, int) and value > 0
                    for value in by_state_dtype.values()
                )
                for transfer in (transfer_per_rank, transfer_all_ranks)
                for by_state_dtype in transfer.values()
            )
        ):
            raise ValueError(
                f"memory preflight has invalid PD transfer sizes for {tp_name}"
            )
        budgets = item.get("available_budget_bytes_by_rank", [])
        if not budgets:
            raise ValueError(f"memory preflight has no rank budgets for {tp_name}")
        required = item["required_free_bytes_per_rank"]
        minimum_budget = min(budgets)
        summaries[tp_name] = {
            "local_parameter_bytes": item["local_parameter_bytes"],
            "max_state_bytes_per_rank": item["max_state_bytes_per_rank"],
            "state_bytes_per_rank_by_dtype": item.get(
                "state_bytes_per_rank_by_dtype"
            ),
            "rotary_cache_bytes_per_rank": item.get(
                "rotary_cache_bytes_per_rank"
            ),
            "minimum_workload_kv_bytes_per_rank": item[
                "minimum_workload_kv_bytes_per_rank"
            ],
            "kv_bytes_per_token_by_dtype": kv_sizes,
            "pd_transfer_bytes_per_sequence_by_dtype": transfer_per_rank,
            "pd_transfer_bytes_all_tp_ranks_by_dtype": transfer_all_ranks,
            "int8_kv_reduction_ratio": 1.0 - int8_bytes / auto_bytes,
            "kv_capacity_by_dtype": capacities,
            "capacity_concurrent_sequences": concurrent_sequences,
            "model_max_position_embeddings": item[
                "model_max_position_embeddings"
            ],
            "configured_max_model_len": item["configured_max_model_len"],
            "required_free_bytes_per_rank": required,
            "minimum_available_budget_bytes_per_rank": minimum_budget,
            "minimum_budget_margin_bytes": minimum_budget - required,
        }
    if not summaries:
        raise ValueError("memory preflight contains no TP results")
    return summaries


def summarize_pd_transfer(
    result: dict,
    *,
    expected_tp_size: int,
    expected_kv_dtype: str,
    expected_state_dtype: str,
    expected_components: dict,
) -> dict:
    profile = result.get("profile", {})
    workload = result.get("workload", {})
    measurements = result.get("results", {})
    receive_pool = measurements.get("receiver_host_staging_pool", {})
    cuda_install = result.get("cuda_install", {})
    samples = measurements.get("latency_ms_samples", [])
    repeats = workload.get("repeats")
    components = workload.get("components_bytes")
    reference_install = cuda_install.get("reference_full_payload_staging", {})
    candidate_install = cuda_install.get("candidate_direct_block_install", {})
    reference_install_samples = reference_install.get("latency_ms_samples", [])
    candidate_install_samples = candidate_install.get("latency_ms_samples", [])
    expected_receive_reuse = (
        repeats + workload.get("warmup") - 1
        if isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and isinstance(workload.get("warmup"), int)
        and not isinstance(workload.get("warmup"), bool)
        else None
    )
    receive_pool_valid = (
        receive_pool.get("valid") is True
        and receive_pool.get("allocation_count") == 1
        and receive_pool.get("reuse_count") == expected_receive_reuse
        and receive_pool.get("expected_reuse_count") == expected_receive_reuse
        and receive_pool.get("transient_allocation_count") == 0
        and receive_pool.get("leased") == 0
        and isinstance(receive_pool.get("storage_bytes"), int)
        and receive_pool["storage_bytes"] >= expected_components["total"]
    )
    install_valid = (
        cuda_install.get("enabled") is True
        and cuda_install.get("valid") is True
        and cuda_install.get("measured_on_cuda") is True
        and cuda_install.get("avoids_full_payload_device_conversion") is True
        and isinstance(repeats, int)
        and len(reference_install_samples) == repeats
        and len(candidate_install_samples) == repeats
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            for value in (*reference_install_samples, *candidate_install_samples)
        )
        and cuda_install.get("peak_device_bytes_reduction", 0) > 0
        and 0 < cuda_install.get("latency_ratio_vs_reference", math.inf)
        <= PD_INSTALL_MAX_LATENCY_RATIO
    )
    valid = (
        result.get("schema_version") == 1
        and result.get("scope")
        == "single-rank synchronous TCP loopback correctness baseline"
        and profile.get("tp_size") == expected_tp_size
        and profile.get("kv_dtype") == expected_kv_dtype
        and profile.get("state_dtype") == expected_state_dtype
        and components == expected_components
        and isinstance(repeats, int)
        and repeats >= 2
        and isinstance(samples, list)
        and len(samples) == repeats
        and all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            for value in samples
        )
        and isinstance(measurements.get("latency_ms_p50"), (int, float))
        and measurements["latency_ms_p50"] > 0
        and isinstance(measurements.get("latency_ms_p95"), (int, float))
        and measurements["latency_ms_p95"] > 0
        and isinstance(
            measurements.get("effective_payload_gib_s_p50"),
            (int, float),
        )
        and measurements["effective_payload_gib_s_p50"] > 0
        and workload.get("receiver_ack_bytes") == 1
        and isinstance(workload.get("payload_tensor_count"), int)
        and workload["payload_tensor_count"] > 1
        and measurements.get("receiver_storage_count") == 1
        and measurements.get("receiver_storage_coalesced") is True
        and receive_pool_valid
        and workload.get("payload_frame_bytes_sent", 0)
        > expected_components["total"]
        and install_valid
    )
    return {
        "valid": valid,
        "profile": profile,
        "components_bytes": components,
        "repeat_count": len(samples) if isinstance(samples, list) else 0,
        "latency_ms_p50": measurements.get("latency_ms_p50"),
        "latency_ms_p95": measurements.get("latency_ms_p95"),
        "effective_payload_gib_s_p50": measurements.get(
            "effective_payload_gib_s_p50"
        ),
        "payload_tensor_count": workload.get("payload_tensor_count"),
        "receiver_storage_count": measurements.get("receiver_storage_count"),
        "receiver_storage_coalesced": measurements.get(
            "receiver_storage_coalesced"
        ),
        "receiver_host_staging_pool": {
            **receive_pool,
            "valid": receive_pool_valid,
        },
        "cuda_install": {
            "valid": install_valid,
            "peak_device_bytes_reduction": cuda_install.get(
                "peak_device_bytes_reduction"
            ),
            "latency_ratio_vs_reference": cuda_install.get(
                "latency_ratio_vs_reference"
            ),
            "max_latency_ratio": PD_INSTALL_MAX_LATENCY_RATIO,
        },
        "limitations": result.get("limitations", []),
    }


def summarize_pd_export(
    result: dict,
    *,
    expected_tp_size: int,
    expected_kv_dtype: str,
    expected_state_dtype: str,
    expected_components: dict,
    expected_allocated_tokens: int,
    expected_cached_tokens: int,
) -> dict:
    profile = result.get("profile", {})
    reference = result.get("reference_gpu_gather_then_host_copy", {})
    candidate = result.get("candidate_direct_host_staging", {})
    candidate_layout = candidate.get("host_layout", {})
    staging_pool = candidate.get("host_staging_pool", {})
    repeats = profile.get("repeats")
    warmup = profile.get("warmup")
    expected_reuse_count = (
        warmup + repeats - 1
        if isinstance(warmup, int)
        and not isinstance(warmup, bool)
        and warmup >= 0
        and isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and repeats > 0
        else None
    )
    staging_pool_valid = (
        staging_pool.get("valid") is True
        and staging_pool.get("allocation_count") == 1
        and staging_pool.get("reuse_count") == expected_reuse_count
        and staging_pool.get("expected_reuse_count") == expected_reuse_count
        and staging_pool.get("transient_allocation_count") == 0
        and staging_pool.get("leased") == 0
        and isinstance(staging_pool.get("storage_bytes"), int)
        and staging_pool["storage_bytes"] >= expected_components["total"]
    )

    def measurements_valid(item: dict) -> bool:
        latency = item.get("latency_ms_samples", [])
        peaks = item.get("peak_extra_device_bytes_samples", [])
        return (
            isinstance(repeats, int)
            and repeats >= 2
            and isinstance(latency, list)
            and len(latency) == repeats
            and all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 0
                for value in latency
            )
            and isinstance(peaks, list)
            and len(peaks) == repeats
            and all(isinstance(value, int) and value >= 0 for value in peaks)
            and item.get("peak_extra_device_bytes_max") == max(peaks)
            and item.get("latency_ms_p50") == statistics.median(latency)
        )

    valid = (
        result.get("schema_version") == 1
        and result.get("scope")
        == "single-rank Qwen3.6 GPU-to-host cache export"
        and profile.get("tp_size") == expected_tp_size
        and profile.get("kv_dtype") == expected_kv_dtype
        and profile.get("state_dtype") == expected_state_dtype
        and profile.get("components") == expected_components
        and profile.get("allocated_tokens") == expected_allocated_tokens
        and profile.get("cached_tokens") == expected_cached_tokens
        and isinstance(result.get("environment", {}).get("device"), str)
        and bool(result["environment"]["device"])
        and result.get("correctness", {}).get("candidate_matches_reference")
        is True
        and measurements_valid(reference)
        and measurements_valid(candidate)
        and isinstance(candidate_layout.get("tensor_count"), int)
        and candidate_layout["tensor_count"] > 1
        and candidate_layout.get("storage_count") == 1
        and candidate_layout.get("all_cpu") is True
        and candidate_layout.get("all_pinned") is True
        and staging_pool_valid
        and candidate["peak_extra_device_bytes_max"]
        < reference["peak_extra_device_bytes_max"]
    )
    return {
        "valid": valid,
        "profile": profile,
        "reference": reference,
        "candidate": candidate,
        "candidate_host_layout": candidate_layout,
        "host_staging_pool": {
            **staging_pool,
            "valid": staging_pool_valid,
        },
        "avoided_peak_device_bytes": (
            reference.get("peak_extra_device_bytes_max", 0)
            - candidate.get("peak_extra_device_bytes_max", 0)
        ),
        "limitations": result.get("limitations", []),
    }


def summarize_long_prefill(result: dict, *, expected_tp_size: int) -> dict:
    configuration = result.get("configuration", {})
    configuration_valid = (
        configuration.get("prefill_only") is True
        and configuration.get("prefill_batch") == 1
        and configuration.get("prefill_tokens") == LONG_PREFILL_TOKENS
        and configuration.get("tp_size") == expected_tp_size
        and str(configuration.get("resolved_device", "")).startswith("cuda")
    )
    cases = {}
    for name in (
        "vectorized_prefill_convolution",
        "grouped_delta_prefill",
    ):
        item = result.get("results", {}).get(name)
        if not isinstance(item, dict):
            raise ValueError(f"long-prefill benchmark is missing {name}")
        candidate = item.get("candidate", {})
        errors = item.get("errors", [])
        max_abs_error = max(
            (error.get("max_abs_error", math.inf) for error in errors),
            default=math.inf,
        )
        median_ms = candidate.get("median_ms", math.nan)
        peak_extra_mib = candidate.get("peak_extra_mib", math.nan)
        compact_state_valid = True
        if name == "vectorized_prefill_convolution":
            compact_state_valid = (
                item.get("next_state_reuses_input_storage") is True
                and item.get("compact_state_storage_mib", 0) > 0
                and item.get("reused_prefill_state_mib")
                == item.get("compact_state_storage_mib")
                and item.get("released_history_storage_mib", 0) > 0
            )
        valid = (
            bool(errors)
            and math.isfinite(max_abs_error)
            and max_abs_error <= LONG_PREFILL_MAX_ABS_ERROR
            and math.isfinite(median_ms)
            and median_ms > 0
            and math.isfinite(peak_extra_mib)
            and peak_extra_mib >= 0
            and compact_state_valid
        )
        cases[name] = {
            "valid": valid,
            "median_ms": median_ms,
            "peak_extra_mib": peak_extra_mib,
            "max_abs_error": max_abs_error,
            "compact_state_valid": compact_state_valid,
            "compact_state_storage_mib": item.get(
                "compact_state_storage_mib"
            ),
            "reused_prefill_state_mib": item.get(
                "reused_prefill_state_mib"
            ),
            "released_history_storage_mib": item.get(
                "released_history_storage_mib"
            ),
        }
        if name == "grouped_delta_prefill":
            reused_mib = item.get("reused_fp32_output_buffer_mib")
            reused_pairwise_mib = item.get(
                "reused_fp32_pairwise_buffer_mib"
            )
            cases[name]["reused_fp32_output_buffer_mib"] = reused_mib
            cases[name]["reused_fp32_pairwise_buffer_mib"] = (
                reused_pairwise_mib
            )
            cases[name]["buffer_reuse_evidence_valid"] = (
                isinstance(reused_mib, (int, float)) and reused_mib > 0
                and isinstance(reused_pairwise_mib, (int, float))
                and reused_pairwise_mib > 0
            )
            cases[name]["valid"] = (
                cases[name]["valid"]
                and cases[name]["buffer_reuse_evidence_valid"]
            )
    configured_chunk_sizes = configuration.get("delta_prefill_chunk_sizes", [])
    sweep_result = result.get("results", {}).get(
        "grouped_delta_prefill_chunk_sweep",
        {},
    )
    sweep = sweep_result.get("candidates", {})
    expected_chunk_names = {str(size) for size in configured_chunk_sizes}
    if (
        sweep_result.get("baseline_chunk_size") != 64
        or "64" not in expected_chunk_names
        or set(sweep) != expected_chunk_names
    ):
        raise ValueError("long-prefill chunk sweep coverage is incomplete")
    chunk_candidates = {}
    for chunk_name, item in sweep.items():
        candidate = item.get("candidate", {})
        errors = item.get("errors_vs_chunk64", [])
        max_abs_error = max(
            (error.get("max_abs_error", math.inf) for error in errors),
            default=math.inf,
        )
        median_ms = candidate.get("median_ms", math.nan)
        peak_extra_mib = candidate.get("peak_extra_mib", math.nan)
        valid = (
            item.get("chunk_size") == int(chunk_name)
            and bool(errors)
            and math.isfinite(max_abs_error)
            and max_abs_error <= LONG_PREFILL_MAX_ABS_ERROR
            and math.isfinite(median_ms)
            and median_ms > 0
            and math.isfinite(peak_extra_mib)
            and peak_extra_mib >= 0
        )
        chunk_candidates[chunk_name] = {
            "valid": valid,
            "median_ms": median_ms,
            "peak_extra_mib": peak_extra_mib,
            "max_abs_error": max_abs_error,
        }
    valid_chunks = {
        name: item for name, item in chunk_candidates.items() if item["valid"]
    }
    fastest = (
        min(valid_chunks, key=lambda name: valid_chunks[name]["median_ms"])
        if valid_chunks
        else None
    )
    lowest_memory = (
        min(
            valid_chunks,
            key=lambda name: valid_chunks[name]["peak_extra_mib"],
        )
        if valid_chunks
        else None
    )
    sweep_valid = len(valid_chunks) == len(chunk_candidates)
    return {
        "valid": (
            configuration_valid
            and all(item["valid"] for item in cases.values())
            and sweep_valid
        ),
        "configuration_valid": configuration_valid,
        "prefill_tokens": configuration.get("prefill_tokens"),
        "max_allowed_abs_error": LONG_PREFILL_MAX_ABS_ERROR,
        "cases": cases,
        "chunk_sweep": {
            "valid": sweep_valid,
            "baseline_chunk_size": 64,
            "fastest_chunk_size": int(fastest) if fastest is not None else None,
            "lowest_memory_chunk_size": (
                int(lowest_memory) if lowest_memory is not None else None
            ),
            "candidates": chunk_candidates,
        },
    }


def summarize(run_dir: Path, run_id: str) -> dict:
    official_audit = load_json(
        run_dir / "preflight" / "official_checkpoint_header_audit.json"
    )
    audit = load_json(run_dir / "preflight" / "checkpoint_mapping_audit.json")
    memory = load_json(run_dir / "preflight" / "memory_preflight.json")
    performance = load_json(
        run_dir / "performance" / f"{run_id}_matrix_summary.json"
    )
    quality = load_json(run_dir / "quality" / f"{run_id}_summary.json")
    gptq = summarize_optional_gptq(run_dir, run_id)
    fp8 = summarize_optional_fp8_audit(run_dir, run_id, performance.get("runs"))
    kernel_paths = sorted((run_dir / "kernels").glob("tp*.json"))
    if not kernel_paths:
        raise ValueError("no kernel benchmark artifacts were found")
    long_prefill_paths = sorted((run_dir / "kernels_long").glob("tp*.json"))
    if not long_prefill_paths:
        raise ValueError("no long-prefill kernel artifacts were found")
    mixed_paths = sorted((run_dir / "mixed").glob("tp*/r*.json"))
    if not mixed_paths:
        raise ValueError("no mixed-workload benchmark artifacts were found")
    pressure_paths = sorted((run_dir / "pressure").glob("tp*/*/r*.json"))
    if not pressure_paths:
        raise ValueError("no KV-pressure benchmark artifacts were found")
    fairness_paths = sorted((run_dir / "fairness").glob("*/tp*/r*.json"))
    if not fairness_paths:
        raise ValueError("no scheduler-fairness benchmark artifacts were found")
    cudagraph_paths = sorted(
        (run_dir / "cudagraph").glob("tp*/*/run_*/summary.json")
    )
    if not cudagraph_paths:
        raise ValueError("no CUDA Graph parity artifacts were found")

    valid_runs = [
        row
        for row in performance["runs"]
        if row["repeat_output_digests_match"]
        and row["execution_paths_valid"]
        and row["generation_valid"]
    ]
    if not valid_runs:
        raise ValueError("performance matrix contains no valid configurations")
    best_throughput = max(
        valid_runs,
        key=lambda row: row["median"]["output_throughput_tok_s"],
    )
    lowest_memory = min(
        valid_runs,
        key=lambda row: row["median"]["peak_torch_allocated_mib"],
    )
    moe_runtime = summarize_moe_runtime(performance["runs"])
    recurrent_state_access = summarize_recurrent_state_access(
        performance["runs"]
    )

    kernels = {}
    normalization = {}
    buffer_reuse = {}
    mixed_moe_dispatch = {}
    moe_weight_buffer_reuse = {}
    moe_route_input_broadcast = {}
    moe_device_scalar = {}
    long_prefill = {}
    mixed_runs = {}
    kv_pressure = {}
    scheduler_fairness = {}
    configured_max_decode_batch = performance["workload"]["max_num_seqs"]
    commits = {
        row["commit"]
        for row in performance["runs"]
    }
    clean_worktrees = True
    cuda_measurements = True
    expected_tp_names = {
        f"tp{row['tensor_parallel_size']}" for row in performance["runs"]
    }
    official_checkpoint_valid = (
        official_audit.get("valid") is True
        and official_audit.get("repo") == OFFICIAL_CHECKPOINT_REPO
        and official_audit.get("resolved_revision")
        == OFFICIAL_CHECKPOINT_REVISION
        and official_audit.get("config_sha256") == OFFICIAL_CONFIG_SHA256
        and official_audit.get("index_sha256") == OFFICIAL_INDEX_SHA256
        and official_audit.get("headers_sha256") == OFFICIAL_HEADERS_SHA256
        and checkpoint_semantic_contract_matches(official_audit, "bf16")
        and set(official_audit.get("results", {})) == expected_tp_names
        and all(
            result.get("valid") is True
            and result.get("skipped_by_prefix")
            == OFFICIAL_SKIPPED_WEIGHT_PREFIXES
            and not result.get("unclassified_skipped_weights")
            for result in official_audit.get("results", {}).values()
        )
    )
    local_checkpoint_manifest = audit.get("checkpoint_manifest", {})
    local_checkpoint_identity_valid = checkpoint_manifest_matches_remote(
        local_checkpoint_manifest,
        official_audit,
    )
    memory_by_tp = summarize_memory_preflight(memory)
    pd_transfer = {}
    pd_export = {}
    for tp_name in sorted(expected_tp_names):
        tp_size = int(tp_name.removeprefix("tp"))
        expected_profiles = (
            ("auto", "float32"),
            ("int8", "model"),
        )
        for kv_dtype, state_dtype in expected_profiles:
            profile_name = f"{kv_dtype}-{state_dtype}"
            result = load_json(
                run_dir / "pd_transfer" / tp_name / f"{profile_name}.json"
            )
            expected_components = memory["results"][tp_name][
                "pd_transfer_components_per_sequence_by_dtype"
            ][kv_dtype][state_dtype]
            pd_transfer.setdefault(tp_name, {})[profile_name] = (
                summarize_pd_transfer(
                    result,
                    expected_tp_size=tp_size,
                    expected_kv_dtype=kv_dtype,
                    expected_state_dtype=state_dtype,
                    expected_components=expected_components,
                )
            )
            export_result = load_json(
                run_dir / "pd_export" / tp_name / f"{profile_name}.json"
            )
            pd_export.setdefault(tp_name, {})[profile_name] = (
                summarize_pd_export(
                    export_result,
                    expected_tp_size=tp_size,
                    expected_kv_dtype=kv_dtype,
                    expected_state_dtype=state_dtype,
                    expected_components=expected_components,
                    expected_allocated_tokens=memory["results"][tp_name][
                        "pd_transfer_allocated_tokens"
                    ],
                    expected_cached_tokens=memory["results"][tp_name][
                        "pd_transfer_context_tokens"
                    ],
                )
            )
    def rank_storage_matches(row, field, expected):
        tp_size = row["tensor_parallel_size"]
        ranks = row.get("storage", {}).get(
            "recurrent_state_storage_by_rank", []
        )
        return (
            len(ranks) == tp_size
            and {item.get("rank") for item in ranks} == set(range(tp_size))
            and all(item.get(field) == expected for item in ranks)
        )

    rotary_storage_matches_preflight = all(
        rank_storage_matches(
            row,
            "rotary_cache_bytes_local_rank",
            memory_by_tp.get(f"tp{row['tensor_parallel_size']}", {}).get(
                "rotary_cache_bytes_per_rank"
            ),
        )
        for row in performance["runs"]
    )
    recurrent_storage_matches_preflight = all(
        rank_storage_matches(
            row,
            "total_bytes_local_rank",
            memory_by_tp.get(f"tp{row['tensor_parallel_size']}", {})
            .get("state_bytes_per_rank_by_dtype", {})
            .get(row["recurrent_state_dtype"]),
        )
        for row in performance["runs"]
    )
    kv_storage_matches_preflight = all(
        len(row.get("storage", {}).get("kv_cache_storage_by_rank", []))
        == row["tensor_parallel_size"]
        and {
            item.get("rank")
            for item in row["storage"]["kv_cache_storage_by_rank"]
        }
        == set(range(row["tensor_parallel_size"]))
        and all(
            item.get("total_bytes")
            == row["storage"].get("num_kvcache_blocks", 0)
            * KVCACHE_BLOCK_SIZE
            * memory_by_tp.get(f"tp{row['tensor_parallel_size']}", {})
            .get("kv_bytes_per_token_by_dtype", {})
            .get(row["kv_cache_dtype"], 0)
            for item in row["storage"]["kv_cache_storage_by_rank"]
        )
        for row in performance["runs"]
    )
    attention = {}
    attention_valid = True
    for tp_name in sorted(expected_tp_names):
        tp_size = int(tp_name.removeprefix("tp"))
        cases = {}
        for case_name, context_len, batch_size, partitioned in (
            ("short", ATTENTION_SHORT_CONTEXT, 4, False),
            ("long", ATTENTION_LONG_CONTEXT, 4, True),
            ("max", ATTENTION_MAX_CONTEXT, 1, True),
        ):
            result = load_json(
                run_dir / "attention" / tp_name / f"{case_name}.json"
            )
            dimensions_valid = (
                result["context_len"] == context_len
                and result["batch_size"] == batch_size
                and result["num_heads"]
                == QWEN35_TOTAL_QUERY_HEADS // tp_size
                and result["num_kv_heads"] == QWEN35_KV_HEADS_PER_RANK
                and result["head_dim"] == QWEN35_HEAD_DIM
            )
            attention_valid = (
                attention_valid
                and dimensions_valid
                and result["cuda_available"]
            )
            cases[case_name] = {
                "dimensions_valid": dimensions_valid,
                **summarize_attention_case(result, partitioned=partitioned),
            }
            attention_valid = (
                attention_valid
                and cases[case_name]["fused_correctness_valid"]
                and cases[case_name]["partitioned_correctness_valid"]
                and cases[case_name]["partitioned_workspace_valid"]
                and cases[case_name]["workspace_reuse_measurement_valid"]
            )
            commits.add(result["commit"])
            clean_worktrees = clean_worktrees and not result["git_dirty"]
        attention[tp_name] = cases
    for path in kernel_paths:
        result = load_json(path)
        dispatch_results = result["results"]["expert_dispatch_torch"]
        measured_decode_batches = sorted(
            int(token_count)
            for token_count in dispatch_results
            if int(token_count) <= configured_max_decode_batch
        )
        if not measured_decode_batches or measured_decode_batches[0] != 1:
            raise ValueError(f"kernel benchmark has no batch-1 MoE result: {path}")
        selected_batches = tuple(
            dict.fromkeys((1, measured_decode_batches[-1]))
        )
        candidates_by_batch = {
            str(batch): dispatch_results[str(batch)][
                "graph_safe_batched_candidate"
            ]
            for batch in selected_batches
        }
        candidate = candidates_by_batch["1"]
        tp_name = path.stem
        moe_weight_buffer_reuse[tp_name] = summarize_moe_weight_buffer_reuse(
            candidate["weight_buffer_reuse"]
        )
        device_scalar = dispatch_results["1"].get("device_scalar_candidate")
        moe_device_scalar[tp_name] = (
            {
                "available": True,
                "promotion": device_scalar["promotion"],
                "median_ms": device_scalar["median_ms"],
                "speedup_vs_current": device_scalar["speedup_vs_current"],
                "peak_extra_mib": device_scalar["peak_extra_mib"],
                "errors_vs_current": device_scalar["errors_vs_current"],
                "avoids_host_route_sync": device_scalar[
                    "avoids_host_route_sync"
                ],
                "estimated_selected_weight_mib": device_scalar[
                    "estimated_selected_weight_mib"
                ],
            }
            if device_scalar is not None
            else {"available": False}
        )
        moe_route_input_broadcast[tp_name] = {
            batch: item["broadcast_route_input"]
            for batch, item in candidates_by_batch.items()
        }
        mixed_moe_dispatch[tp_name] = summarize_mixed_moe_dispatch_sweep(
            result["results"]["mixed_expert_dispatch"]
        )
        normalization[tp_name] = {
            "rmsnorm": summarize_normalization_candidate(
                result["results"]["rmsnorm_fp32_reuse"],
                "candidate_reuses_fp32_workspace",
                required_flags=("candidate_uses_precomputed_gain",),
            ),
            "gated_rmsnorm": summarize_normalization_candidate(
                result["results"]["gated_rmsnorm_fp32_reuse"],
                "candidate_reuses_fp32_workspaces",
            ),
        }
        buffer_reuse[tp_name] = {
            "sampling_unfiltered": summarize_buffer_reuse_candidate(
                result["results"]["sampling_filter_fast_paths"]["unfiltered"],
                ("avoided_full_sort_workspace_mib",),
                {"uses_host_sampling_metadata": True},
            ),
            "sampling_top_k": summarize_buffer_reuse_candidate(
                result["results"]["sampling_filter_fast_paths"]["top_k"],
                ("avoided_full_sort_workspace_mib",),
                {"uses_host_sampling_metadata": True},
            ),
            "sampling_top_p": summarize_buffer_reuse_candidate(
                result["results"]["sampling_filter_fast_paths"]["top_p"],
                (
                    "avoided_top_k_mask_workspace_mib",
                    "avoided_top_p_shift_clone_mib",
                ),
                {
                    "uses_host_sampling_metadata": True,
                    "eliminated_top_p_mask_clones_per_step": 1,
                },
            ),
            "sampling_top_k_top_p": summarize_buffer_reuse_candidate(
                result["results"]["sampling_filter_fast_paths"]["top_k_top_p"],
                (
                    "avoided_full_sort_workspace_mib",
                    "avoided_top_p_shift_clone_mib",
                ),
                {
                    "uses_host_sampling_metadata": True,
                    "eliminated_top_p_mask_clones_per_step": 1,
                },
            ),
            "sampling_compact_top_k": summarize_buffer_reuse_candidate(
                result["results"]["compact_top_k_sampling"],
                ("avoided_fp32_logits_mib",),
            ),
            "sampling_filter_output": summarize_buffer_reuse_candidate(
                result["results"]["sampling_filter_output_reuse"],
                ("avoided_fp32_logits_mib",),
                {
                    "eliminated_tensor_allocations_per_sampling_step": 2,
                    "candidate_reuses_temperature_and_filter_storage": True,
                },
            ),
            "gated_delta_packed_projection": summarize_buffer_reuse_candidate(
                result["results"]["gated_delta_packed_projection"],
                ("avoided_gemm_launches",),
                {
                    "reference_gemm_launches": 3,
                    "candidate_gemm_launches": 1,
                },
            ),
            "attention_packed_qkv": summarize_buffer_reuse_candidate(
                result["results"]["attention_packed_qkv"],
                ("avoided_gemm_launches",),
                {
                    "reference_gemm_launches": 3,
                    "candidate_gemm_launches": 1,
                },
            ),
            "contiguous_decode_state": summarize_buffer_reuse_candidate(
                result["results"]["contiguous_decode_state"],
                (
                    "avoided_state_gather_mib",
                    "avoided_state_scatter_mib",
                ),
                {"candidate_uses_cache_views": True},
            ),
            "sampling_greedy_precision": summarize_buffer_reuse_candidate(
                result["results"]["greedy_sampler_precision_fast_path"],
                ("avoided_fp32_logits_mib",),
                {"uses_host_sampling_metadata": True},
            ),
            "sampling_inputs": summarize_buffer_reuse_candidate(
                result["results"]["sampling_input_buffer_reuse"],
                (
                    "eliminated_tensor_allocations_per_step",
                    "persistent_sampling_input_mib",
                ),
                {"candidate_reuses_host_device_storage": True},
            ),
            "sampling_noise": summarize_buffer_reuse_candidate(
                result["results"]["sampling_noise_buffer_reuse"],
                ("reused_filtered_logits_mib",),
                {
                    "eliminated_tensor_allocations_per_sampling_step": 1,
                    "persistent_sampling_noise_mib": 0.0,
                    "candidate_reuses_filtered_logits_storage": True,
                },
            ),
            "packed_block_metadata": summarize_buffer_reuse_candidate(
                result["results"]["packed_block_metadata_buffer_reuse"],
                ("persistent_metadata_buffers_mib",),
                {
                    "eliminated_tensor_allocations_per_update": 4,
                    "candidate_reuses_two_isolated_buffer_banks": True,
                },
            ),
            "decode_convolution_state": summarize_buffer_reuse_candidate(
                result["results"]["decode_convolution_state_reuse"],
                (
                    "reused_convolution_state_mib",
                    "eliminated_weighted_state_temporary_mib",
                ),
                {"candidate_uses_inplace_channel_accumulation": True},
            ),
            "router_softmax": summarize_buffer_reuse_candidate(
                result["results"]["router_topk_first"],
                ("reused_selected_logits_mib",),
            ),
            "delta_l2_normalization": summarize_buffer_reuse_candidate(
                result["results"]["delta_l2_normalization_reuse"],
                ("reused_query_key_fp32_mib",),
            ),
            "delta_causal_mask": summarize_delta_causal_mask_cache(
                result["results"].get("delta_causal_mask_cache"),
                measured_on_cuda=result["cuda_available"],
            ),
            "attention_norm_output": summarize_buffer_reuse_candidate(
                result["results"]["attention_norm_output_reuse"],
                ("reused_projection_output_mib",),
            ),
            "rotary_output": summarize_buffer_reuse_candidate(
                result["results"]["rotary_output_reuse"],
                ("reused_query_key_output_mib",),
            ),
            "vocab_gather": summarize_buffer_reuse_candidate(
                result["results"]["vocab_gather_layout"],
                ("avoided_full_vocab_copy_mib",),
                {"candidate_returns_transpose_view": True},
            ),
            "moe_output_merge": summarize_buffer_reuse_candidate(
                result["results"]["moe_output_buffer_reuse"],
                (
                    "reused_routed_output_mib",
                    "reused_shared_output_mib",
                    "reused_gate_mib",
                ),
            ),
            "sorted_route_weighting": summarize_buffer_reuse_candidate(
                result["results"]["sorted_route_weighting_reuse"],
                ("avoided_weighted_expert_output_mib",),
            ),
            "batched_route_sum_output": summarize_buffer_reuse_candidate(
                result["results"]["batched_route_sum_output_reuse"],
                ("avoided_route_sum_output_mib",),
                {"candidate_reuses_dispatch_output": True},
            ),
            "residual_merge": summarize_buffer_reuse_candidate(
                result["results"]["residual_output_buffer_reuse"],
                ("reused_branch_output_mib_per_merge",),
                {"residual_merges_per_decoder_layer": 2},
            ),
            "torch_kv_dequant": summarize_buffer_reuse_candidate(
                result["results"]["torch_kv_dequant_buffer_reuse"],
                (
                    "avoided_output_workspace_mib",
                    "avoided_block_id_cast_mib",
                ),
            ),
            "recurrent_decode": summarize_buffer_reuse_candidate(
                result["results"]["specialized_delta_decode"],
                (
                    "reused_recurrent_state_mib",
                    "reused_prediction_workspace_mib",
                    "reused_decay_exp_mib",
                ),
                {"avoided_full_state_intermediates": 2},
            ),
            "recurrent_state_contraction": summarize_buffer_reuse_candidate(
                result["results"]["delta_state_contraction"],
                ("avoided_state_product_mib_per_contraction",),
                {"state_contractions_per_decode": 2},
            ),
            "recurrent_prefill": summarize_buffer_reuse_candidate(
                result["results"]["delta_prefill_state_reuse"],
                ("reused_recurrent_state_mib",),
            ),
            "gated_delta_beta": summarize_buffer_reuse_candidate(
                result["results"]["gated_delta_beta_buffer_reuse"],
                ("reused_beta_projection_mib",),
            ),
            "gated_delta_decay": summarize_buffer_reuse_candidate(
                result["results"]["gated_delta_decay_rate_precompute"],
                (
                    "precomputed_decay_rate_mib",
                    "reused_decay_projection_fp32_mib",
                    "reused_softplus_output_mib",
                ),
            ),
        }
        delta_prefill_decay = result["results"].get(
            "delta_prefill_decay_workspace_reuse"
        )
        if delta_prefill_decay is not None:
            buffer_reuse[tp_name]["delta_prefill_decay_workspace"] = (
                summarize_buffer_reuse_candidate(
                    delta_prefill_decay,
                    ("avoided_expanded_fp32_qk_mib",),
                    {"eliminated_expanded_qk_allocations": 2},
                )
            )
        kernels[tp_name] = {
            "promotion": {
                "promote_to_runtime": all(
                    item["promotion"]["promote_to_runtime"]
                    for item in candidates_by_batch.values()
                ),
                "selected_decode_batches": list(selected_batches),
            },
            "median_ms": candidate["median_ms"],
            "speedup_vs_current": candidate["speedup_vs_current"],
            "peak_extra_mib": candidate["peak_extra_mib"],
            "errors_vs_current": candidate["errors_vs_current"],
            "reused_weighted_route_mib": candidate[
                "reused_weighted_route_mib"
            ],
            "by_decode_batch": {
                batch: {
                    "promotion": item["promotion"],
                    "median_ms": item["median_ms"],
                    "speedup_vs_current": item["speedup_vs_current"],
                    "peak_extra_mib": item["peak_extra_mib"],
                    "errors_vs_current": item["errors_vs_current"],
                    "reused_weighted_route_mib": item[
                        "reused_weighted_route_mib"
                    ],
                }
                for batch, item in candidates_by_batch.items()
            },
            "chunk_recommendation": result["results"].get(
                "moe_decode_chunk_recommendation"
            ),
        }
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    for path in long_prefill_paths:
        result = load_json(path)
        tp_name = path.stem
        tp_size = int(tp_name.removeprefix("tp"))
        long_prefill[tp_name] = summarize_long_prefill(
            result,
            expected_tp_size=tp_size,
        )

    pressure_results = {}
    for path in pressure_paths:
        result = load_json(path)
        tp_size = int(path.parents[1].name.removeprefix("tp"))
        tp_name = f"tp{tp_size}"
        policy = path.parent.name
        pressure_results.setdefault((tp_name, policy), []).append(result)
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]
    for (tp_name, policy), results in sorted(pressure_results.items()):
        tp_size = int(tp_name.removeprefix("tp"))
        expected_policy = (
            "min_recompute"
            if policy == "min_recompute_reserved"
            else policy
        )
        kv_pressure.setdefault(tp_name, {})[policy] = (
            summarize_kv_pressure_repeats(
                results,
                expected_tp_size=tp_size,
                expected_policy=expected_policy,
                expected_decode_reservation=(
                    policy == "min_recompute_reserved"
                ),
            )
        )
    kv_pressure_comparisons = {}
    for tp_name, by_policy in kv_pressure.items():
        if set(by_policy) != {
            "fcfs",
            "min_recompute",
            "min_recompute_reserved",
        }:
            continue
        baseline = by_policy["fcfs"]
        candidate = by_policy["min_recompute"]
        reserved = by_policy["min_recompute_reserved"]
        output_parity = (
            baseline["generated_token_ids_digest"] is not None
            and baseline["generated_token_ids_digest"]
            == candidate["generated_token_ids_digest"]
        )
        baseline_progress = baseline["preempted_token_progress"]
        candidate_progress = candidate["preempted_token_progress"]
        reserved_progress = reserved["preempted_token_progress"]
        reservation_output_parity = (
            candidate["generated_token_ids_digest"] is not None
            and candidate["generated_token_ids_digest"]
            == reserved["generated_token_ids_digest"]
        )
        reservation_implementation_parity = (
            candidate["commit"] is not None
            and candidate["commit"] == reserved["commit"]
        )
        reservation_checkpoint_parity = (
            candidate["checkpoint_digest"] is not None
            and candidate["checkpoint_digest"]
            == reserved["checkpoint_digest"]
        )
        reservation_progress_valid = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (candidate_progress, reserved_progress)
        )
        latency_ratios = {
            name: (
                candidate[name] / baseline[name]
                if baseline[name]
                else None
            )
            for name in (
                "p50_ttft_s",
                "p95_ttft_s",
                "p99_ttft_s",
                "p50_tpot_s",
                "p95_tpot_s",
                "p99_tpot_s",
                "p50_request_latency_s",
                "p95_request_latency_s",
                "p99_request_latency_s",
            )
        }
        kv_pressure_comparisons[tp_name] = {
            "valid": (
                baseline["valid"]
                and candidate["valid"]
                and output_parity
                and candidate_progress < baseline_progress
            ),
            "output_parity": output_parity,
            "recomputed_token_reduction": baseline_progress - candidate_progress,
            "recomputed_token_reduction_ratio": (
                1.0 - candidate_progress / baseline_progress
            ),
            "elapsed_speedup": (
                baseline["total_time_s"] / candidate["total_time_s"]
                if baseline["total_time_s"] and candidate["total_time_s"]
                else None
            ),
            "candidate_latency_vs_fcfs": latency_ratios,
            "tail_latency_non_regressing": all(
                latency_ratios[name] is not None
                and latency_ratios[name] <= 1.0
                for name in (
                    "p95_ttft_s",
                    "p99_ttft_s",
                    "p95_tpot_s",
                    "p99_tpot_s",
                    "p95_request_latency_s",
                    "p99_request_latency_s",
                )
            ),
            "decode_kv_reservation": {
                "valid": (
                    candidate["valid"]
                    and reserved["valid"]
                    and reserved["decode_kv_reservation_observed"]
                    and reservation_output_parity
                    and reservation_implementation_parity
                    and reservation_checkpoint_parity
                    and reservation_progress_valid
                    and reserved_progress <= candidate_progress
                ),
                "output_parity": reservation_output_parity,
                "implementation_parity": reservation_implementation_parity,
                "checkpoint_parity": reservation_checkpoint_parity,
                "recomputed_token_reduction": (
                    candidate_progress - reserved_progress
                    if reservation_progress_valid
                    else None
                ),
                "reservation_stop_count": reserved[
                    "prefill_stopped_by_decode_kv_reservation"
                ],
                "elapsed_ratio": (
                    reserved["total_time_s"] / candidate["total_time_s"]
                    if candidate["total_time_s"]
                    and reserved["total_time_s"]
                    else None
                ),
                "peak_memory_delta_mib": (
                    reserved["peak_torch_allocated_mib"]
                    - candidate["peak_torch_allocated_mib"]
                    if candidate["peak_torch_allocated_mib"] is not None
                    and reserved["peak_torch_allocated_mib"] is not None
                    else None
                ),
            },
        }

    fairness_results = {}
    for path in fairness_paths:
        result = load_json(path)
        mode = path.parents[1].name
        tp_name = path.parent.name
        if mode not in {"disabled", "enabled"}:
            raise ValueError(f"unknown scheduler-fairness mode: {mode}")
        fairness_results.setdefault((tp_name, mode), []).append(result)
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    for (tp_name, mode), results in sorted(fairness_results.items()):
        scheduler_fairness.setdefault(tp_name, {})[mode] = (
            summarize_fairness_repeats(
                results,
                expected_tp_size=int(tp_name.removeprefix("tp")),
                mode=mode,
            )
        )
    fairness_comparisons = {
        tp_name: compare_fairness_modes(
            by_mode["disabled"],
            by_mode["enabled"],
        )
        for tp_name, by_mode in scheduler_fairness.items()
        if set(by_mode) == {"disabled", "enabled"}
    }
    for path in mixed_paths:
        result = load_json(path)
        tp_name = path.parent.name
        tp_size = int(tp_name.removeprefix("tp"))
        model_paths = result.get("execution_stats", {}).get(
            "model_path_counts",
            {},
        )
        valid = (
            result.get("tensor_parallel_size") == tp_size
            and result.get("qwen35_moe_decode_backend") == "batched"
            and result.get("enable_dynamic_chunked_prefill") is True
            and result.get("injected") is True
            and result.get("execution_validation", {}).get("valid") is True
            and result.get("generation_validation", {}).get("valid") is True
            and model_paths.get("mixed_eager", 0) > 0
            and result.get("cuda_available") is True
        )
        mixed_runs.setdefault(tp_name, []).append({
            "valid": valid,
            "mixed_steps": model_paths.get("mixed_eager", 0),
            "initial_p95_decode_gap_s": result.get(
                "initial_p95_decode_gap_s"
            ),
            "peak_torch_allocated_mib": result.get(
                "peak_torch_allocated_mib"
            ),
            "generated_token_ids_digest": result.get(
                "generated_token_ids",
                {},
            ).get("digest"),
        })
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    mixed_workload = {}
    for tp_name, rows in sorted(mixed_runs.items()):
        p95_samples = [row["initial_p95_decode_gap_s"] for row in rows]
        digests = {row["generated_token_ids_digest"] for row in rows}
        mean_p95 = statistics.mean(p95_samples)
        p95_cv = (
            statistics.pstdev(p95_samples) / mean_p95
            if mean_p95 > 0
            else math.inf
        )
        output_parity = None not in digests and len(digests) == 1
        mixed_workload[tp_name] = {
            "valid": (
                len(rows) >= 2
                and all(row["valid"] for row in rows)
                and output_parity
                and p95_cv <= MIXED_MAX_COEFFICIENT_OF_VARIATION
            ),
            "repeat_count": len(rows),
            "output_parity": output_parity,
            "generated_token_ids_digest": (
                next(iter(digests)) if output_parity else None
            ),
            "median_mixed_steps": statistics.median(
                row["mixed_steps"] for row in rows
            ),
            "median_initial_p95_decode_gap_s": statistics.median(p95_samples),
            "initial_p95_decode_gap_cv": p95_cv,
            "max_allowed_cv": MIXED_MAX_COEFFICIENT_OF_VARIATION,
            "max_peak_torch_allocated_mib": max(
                row["peak_torch_allocated_mib"] for row in rows
            ),
            "runs": rows,
        }

    cudagraph = {}
    for path in cudagraph_paths:
        result = load_json(path)
        tp_name = path.parents[2].name
        context_name = path.parents[1].name
        cases = cudagraph.setdefault(tp_name, {})
        if context_name in cases:
            raise ValueError(
                f"multiple {context_name} CUDA Graph summaries found for {tp_name}"
            )
        expected_attention_path = (
            "int8_partitioned_decode"
            if context_name == "long"
            else "int8_fused_decode"
        )
        cases[context_name] = {
            "passed": result["passed"],
            "hybrid_graph_captured": result.get(
                "hybrid_graph_captured",
                False,
            ),
            "kv_cache_dtype": result["kv_cache_dtype"],
            "expected_attention_path": expected_attention_path,
            "attention_path_observed": all(
                scenario["comparison"].get("expected_attention_path")
                == expected_attention_path
                and scenario["comparison"].get(
                    "expected_eager_attention_path", False
                )
                and scenario["comparison"].get(
                    "expected_graph_attention_path", False
                )
                for scenario in result["scenarios"]
            ),
            "scratch_isolation_observed": all(
                scenario["comparison"].get(
                    "scratch_primed_across_bucket", False
                )
                for scenario in result["scenarios"]
            ),
            "scenario_count": len(result["scenarios"]),
            "batch_sizes": [
                scenario["batch_size"] for scenario in result["scenarios"]
            ],
        }
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

    quality_case_dirs = sorted(
        path
        for path in (run_dir / "quality").glob(f"{run_id}_qwen35_*")
        if path.is_dir()
    )
    quality_case_paths = [
        case_dir / f"{case_dir.name}.json"
        for case_dir in quality_case_dirs
    ]
    if not quality_case_paths:
        raise ValueError("no quality case artifacts were found")
    checkpoint_digests = {performance["workload"]["checkpoint_manifest_digest"]}
    for path in quality_case_paths:
        result = load_json(path)
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        checkpoint_digests.add(result["checkpoint_manifest"]["digest"])

    mixed_digests = {
        item["generated_token_ids_digest"]
        for item in mixed_workload.values()
        if item["generated_token_ids_digest"]
    }
    mixed_cross_tp_parity = (
        len(mixed_workload) == len(expected_tp_names)
        and len(mixed_digests) == 1
    )
    cudagraph_valid = (
        set(cudagraph) == set(kernels)
        and all(set(cases) == {"short", "long"} for cases in cudagraph.values())
        and all(
            item["passed"]
            and item["hybrid_graph_captured"]
            and item["kv_cache_dtype"] == "int8"
            and item["attention_path_observed"]
            and item["scratch_isolation_observed"]
            for cases in cudagraph.values()
            for item in cases.values()
        )
    )
    evidence = {
        "official_checkpoint_headers_valid": official_checkpoint_valid,
        "local_checkpoint_matches_official": local_checkpoint_identity_valid,
        "checkpoint_mapping_valid": (
            audit["valid"]
            and audit["complete"]
            and all(
                result.get("skipped_tensor_groups")
                == OFFICIAL_SKIPPED_WEIGHT_GROUPS
                for result in audit.get("results", {}).values()
            )
        ),
        "memory_preflight_valid": (
            memory["valid"] and set(memory_by_tp) == expected_tp_names
        ),
        "pd_transfer_baseline_valid": (
            set(pd_transfer) == expected_tp_names
            and all(
                set(profiles) == {"auto-float32", "int8-model"}
                and all(item["valid"] for item in profiles.values())
                for profiles in pd_transfer.values()
            )
        ),
        "pd_export_memory_evidence": (
            set(pd_export) == expected_tp_names
            and all(
                set(profiles) == {"auto-float32", "int8-model"}
                and all(item["valid"] for item in profiles.values())
                for profiles in pd_export.values()
            )
        ),
        "rotary_storage_matches_preflight": rotary_storage_matches_preflight,
        "recurrent_storage_matches_preflight": (
            recurrent_storage_matches_preflight
        ),
        "kv_storage_matches_preflight": kv_storage_matches_preflight,
        "performance_paths_valid": (
            performance["all_execution_paths_valid"]
            and recurrent_state_access["all_configurations_valid"]
        ),
        "performance_generation_valid": performance["all_generation_valid"],
        "performance_output_parity": performance["all_output_digests_match"],
        "moe_runtime_output_parity": all(
            item["output_digest_matches"]
            for by_kv in moe_runtime.values()
            for item in by_kv.values()
        ),
        "hybrid_cudagraph_parity": cudagraph_valid,
        "attention_kernel_evidence": (
            set(attention) == expected_tp_names and attention_valid
        ),
        "long_prefill_kernel_evidence": (
            set(long_prefill) == expected_tp_names
            and all(item["valid"] for item in long_prefill.values())
        ),
        "mixed_workload_evidence": (
            set(mixed_workload) == expected_tp_names
            and all(item["valid"] for item in mixed_workload.values())
            and mixed_cross_tp_parity
        ),
        "kv_pressure_evidence": (
            set(kv_pressure) == expected_tp_names
            and all(
                set(by_policy)
                == {"fcfs", "min_recompute", "min_recompute_reserved"}
                and all(item["valid"] for item in by_policy.values())
                and kv_pressure_comparisons.get(tp_name, {}).get("valid") is True
                and kv_pressure_comparisons.get(tp_name, {})
                .get("decode_kv_reservation", {})
                .get("valid")
                is True
                for tp_name, by_policy in kv_pressure.items()
            )
        ),
        "scheduler_fairness_evidence": (
            set(scheduler_fairness) == expected_tp_names
            and all(
                set(by_mode) == {"disabled", "enabled"}
                and all(item["valid"] for item in by_mode.values())
                and fairness_comparisons.get(tp_name, {}).get("valid") is True
                for tp_name, by_mode in scheduler_fairness.items()
            )
        ),
        "normalization_workspace_evidence": (
            set(normalization) == expected_tp_names
            and all(
                item["valid"]
                for by_kind in normalization.values()
                for item in by_kind.values()
            )
        ),
        "buffer_reuse_evidence": (
            set(buffer_reuse) == expected_tp_names
            and all(
                item["valid"]
                for by_kind in buffer_reuse.values()
                for item in by_kind.values()
            )
        ),
        "mixed_moe_dispatch_evidence": (
            set(mixed_moe_dispatch) == expected_tp_names
            and all(item["valid"] for item in mixed_moe_dispatch.values())
        ),
        "moe_weight_buffer_reuse_evidence": (
            set(moe_weight_buffer_reuse) == expected_tp_names
            and all(item["valid"] for item in moe_weight_buffer_reuse.values())
        ),
        "moe_route_input_broadcast_evidence": (
            set(moe_route_input_broadcast) == expected_tp_names
            and all(
                item["valid"]
                for by_batch in moe_route_input_broadcast.values()
                for item in by_batch.values()
            )
        ),
        "quality_reads_stored_kv": all(
            row["kv_sensitive_token_rows"] > 0 for row in quality["cases"]
        ),
        "quality_partitioned_int8_path": all(
            row["int8_partitioned_decode_observed"]
            for row in quality["cases"]
        ),
        "quality_cross_tp_parity": quality["cross_tp"]["all_passed"],
        "quality_thresholds_passed": quality["quality_gates"]["all_passed"],
        "gptq_validation": gptq["valid"],
        "cuda_measurements": cuda_measurements,
        "clean_worktrees": clean_worktrees,
        "single_commit": len(commits) == 1,
        "single_checkpoint": len(checkpoint_digests) == 1,
    }
    microbenchmark_promoted = all(
        item["promotion"]["promote_to_runtime"]
        for item in kernels.values()
    )
    runtime_promoted = all(
        item["promotion"]["promote_to_default"]
        for by_kv in moe_runtime.values()
        for item in by_kv.values()
    )
    same_tp_coverage = set(kernels) == set(moe_runtime)
    device_scalar_all_tp_promoted = bool(moe_device_scalar) and all(
        item["available"] and item["promotion"]["promote_to_runtime"]
        for item in moe_device_scalar.values()
    )
    return {
        "run_id": run_id,
        "model": quality["model"],
        "evidence": evidence,
        "valid": all(evidence.values()),
        "commits": sorted(commits),
        "checkpoint_digests": sorted(checkpoint_digests),
        "official_checkpoint": {
            key: official_audit[key]
            for key in (
                "repo",
                "resolved_revision",
                "config_sha256",
                "index_sha256",
                "headers_sha256",
                "source_tensor_count",
                "semantic_contract",
                "shard_count",
                "checkpoint_shards",
            )
        },
        "local_checkpoint_manifest": {
            key: local_checkpoint_manifest.get(key)
            for key in (
                "digest",
                "strength",
                "config_sha256",
                "index_sha256",
                "shard_count",
                "present_shard_count",
                "missing_shards",
                "total_size_bytes",
            )
        },
        "performance": {
            "best_throughput": best_throughput,
            "lowest_peak_memory": lowest_memory,
            "recurrent_state_access": recurrent_state_access,
        },
        "memory": {
            "by_tp": memory_by_tp,
        },
        "pd_transfer": {
            "scope": "single-rank TCP loopback; not cross-node GPU evidence",
            "by_tp": pd_transfer,
        },
        "pd_export": {
            "scope": "synthetic single-rank GPU-to-host export evidence",
            "by_tp": pd_export,
        },
        "quality": {
            "scope": quality["quality_scope"],
            "comparisons_by_tp": quality["comparisons_by_tp"],
            "cross_tp": quality["cross_tp"],
            "gates": quality["quality_gates"],
        },
        "gptq": gptq,
        "fp8": fp8,
        "graph_safe_moe": {
            "all_tp_promoted": (
                same_tp_coverage
                and microbenchmark_promoted
                and runtime_promoted
            ),
            "same_tp_coverage": same_tp_coverage,
            "microbenchmark_all_tp_promoted": microbenchmark_promoted,
            "runtime_all_tp_promoted": runtime_promoted,
            "by_tp": kernels,
            "runtime_by_tp": moe_runtime,
            "mixed_dispatch_by_tp": mixed_moe_dispatch,
            "weight_buffer_reuse_by_tp": moe_weight_buffer_reuse,
            "route_input_broadcast_by_tp": moe_route_input_broadcast,
            "device_scalar_all_tp_promoted": device_scalar_all_tp_promoted,
            "device_scalar_by_tp": moe_device_scalar,
        },
        "normalization": {
            "by_tp": normalization,
        },
        "buffer_reuse": {
            "by_tp": buffer_reuse,
        },
        "hybrid_cudagraph": {
            "all_tp_passed": cudagraph_valid,
            "same_tp_coverage": set(cudagraph) == set(kernels),
            "by_tp": cudagraph,
        },
        "int8_attention": {
            "short_context": ATTENTION_SHORT_CONTEXT,
            "long_context": ATTENTION_LONG_CONTEXT,
            "max_context": ATTENTION_MAX_CONTEXT,
            "by_tp": attention,
        },
        "long_prefill": {
            "tokens": LONG_PREFILL_TOKENS,
            "by_tp": long_prefill,
        },
        "mixed_workload": {
            "cross_tp_output_parity": mixed_cross_tp_parity,
            "by_tp": mixed_workload,
        },
        "kv_pressure": {
            "configured_kv_blocks": PRESSURE_KV_BLOCKS,
            "by_tp": kv_pressure,
            "comparisons": kv_pressure_comparisons,
        },
        "scheduler_fairness": {
            "workload": {
                "initial_sequences": FAIRNESS_INITIAL_SEQUENCES,
                "injected_sequences": FAIRNESS_INJECTED_SEQUENCES,
                "max_num_batched_tokens": FAIRNESS_MAX_BATCHED_TOKENS,
                "enabled_threshold": FAIRNESS_THRESHOLD,
                "enabled_token_budget": FAIRNESS_TOKEN_BUDGET,
            },
            "by_tp": scheduler_fairness,
            "comparisons": fairness_comparisons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir, args.run_id)
    output = args.output or args.run_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
