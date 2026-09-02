"""Summarize one complete Qwen3.5 rental validation run."""

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
ATTENTION_LONG_CONTEXT = 16384
ATTENTION_MAX_ABS_ERROR = 0.05
PRODUCTION_PARTITION_SIZE = 512
LONG_PREFILL_TOKENS = 8192
LONG_PREFILL_MAX_ABS_ERROR = 0.05
MIXED_MAX_COEFFICIENT_OF_VARIATION = 0.10
NORMALIZATION_MAX_ABS_ERROR = 0.05
BUFFER_REUSE_MAX_ABS_ERROR = 0.05
OFFICIAL_CHECKPOINT_REPO = "Qwen/Qwen3.5-35B-A3B"
OFFICIAL_CHECKPOINT_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
OFFICIAL_CONFIG_SHA256 = (
    "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944"
)
OFFICIAL_INDEX_SHA256 = (
    "d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205"
)
OFFICIAL_HEADERS_SHA256 = (
    "39753f429d8ce99ba181f00e068b36df4ecd2603c34df5352492b21d5a32878b"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required validation artifact is missing: {path}")
    return json.loads(path.read_text())


def summarize_normalization_candidate(result: dict, reuse_key: str) -> dict:
    max_abs_error = max(item["max_abs_error"] for item in result["errors"])
    reference = result["reference"]
    candidate = result["candidate"]
    return {
        "valid": result[reuse_key] and max_abs_error <= NORMALIZATION_MAX_ABS_ERROR,
        "speedup": result["speedup"],
        "reference_peak_extra_mib": reference["peak_extra_mib"],
        "candidate_peak_extra_mib": candidate["peak_extra_mib"],
        "peak_extra_mib_delta": (
            candidate["peak_extra_mib"] - reference["peak_extra_mib"]
        ),
        "max_abs_error": max_abs_error,
        "max_allowed_abs_error": NORMALIZATION_MAX_ABS_ERROR,
        "workspace": {
            key: value
            for key, value in result.items()
            if key.endswith("_workspace_mib") or key == "avoided_fp32_copy_mib"
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
    metadata = {
        key: result.get(key) for key in (required_metadata or {})
    }
    metadata_valid = all(
        metadata[key] == expected
        for key, expected in (required_metadata or {}).items()
    )
    workspace = {key: result[key] for key in workspace_keys}
    return {
        "valid": (
            all(value > 0 for value in workspace.values())
            and metadata_valid
            and max_abs_error <= BUFFER_REUSE_MAX_ABS_ERROR
        ),
        "speedup": result["speedup"],
        "reference_peak_extra_mib": reference["peak_extra_mib"],
        "candidate_peak_extra_mib": candidate["peak_extra_mib"],
        "peak_extra_mib_delta": (
            candidate["peak_extra_mib"] - reference["peak_extra_mib"]
        ),
        "max_abs_error": max_abs_error,
        "max_allowed_abs_error": BUFFER_REUSE_MAX_ABS_ERROR,
        "workspace": workspace,
        "metadata": metadata,
    }


def evaluate_moe_runtime_candidate(
    *,
    output_digest_matches: bool,
    throughput_speedup: float,
    tpot_speedup: float,
    peak_memory_delta_mib: float,
    max_coefficient_of_variation: float,
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
            "promotion": evaluate_moe_runtime_candidate(
                output_digest_matches=output_digest_matches,
                throughput_speedup=throughput_speedup,
                tpot_speedup=tpot_speedup,
                peak_memory_delta_mib=peak_memory_delta_mib,
                max_coefficient_of_variation=max_cv,
            ),
        }
    return comparisons


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
    production_item = result["results"].get(production_name)
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
        "best_fused": best_accurate(fused),
        "best_partitioned": best_accurate(partitioned_results),
        "production_partition_size": PRODUCTION_PARTITION_SIZE,
        "production_partitioned": production_summary,
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
        budgets = item.get("available_budget_bytes_by_rank", [])
        if not budgets:
            raise ValueError(f"memory preflight has no rank budgets for {tp_name}")
        required = item["required_free_bytes_per_rank"]
        minimum_budget = min(budgets)
        summaries[tp_name] = {
            "local_parameter_bytes": item["local_parameter_bytes"],
            "max_state_bytes_per_rank": item["max_state_bytes_per_rank"],
            "minimum_workload_kv_bytes_per_rank": item[
                "minimum_workload_kv_bytes_per_rank"
            ],
            "kv_bytes_per_token_by_dtype": kv_sizes,
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
        valid = (
            bool(errors)
            and math.isfinite(max_abs_error)
            and max_abs_error <= LONG_PREFILL_MAX_ABS_ERROR
            and math.isfinite(median_ms)
            and median_ms > 0
            and math.isfinite(peak_extra_mib)
            and peak_extra_mib >= 0
        )
        cases[name] = {
            "valid": valid,
            "median_ms": median_ms,
            "peak_extra_mib": peak_extra_mib,
            "max_abs_error": max_abs_error,
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
    kernel_paths = sorted((run_dir / "kernels").glob("tp*.json"))
    if not kernel_paths:
        raise ValueError("no kernel benchmark artifacts were found")
    long_prefill_paths = sorted((run_dir / "kernels_long").glob("tp*.json"))
    if not long_prefill_paths:
        raise ValueError("no long-prefill kernel artifacts were found")
    mixed_paths = sorted((run_dir / "mixed").glob("tp*/r*.json"))
    if not mixed_paths:
        raise ValueError("no mixed-workload benchmark artifacts were found")
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

    kernels = {}
    normalization = {}
    buffer_reuse = {}
    long_prefill = {}
    mixed_runs = {}
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
        and set(official_audit.get("results", {})) == expected_tp_names
        and all(
            result.get("valid") is True
            for result in official_audit.get("results", {}).values()
        )
    )
    memory_by_tp = summarize_memory_preflight(memory)
    attention = {}
    attention_valid = True
    for tp_name in sorted(expected_tp_names):
        tp_size = int(tp_name.removeprefix("tp"))
        cases = {}
        for case_name, context_len, partitioned in (
            ("short", ATTENTION_SHORT_CONTEXT, False),
            ("long", ATTENTION_LONG_CONTEXT, True),
        ):
            result = load_json(
                run_dir / "attention" / tp_name / f"{case_name}.json"
            )
            dimensions_valid = (
                result["context_len"] == context_len
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
        normalization[tp_name] = {
            "rmsnorm": summarize_normalization_candidate(
                result["results"]["rmsnorm_fp32_reuse"],
                "candidate_reuses_fp32_workspace",
            ),
            "gated_rmsnorm": summarize_normalization_candidate(
                result["results"]["gated_rmsnorm_fp32_reuse"],
                "candidate_reuses_fp32_workspaces",
            ),
        }
        buffer_reuse[tp_name] = {
            "moe_output_merge": summarize_buffer_reuse_candidate(
                result["results"]["moe_output_buffer_reuse"],
                (
                    "reused_routed_output_mib",
                    "reused_shared_output_mib",
                    "reused_gate_mib",
                ),
            ),
            "residual_merge": summarize_buffer_reuse_candidate(
                result["results"]["residual_output_buffer_reuse"],
                ("reused_branch_output_mib_per_merge",),
                {"residual_merges_per_decoder_layer": 2},
            ),
            "torch_kv_dequant": summarize_buffer_reuse_candidate(
                result["results"]["torch_kv_dequant_buffer_reuse"],
                ("avoided_output_workspace_mib",),
            ),
            "recurrent_decode": summarize_buffer_reuse_candidate(
                result["results"]["specialized_delta_decode"],
                ("reused_recurrent_state_mib",),
                {"avoided_full_state_intermediates": 2},
            ),
        }
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
            "by_decode_batch": {
                batch: {
                    "promotion": item["promotion"],
                    "median_ms": item["median_ms"],
                    "speedup_vs_current": item["speedup_vs_current"],
                    "peak_extra_mib": item["peak_extra_mib"],
                    "errors_vs_current": item["errors_vs_current"],
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
        commits.add(result["commit"])
        clean_worktrees = clean_worktrees and not result["git_dirty"]
        cuda_measurements = cuda_measurements and result["cuda_available"]

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
        "checkpoint_mapping_valid": audit["valid"] and audit["complete"],
        "memory_preflight_valid": (
            memory["valid"] and set(memory_by_tp) == expected_tp_names
        ),
        "performance_paths_valid": performance["all_execution_paths_valid"],
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
        "quality_reads_stored_kv": all(
            row["kv_sensitive_token_rows"] > 0 for row in quality["cases"]
        ),
        "quality_partitioned_int8_path": all(
            row["int8_partitioned_decode_observed"]
            for row in quality["cases"]
        ),
        "quality_cross_tp_parity": quality["cross_tp"]["all_passed"],
        "quality_thresholds_passed": quality["quality_gates"]["all_passed"],
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
                "shard_count",
            )
        },
        "performance": {
            "best_throughput": best_throughput,
            "lowest_peak_memory": lowest_memory,
        },
        "memory": {
            "by_tp": memory_by_tp,
        },
        "quality": {
            "scope": quality["quality_scope"],
            "comparisons_by_tp": quality["comparisons_by_tp"],
            "cross_tp": quality["cross_tp"],
            "gates": quality["quality_gates"],
        },
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
