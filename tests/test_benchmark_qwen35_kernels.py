from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_qwen35_kernels",
    ROOT / "scripts" / "benchmark_qwen35_kernels.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_measure_preserves_every_raw_repeat():
    result = MODULE.measure(
        lambda: None,
        device=torch.device("cpu"),
        warmup=1,
        iterations=1,
        repeats=3,
    )

    assert len(result["samples_ms"]) == 3
    assert result["peak_extra_mib_samples"] == []
    assert result["median_ms"] == sorted(result["samples_ms"])[1]


def test_measure_runs_the_inference_path():
    grad_modes = []

    MODULE.measure(
        lambda: grad_modes.append(torch.is_grad_enabled()),
        device=torch.device("cpu"),
        warmup=1,
        iterations=2,
        repeats=1,
    )

    assert grad_modes == [False, False, False]


def test_partitioned_decode_benchmark_measures_shared_buffer_reuse():
    args = SimpleNamespace(
        decode_batch=2,
        attention_head_dim=8,
        int8_context_len=17,
        int8_partition_size=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_partitioned_decode_buffer_reuse(
        args,
        torch.device("cpu"),
        torch.float32,
        local_query_heads=3,
    )

    assert result["configuration"]["num_partitions"] == 3
    assert result["persistent_workspace_mib"] > 0
    assert result["persistent_output_mib"] > 0
    assert result["eliminated_tensor_allocations_per_attention_layer"] == 2
    assert result["candidate_reuses_workspace_and_output"]


def test_int8_dequant_benchmark_measures_shared_kv_storage():
    args = SimpleNamespace(
        attention_head_dim=8,
        int8_context_len=17,
        kvcache_block_size=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_int8_dequant_buffer_reuse(
        args,
        torch.device("cpu"),
        torch.float32,
        local_kv_heads=2,
    )

    assert result["configuration"]["selected_blocks"] == 3
    assert result["packed_k_shape"] == [3, 8, 2, 8]
    assert result["packed_v_shape"] == [3, 8, 2, 8]
    assert result["persistent_buffer_mib"] > 0
    assert result["eliminated_tensor_allocations_per_attention_layer"] == 2
    assert result["candidate_reuses_one_storage_for_kv"]


def test_packed_block_metadata_benchmark_measures_buffer_reuse():
    args = SimpleNamespace(
        sampling_batch=2,
        prefill_tokens=512,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_packed_block_metadata_reuse(
        args,
        torch.device("cpu"),
    )

    assert result["errors"] == [
        {"max_abs_error": 0.0, "max_relative_error": 0.0, "rmse": 0.0},
        {"max_abs_error": 0.0, "max_relative_error": 0.0, "rmse": 0.0},
    ]
    assert result["eliminated_tensor_allocations_per_update"] == 4
    assert result["persistent_metadata_buffers_mib"] > 0
    assert result["candidate_reuses_two_isolated_buffer_banks"]


def test_error_treats_matching_negative_infinity_as_equal():
    result = MODULE.error(
        torch.tensor([1.0, float("-inf")]),
        torch.tensor([1.0, float("-inf")]),
    )

    assert result == {"max_abs_error": 0.0, "max_relative_error": 0.0, "rmse": 0.0}


def test_sampling_filter_benchmark_covers_unfiltered_and_top_k_paths():
    args = SimpleNamespace(
        sampling_batch=4,
        vocab_size=16,
        sampling_top_k=3,
        sampling_top_p=0.9,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_sampling_filter(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert set(result) == {"unfiltered", "top_k", "top_p", "top_k_top_p"}
    assert all(
        result[name]["avoided_full_sort_workspace_mib"]
        == 4 * 16 * 10 / 1024 / 1024
        for name in ("unfiltered", "top_k", "top_k_top_p")
    )
    assert result["top_p"]["avoided_top_k_mask_workspace_mib"] == (
        4 * 16 * 3 + 16 * 8
    ) / 1024 / 1024
    assert all(item["errors"][0]["max_abs_error"] == 0 for item in result.values())
    assert all(item["uses_host_sampling_metadata"] for item in result.values())


def test_compact_top_k_sampling_benchmark_tracks_fp32_reduction():
    args = SimpleNamespace(
        sampling_batch=4,
        vocab_size=16,
        sampling_top_k=3,
        sampling_top_p=0.9,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_compact_top_k_sampling(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["full_fp32_logits_mib"] == 4 * 16 * 4 / 1024 / 1024
    assert result["compact_fp32_logits_mib"] == 4 * 3 * 4 / 1024 / 1024
    assert result["avoided_fp32_logits_mib"] == 4 * 13 * 4 / 1024 / 1024


def test_sampling_filter_output_reuse_tracks_eliminated_workspaces():
    args = SimpleNamespace(
        sampling_batch=4,
        vocab_size=16,
        sampling_top_p=0.9,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_sampling_filter_output_reuse(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["errors"][0]["max_abs_error"] == 0
    assert result["avoided_fp32_logits_mib"] == 2 * 4 * 16 * 4 / 1024 / 1024
    assert result["eliminated_tensor_allocations_per_sampling_step"] == 2
    assert result["candidate_reuses_temperature_and_filter_storage"]


def test_sampling_input_benchmark_tracks_persistent_storage():
    args = SimpleNamespace(
        sampling_batch=4,
        sampling_top_k=3,
        sampling_top_p=0.9,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_sampling_input_reuse(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["eliminated_tensor_allocations_per_step"] == 6
    assert result["persistent_sampling_input_mib"] == 4 * 4 * 6 / 1024 / 1024
    assert result["candidate_reuses_host_device_storage"]
    assert all(item["max_abs_error"] == 0 for item in result["errors"])
    assert result["errors"][0]["max_abs_error"] <= 1e-7


def test_sampling_noise_benchmark_tracks_reused_storage():
    args = SimpleNamespace(
        sampling_batch=4,
        sampling_top_k=3,
        seed=47,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_sampling_noise_reuse(
        args,
        torch.device("cpu"),
    )

    assert result["errors"][0]["max_abs_error"] == 0
    assert result["eliminated_tensor_allocations_per_sampling_step"] == 1
    assert result["persistent_sampling_noise_mib"] == 4 * 3 * 4 / 1024 / 1024
    assert result["candidate_reuses_noise_storage"]


def test_gated_delta_packed_projection_replaces_three_gemms():
    args = SimpleNamespace(
        decode_batch=2,
        hidden_size=8,
        value_head_dim=4,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_gated_delta_packed_projection(
        args,
        torch.device("cpu"),
        torch.float32,
        local_value_heads=2,
    )

    assert result["reference_gemm_launches"] == 3
    assert result["candidate_gemm_launches"] == 1
    assert result["avoided_gemm_launches"] == 2
    assert all(item["max_abs_error"] <= 1e-6 for item in result["errors"])


def test_attention_packed_qkv_replaces_three_gemms_and_tracks_key_copy():
    args = SimpleNamespace(
        decode_batch=2,
        hidden_size=8,
        attention_head_dim=4,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_attention_packed_qkv(
        args,
        torch.device("cpu"),
        torch.float32,
        local_query_heads=2,
        local_kv_heads=1,
    )

    assert result["reference_gemm_launches"] == 3
    assert result["candidate_gemm_launches"] == 1
    assert result["avoided_gemm_launches"] == 2
    assert result["key_alias_break_copy_mib"] == 2 * 1 * 4 * 4 / 1024 / 1024
    assert all(item["max_abs_error"] <= 1e-6 for item in result["errors"])


def test_contiguous_decode_state_benchmark_tracks_avoided_copy():
    args = SimpleNamespace(
        decode_batch=2,
        key_head_dim=3,
        value_head_dim=4,
        conv_kernel_size=2,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_contiguous_decode_state(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        local_value_heads=2,
        local_conv_channels=5,
    )

    expected_bytes = 2 * 2 * 3 * 4 * 4 + 2 * 5 * 2 * 2
    assert result["avoided_state_gather_mib"] == expected_bytes / 1024 / 1024
    assert result["avoided_state_scatter_mib"] == expected_bytes / 1024 / 1024
    assert result["candidate_uses_cache_views"]
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_greedy_sampler_benchmark_tracks_avoided_fp32_logits():
    args = SimpleNamespace(
        sampling_batch=4,
        vocab_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_greedy_sampler(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["avoided_fp32_logits_mib"] == 4 * 16 * 2 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0
    assert result["uses_host_sampling_metadata"]


def test_attention_norm_benchmark_compares_projection_output_reuse():
    args = SimpleNamespace(
        router_tokens=4,
        hidden_size=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_attention_norm_output_reuse(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["reused_projection_output_mib"] == 4 * 8 * 2 / 1024 / 1024
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_delta_l2_benchmark_tracks_reused_fp32_workspaces():
    args = SimpleNamespace(
        router_tokens=4,
        key_head_dim=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_l2_normalization(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        local_key_heads=2,
    )

    assert result["reused_query_key_fp32_mib"] == (
        2 * 4 * 2 * 8 * 4 / 1024 / 1024
    )
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_rotary_benchmark_compares_query_key_output_reuse():
    args = SimpleNamespace(
        router_tokens=4,
        total_key_heads=8,
        tp_size=4,
        key_head_dim=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_rotary_output_reuse(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["rotary_dim"] == 2
    assert result["reused_query_key_output_mib"] == (
        2 * 4 * 2 * 8 * 2 / 1024 / 1024
    )
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_vocab_gather_layout_benchmark_tracks_avoided_full_copy():
    args = SimpleNamespace(
        vocab_size=16,
        tp_size=4,
        sampling_batch=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_vocab_gather_layout(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["avoided_full_vocab_copy_mib"] == 3 * 16 * 2 / 1024 / 1024
    assert result["candidate_returns_transpose_view"]
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_decode_convolution_benchmark_tracks_reused_state():
    args = SimpleNamespace(
        decode_batch=4,
        conv_kernel_size=4,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_decode_convolution(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        local_conv_channels=6,
    )

    assert result["reused_convolution_state_mib"] == 4 * 6 * 4 * 2 / 1024 / 1024
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_router_benchmark_reuses_selected_logits_for_probabilities():
    args = SimpleNamespace(
        router_tokens=8,
        num_experts=16,
        top_k=2,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_router(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["reused_selected_logits_mib"] == 8 * 2 * 4 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] <= 1e-6


def test_rmsnorm_benchmark_compares_workspace_reuse_to_baseline():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_rmsnorm(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["candidate_reuses_fp32_workspace"]
    assert result["avoided_fp32_copy_mib"] == 8 * 16 * 4 / 1024 / 1024
    assert result["eliminated_per_call_gain_materialization_mib"] == (
        16 * 4 / 1024 / 1024
    )
    assert result["persistent_gain_storage_mib"] == 16 * 4 / 1024 / 1024
    assert result["persistent_storage_delta_mib"] == 16 * 2 / 1024 / 1024
    assert result["candidate_uses_precomputed_gain"]
    assert result["errors"][0]["max_abs_error"] == 0


def test_gated_rmsnorm_benchmark_compares_both_reused_workspaces():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_gated_rmsnorm(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    workspace_mib = 8 * 16 * 4 / 1024 / 1024
    assert result["candidate_reuses_fp32_workspaces"]
    assert result["reused_hidden_fp32_workspace_mib"] == workspace_mib
    assert result["reused_gate_fp32_workspace_mib"] == workspace_mib
    assert result["errors"][0]["max_abs_error"] == 0


def test_beta_gate_benchmark_reuses_projection_buffer():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_beta_gate(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        local_value_heads=4,
    )

    assert result["reused_beta_projection_mib"] == 8 * 4 * 2 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0


def test_decay_rate_benchmark_reports_precomputed_storage():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_decay_rate(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        local_value_heads=4,
    )

    assert result["precomputed_decay_rate_mib"] == 4 * 4 / 1024 / 1024
    assert result["reused_decay_projection_fp32_mib"] == 8 * 4 * 4 / 1024 / 1024
    assert result["reused_softplus_output_mib"] == 8 * 4 * 4 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0


def test_moe_output_merge_benchmark_measures_buffer_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_moe_output_merge(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    output_mib = 8 * 16 * 2 / 1024 / 1024
    assert result["reused_routed_output_mib"] == output_mib
    assert result["reused_shared_output_mib"] == output_mib
    assert result["reused_gate_mib"] == 8 * 2 / 1024 / 1024
    assert result["errors"][0]["max_abs_error"] == 0


def test_residual_merge_benchmark_measures_branch_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_residual_merge(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["reused_branch_output_mib_per_merge"] == (
        8 * 16 * 2 / 1024 / 1024
    )
    assert result["residual_merges_per_decoder_layer"] == 2
    assert result["errors"][0]["max_abs_error"] == 0


def test_sorted_route_weighting_benchmark_measures_output_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_sorted_route_weighting(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["avoided_weighted_expert_output_mib"] == (
        8 * 16 * 2 / 1024 / 1024
    )
    assert result["errors"][0]["max_abs_error"] == 0


def test_batched_route_sum_benchmark_measures_dispatch_output_reuse():
    args = SimpleNamespace(
        router_tokens=8,
        top_k=2,
        hidden_size=16,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_batched_route_sum_output(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert result["avoided_route_sum_output_mib"] == (
        8 * 16 * 2 / 1024 / 1024
    )
    assert result["candidate_reuses_dispatch_output"]
    assert result["errors"][0]["max_abs_error"] == 0


def test_torch_kv_dequant_benchmark_measures_output_reuse():
    args = SimpleNamespace(
        prefill_tokens=256,
        key_head_dim=8,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_torch_kv_dequant(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        2,
    )

    assert result["selected_blocks"] == 1
    assert result["avoided_output_workspace_mib"] == (
        2 * 256 * 2 * 8 * 2 / 1024 / 1024
    )
    assert result["avoided_block_id_cast_mib"] == 8 / 1024 / 1024
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_expert_dispatch_matches_naive_route_accumulation():
    torch.manual_seed(101)
    hidden = torch.randn(4, 3)
    topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1]])
    topk_weights = torch.rand(4, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(3, 4, 3)
    down = torch.randn(3, 3, 2)

    actual = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = topk_ids[token, route]
            projected = torch.nn.functional.linear(hidden[token], gate_up[expert])
            gate, up = projected.chunk(2)
            value = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                down[expert],
            )
            expected[token] += value * topk_weights[token, route]

    torch.testing.assert_close(actual, expected)
    reference = MODULE.expert_dispatch_reference(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    torch.testing.assert_close(actual, reference)


def test_graph_safe_batched_decode_matches_current_single_token_path():
    torch.manual_seed(109)
    hidden = torch.randn(1, 4)
    topk_ids = torch.tensor([[3, 1]])
    topk_weights = torch.tensor([[0.25, 0.75]])
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    actual = MODULE.expert_dispatch_batched_decode(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )

    torch.testing.assert_close(actual, expected)


def test_graph_safe_batched_decode_matches_multi_token_path():
    torch.manual_seed(113)
    hidden = torch.randn(5, 4)
    topk_ids = torch.randint(0, 4, (5, 2))
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    actual = MODULE.expert_dispatch_batched_decode(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
        chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_broadcast_batched_decode_matches_repeated_input_oracle():
    torch.manual_seed(119)
    hidden = torch.randn(5, 4)
    topk_ids = torch.randint(0, 4, (5, 2))
    topk_weights = torch.rand(5, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch_batched_repeated_input(
        hidden, topk_ids, topk_weights, gate_up, down, chunk_size=2
    )
    actual = MODULE.expert_dispatch_batched_decode(
        hidden, topk_ids, topk_weights, gate_up, down, chunk_size=2
    )

    torch.testing.assert_close(actual, expected)


def test_mixed_expert_dispatch_matches_whole_batch_grouped_path():
    torch.manual_seed(127)
    hidden = torch.randn(7, 4)
    topk_ids = torch.randint(0, 4, (7, 2))
    topk_weights = torch.rand(7, 2)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)

    expected = MODULE.expert_dispatch_general(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )
    actual = MODULE.expert_dispatch_mixed(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
        decode_token_count=2,
        chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_fp8_expert_shard_benchmark_covers_non_aligned_tp_slice():
    args = SimpleNamespace(
        moe_intermediate_size=12,
        hidden_size=8,
        fp8_weight_block_size=8,
        tp_size=4,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_fp8_expert_shard_dequantization(
        args,
        torch.device("cpu"),
        torch.float32,
    )

    assert result["local_weight_shape"] == [3, 8]
    assert result["row_block_offset"] == 1
    assert result["dequantized_temporary_reduction"] == 4
    assert result["candidate"]["median_ms"] > 0
    assert result["errors"][0]["max_abs_error"] == 0


def test_mixed_expert_benchmark_records_cuda_evidence_boundary():
    args = SimpleNamespace(
        mixed_decode_tokens=2,
        mixed_prefill_tokens=3,
        mixed_decode_token_counts=(2, 4),
        mixed_prefill_token_counts=(3, 6),
        moe_intermediate_size=6,
        tp_size=1,
        hidden_size=4,
        num_experts=4,
        top_k=2,
        moe_decode_chunk_size=2,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_mixed_expert_dispatch(
        args,
        torch.device("cpu"),
        torch.float32,
    )

    assert result["decode_tokens"] == 2
    assert result["prefill_tokens"] == 3
    assert result["speedup_vs_grouped"] > 0
    assert result["avoided_route_hidden_allocation_mib_per_step"] > 0
    assert result["avoided_redundant_output_zero_mib_per_step"] > 0
    assert not result["measured_on_cuda"]
    assert result["errors"][0]["max_abs_error"] < 1e-5

    sweep = MODULE.benchmark_mixed_expert_dispatch_sweep(
        args,
        torch.device("cpu"),
        torch.float32,
    )
    assert set(sweep) == {"decode2_prefill3", "decode4_prefill6"}


def test_expert_dispatch_sweep_preserves_requested_token_counts(monkeypatch):
    calls = []

    def fake_benchmark(args, device, dtype, token_count):
        calls.append(token_count)
        return {"tokens": token_count}

    monkeypatch.setattr(MODULE, "benchmark_expert_dispatch", fake_benchmark)
    args = SimpleNamespace(expert_token_counts=(1, 8, 32, 64, 128, 512))

    result = MODULE.benchmark_expert_dispatch_sweep(
        args,
        torch.device("cpu"),
        torch.float32,
    )

    assert calls == [1, 8, 32, 64, 128, 512]
    assert list(result) == ["1", "8", "32", "64", "128", "512"]
    assert result["128"] == {"tokens": 128}


def test_single_token_dispatch_reports_general_path_baseline():
    args = SimpleNamespace(
        moe_intermediate_size=8,
        tp_size=1,
        hidden_size=4,
        num_experts=4,
        top_k=2,
        num_hidden_layers=1,
        warmup=0,
        iterations=1,
        repeats=1,
        moe_decode_chunk_size=2,
        moe_decode_chunk_sizes=(1, 2),
        max_decode_tokens=64,
        moe_graph_safe_min_speedup=1.05,
        moe_graph_safe_max_peak_extra_mib=64.0,
        moe_graph_safe_max_abs_error=0.05,
    )

    result = MODULE.benchmark_expert_dispatch(
        args,
        torch.device("cpu"),
        torch.float32,
        1,
    )

    assert result["single_token_decode_fast_path"]
    assert result["general_dispatch_baseline"]["median_ms"] > 0
    assert result["decode_fast_path_speedup"] > 0
    device_scalar = result["device_scalar_candidate"]
    assert device_scalar["avoids_host_route_sync"]
    assert device_scalar["speedup_vs_current"] > 0
    assert device_scalar["estimated_selected_weight_mib"] > 0
    assert device_scalar["errors_vs_current"]["max_abs_error"] < 1e-5
    assert not device_scalar["promotion"]["promote_to_runtime"]
    graph_safe = result["graph_safe_batched_candidate"]
    assert graph_safe["speedup_vs_current"] > 0
    assert graph_safe["estimated_selected_weight_mib"] > 0
    assert graph_safe["reused_weighted_route_mib"] > 0
    assert graph_safe["errors_vs_current"]["max_abs_error"] < 1e-5
    assert not graph_safe["promotion"]["promote_to_runtime"]
    assert not graph_safe["promotion"]["checks"]["cuda_measurement"]
    assert set(result["graph_safe_chunk_sweep"]["candidates"]) == {"1", "2"}


def test_device_scalar_decode_matches_sorted_without_host_sync(monkeypatch):
    torch.manual_seed(131)
    hidden = torch.randn(1, 4)
    topk_ids = torch.tensor([[3, 0]])
    topk_weights = torch.tensor([[0.4, 0.6]])
    gate_up = torch.randn(4, 6, 4)
    down = torch.randn(4, 4, 3)
    expected = MODULE.expert_dispatch(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )

    monkeypatch.setattr(
        torch.Tensor,
        "cpu",
        lambda _tensor: (_ for _ in ()).throw(
            AssertionError("candidate must keep route metadata on device")
        ),
    )
    actual = MODULE.expert_dispatch_device_scalar_decode(
        hidden,
        topk_ids,
        topk_weights,
        gate_up,
        down,
    )

    torch.testing.assert_close(actual, expected)


def test_chunk_recommendation_uses_worst_decode_batch_speedup():
    def candidate(speedup, promoted=True):
        return {
            "promotion": {"promote_to_runtime": promoted},
            "speedup_vs_current": speedup,
            "peak_extra_mib": 4.0,
            "median_ms": 1.0 / speedup,
        }

    results = {
        "1": {"graph_safe_chunk_sweep": {"candidates": {
            "4": candidate(1.3),
            "8": candidate(1.2),
            "16": candidate(1.4, promoted=False),
        }}},
        "64": {"graph_safe_chunk_sweep": {"candidates": {
            "4": candidate(1.1),
            "8": candidate(1.15),
            "16": candidate(1.5),
        }}},
        "128": {"tokens": 128},
    }

    recommendation = MODULE.recommend_moe_decode_chunk_size(results, 64)

    assert recommendation["measured_decode_batches"] == [1, 64]
    assert recommendation["recommended_chunk_size"] == 8
    assert not recommendation["candidates"]["16"]["all_batches_promoted"]


def test_delta_prefill_chunk_sweep_compares_shared_input_to_chunk64():
    args = SimpleNamespace(
        delta_prefill_chunk_sizes=(4, 2, 4),
        prefill_batch=1,
        prefill_tokens=4,
        key_head_dim=2,
        value_head_dim=2,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_prefill_chunk_sweep(
        args,
        torch.device("cpu"),
        torch.float32,
        2,
        4,
    )

    assert result["baseline_chunk_size"] == 64
    assert list(result["candidates"]) == ["2", "4"]
    for chunk_size, candidate in result["candidates"].items():
        assert candidate["chunk_size"] == int(chunk_size)
        assert candidate["candidate"]["median_ms"] > 0
        assert max(
            item["max_abs_error"]
            for item in candidate["errors_vs_chunk64"]
        ) < 1e-4


def test_delta_causal_mask_benchmark_records_bounded_reuse():
    args = SimpleNamespace(
        delta_prefill_chunk_sizes=(4, 2, 4),
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_causal_mask_cache(
        args,
        torch.device("cpu"),
    )

    assert set(result["candidates"]) == {"2", "4"}
    assert result["cache_max_entries"] == 32
    assert result["maximum_cached_chunk_size"] == 1024
    for candidate in result["candidates"].values():
        assert candidate["cache_reuses_storage"]
        assert candidate["persistent_mask_mib"] > 0
        assert candidate["eliminated_allocations_per_additional_layer"] == 1
        assert candidate["errors"][0]["max_abs_error"] == 0.0


def test_grouped_delta_prefill_records_reused_correction_workspace():
    args = SimpleNamespace(
        prefill_batch=2,
        prefill_tokens=5,
        key_head_dim=2,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
        prefill_only=True,
    )

    result = MODULE.benchmark_delta_prefill_head_groups(
        args,
        torch.device("cpu"),
        torch.float32,
        local_key_heads=1,
        local_value_heads=2,
        chunk_size=4,
    )

    # The peak correction workspace covers one four-token chunk in FP32.
    expected_mib = 2 * 2 * 4 * 3 * 4 / 1024 / 1024
    assert result["reused_fp32_correction_buffer_mib"] == expected_mib


def test_delta_prefill_state_reuse_compares_allocation_paths():
    args = SimpleNamespace(
        prefill_batch=2,
        prefill_tokens=5,
        key_head_dim=4,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_prefill_state_reuse(
        args,
        torch.device("cpu"),
        torch.float32,
        2,
        6,
        chunk_size=4,
    )

    assert result["num_chunks"] == 2
    assert result["avoided_state_reallocations"] == 1
    assert result["reused_recurrent_state_mib"] == (
        2 * 6 * 4 * 3 * 4 / 1024 / 1024
    )
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_delta_prefill_decay_workspace_benchmark_keeps_exact_baseline():
    args = SimpleNamespace(
        prefill_batch=2,
        prefill_tokens=5,
        key_head_dim=4,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_prefill_decay_workspace_reuse(
        args,
        torch.device("cpu"),
        torch.float32,
        2,
        6,
        chunk_size=4,
    )

    assert result["eliminated_expanded_qk_allocations"] == 2
    assert result["avoided_expanded_fp32_qk_mib"] == (
        2 * 2 * 5 * 6 * 4 * 4 / 1024 / 1024
    )
    assert all(item["max_abs_error"] == 0 for item in result["errors"])


def test_delta_decode_benchmark_records_state_workspace_reuse():
    args = SimpleNamespace(
        decode_batch=2,
        key_head_dim=4,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_decode(
        args,
        torch.device("cpu"),
        torch.bfloat16,
        2,
        6,
    )

    assert result["reused_recurrent_state_mib"] == (
        2 * 6 * 4 * 3 * 4 / 1024 / 1024
    )
    assert result["avoided_full_state_intermediates"] == 2
    assert result["reused_prediction_workspace_mib"] == (
        2 * 6 * 3 * 4 / 1024 / 1024
    )
    assert result["reused_decay_exp_mib"] == (
        args.decode_batch * 6 * 4 / 1024 / 1024
    )
    assert all(item["max_abs_error"] < 1e-4 for item in result["errors"])


def test_delta_state_contraction_benchmark_isolates_full_state_products():
    args = SimpleNamespace(
        decode_batch=2,
        key_head_dim=4,
        value_head_dim=3,
        warmup=0,
        iterations=1,
        repeats=1,
    )

    result = MODULE.benchmark_delta_state_contraction(
        args,
        torch.device("cpu"),
        local_key_heads=2,
        local_value_heads=6,
    )

    assert result["avoided_state_product_mib_per_contraction"] == (
        2 * 6 * 4 * 3 * 4 / 1024 / 1024
    )
    assert result["state_contractions_per_decode"] == 2
    assert all(item["max_abs_error"] < 1e-5 for item in result["errors"])


def test_graph_safe_candidate_requires_every_promotion_gate():
    promoted = MODULE.evaluate_graph_safe_moe_candidate(
        device_type="cuda",
        speedup=1.2,
        peak_extra_mib=32.0,
        max_abs_error=0.01,
        min_speedup=1.05,
        max_peak_extra_mib=64.0,
        max_allowed_abs_error=0.05,
    )
    too_slow = MODULE.evaluate_graph_safe_moe_candidate(
        device_type="cuda",
        speedup=1.0,
        peak_extra_mib=32.0,
        max_abs_error=0.01,
        min_speedup=1.05,
        max_peak_extra_mib=64.0,
        max_allowed_abs_error=0.05,
    )

    assert promoted["promote_to_runtime"]
    assert not too_slow["promote_to_runtime"]
    assert not too_slow["checks"]["speedup"]
