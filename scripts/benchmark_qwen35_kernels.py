"""Reproducible microbenchmarks for Qwen3.5 routing and DeltaNet paths."""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_source_module(name: str, relative_path: str):
    spec = spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GDN = load_source_module(
    "qwen35_gated_delta_benchmark",
    "nanovllm/models/qwen35_gated_delta.py",
)
METADATA = load_source_module(
    "qwen35_benchmark_metadata",
    "nanovllm/benchmark_metadata.py",
)
MOE_DISPATCH = load_source_module(
    "qwen35_moe_dispatch_benchmark",
    "nanovllm/models/moe_dispatch.py",
)
KV_QUANT = load_source_module(
    "qwen35_kv_quant_benchmark",
    "nanovllm/layers/kv_cache_quant.py",
)
SAMPLER = load_source_module(
    "qwen35_sampler_benchmark",
    "nanovllm/layers/sampler.py",
)
ROTARY = load_source_module(
    "qwen35_rotary_benchmark",
    "nanovllm/layers/rotary_embedding.py",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    peak_extra_bytes = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        else:
            baseline = 0
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1_000 / iterations)
        if device.type == "cuda":
            peak_extra_bytes.append(
                max(torch.cuda.max_memory_allocated(device) - baseline, 0)
            )

    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "peak_extra_mib_samples": [
            value / 1024 / 1024 for value in peak_extra_bytes
        ],
        "peak_extra_mib": (
            max(peak_extra_bytes) / 1024 / 1024 if peak_extra_bytes else 0.0
        ),
    }


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    matching_nonfinite = (~torch.isfinite(actual_float)) & (
        actual_float == expected_float
    )
    difference = torch.where(
        matching_nonfinite,
        torch.zeros_like(actual_float),
        actual_float - expected_float,
    )
    denominator = expected_float.abs().clamp_min(1e-6)
    denominator = torch.where(
        matching_nonfinite,
        torch.ones_like(denominator),
        denominator,
    )
    return {
        "max_abs_error": difference.abs().max().item(),
        "max_relative_error": (difference.abs() / denominator).max().item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def compare(
    reference: Callable[[], tuple[torch.Tensor, ...]],
    candidate: Callable[[], tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
    measure_reference: bool = True,
) -> dict:
    expected = reference()
    actual = candidate()
    if len(actual) != len(expected):
        raise RuntimeError("candidate and reference output counts differ")
    for index, (value, target) in enumerate(zip(actual, expected)):
        if value.shape != target.shape:
            raise RuntimeError(
                f"output {index} shape differs: {tuple(value.shape)} != "
                f"{tuple(target.shape)}"
            )
    errors = [error(value, target) for value, target in zip(actual, expected)]
    reference_timing = (
        measure(
            reference,
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        if measure_reference
        else None
    )
    candidate_timing = measure(
        candidate,
        device=device,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    return {
        "reference": reference_timing,
        "candidate": candidate_timing,
        "speedup": (
            reference_timing["median_ms"] / candidate_timing["median_ms"]
            if reference_timing is not None
            else None
        ),
        "errors": errors,
    }


def benchmark_router(args, device, dtype) -> dict:
    logits = torch.randn(
        args.router_tokens,
        args.num_experts,
        device=device,
        dtype=dtype,
    )

    def reference():
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, args.top_k, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), ids

    def candidate():
        values, ids = torch.topk(logits, args.top_k, dim=-1)
        values = values.float()
        torch.softmax(values, dim=-1, out=values)
        return values, ids

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["full_probability_mib"] = (
        args.router_tokens * args.num_experts * 4 / 1024 / 1024
    )
    result["selected_probability_mib"] = (
        args.router_tokens * args.top_k * 4 / 1024 / 1024
    )
    result["reused_selected_logits_mib"] = result["selected_probability_mib"]
    return result


def _full_sort_sampling_filter(logits, top_ks, top_ps):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    vocab_size = logits.size(1)
    full_vocab = torch.full_like(top_ks, vocab_size)
    effective_top_ks = torch.where(
        top_ks > 0,
        torch.minimum(top_ks, full_vocab),
        full_vocab,
    )
    ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    top_k_keep = ranks < effective_top_ks.unsqueeze(1)
    top_k_logits = sorted_logits.masked_fill(~top_k_keep, float("-inf"))
    sorted_probs = torch.softmax(top_k_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_ps.unsqueeze(1)
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False
    filtered = torch.empty_like(logits)
    filtered.scatter_(
        -1,
        sorted_indices,
        sorted_logits.masked_fill(~(top_k_keep & ~sorted_remove), float("-inf")),
    )
    return filtered


def benchmark_sampling_filter(args, device, dtype) -> dict:
    logits = torch.randn(
        args.sampling_batch,
        args.vocab_size,
        device=device,
        dtype=dtype,
    )
    full_sort_workspace_mib = (
        logits.numel()
        * (logits.element_size() + 8)
        / 1024
        / 1024
    )
    results = {}
    cases = (
        ("unfiltered", -1, 1.0),
        ("top_k", args.sampling_top_k, 1.0),
        ("top_k_top_p", args.sampling_top_k, args.sampling_top_p),
    )
    for name, top_k, top_p in cases:
        metadata = SAMPLER.build_sampling_metadata(
            [1.0] * args.sampling_batch,
            [top_k] * args.sampling_batch,
            [top_p] * args.sampling_batch,
            args.vocab_size,
        )
        top_ks = torch.full(
            (args.sampling_batch,),
            top_k,
            device=device,
            dtype=torch.int32,
        )
        top_ps = torch.full(
            (args.sampling_batch,),
            top_p,
            device=device,
            dtype=torch.float32,
        )

        def reference():
            return (_full_sort_sampling_filter(logits, top_ks, top_ps),)

        def candidate():
            return (
                SAMPLER.apply_top_k_top_p(
                    logits,
                    top_ks,
                    top_ps,
                    metadata,
                ),
            )

        result = compare(
            reference,
            candidate,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        # BF16 logits contain many ties. ``sort`` and ``topk`` may retain
        # different token ids at a tied cutoff while producing the same
        # filtered distribution. Compare the sorted value multisets for
        # correctness without adding this canonicalization to timed calls.
        expected_values = torch.sort(reference()[0], descending=True, dim=-1).values
        actual_values = torch.sort(candidate()[0], descending=True, dim=-1).values
        result["errors"] = [error(actual_values, expected_values)]
        result["avoided_full_sort_workspace_mib"] = full_sort_workspace_mib
        result["uses_host_sampling_metadata"] = True
        results[name] = result
    return results


def benchmark_compact_top_k_sampling(args, device, dtype) -> dict:
    logits = torch.randn(
        args.sampling_batch,
        args.vocab_size,
        device=device,
        dtype=dtype,
    )
    temperatures = torch.ones(args.sampling_batch, device=device)
    top_ks = torch.full(
        (args.sampling_batch,),
        args.sampling_top_k,
        device=device,
        dtype=torch.int32,
    )
    top_ps = torch.full(
        (args.sampling_batch,),
        args.sampling_top_p,
        device=device,
    )

    def reference():
        filtered = SAMPLER.apply_top_k_top_p(
            logits.float().div(temperatures.unsqueeze(1)),
            top_ks,
            top_ps,
        )
        return torch.softmax(filtered, dim=-1)

    def candidate():
        selected, indices = SAMPLER.compact_top_k_logits(
            logits,
            temperatures,
            top_ks,
            top_ps,
            args.sampling_top_k,
            args.sampling_top_p < 1.0,
        )
        return torch.softmax(selected, dim=-1), indices

    expected = reference()
    selected_probs, selected_indices = candidate()
    actual = torch.zeros_like(expected).scatter_(
        -1,
        selected_indices,
        selected_probs,
    )
    reference_timing = measure(
        reference,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    candidate_timing = measure(
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    full_fp32_mib = logits.numel() * 4 / 1024 / 1024
    compact_fp32_mib = (
        args.sampling_batch * args.sampling_top_k * 4 / 1024 / 1024
    )
    return {
        "reference": reference_timing,
        "candidate": candidate_timing,
        "speedup": reference_timing["median_ms"] / candidate_timing["median_ms"],
        "errors": [error(actual, expected)],
        "full_fp32_logits_mib": full_fp32_mib,
        "compact_fp32_logits_mib": compact_fp32_mib,
        "avoided_fp32_logits_mib": full_fp32_mib - compact_fp32_mib,
    }


def benchmark_greedy_sampler(args, device, dtype) -> dict:
    logits = torch.randn(
        args.sampling_batch,
        args.vocab_size,
        device=device,
        dtype=dtype,
    )
    temperatures = torch.zeros(args.sampling_batch, device=device)
    top_ks = torch.full(
        (args.sampling_batch,),
        -1,
        device=device,
        dtype=torch.int32,
    )
    top_ps = torch.ones(args.sampling_batch, device=device)
    sampler = SAMPLER.Sampler()
    metadata = SAMPLER.build_sampling_metadata(
        [0.0] * args.sampling_batch,
        [-1] * args.sampling_batch,
        [1.0] * args.sampling_batch,
        args.vocab_size,
    )

    def reference():
        return (logits.float().argmax(dim=-1),)

    def candidate():
        return (sampler(logits, temperatures, top_ks, top_ps, metadata),)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["avoided_fp32_logits_mib"] = (
        logits.numel() * max(4 - logits.element_size(), 0) / 1024 / 1024
    )
    result["uses_host_sampling_metadata"] = True
    return result


def benchmark_moe_output_merge(args, device, dtype) -> dict:
    routed_source = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    shared_source = torch.randn_like(routed_source)
    gate_source = torch.randn(
        args.router_tokens,
        1,
        device=device,
        dtype=dtype,
    )

    def reference():
        routed = routed_source.clone()
        shared = shared_source.clone()
        gate = gate_source.clone()
        return (routed + torch.sigmoid(gate) * shared,)

    def candidate():
        routed = routed_source.clone()
        shared = shared_source.clone()
        gate = gate_source.clone()
        gate.sigmoid_()
        shared.mul_(gate)
        routed.add_(shared)
        return (routed,)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    element_size = routed_source.element_size()
    result["reused_routed_output_mib"] = (
        routed_source.numel() * element_size / 1024 / 1024
    )
    result["reused_shared_output_mib"] = (
        shared_source.numel() * element_size / 1024 / 1024
    )
    result["reused_gate_mib"] = gate_source.numel() * element_size / 1024 / 1024
    return result


def benchmark_sorted_route_weighting(args, device, dtype) -> dict:
    expert_output_source = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    route_weights = torch.rand(
        args.router_tokens,
        device=device,
        dtype=dtype,
    )

    def reference():
        expert_output = expert_output_source.clone()
        return (expert_output * route_weights.unsqueeze(-1),)

    def candidate():
        expert_output = expert_output_source.clone()
        return (
            MOE_DISPATCH.weight_expert_output(
                expert_output,
                route_weights,
            ),
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["avoided_weighted_expert_output_mib"] = (
        expert_output_source.numel()
        * expert_output_source.element_size()
        / 1024
        / 1024
    )
    return result


def benchmark_residual_merge(args, device, dtype) -> dict:
    residual_source = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    branch_source = torch.randn_like(residual_source)

    def reference():
        residual = residual_source.clone()
        branch = branch_source.clone()
        return (residual + branch,)

    def candidate():
        residual = residual_source.clone()
        branch = branch_source.clone()
        return (branch.add_(residual),)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reused_branch_output_mib_per_merge"] = (
        branch_source.numel()
        * branch_source.element_size()
        / 1024
        / 1024
    )
    result["residual_merges_per_decoder_layer"] = 2
    return result


def benchmark_torch_kv_dequant(args, device, dtype, num_kv_heads: int) -> dict:
    block_size = 256
    num_blocks = max(
        1,
        (args.prefill_tokens + block_size - 1) // block_size,
    )
    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
        args.key_head_dim,
    )
    k_cache = torch.randint(-127, 128, cache_shape, device=device, dtype=torch.int8)
    v_cache = torch.randint(-127, 128, cache_shape, device=device, dtype=torch.int8)
    scale_shape = cache_shape[:-1]
    k_scale = torch.rand(scale_shape, device=device, dtype=torch.float16)
    v_scale = torch.rand(scale_shape, device=device, dtype=torch.float16)
    block_ids = torch.arange(num_blocks, device=device, dtype=torch.int32)

    def reference():
        ids = block_ids.long()
        k = k_cache.index_select(0, ids).to(dtype)
        v = v_cache.index_select(0, ids).to(dtype)
        return (
            k * k_scale.index_select(0, ids).unsqueeze(-1).to(dtype),
            v * v_scale.index_select(0, ids).unsqueeze(-1).to(dtype),
        )

    def candidate():
        return KV_QUANT.dequant_selected_kvcache_torch(
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_ids,
            dtype,
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    output_bytes = math.prod(cache_shape) * torch.empty(
        (), dtype=dtype
    ).element_size()
    result["avoided_output_workspace_mib"] = 2 * output_bytes / 1024 / 1024
    result["avoided_block_id_cast_mib"] = (
        block_ids.numel() * 8 / 1024 / 1024
    )
    result["selected_blocks"] = num_blocks
    return result


def expert_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    if hidden_states.shape[0] == 1:
        route_order = torch.argsort(topk_ids[0], stable=True)
        routes = torch.stack(
            (topk_ids[0, route_order], route_order), dim=1
        ).cpu().tolist()
        for expert_id, route_index in routes:
            gate_up = F.linear(hidden_states, gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                down_proj[expert_id],
            )
            output.add_(expert_output * topk_weights[0, route_index])
        return output

    return expert_dispatch_general(
        hidden_states,
        topk_ids,
        topk_weights,
        gate_up_proj,
        down_proj,
    )


def expert_dispatch_general(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Previous general sorted dispatch, retained as a benchmark baseline."""

    if output is None:
        output = torch.zeros_like(hidden_states)
    else:
        output.zero_()

    assignments = topk_ids.reshape(-1)
    routing_weights = topk_weights.reshape(-1)
    order = torch.argsort(assignments, stable=True)
    sorted_experts = assignments[order]
    sorted_tokens = torch.div(order, topk_ids.shape[1], rounding_mode="floor")
    sorted_weights = routing_weights[order]
    active_experts, counts = torch.unique_consecutive(
        sorted_experts,
        return_counts=True,
    )
    groups = torch.stack((active_experts, counts), dim=1).cpu().tolist()
    offset = 0
    for expert_id, count in groups:
        end = offset + count
        token_index = sorted_tokens[offset:end]
        gate_up = F.linear(hidden_states[token_index], gate_up_proj[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, down_proj[expert_id])
        output.index_add_(
            0,
            token_index,
            expert_output * sorted_weights[offset:end].unsqueeze(-1),
        )
        offset = end
    return output


def expert_dispatch_mixed(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    decode_token_count: int,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Mirror runtime mixed dispatch with one shared output buffer."""

    if not 0 < decode_token_count < hidden_states.shape[0]:
        raise ValueError("mixed decode token count must split the batch")
    output = torch.zeros_like(hidden_states)
    MOE_DISPATCH.batched_expert_dispatch(
        hidden_states[:decode_token_count],
        topk_ids[:decode_token_count],
        topk_weights[:decode_token_count],
        gate_up_proj,
        down_proj,
        chunk_size,
        output=output[:decode_token_count],
    )
    expert_dispatch_general(
        hidden_states[decode_token_count:],
        topk_ids[decode_token_count:],
        topk_weights[decode_token_count:],
        gate_up_proj,
        down_proj,
        output=output[decode_token_count:],
    )
    return output


def expert_dispatch_batched_decode(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    return MOE_DISPATCH.batched_expert_dispatch(
        hidden_states,
        topk_ids,
        topk_weights,
        gate_up_proj,
        down_proj,
        chunk_size,
    )


def expert_dispatch_batched_repeated_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Previous batched path that materialized one hidden row per route."""

    output = torch.empty_like(hidden_states)
    top_k = topk_ids.shape[1]
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        expert_ids = topk_ids[start:end].reshape(-1)
        selected_gate_up = gate_up_proj.index_select(0, expert_ids)
        route_hidden = (
            hidden_states[start:end]
            .unsqueeze(1)
            .expand(-1, top_k, -1)
            .reshape(expert_ids.numel(), -1, 1)
        )
        gate_up = torch.bmm(selected_gate_up, route_hidden).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = MOE_DISPATCH.silu_and_mul(gate, up)
        selected_down = down_proj.index_select(0, expert_ids)
        expert_output = torch.bmm(
            selected_down,
            activated.unsqueeze(-1),
        ).squeeze(-1)
        output[start:end] = MOE_DISPATCH.weighted_route_sum(
            expert_output,
            topk_weights[start:end],
        )
    return output


def evaluate_graph_safe_moe_candidate(
    *,
    device_type: str,
    speedup: float,
    peak_extra_mib: float,
    max_abs_error: float,
    min_speedup: float,
    max_peak_extra_mib: float,
    max_allowed_abs_error: float,
) -> dict:
    checks = {
        "cuda_measurement": device_type == "cuda",
        "speedup": speedup >= min_speedup,
        "peak_memory": peak_extra_mib <= max_peak_extra_mib,
        "accuracy": max_abs_error <= max_allowed_abs_error,
    }
    return {
        "promote_to_runtime": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_speedup": min_speedup,
            "max_peak_extra_mib": max_peak_extra_mib,
            "max_abs_error": max_allowed_abs_error,
        },
    }


def expert_dispatch_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Match the Transformers one-hot expert dispatch implementation."""

    output = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(
            topk_ids,
            num_classes=gate_up_proj.shape[0],
        ).permute(2, 1, 0)
        active_experts = torch.greater(
            expert_mask.sum(dim=(-1, -2)),
            0,
        ).nonzero()
    for expert_index in active_experts:
        expert_id = expert_index[0]
        route_index, token_index = torch.where(expert_mask[expert_id])
        gate_up = F.linear(hidden_states[token_index], gate_up_proj[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, down_proj[expert_id])
        output.index_add_(
            0,
            token_index,
            expert_output * topk_weights[token_index, route_index, None],
        )
    return output


def benchmark_expert_dispatch(args, device, dtype, token_count: int) -> dict:
    local_intermediate_size = args.moe_intermediate_size // args.tp_size
    hidden = torch.randn(
        token_count,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    topk_ids = torch.randint(
        args.num_experts,
        (token_count, args.top_k),
        device=device,
    )
    topk_weights = torch.rand(
        token_count,
        args.top_k,
        device=device,
        dtype=dtype,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up_proj = torch.randn(
        args.num_experts,
        2 * local_intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    down_proj = torch.randn(
        args.num_experts,
        args.hidden_size,
        local_intermediate_size,
        device=device,
        dtype=dtype,
    )

    def reference():
        return (
            expert_dispatch_reference(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
            ),
        )

    def candidate():
        return (
            expert_dispatch(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
            ),
        )

    output = candidate()[0]
    if not torch.isfinite(output).all():
        raise RuntimeError("expert dispatch produced non-finite output")
    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result.update(
        {
            "tokens": token_count,
            "single_token_decode_fast_path": token_count == 1,
            "routes": token_count * args.top_k,
            "active_experts": torch.unique(topk_ids).numel(),
            "local_intermediate_size": local_intermediate_size,
            "estimated_model_moe_ms": (
                result["candidate"]["median_ms"] * args.num_hidden_layers
            ),
        }
    )
    if token_count == 1:
        general_timing = measure(
            lambda: expert_dispatch_general(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
            ),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        result["general_dispatch_baseline"] = general_timing
        result["decode_fast_path_speedup"] = (
            general_timing["median_ms"] / result["candidate"]["median_ms"]
        )
    if token_count <= args.max_decode_tokens:
        chunk_sizes = tuple(
            sorted(
                set(args.moe_decode_chunk_sizes)
                | {args.moe_decode_chunk_size}
            )
        )
        graph_safe_candidates = {}
        for chunk_size in chunk_sizes:
            graph_safe_output = expert_dispatch_batched_decode(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
                chunk_size,
            )
            graph_safe_timing = measure(
                lambda chunk_size=chunk_size: expert_dispatch_batched_decode(
                    hidden,
                    topk_ids,
                    topk_weights,
                    gate_up_proj,
                    down_proj,
                    chunk_size,
                ),
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            repeated_output = expert_dispatch_batched_repeated_input(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
                chunk_size,
            )
            repeated_timing = measure(
                lambda chunk_size=chunk_size: (
                    expert_dispatch_batched_repeated_input(
                        hidden,
                        topk_ids,
                        topk_weights,
                        gate_up_proj,
                        down_proj,
                        chunk_size,
                    )
                ),
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            broadcast_error = error(graph_safe_output, repeated_output)
            broadcast_speedup = (
                repeated_timing["median_ms"] / graph_safe_timing["median_ms"]
            )
            broadcast_peak_delta = (
                graph_safe_timing["peak_extra_mib"]
                - repeated_timing["peak_extra_mib"]
            )
            graph_safe_timing["broadcast_route_input"] = {
                "valid": (
                    device.type == "cuda"
                    and broadcast_speedup >= 1.0
                    and broadcast_peak_delta <= 0.0
                    and broadcast_error["max_abs_error"]
                    <= args.moe_graph_safe_max_abs_error
                ),
                "measured_on_cuda": device.type == "cuda",
                "speedup_vs_repeated_input": broadcast_speedup,
                "peak_extra_mib_delta": broadcast_peak_delta,
                "errors": broadcast_error,
                "reference": repeated_timing,
            }
            graph_safe_timing.update(
                {
                    "chunk_size": chunk_size,
                    "speedup_vs_current": (
                        result["candidate"]["median_ms"]
                        / graph_safe_timing["median_ms"]
                    ),
                    "errors_vs_current": error(graph_safe_output, output),
                    "estimated_selected_weight_mib": (
                        min(token_count, chunk_size)
                        * args.top_k
                        * 2
                        * local_intermediate_size
                        * args.hidden_size
                        * hidden.element_size()
                        / 1024
                        / 1024
                    ),
                    "reused_weighted_route_mib": (
                        min(token_count, chunk_size)
                        * args.top_k
                        * args.hidden_size
                        * hidden.element_size()
                        / 1024
                        / 1024
                    ),
                }
            )
            graph_safe_timing["promotion"] = evaluate_graph_safe_moe_candidate(
                device_type=device.type,
                speedup=graph_safe_timing["speedup_vs_current"],
                peak_extra_mib=graph_safe_timing["peak_extra_mib"],
                max_abs_error=graph_safe_timing["errors_vs_current"][
                    "max_abs_error"
                ],
                min_speedup=args.moe_graph_safe_min_speedup,
                max_peak_extra_mib=args.moe_graph_safe_max_peak_extra_mib,
                max_allowed_abs_error=args.moe_graph_safe_max_abs_error,
            )
            graph_safe_candidates[str(chunk_size)] = graph_safe_timing
        promoted = [
            item
            for item in graph_safe_candidates.values()
            if item["promotion"]["promote_to_runtime"]
        ]
        result["graph_safe_chunk_sweep"] = {
            "candidates": graph_safe_candidates,
            "recommended_chunk_size": (
                min(promoted, key=lambda item: item["median_ms"])["chunk_size"]
                if promoted
                else None
            ),
        }
        result["graph_safe_batched_candidate"] = graph_safe_candidates[
            str(args.moe_decode_chunk_size)
        ]
    return result


def benchmark_expert_dispatch_sweep(args, device, dtype) -> dict[str, dict]:
    """Measure decode- through prefill-sized expert routing workloads."""

    return {
        str(token_count): benchmark_expert_dispatch(
            args,
            device,
            dtype,
            token_count,
        )
        for token_count in args.expert_token_counts
    }


def benchmark_mixed_expert_dispatch(
    args,
    device,
    dtype,
    decode_tokens: int | None = None,
    prefill_tokens: int | None = None,
) -> dict:
    """Compare whole-batch grouped dispatch with decode/prefill splitting."""

    decode_tokens = (
        args.mixed_decode_tokens if decode_tokens is None else decode_tokens
    )
    prefill_tokens = (
        args.mixed_prefill_tokens if prefill_tokens is None else prefill_tokens
    )
    token_count = decode_tokens + prefill_tokens
    local_intermediate_size = args.moe_intermediate_size // args.tp_size
    hidden = torch.randn(
        token_count,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    topk_ids = torch.randint(
        args.num_experts,
        (token_count, args.top_k),
        device=device,
    )
    topk_weights = torch.rand(
        token_count,
        args.top_k,
        device=device,
        dtype=dtype,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    gate_up_proj = torch.randn(
        args.num_experts,
        2 * local_intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    down_proj = torch.randn(
        args.num_experts,
        args.hidden_size,
        local_intermediate_size,
        device=device,
        dtype=dtype,
    )

    def baseline():
        return (
            expert_dispatch_general(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
            ),
        )

    def candidate():
        return (
            expert_dispatch_mixed(
                hidden,
                topk_ids,
                topk_weights,
                gate_up_proj,
                down_proj,
                decode_tokens,
                args.moe_decode_chunk_size,
            ),
        )

    result = compare(
        baseline,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result.update(
        {
            "decode_tokens": decode_tokens,
            "prefill_tokens": prefill_tokens,
            "avoided_route_hidden_allocation_mib_per_step": (
                decode_tokens
                * args.top_k
                * args.hidden_size
                * hidden.element_size()
                / (1024**2)
            ),
            "speedup_vs_grouped": (
                result["reference"]["median_ms"]
                / result["candidate"]["median_ms"]
            ),
            "measured_on_cuda": device.type == "cuda",
        }
    )
    return result


def benchmark_mixed_expert_dispatch_sweep(args, device, dtype) -> dict[str, dict]:
    """Measure representative mixed batches from latency to throughput loads."""

    return {
        f"decode{decode_tokens}_prefill{prefill_tokens}": (
            benchmark_mixed_expert_dispatch(
                args,
                device,
                dtype,
                decode_tokens,
                prefill_tokens,
            )
        )
        for decode_tokens, prefill_tokens in zip(
            args.mixed_decode_token_counts,
            args.mixed_prefill_token_counts,
            strict=True,
        )
    }


def recommend_moe_decode_chunk_size(
    dispatch_results: dict[str, dict],
    max_decode_tokens: int,
) -> dict:
    measured = {
        batch: result
        for batch, result in dispatch_results.items()
        if int(batch) <= max_decode_tokens and "graph_safe_chunk_sweep" in result
    }
    if not measured:
        raise ValueError("no graph-safe decode batches were measured")
    common_chunks = set.intersection(
        *(
            set(result["graph_safe_chunk_sweep"]["candidates"])
            for result in measured.values()
        )
    )
    candidates = {}
    for chunk in sorted(common_chunks, key=int):
        measurements = [
            result["graph_safe_chunk_sweep"]["candidates"][chunk]
            for result in measured.values()
        ]
        candidates[chunk] = {
            "all_batches_promoted": all(
                item["promotion"]["promote_to_runtime"]
                for item in measurements
            ),
            "worst_speedup": min(
                item["speedup_vs_current"] for item in measurements
            ),
            "max_peak_extra_mib": max(
                item["peak_extra_mib"] for item in measurements
            ),
            "total_median_ms": sum(item["median_ms"] for item in measurements),
        }
    eligible = [
        (int(chunk), item)
        for chunk, item in candidates.items()
        if item["all_batches_promoted"]
    ]
    recommended = (
        max(
            eligible,
            key=lambda pair: (
                pair[1]["worst_speedup"],
                -pair[1]["total_median_ms"],
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "measured_decode_batches": sorted(map(int, measured)),
        "recommended_chunk_size": recommended,
        "candidates": candidates,
    }


def benchmark_rmsnorm(args, device, dtype) -> dict:
    x = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)
    gain = 1.0 + weight.float()
    eps = 1e-6

    def reference():
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        return ((normalized * (1.0 + weight.float())).to(dtype),)

    def candidate():
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        if x_float is not x:
            x_float.mul_(inverse_rms)
            x_float.mul_(gain)
            return (x_float.to(dtype),)
        normalized = x_float * inverse_rms
        return ((normalized * gain).to(dtype),)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["avoided_fp32_copy_mib"] = x.numel() * 4 / 1024 / 1024
    result["eliminated_per_call_gain_materialization_mib"] = (
        gain.numel() * gain.element_size() / 1024 / 1024
    )
    result["persistent_gain_storage_mib"] = (
        gain.numel() * gain.element_size() / 1024 / 1024
    )
    result["persistent_storage_delta_mib"] = (
        gain.numel() * (gain.element_size() - weight.element_size())
        / 1024
        / 1024
    )
    result["candidate_reuses_fp32_workspace"] = dtype != torch.float32
    result["candidate_uses_precomputed_gain"] = True
    return result


def benchmark_delta_l2_normalization(
    args,
    device,
    dtype,
    local_key_heads: int,
) -> dict:
    shape = (
        args.router_tokens,
        local_key_heads,
        args.key_head_dim,
    )
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn_like(query)

    def reference():
        return (
            GDN.l2_normalize(query.float()),
            GDN.l2_normalize(key.float()),
        )

    def candidate():
        query_workspace = query.float()
        key_workspace = key.float()
        return (
            GDN.l2_normalize(query_workspace, inplace_output=True),
            GDN.l2_normalize(key_workspace, inplace_output=True),
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reused_query_key_fp32_mib"] = (
        2 * query.numel() * 4 / 1024 / 1024
    )
    return result


def benchmark_attention_norm_output_reuse(args, device, dtype) -> dict:
    x = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        args.hidden_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    gain = torch.randn(args.hidden_size, device=device, dtype=torch.float32)
    eps = 1e-6

    def reference():
        projected = F.linear(x, weight)
        projected_float = projected.float()
        projected_float.mul_(
            torch.rsqrt(
                projected_float.square().mean(dim=-1, keepdim=True) + eps
            )
        )
        projected_float.mul_(gain)
        return (projected_float.to(dtype),)

    def candidate():
        projected = F.linear(x, weight)
        projected_float = projected.float()
        projected_float.mul_(
            torch.rsqrt(
                projected_float.square().mean(dim=-1, keepdim=True) + eps
            )
        )
        projected_float.mul_(gain)
        projected.copy_(projected_float)
        return (projected,)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reused_projection_output_mib"] = (
        x.numel() * x.element_size() / 1024 / 1024
    )
    return result


def benchmark_rotary_output_reuse(args, device, dtype) -> dict:
    local_heads = args.total_key_heads // args.tp_size
    query = torch.randn(
        args.router_tokens,
        local_heads,
        args.key_head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    rotary_dim = max(2, args.key_head_dim // 4)
    rotary_dim -= rotary_dim % 2
    cos = torch.randn(
        args.router_tokens,
        1,
        rotary_dim // 2,
        device=device,
    )
    sin = torch.randn_like(cos)

    def run(inplace_output):
        q = query.clone()
        k = key.clone()
        return (
            ROTARY.apply_rotary_emb(
                q, cos, sin, inplace_output=inplace_output
            ),
            ROTARY.apply_rotary_emb(
                k, cos, sin, inplace_output=inplace_output
            ),
        )

    with torch.inference_mode():
        result = compare(
            lambda: run(False),
            lambda: run(True),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    result["reused_query_key_output_mib"] = (
        (query.numel() + key.numel()) * query.element_size() / 1024 / 1024
    )
    result["rotary_dim"] = rotary_dim
    return result


def benchmark_vocab_gather_layout(args, device, dtype) -> dict:
    if args.vocab_size % args.tp_size:
        raise ValueError("vocabulary size must divide TP size")
    local_vocab_size = args.vocab_size // args.tp_size
    shards = tuple(
        torch.randn(
            args.sampling_batch,
            local_vocab_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(args.tp_size)
    )

    def reference():
        gathered = [shard.clone() for shard in shards]
        return (torch.cat(gathered, dim=-1),)

    def candidate():
        gathered = torch.empty(
            args.vocab_size,
            args.sampling_batch,
            device=device,
            dtype=dtype,
        )
        for destination, shard in zip(
            gathered.split(local_vocab_size, dim=0),
            shards,
        ):
            destination.copy_(shard.transpose(0, 1))
        return (gathered.transpose(0, 1),)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["avoided_full_vocab_copy_mib"] = (
        args.sampling_batch
        * args.vocab_size
        * torch.empty((), dtype=dtype).element_size()
        / 1024
        / 1024
    )
    result["candidate_returns_transpose_view"] = True
    return result


def benchmark_beta_gate(args, device, dtype, local_value_heads: int) -> dict:
    hidden = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        local_value_heads,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    def reference():
        return (torch.sigmoid(F.linear(hidden, weight)),)

    def candidate():
        beta = F.linear(hidden, weight)
        beta.sigmoid_()
        return (beta,)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reused_beta_projection_mib"] = (
        args.router_tokens
        * local_value_heads
        * torch.empty((), dtype=dtype).element_size()
        / 1024
        / 1024
    )
    return result


def benchmark_gated_delta_packed_projection(
    args,
    device,
    dtype,
    local_value_heads: int,
) -> dict:
    hidden = torch.randn(
        args.decode_batch,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    local_value_dim = local_value_heads * args.value_head_dim
    output_sizes = (local_value_dim, local_value_heads, local_value_heads)
    weights = tuple(
        torch.randn(size, args.hidden_size, device=device, dtype=dtype)
        for size in output_sizes
    )
    packed_weight = torch.cat(weights, dim=0)

    def reference():
        return tuple(F.linear(hidden, weight) for weight in weights)

    def candidate():
        return F.linear(hidden, packed_weight).split(output_sizes, dim=-1)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reference_gemm_launches"] = 3
    result["candidate_gemm_launches"] = 1
    result["avoided_gemm_launches"] = 2
    return result


def benchmark_attention_packed_qkv(
    args,
    device,
    dtype,
    local_query_heads: int,
    local_kv_heads: int,
) -> dict:
    hidden = torch.randn(
        args.decode_batch,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    output_sizes = (
        2 * local_query_heads * args.attention_head_dim,
        local_kv_heads * args.attention_head_dim,
        local_kv_heads * args.attention_head_dim,
    )
    weights = tuple(
        torch.randn(size, args.hidden_size, device=device, dtype=dtype)
        for size in output_sizes
    )
    packed_weight = torch.cat(weights, dim=0)

    def reference():
        return tuple(F.linear(hidden, weight) for weight in weights)

    def candidate():
        query, key, value = F.linear(hidden, packed_weight).split(
            output_sizes,
            dim=-1,
        )
        return query, key.clone(), value

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reference_gemm_launches"] = 3
    result["candidate_gemm_launches"] = 1
    result["avoided_gemm_launches"] = 2
    result["key_alias_break_copy_mib"] = (
        args.decode_batch
        * local_kv_heads
        * args.attention_head_dim
        * torch.empty((), dtype=dtype).element_size()
        / 1024
        / 1024
    )
    return result


def benchmark_contiguous_decode_state(
    args,
    device,
    dtype,
    local_value_heads: int,
    local_conv_channels: int,
) -> dict:
    state_shape = (
        args.decode_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
    )
    conv_shape = (
        args.decode_batch,
        local_conv_channels,
        args.conv_kernel_size,
    )
    indexed_state = torch.zeros(state_shape, device=device, dtype=torch.float32)
    indexed_conv = torch.zeros(conv_shape, device=device, dtype=dtype)
    view_state = indexed_state.clone()
    view_conv = indexed_conv.clone()
    slots = torch.arange(args.decode_batch, device=device)
    state_delta = torch.randn_like(indexed_state)
    conv_delta = torch.randn_like(indexed_conv)

    def reference():
        recurrent = indexed_state[slots]
        convolution = indexed_conv[slots]
        recurrent.add_(state_delta)
        convolution.add_(conv_delta)
        indexed_state[slots] = recurrent
        indexed_conv[slots] = convolution
        return recurrent, convolution

    def candidate():
        recurrent = view_state.narrow(0, 0, args.decode_batch)
        convolution = view_conv.narrow(0, 0, args.decode_batch)
        recurrent.add_(state_delta)
        convolution.add_(conv_delta)
        return recurrent, convolution

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    gathered_bytes = (
        indexed_state.numel() * indexed_state.element_size()
        + indexed_conv.numel() * indexed_conv.element_size()
    )
    result["avoided_state_gather_mib"] = gathered_bytes / 1024 / 1024
    result["avoided_state_scatter_mib"] = gathered_bytes / 1024 / 1024
    result["candidate_uses_cache_views"] = True
    return result


def benchmark_decay_rate(args, device, dtype, local_value_heads: int) -> dict:
    hidden = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        local_value_heads,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    dt_bias = torch.randn(local_value_heads, device=device, dtype=dtype)
    a_log = torch.randn(local_value_heads, device=device, dtype=torch.float32)
    decay_rate = -a_log.exp()

    def reference():
        a = F.linear(hidden, weight)
        return (-a_log.exp() * F.softplus(a.float() + dt_bias),)

    def candidate():
        a = F.linear(hidden, weight)
        a_float = a.float()
        a_float.add_(dt_bias)
        log_decay = F.softplus(a_float)
        log_decay.mul_(decay_rate)
        return (log_decay,)

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["precomputed_decay_rate_mib"] = (
        decay_rate.numel() * decay_rate.element_size() / 1024 / 1024
    )
    reused_workspace_mib = (
        args.router_tokens * local_value_heads * 4 / 1024 / 1024
    )
    result["reused_decay_projection_fp32_mib"] = reused_workspace_mib
    result["reused_softplus_output_mib"] = reused_workspace_mib
    return result


def benchmark_gated_rmsnorm(args, device, dtype) -> dict:
    hidden = torch.randn(
        args.router_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    gate = torch.randn_like(hidden)
    weight = torch.randn(args.hidden_size, device=device)
    eps = 1e-6

    def reference():
        hidden_float = hidden.float()
        normalized = hidden_float * torch.rsqrt(
            hidden_float.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        return (
            (
                normalized.to(dtype)
                * weight
                * F.silu(gate.float())
            ).to(dtype),
        )

    def candidate():
        hidden_float = hidden.float()
        inverse_rms = torch.rsqrt(
            hidden_float.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        if hidden_float is not hidden and gate.dtype != torch.float32:
            hidden_float.mul_(inverse_rms)
            normalized = hidden_float.to(dtype)
            gate_float = gate.float()
            F.silu(gate_float, inplace=True)
            torch.mul(normalized, weight, out=hidden_float)
            hidden_float.mul_(gate_float)
            return (hidden_float.to(dtype),)
        normalized = hidden_float * inverse_rms
        return (
            (
                normalized.to(dtype)
                * weight
                * F.silu(gate.float())
            ).to(dtype),
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    workspace_mib = hidden.numel() * 4 / 1024 / 1024
    result.update(
        {
            "reused_hidden_fp32_workspace_mib": workspace_mib,
            "reused_gate_fp32_workspace_mib": workspace_mib,
            "candidate_reuses_fp32_workspaces": dtype != torch.float32,
        }
    )
    return result


def benchmark_convolution(args, device, dtype, local_conv_channels) -> dict:
    x = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_conv_channels,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.prefill_batch,
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    result = compare(
        lambda: GDN.causal_conv1d_scan(x, state, weight),
        lambda: GDN.causal_conv1d_prefill(x, state, weight),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        measure_reference=not args.prefill_only,
    )
    history_elements = (
        args.prefill_batch
        * local_conv_channels
        * (args.conv_kernel_size - 1 + args.prefill_tokens)
    )
    state_elements = (
        args.prefill_batch * local_conv_channels * args.conv_kernel_size
    )
    result.update(
        {
            "compact_state_storage_mib": (
                state_elements * x.element_size() / (1024**2)
            ),
            "released_history_storage_mib": (
                (history_elements - state_elements)
                * x.element_size()
                / (1024**2)
            ),
            "next_state_owns_compact_storage": True,
        }
    )
    return result


def benchmark_decode_convolution(args, device, dtype, local_conv_channels) -> dict:
    x = torch.randn(
        args.decode_batch,
        local_conv_channels,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.decode_batch,
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(
        local_conv_channels,
        args.conv_kernel_size,
        device=device,
        dtype=dtype,
    )
    candidate_state = state.clone()

    def reference():
        return GDN.causal_conv1d_step(x, state, weight)

    def candidate():
        return GDN.causal_conv1d_step(
            x,
            candidate_state,
            weight,
            inplace_state=True,
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["reused_convolution_state_mib"] = (
        state.numel() * state.element_size() / 1024 / 1024
    )
    return result


def benchmark_delta_decode(
    args,
    device,
    dtype,
    local_key_heads,
    local_value_heads,
) -> dict:
    shape = (args.decode_batch, local_key_heads, args.key_head_dim)
    query = torch.randn(*shape, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(
        args.decode_batch,
        local_value_heads,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    decay = -torch.rand(
        args.decode_batch,
        local_value_heads,
        device=device,
    )
    beta = torch.rand(
        args.decode_batch,
        local_value_heads,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.decode_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    def reference():
        repeat_factor = local_value_heads // local_key_heads
        repeated_query = query.repeat_interleave(repeat_factor, dim=1)
        repeated_key = key.repeat_interleave(repeat_factor, dim=1)
        output, next_state = GDN.recurrent_gated_delta_rule(
            repeated_query.unsqueeze(1),
            repeated_key.unsqueeze(1),
            value.unsqueeze(1),
            decay.unsqueeze(1),
            beta.unsqueeze(1),
            state.clone(),
        )
        return output.squeeze(1), next_state

    def candidate():
        return GDN.recurrent_gated_delta_step(
            query,
            key,
            value,
            decay,
            beta,
            state.clone(),
            inplace_state=True,
            inplace_decay=True,
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    repeated_qk_elements = (
        2
        * args.decode_batch
        * (local_value_heads - local_key_heads)
        * args.key_head_dim
    )
    result["avoided_qk_replication_mib"] = (
        repeated_qk_elements * torch.empty((), dtype=dtype).element_size()
        / 1024
        / 1024
    )
    state_workspace_mib = state.numel() * 4 / 1024 / 1024
    result["reused_recurrent_state_mib"] = state_workspace_mib
    result["avoided_full_state_intermediates"] = 2
    result["reused_prediction_workspace_mib"] = (
        args.decode_batch
        * local_value_heads
        * args.value_head_dim
        * 4
        / 1024
        / 1024
    )
    result["reused_decay_exp_mib"] = (
        decay.numel() * decay.element_size() / 1024 / 1024
    )
    return result


def benchmark_delta_state_contraction(
    args,
    device,
    local_key_heads: int,
    local_value_heads: int,
) -> dict:
    groups = local_value_heads // local_key_heads
    state = torch.randn(
        args.decode_batch,
        local_key_heads,
        groups,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    query = torch.randn(
        args.decode_batch,
        local_key_heads,
        1,
        args.key_head_dim,
        device=device,
        dtype=torch.float32,
    )
    key = torch.randn_like(query)

    def reference():
        prediction = (state * key.unsqueeze(-1)).sum(dim=-2)
        output = (state * query.unsqueeze(-1)).sum(dim=-2)
        return prediction, output

    def candidate():
        prediction = torch.matmul(key.unsqueeze(-2), state).squeeze(-2)
        output = torch.matmul(query.unsqueeze(-2), state).squeeze(-2)
        return prediction, output

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    result["avoided_state_product_mib_per_contraction"] = (
        state.numel() * state.element_size() / 1024 / 1024
    )
    result["state_contractions_per_decode"] = 2
    return result


def benchmark_delta_prefill_head_groups(
    args,
    device,
    dtype,
    local_key_heads,
    local_value_heads,
    chunk_size=64,
) -> dict:
    query = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_key_heads,
        args.key_head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    decay = -torch.rand(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        device=device,
    )
    beta = torch.rand(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.prefill_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    repeat_factor = local_value_heads // local_key_heads

    def reference():
        return GDN.chunk_gated_delta_rule(
            query.repeat_interleave(repeat_factor, dim=2),
            key.repeat_interleave(repeat_factor, dim=2),
            value,
            decay,
            beta,
            state,
            chunk_size=chunk_size,
        )

    def candidate():
        return GDN.chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            state,
            chunk_size=chunk_size,
        )

    result = compare(
        reference,
        candidate,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        measure_reference=not args.prefill_only,
    )
    repeated_qk_elements = (
        2
        * args.prefill_batch
        * args.prefill_tokens
        * (local_value_heads - local_key_heads)
        * args.key_head_dim
    )
    result["avoided_qk_replication_mib"] = (
        repeated_qk_elements * torch.empty((), dtype=dtype).element_size()
        / 1024
        / 1024
    )
    effective_chunk = GDN.effective_chunk_size(
        args.prefill_tokens,
        chunk_size,
    )
    padded_tokens = (
        (args.prefill_tokens + effective_chunk - 1)
        // effective_chunk
        * effective_chunk
    )
    output_elements = (
        args.prefill_batch
        * local_value_heads
        * padded_tokens
        * args.value_head_dim
    )
    result["reused_fp32_output_buffer_mib"] = (
        output_elements * torch.empty((), dtype=torch.float32).element_size()
        / 1024
        / 1024
    )
    result["reused_fp32_correction_buffer_mib"] = (
        args.prefill_batch
        * local_value_heads
        * effective_chunk
        * args.value_head_dim
        * torch.empty((), dtype=torch.float32).element_size()
        / 1024
        / 1024
    )
    pairwise_elements = (
        args.prefill_batch
        * local_value_heads
        * (padded_tokens // effective_chunk)
        * effective_chunk
        * effective_chunk
    )
    result["reused_fp32_pairwise_buffer_mib"] = (
        pairwise_elements
        * torch.empty((), dtype=torch.float32).element_size()
        / 1024
        / 1024
    )
    result["chunk_size"] = chunk_size
    return result


def benchmark_delta_prefill_state_reuse(
    args,
    device,
    dtype,
    local_key_heads,
    local_value_heads,
    chunk_size=64,
) -> dict:
    query = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_key_heads,
        args.key_head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    decay = -torch.rand(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        device=device,
    )
    beta = torch.rand_like(decay, dtype=dtype)
    state = torch.randn(
        args.prefill_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=torch.float32,
    )

    def run(inplace_state):
        return GDN.chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            state,
            chunk_size=chunk_size,
            inplace_state=inplace_state,
        )

    with torch.inference_mode():
        result = compare(
            lambda: run(False),
            lambda: run(True),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    effective_chunk = GDN.effective_chunk_size(
        args.prefill_tokens,
        chunk_size,
    )
    num_chunks = (
        args.prefill_tokens + effective_chunk - 1
    ) // effective_chunk
    result.update(
        {
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "reused_recurrent_state_mib": (
                state.numel() * state.element_size() / 1024 / 1024
            ),
            "avoided_state_reallocations": max(num_chunks - 1, 0),
        }
    )
    return result


def benchmark_delta_prefill_chunk_sweep(
    args,
    device,
    dtype,
    local_key_heads,
    local_value_heads,
) -> dict:
    query = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_key_heads,
        args.key_head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )
    decay = -torch.rand(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        device=device,
    )
    beta = torch.rand(
        args.prefill_batch,
        args.prefill_tokens,
        local_value_heads,
        device=device,
        dtype=dtype,
    )
    state = torch.randn(
        args.prefill_batch,
        local_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=dtype,
    )

    def run(chunk_size):
        return GDN.chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            state,
            chunk_size=chunk_size,
        )

    baseline_chunk_size = 64
    baseline = run(baseline_chunk_size)
    candidates = {}
    for chunk_size in sorted(set(args.delta_prefill_chunk_sizes)):
        candidate = lambda chunk_size=chunk_size: run(chunk_size)
        actual = candidate()
        errors_vs_chunk64 = [
            error(value, target)
            for value, target in zip(actual, baseline)
        ]
        del actual
        timing = measure(
            candidate,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        candidates[str(chunk_size)] = {
            "chunk_size": chunk_size,
            "candidate": timing,
            "errors_vs_chunk64": errors_vs_chunk64,
        }
    return {
        "baseline_chunk_size": baseline_chunk_size,
        "candidates": candidates,
    }


def evaluate_recurrent_storage_drift(
    args,
    device,
    dtype,
    local_value_heads,
) -> dict:
    shape = (
        args.decode_batch,
        local_value_heads,
        args.key_head_dim,
    )
    scenarios = (
        ("slow_decay", -1e-4, 0.5),
        ("medium_decay", -1e-2, 0.5),
        ("fast_decay", -0.15, 0.5),
    )
    results = {}
    for scenario_index, (name, decay_value, beta_value) in enumerate(scenarios):
        generator = torch.Generator(device=device).manual_seed(
            args.seed + scenario_index
        )
        state_fp32 = torch.zeros(
            *shape,
            args.value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        state_model = state_fp32.to(dtype)
        squared_error = torch.zeros((), device=device)
        max_error = torch.zeros((), device=device)
        element_count = 0
        decay = torch.full(
            shape[:2],
            decay_value,
            device=device,
            dtype=torch.float32,
        )
        beta = torch.full(
            shape[:2],
            beta_value,
            device=device,
            dtype=dtype,
        )
        for _ in range(args.drift_steps):
            query = torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            key = torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            value = torch.randn(
                args.decode_batch,
                local_value_heads,
                args.value_head_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            output_fp32, state_fp32 = GDN.recurrent_gated_delta_step(
                query,
                key,
                value,
                decay,
                beta,
                state_fp32,
            )
            output_model, next_state = GDN.recurrent_gated_delta_step(
                query,
                key,
                value,
                decay,
                beta,
                state_model,
            )
            state_model = next_state.to(dtype)
            difference = output_model.float() - output_fp32.float()
            squared_error += difference.square().sum()
            max_error = torch.maximum(max_error, difference.abs().max())
            element_count += difference.numel()

        state_difference = state_model.float() - state_fp32
        results[name] = {
            "log_decay": decay_value,
            "beta": beta_value,
            "steps": args.drift_steps,
            "output_max_abs_error": max_error.item(),
            "output_rmse": (squared_error / element_count).sqrt().item(),
            "final_state_max_abs_error": state_difference.abs().max().item(),
            "final_state_mean_abs_error": state_difference.abs().mean().item(),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--router-tokens", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--attention-kv-heads", type=int, default=2)
    parser.add_argument("--attention-head-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--sampling-batch", type=int, default=64)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument("--sampling-top-p", type=float, default=0.9)
    parser.add_argument("--moe-intermediate-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=40)
    parser.add_argument(
        "--expert-token-counts",
        type=int,
        nargs="+",
        default=(1, 8, 32, 64, 128, 512),
        metavar="N",
        help="MoE token counts to scan from decode to prefill workloads",
    )
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Measure only convolution and grouped DeltaNet prefill paths.",
    )
    parser.add_argument(
        "--delta-prefill-chunk-sizes",
        type=int,
        nargs="+",
        default=(32, 64, 128),
    )
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--total-key-heads", type=int, default=16)
    parser.add_argument("--total-value-heads", type=int, default=32)
    parser.add_argument("--key-head-dim", type=int, default=128)
    parser.add_argument("--value-head-dim", type=int, default=128)
    parser.add_argument("--conv-kernel-size", type=int, default=4)
    parser.add_argument("--drift-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--moe-decode-chunk-size", type=int, default=8)
    parser.add_argument(
        "--moe-decode-chunk-sizes",
        type=int,
        nargs="+",
        default=(4, 8, 16),
    )
    parser.add_argument("--max-decode-tokens", type=int, default=64)
    parser.add_argument("--mixed-decode-tokens", type=int, default=32)
    parser.add_argument("--mixed-prefill-tokens", type=int, default=512)
    parser.add_argument(
        "--mixed-decode-token-counts",
        type=int,
        nargs="+",
        default=(8, 32, 64),
    )
    parser.add_argument(
        "--mixed-prefill-token-counts",
        type=int,
        nargs="+",
        default=(128, 512, 2048),
    )
    parser.add_argument("--moe-graph-safe-min-speedup", type=float, default=1.05)
    parser.add_argument(
        "--moe-graph-safe-max-peak-extra-mib",
        type=float,
        default=64.0,
    )
    parser.add_argument(
        "--moe-graph-safe-max-abs-error",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/qwen35_kernels.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = {
        "router_tokens": args.router_tokens,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "hidden_size": args.hidden_size,
        "attention_heads": args.attention_heads,
        "attention_kv_heads": args.attention_kv_heads,
        "attention_head_dim": args.attention_head_dim,
        "vocab_size": args.vocab_size,
        "sampling_batch": args.sampling_batch,
        "sampling_top_k": args.sampling_top_k,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "prefill_batch": args.prefill_batch,
        "prefill_tokens": args.prefill_tokens,
        "decode_batch": args.decode_batch,
        "drift_steps": args.drift_steps,
        "warmup": args.warmup,
        "moe_decode_chunk_size": args.moe_decode_chunk_size,
        "max_decode_tokens": args.max_decode_tokens,
        "mixed_decode_tokens": args.mixed_decode_tokens,
        "mixed_prefill_tokens": args.mixed_prefill_tokens,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "moe_graph_safe_min_speedup": args.moe_graph_safe_min_speedup,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if any(value <= 0 for value in args.expert_token_counts):
        invalid.append("expert_token_counts")
    if any(value <= 0 for value in args.moe_decode_chunk_sizes):
        invalid.append("moe_decode_chunk_sizes")
    if any(value <= 0 for value in args.delta_prefill_chunk_sizes):
        invalid.append("delta_prefill_chunk_sizes")
    if any(value <= 0 for value in args.mixed_decode_token_counts):
        invalid.append("mixed_decode_token_counts")
    if any(value <= 0 for value in args.mixed_prefill_token_counts):
        invalid.append("mixed_prefill_token_counts")
    if invalid:
        raise ValueError(f"benchmark values must be positive: {', '.join(invalid)}")
    if len(args.mixed_decode_token_counts) != len(
        args.mixed_prefill_token_counts
    ):
        raise ValueError("mixed MoE decode and prefill scans must have equal lengths")
    if args.moe_graph_safe_max_peak_extra_mib < 0:
        raise ValueError("MoE graph-safe peak-memory limit must be non-negative")
    if args.moe_graph_safe_max_abs_error < 0:
        raise ValueError("MoE graph-safe error limit must be non-negative")
    if args.top_k > args.num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    if args.sampling_top_k > args.vocab_size:
        raise ValueError("sampling_top_k cannot exceed vocab_size")
    if not 0.0 < args.sampling_top_p < 1.0:
        raise ValueError("sampling_top_p must be in (0, 1)")
    if args.moe_intermediate_size % args.tp_size:
        raise ValueError("Qwen3.5 MoE intermediate size must divide TP size")
    if args.attention_heads % args.tp_size:
        raise ValueError("Qwen3.5 query heads must divide TP size")
    if (
        args.attention_kv_heads >= args.tp_size
        and args.attention_kv_heads % args.tp_size
    ) or (
        args.attention_kv_heads < args.tp_size
        and args.tp_size % args.attention_kv_heads
    ):
        raise ValueError("Qwen3.5 KV heads must shard or replicate across TP")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and args.dtype == "float16":
        raise ValueError("float16 benchmark requires CUDA")
    dtype = getattr(torch, args.dtype)
    if args.total_key_heads % args.tp_size or args.total_value_heads % args.tp_size:
        raise ValueError("Qwen3.5 linear-attention heads must divide TP size")
    local_key_heads = args.total_key_heads // args.tp_size
    local_value_heads = args.total_value_heads // args.tp_size
    local_query_heads = args.attention_heads // args.tp_size
    local_kv_heads = max(args.attention_kv_heads // args.tp_size, 1)
    local_conv_channels = (
        2 * local_key_heads * args.key_head_dim
        + local_value_heads * args.value_head_dim
    )

    torch.manual_seed(args.seed)
    if args.prefill_only:
        benchmark_results = {
            "vectorized_prefill_convolution": benchmark_convolution(
                args,
                device,
                dtype,
                local_conv_channels,
            ),
            "decode_convolution_state_reuse": benchmark_decode_convolution(
                args,
                device,
                dtype,
                local_conv_channels,
            ),
            "grouped_delta_prefill": benchmark_delta_prefill_head_groups(
                args,
                device,
                dtype,
                local_key_heads,
                local_value_heads,
            ),
            "delta_prefill_state_reuse": benchmark_delta_prefill_state_reuse(
                args,
                device,
                dtype,
                local_key_heads,
                local_value_heads,
            ),
            "grouped_delta_prefill_chunk_sweep": (
                benchmark_delta_prefill_chunk_sweep(
                    args,
                    device,
                    dtype,
                    local_key_heads,
                    local_value_heads,
                )
            ),
        }
    else:
        expert_dispatch = benchmark_expert_dispatch_sweep(args, device, dtype)
        benchmark_results = {
            "router_topk_first": benchmark_router(args, device, dtype),
            "sampling_filter_fast_paths": benchmark_sampling_filter(
                args,
                device,
                dtype,
            ),
            "compact_top_k_sampling": benchmark_compact_top_k_sampling(
                args,
                device,
                dtype,
            ),
            "greedy_sampler_precision_fast_path": benchmark_greedy_sampler(
                args,
                device,
                dtype,
            ),
            "moe_output_buffer_reuse": benchmark_moe_output_merge(
                args,
                device,
                dtype,
            ),
            "sorted_route_weighting_reuse": benchmark_sorted_route_weighting(
                args,
                device,
                dtype,
            ),
            "residual_output_buffer_reuse": benchmark_residual_merge(
                args,
                device,
                dtype,
            ),
            "torch_kv_dequant_buffer_reuse": benchmark_torch_kv_dequant(
                args,
                device,
                dtype,
                local_key_heads,
            ),
            "expert_dispatch_torch": expert_dispatch,
            "moe_decode_chunk_recommendation": recommend_moe_decode_chunk_size(
                expert_dispatch,
                args.max_decode_tokens,
            ),
            "mixed_expert_dispatch": benchmark_mixed_expert_dispatch_sweep(
                args,
                device,
                dtype,
            ),
            "rmsnorm_fp32_reuse": benchmark_rmsnorm(args, device, dtype),
            "delta_l2_normalization_reuse": (
                benchmark_delta_l2_normalization(
                    args,
                    device,
                    dtype,
                    local_key_heads,
                )
            ),
            "attention_norm_output_reuse": benchmark_attention_norm_output_reuse(
                args,
                device,
                dtype,
            ),
            "rotary_output_reuse": benchmark_rotary_output_reuse(
                args,
                device,
                dtype,
            ),
            "vocab_gather_layout": benchmark_vocab_gather_layout(
                args,
                device,
                dtype,
            ),
            "gated_delta_beta_buffer_reuse": benchmark_beta_gate(
                args,
                device,
                dtype,
                local_value_heads,
            ),
            "gated_delta_packed_projection": benchmark_gated_delta_packed_projection(
                args,
                device,
                dtype,
                local_value_heads,
            ),
            "attention_packed_qkv": benchmark_attention_packed_qkv(
                args,
                device,
                dtype,
                local_query_heads,
                local_kv_heads,
            ),
            "contiguous_decode_state": benchmark_contiguous_decode_state(
                args,
                device,
                dtype,
                local_value_heads,
                local_conv_channels,
            ),
            "gated_delta_decay_rate_precompute": benchmark_decay_rate(
                args,
                device,
                dtype,
                local_value_heads,
            ),
            "gated_rmsnorm_fp32_reuse": benchmark_gated_rmsnorm(
                args,
                device,
                dtype,
            ),
            "vectorized_prefill_convolution": benchmark_convolution(
                args,
                device,
                dtype,
                local_conv_channels,
            ),
            "grouped_delta_prefill": benchmark_delta_prefill_head_groups(
                args,
                device,
                dtype,
                local_key_heads,
                local_value_heads,
            ),
            "delta_prefill_state_reuse": benchmark_delta_prefill_state_reuse(
                args,
                device,
                dtype,
                local_key_heads,
                local_value_heads,
            ),
            "specialized_delta_decode": benchmark_delta_decode(
                args,
                device,
                dtype,
                local_key_heads,
                local_value_heads,
            ),
            "delta_state_contraction": benchmark_delta_state_contraction(
                args,
                device,
                local_key_heads,
                local_value_heads,
            ),
            "recurrent_storage_drift": evaluate_recurrent_storage_drift(
                args,
                device,
                dtype,
                local_value_heads,
            ),
        }
    result = {
        **METADATA.collect_benchmark_metadata(torch),
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "resolved_device": str(device),
            "local_key_heads": local_key_heads,
            "local_value_heads": local_value_heads,
            "local_conv_channels": local_conv_channels,
        },
        "results": benchmark_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
