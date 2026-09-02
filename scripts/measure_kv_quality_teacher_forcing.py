"""Measure BF16-KV versus INT8-KV model quality on identical token paths.

This script is deliberately separate from ``compare_kv_quality.py``.  The
older script compares two independent free-running generations; once the
generations diverge, later differences no longer isolate KV-cache
quantization.  Here the sampler is replaced with a deterministic teacher-
forcing sampler, so both modes receive exactly the same continuation token at
every decode step.

The experiment records:

* logits error, cosine similarity, Top-1 agreement and Top-5 overlap;
* both KL directions and Jensen-Shannon divergence;
* target-token log probability and PPL;
* the actual INT8 cache values/scales written by the runtime for real model
  K/V tensors;
* execution paths, shapes, strides and bounded shape traces.

Each KV mode runs in a separate worker process.  The parent process never keeps
two model copies on the GPU at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.benchmark_metadata import (
    checkpoint_manifest_metadata,
    collect_benchmark_metadata,
)
from nanovllm.utils.context import get_context


CORPUS = [
    (
        "Large language model inference is a systems problem. "
        "The runtime must coordinate memory management, scheduling, tensor "
        "layouts, kernel launches, and numerical precision. "
    ),
    (
        "在大模型推理中，KV Cache 会随着上下文长度增长。"
        "分页式存储可以减少连续显存分配带来的碎片，"
        "而量化可以降低每个 KV token 的存储字节数。"
    ),
    (
        "FlashAttention avoids materializing the full attention score matrix. "
        "It loads tiles from global memory, maintains an online softmax state, "
        "and accumulates the result while controlling the working set. "
    ),
    (
        "工程 benchmark 不能只记录一次最好成绩。"
        "应当固定模型、输入形状、数据类型、软件版本和随机种子，"
        "进行 warmup 和多次独立运行，并保留原始日志。"
    ),
]


def iter_full_attention_modules(layers):
    """Yield cache-backed attention modules, skipping recurrent GDN layers."""

    for layer_id, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        attention = getattr(self_attn, "attn", None)
        if attention is not None:
            yield layer_id, attention


def tensor_metadata(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "bytes": tensor.numel() * tensor.element_size(),
        "contiguous": tensor.is_contiguous(),
    }


def build_cases(
    tokenizer,
    *,
    prompt_lengths: list[int],
    cases_per_length: int,
    continuation_len: int,
) -> list[dict[str, Any]]:
    """Build deterministic natural-language token cases offline."""

    cases: list[dict[str, Any]] = []
    for length_index, prompt_len in enumerate(prompt_lengths):
        for case_index in range(cases_per_length):
            pieces: list[str] = []
            cursor = (length_index + case_index) % len(CORPUS)
            while True:
                pieces.append(CORPUS[cursor])
                text = "\n".join(pieces)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) >= prompt_len + continuation_len:
                    break
                cursor = (cursor + 1) % len(CORPUS)
            prompt_ids = token_ids[:prompt_len]
            target_ids = token_ids[prompt_len : prompt_len + continuation_len]
            cases.append(
                {
                    "case_name": f"len{prompt_len}_case{case_index}",
                    "prompt_length": prompt_len,
                    "continuation_length": continuation_len,
                    "prompt_ids": prompt_ids,
                    "target_ids": target_ids,
                    "prompt_text_preview": tokenizer.decode(prompt_ids[:128]),
                    "target_text": tokenizer.decode(target_ids),
                }
            )
    return cases


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _quantization_metrics(
    source: torch.Tensor,
    stored_q: torch.Tensor,
    stored_scale: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize the values actually present in the INT8 KV cache."""

    source = source.float()
    stored_q = stored_q.float()
    stored_scale = stored_scale.float()
    dequant = stored_q * stored_scale.unsqueeze(-1)
    diff = dequant - source
    abs_diff = diff.abs()
    source_flat = source.reshape(-1)
    dequant_flat = dequant.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        source_flat.unsqueeze(0),
        dequant_flat.unsqueeze(0),
    ).item()
    return {
        "token_count": int(source.shape[0]),
        "value_count": int(source.numel()),
        "scale_min": _safe_float(stored_scale.min()),
        "scale_max": _safe_float(stored_scale.max()),
        "scale_mean": _safe_float(stored_scale.mean()),
        "abs_error_max": _safe_float(abs_diff.max()),
        "abs_error_mean": _safe_float(abs_diff.mean()),
        "rmse": _safe_float(diff.square().mean().sqrt()),
        "cosine_similarity": cosine,
        "saturation_ratio": _safe_float((stored_q.abs() >= 127).float().mean()),
    }


def _merge_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {"token_count": 0, "value_count": 0}
    value_keys = (
        "scale_min",
        "scale_max",
        "scale_mean",
        "abs_error_max",
        "abs_error_mean",
        "rmse",
        "cosine_similarity",
        "saturation_ratio",
    )
    result: dict[str, float | int] = {
        "token_count": sum(int(row["token_count"]) for row in rows),
        "value_count": sum(int(row["value_count"]) for row in rows),
    }
    for key in value_keys:
        values = [float(row[key]) for row in rows]
        result[key] = {
            "scale_min": min,
            "scale_max": max,
            "abs_error_max": max,
        }.get(key, mean)(values)
    return result


class KVMetricCollector:
    """Collect actual post-store INT8 K/V errors through Attention hooks."""

    def __init__(self, mode: str):
        self.mode = mode
        self.records: list[dict[str, Any]] = []
        self.current_stage: str | None = None
        self.current_context: Any | None = None
        self.hooks = []

    def attach(self, runner) -> None:
        if self.mode != "int8":
            return
        for layer_id, attention in iter_full_attention_modules(
            runner.model.model.layers
        ):
            self.hooks.append(
                attention.register_forward_hook(self._make_hook(layer_id))
            )

    def _make_hook(self, layer_id: int):
        def hook(module, inputs, _output):
            if self.mode != "int8" or not inputs:
                return
            if len(inputs) < 3:
                return
            key, value = inputs[1], inputs[2]
            context = self.current_context or get_context()
            slot_mapping = getattr(context, "slot_mapping", None)
            if getattr(context, "is_mixed", False):
                # Mixed mode has separate decode/prefill index tensors and the
                # single concatenated slot_mapping needs stage-specific slicing.
                if self.current_stage == "decode":
                    slot_mapping = slot_mapping[: context.decode_token_count]
                elif self.current_stage == "prefill":
                    slot_mapping = slot_mapping[context.decode_token_count :]
            if slot_mapping is None or module.k_scale.numel() == 0:
                return
            slots = slot_mapping.detach()
            valid = slots >= 0
            if not bool(valid.any().item()):
                return
            slots = slots[valid].long()
            flat_k = module.k_cache.view(-1, module.k_cache.size(2), module.k_cache.size(3))
            flat_v = module.v_cache.view(-1, module.v_cache.size(2), module.v_cache.size(3))
            flat_ks = module.k_scale.view(-1, module.k_scale.size(2))
            flat_vs = module.v_scale.view(-1, module.v_scale.size(2))
            key = key.detach()[valid]
            value = value.detach()[valid]
            key_q = flat_k[slots]
            value_q = flat_v[slots]
            key_scale = flat_ks[slots]
            value_scale = flat_vs[slots]
            self.records.append(
                {
                    "layer_id": layer_id,
                    "stage": self.current_stage,
                    "key": _quantization_metrics(key, key_q, key_scale),
                    "value": _quantization_metrics(
                        value,
                        value_q,
                        value_scale,
                    ),
                    "input": {
                        "key": tensor_metadata("key", key),
                        "value": tensor_metadata("value", value),
                        "slot_mapping": tensor_metadata("slot_mapping", slots),
                        "cache": tensor_metadata("k_cache", module.k_cache),
                        "scale": tensor_metadata("k_scale", module.k_scale),
                    },
                    "slot_summary": {
                        "valid_count": int(slots.numel()),
                        "min": int(slots.min().item()),
                        "max": int(slots.max().item()),
                    },
                }
            )

        return hook

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def summary(self) -> dict[str, Any]:
        key_rows = [record["key"] for record in self.records]
        value_rows = [record["value"] for record in self.records]
        by_stage: dict[str, dict[str, Any]] = {}
        for stage in ("prefill", "decode"):
            stage_key = [
                record["key"]
                for record in self.records
                if record["stage"] == stage
            ]
            stage_value = [
                record["value"]
                for record in self.records
                if record["stage"] == stage
            ]
            by_stage[stage] = {
                "key": _merge_metric_rows(stage_key),
                "value": _merge_metric_rows(stage_value),
            }
        return {
            "mode": self.mode,
            "record_count": len(self.records),
            "key": _merge_metric_rows(key_rows),
            "value": _merge_metric_rows(value_rows),
            "by_stage": by_stage,
            "records": self.records,
        }


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for quality measurement")
    from transformers import AutoTokenizer

    cases = json.loads(Path(args.worker_cases_file).read_text())
    mode = args.worker_mode
    model = args.model
    target_matrix = torch.tensor(
        [case["target_ids"] for case in cases],
        dtype=torch.long,
    )
    batch_size, continuation_len = target_matrix.shape
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    del tokenizer

    llm = LLM(
        model,
        tensor_parallel_size=args.tensor_parallel_size,
        recurrent_state_dtype=args.recurrent_state_dtype,
        weight_quant_backend=getattr(args, "weight_quant_backend", "auto"),
        qwen35_moe_decode_backend=getattr(
            args, "qwen35_moe_decode_backend", "sorted"
        ),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=batch_size,
        kv_cache_dtype="auto" if mode == "auto" else "int8",
        kv_dequant_backend="fused",
        int8_partitioned_decode_threshold=args.partition_threshold,
        int8_partitioned_decode_partition_size=args.partition_size,
    )
    runner = llm.model_runner
    collector = KVMetricCollector(mode)
    collector.attach(runner)
    state: dict[str, Any] = {
        "stage": None,
        "step": 0,
        "logits": [],
        "stage_records": [],
    }
    original_run_model = runner.run_model
    original_compute_logits = runner.model.compute_logits
    original_sampler_forward = runner.sampler.forward

    def wrapped_run_model(input_ids, positions, is_prefill):
        state["stage"] = "prefill" if is_prefill else "decode"
        collector.current_stage = state["stage"]
        collector.current_context = get_context()
        return original_run_model(input_ids, positions, is_prefill)

    def wrapped_compute_logits(hidden_states):
        logits = original_compute_logits(hidden_states)
        state["logits"].append(logits.detach().float().cpu())
        state["stage_records"].append(
            {
                "step": state["step"],
                "stage": state["stage"],
                "logits": tensor_metadata("logits", logits),
                "hidden_states": tensor_metadata(
                    "hidden_states",
                    hidden_states,
                ),
            }
        )
        return logits

    def forced_sampler(logits, temperatures, top_ks, top_ps):
        del temperatures, top_ks, top_ps
        step = state["step"]
        if step >= continuation_len:
            raise RuntimeError(
                f"sampler called for step {step}, only {continuation_len} "
                "teacher-forced targets are available"
            )
        if logits.size(0) != batch_size:
            raise RuntimeError(
                f"teacher-forcing batch changed from {batch_size} to "
                f"{logits.size(0)}"
            )
        tokens = target_matrix[:, step].to(logits.device)
        state["step"] += 1
        return tokens

    runner.run_model = wrapped_run_model
    runner.model.compute_logits = wrapped_compute_logits
    runner.sampler.forward = forced_sampler
    runner.call("reset_execution_stats")
    runner.call("reset_shape_trace")
    try:
        for case in cases:
            llm.add_request(
                case["prompt_ids"],
                SamplingParams(
                    temperature=0.0,
                    ignore_eos=True,
                    max_tokens=continuation_len,
                ),
            )
        while not llm.is_finished():
            llm.step()
        outputs = [
            sequence.completion_token_ids
            for sequence in llm.scheduler.running
        ]
        # Finished sequences are removed from ``running``.  The forced sampler
        # already guarantees the continuation, so retain the target matrix as
        # the auditable output trajectory instead.
        execution_stats = runner.call("get_execution_stats")
        shape_trace = runner.call("get_shape_trace")
        result = {
            "mode": mode,
            "tensor_parallel_size": args.tensor_parallel_size,
            "recurrent_state_dtype": args.recurrent_state_dtype,
            "requested_weight_quant_backend": getattr(
                args, "weight_quant_backend", "auto"
            ),
            "weight_quant_backend": runner.config.weight_quant_backend,
            "partition_threshold": args.partition_threshold,
            "partition_size": args.partition_size,
            "cases": cases,
            "target_matrix": target_matrix.tolist(),
            "forced_steps": state["step"],
            "logits": state["logits"],
            "stage_records": state["stage_records"],
            "execution_stats": execution_stats,
            "shape_trace": shape_trace,
            "kv_metrics": collector.summary(),
            "outputs_note": (
                "The sampler was replaced with a fixed target-token sampler; "
                "target_matrix is the exact continuation consumed by every "
                "decode step."
            ),
            "unused_outputs_length": len(outputs),
        }
        output_path = Path(args.worker_output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
    finally:
        collector.close()
        runner.sampler.forward = original_sampler_forward
        llm.exit()


def _row_metrics(left: torch.Tensor, right: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    left = left.float()
    right = right.float()
    diff = left - right
    left_logp = torch.log_softmax(left, dim=-1)
    right_logp = torch.log_softmax(right, dim=-1)
    left_prob = left_logp.exp()
    right_prob = right_logp.exp()
    midpoint = (left_prob + right_prob).clamp_min(1e-30) * 0.5
    midpoint_logp = midpoint.log()
    kl_left = (left_prob * (left_logp - right_logp)).sum(-1).clamp_min(0)
    kl_right = (right_prob * (right_logp - left_logp)).sum(-1).clamp_min(0)
    js = (
        0.5 * (left_prob * (left_logp - midpoint_logp)).sum(-1)
        + 0.5 * (right_prob * (right_logp - midpoint_logp)).sum(-1)
    ).clamp_min(0)
    top1_left = left.argmax(-1)
    top1_right = right.argmax(-1)
    top5_left = torch.topk(left, 5, dim=-1).indices
    top5_right = torch.topk(right, 5, dim=-1).indices
    top5_overlap = torch.stack(
        [
            torch.isin(top5_left[i], top5_right[i]).float().mean()
            for i in range(left.size(0))
        ]
    )
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    target_left = left_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_right = right_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {
        "logits_max_abs": _safe_float(diff.abs().max()),
        "logits_mean_abs": _safe_float(diff.abs().mean()),
        "logits_rmse": _safe_float(diff.square().mean().sqrt()),
        "logits_cosine_mean": _safe_float(cosine.mean()),
        "top1_agreement": _safe_float((top1_left == top1_right).float().mean()),
        "top5_overlap": _safe_float(top5_overlap.mean()),
        "kl_bf16_to_int8": _safe_float(kl_left.mean()),
        "kl_int8_to_bf16": _safe_float(kl_right.mean()),
        "js_divergence": _safe_float(js.mean()),
        "target_logprob_bf16_mean": _safe_float(target_left.mean()),
        "target_logprob_int8_mean": _safe_float(target_right.mean()),
        "target_nll_bf16": _safe_float((-target_left).mean()),
        "target_nll_int8": _safe_float((-target_right).mean()),
        "row_count": int(left.size(0)),
    }


def validate_worker_execution(worker: dict[str, Any]) -> dict[str, Any]:
    mode = worker.get("mode")
    max_prompt_length = max(
        len(case["prompt_ids"]) for case in worker.get("cases", [])
    )
    partition_threshold = worker.get("partition_threshold", 8192)
    int8_decode_path = (
        "int8_partitioned_decode"
        if max_prompt_length >= partition_threshold
        else "int8_fused_decode"
    )
    expected_attention = {
        "auto": {"float_flash_prefill", "float_flash_decode"},
        "int8": {"int8_prefill", int8_decode_path},
    }
    if mode not in expected_attention:
        raise ValueError(f"unsupported worker mode: {mode!r}")
    stats = worker.get("execution_stats", {})
    model_counts = stats.get("model_path_counts", {})
    attention_counts = stats.get("attention_path_counts", {})
    observed = {
        name
        for counts in (model_counts, attention_counts)
        for name, count in counts.items()
        if count > 0
    }
    expected = {"prefill_eager", "decode_eager"} | expected_attention[mode]
    stages = [record.get("stage") for record in worker.get("stage_records", [])]
    missing = sorted(expected - observed)
    valid = (
        not missing
        and stages[:1] == ["prefill"]
        and "decode" in stages
        and worker.get("forced_steps", 0) >= 2
        and stats.get("dropped_execution_signature_steps", 0) == 0
    )
    return {
        "valid": valid,
        "expected_paths": sorted(expected),
        "observed_paths": sorted(observed),
        "missing_paths": missing,
        "stages": stages,
        "forced_steps": worker.get("forced_steps", 0),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric row list")
    total_rows = sum(int(row["row_count"]) for row in rows)
    if total_rows <= 0:
        raise ValueError("metric rows must contain observations")
    metric_names = [
        key
        for key in rows[0]
        if key not in {"step", "stage", "row_count"}
    ]
    result = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        if name.endswith("_max_abs"):
            result[name] = max(values)
        elif name.endswith("_rmse"):
            result[name] = math.sqrt(
                sum(
                    value * value * int(row["row_count"])
                    for value, row in zip(values, rows)
                )
                / total_rows
            )
        else:
            result[name] = sum(
                value * int(row["row_count"])
                for value, row in zip(values, rows)
            ) / total_rows
    return result


def perplexity_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot calculate perplexity without metric rows")
    count = sum(int(row["row_count"]) for row in rows)
    nll_bf16 = sum(
        float(row["target_nll_bf16"]) * int(row["row_count"])
        for row in rows
    )
    nll_int8 = sum(
        float(row["target_nll_int8"]) * int(row["row_count"])
        for row in rows
    )
    bf16 = math.exp(nll_bf16 / count)
    int8 = math.exp(nll_int8 / count)
    return {
        "token_count": count,
        "bf16": bf16,
        "int8": int8,
        "relative_change": int8 / bf16 - 1.0,
    }


def summarize_batch_comparisons(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("cannot summarize an empty quality run")
    rows = [row for comparison in comparisons for row in comparison["rows"]]
    decode_rows = [row for row in rows if row["stage"] == "decode"]
    if not decode_rows:
        raise ValueError("quality run contains no KV-sensitive decode rows")
    return {
        "batch_count": len(comparisons),
        "steps_compared": len(rows),
        "kv_sensitive_steps_compared": len(decode_rows),
        "token_rows_compared": sum(int(row["row_count"]) for row in rows),
        "kv_sensitive_token_rows_compared": sum(
            int(row["row_count"]) for row in decode_rows
        ),
        "aggregate": aggregate_metric_rows(rows),
        "decode_aggregate": aggregate_metric_rows(decode_rows),
        "ppl": perplexity_summary(rows),
        "decode_ppl": perplexity_summary(decode_rows),
    }


def compare_workers(auto: dict[str, Any], int8: dict[str, Any]) -> dict[str, Any]:
    if auto.get("target_matrix") != int8.get("target_matrix"):
        raise RuntimeError("BF16 and INT8 workers used different target trajectories")
    if auto.get("tensor_parallel_size") != int8.get("tensor_parallel_size"):
        raise RuntimeError("BF16 and INT8 workers used different tensor parallel sizes")
    if auto.get("recurrent_state_dtype") != int8.get("recurrent_state_dtype"):
        raise RuntimeError("BF16 and INT8 workers used different recurrent state dtypes")
    if auto.get("weight_quant_backend") != int8.get("weight_quant_backend"):
        raise RuntimeError("BF16 and INT8 workers used different weight backends")
    execution_validation = {
        "auto": validate_worker_execution(auto),
        "int8": validate_worker_execution(int8),
    }
    if not all(item["valid"] for item in execution_validation.values()):
        raise RuntimeError(
            f"quality workers did not exercise required KV paths: "
            f"{execution_validation}"
        )
    left = auto["logits"]
    right = int8["logits"]
    if len(left) != len(right):
        raise RuntimeError("BF16 and INT8 worker produced different step counts")
    target_matrix = torch.tensor(auto["target_matrix"], dtype=torch.long)
    rows = []
    decode_trajectories = {
        "bf16_top1_token_ids": [],
        "int8_top1_token_ids": [],
        "bf16_target_logprobs": [],
        "int8_target_logprobs": [],
    }
    for step, (left_step, right_step) in enumerate(zip(left, right)):
        if left_step.shape != right_step.shape:
            raise RuntimeError(
                f"logit shape mismatch at step {step}: "
                f"{left_step.shape} vs {right_step.shape}"
            )
        stage = auto["stage_records"][step]["stage"]
        targets = target_matrix[:, step]
        rows.append(
            {
                "step": step,
                "stage": stage,
                **_row_metrics(
                    left_step,
                    right_step,
                    targets,
                ),
            }
        )
        if stage == "decode":
            left_logp = torch.log_softmax(left_step.float(), dim=-1)
            right_logp = torch.log_softmax(right_step.float(), dim=-1)
            decode_trajectories["bf16_top1_token_ids"].append(
                left_step.argmax(-1).tolist()
            )
            decode_trajectories["int8_top1_token_ids"].append(
                right_step.argmax(-1).tolist()
            )
            decode_trajectories["bf16_target_logprobs"].append(
                left_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).tolist()
            )
            decode_trajectories["int8_target_logprobs"].append(
                right_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).tolist()
            )
    decode_rows = [row for row in rows if row["stage"] == "decode"]
    if not decode_rows:
        raise RuntimeError("quality comparison produced no KV-sensitive decode rows")
    result = {
        "weight_quant_backend": auto.get("weight_quant_backend"),
        "steps_compared": len(rows),
        "kv_sensitive_steps_compared": len(decode_rows),
        "quality_scope": (
            "decode-only metrics read previously stored KV cache; prefill is "
            "retained separately as a model-path control"
        ),
        "rows": rows,
        "aggregate": aggregate_metric_rows(rows),
        "decode_aggregate": aggregate_metric_rows(decode_rows),
        "ppl": perplexity_summary(rows),
        "decode_ppl": perplexity_summary(decode_rows),
        "execution_validation": execution_validation,
        "execution_stats": {
            "auto": auto["execution_stats"],
            "int8": int8["execution_stats"],
        },
        "kv_metrics": {
            "auto": auto["kv_metrics"],
            "int8": int8["kv_metrics"],
        },
        "shape_trace": {
            "auto": auto["shape_trace"],
            "int8": int8["shape_trace"],
        },
        "decode_trajectories": decode_trajectories,
    }
    return result


def run_worker_process(
    args: argparse.Namespace,
    *,
    mode: str,
    cases_file: Path,
    result_dir: Path,
    batch_name: str,
) -> Path:
    output_path = result_dir / "workers" / f"{batch_name}_{mode}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-cases-file",
        str(cases_file),
        "--worker-output-file",
        str(output_path),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--recurrent-state-dtype",
        args.recurrent_state_dtype,
        "--weight-quant-backend",
        getattr(args, "weight_quant_backend", "auto"),
        "--qwen35-moe-decode-backend",
        getattr(args, "qwen35_moe_decode_backend", "sorted"),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--partition-threshold",
        str(args.partition_threshold),
        "--partition-size",
        str(args.partition_size),
    ]
    subprocess.run(command, check=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced BF16-KV versus INT8-KV quality measurement."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--recurrent-state-dtype",
        choices=("float32", "model"),
        default="float32",
    )
    parser.add_argument(
        "--weight-quant-backend",
        choices=("auto", "reference", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--qwen35-moe-decode-backend",
        choices=("sorted", "batched"),
        default="sorted",
    )
    parser.add_argument(
        "--cases-file",
        default=None,
        help=(
            "Optional JSON file containing pre-tokenized cases. Each case must "
            "have prompt_ids and target_ids."
        ),
    )
    parser.add_argument("--prompt-lengths", default="128,1024,3072")
    parser.add_argument("--cases-per-length", type=int, default=2)
    parser.add_argument("--continuation-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--partition-threshold", type=int, default=8192)
    parser.add_argument("--partition-size", type=int, default=512)
    parser.add_argument("--trace-max-events", type=int, default=2048)
    parser.add_argument("--trace-max-index-values", type=int, default=64)
    parser.add_argument("--result-dir", default="benchmark_results/kv_quality_teacher_forcing")
    parser.add_argument("--name", default="kv_quality_teacher_forcing")
    parser.add_argument("--worker-mode", choices=("auto", "int8"), default=None)
    parser.add_argument("--worker-cases-file", default=None)
    parser.add_argument("--worker-output-file", default=None)
    args = parser.parse_args()
    if args.worker_mode is not None:
        if not args.worker_cases_file or not args.worker_output_file:
            parser.error("worker mode requires cases and output files")
        run_worker(args)
        return
    if args.cases_per_length <= 0 or args.continuation_len < 2:
        parser.error(
            "cases-per-length must be positive and continuation-len must be "
            "at least 2 so the experiment reads KV cache during decode"
        )
    if args.trace_max_events <= 0 or args.trace_max_index_values <= 0:
        parser.error("trace limits must be positive")
    if args.partition_threshold <= 0 or args.partition_size <= 0:
        parser.error("partition settings must be positive")
    prompt_lengths = [int(item) for item in args.prompt_lengths.split(",") if item]
    if not prompt_lengths or any(item <= 0 for item in prompt_lengths):
        parser.error("prompt-lengths must contain positive integers")
    if not torch.cuda.is_available():
        # Parent only tokenizes, but this catches accidental local execution
        # before spawning remote workers.
        raise RuntimeError("CUDA is required for this experiment")
    # Enable a sufficiently large metadata trace for the formal quality run.
    # The trace stores shapes/strides/index metadata, not model values.
    os.environ["NANOVLLM_SHAPE_TRACE"] = "1"
    os.environ["NANOVLLM_SHAPE_TRACE_MAX_EVENTS"] = str(args.trace_max_events)
    os.environ["NANOVLLM_SHAPE_TRACE_MAX_INDEX_VALUES"] = str(
        args.trace_max_index_values
    )
    if args.cases_file is not None:
        cases_path = Path(args.cases_file)
        try:
            all_cases = json.loads(cases_path.read_text())
        except Exception as exc:
            parser.error(f"failed to load --cases-file: {exc}")
        if not isinstance(all_cases, list) or not all_cases:
            parser.error("--cases-file must contain a non-empty JSON list")
        continuation_lengths = {
            len(case.get("target_ids", []))
            for case in all_cases
            if isinstance(case, dict)
        }
        if (
            len(continuation_lengths) != 1
            or not all(isinstance(case, dict) for case in all_cases)
            or any(
                not isinstance(case.get("prompt_ids"), list)
                or not isinstance(case.get("target_ids"), list)
                or not case["prompt_ids"]
                or not case["target_ids"]
                for case in all_cases
            )
        ):
            parser.error(
                "--cases-file cases must be objects with non-empty prompt_ids "
                "and equal-length target_ids"
            )
        if any(
            len(case["prompt_ids"]) + len(case["target_ids"])
            > args.max_model_len
            for case in all_cases
        ):
            parser.error(
                "a case in --cases-file exceeds --max-model-len"
            )
        args.continuation_len = next(iter(continuation_lengths))
        if args.continuation_len < 2:
            parser.error(
                "--cases-file target_ids must contain at least two tokens so "
                "the experiment reads KV cache during decode"
            )
        # The batch loop uses this value as the number of cases per worker
        # process. It is also useful for corpus cases, where all prompts share
        # one target length but do not share one prompt length.
        args.cases_per_length = max(1, args.cases_per_length)
    else:
        if max(prompt_lengths) + args.continuation_len > args.max_model_len:
            parser.error("prompt lengths plus continuation exceed max-model-len")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        all_cases = build_cases(
            tokenizer,
            prompt_lengths=prompt_lengths,
            cases_per_length=args.cases_per_length,
            continuation_len=args.continuation_len,
        )
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    actual_prompt_lengths = [
        int(case.get("prompt_length", len(case["prompt_ids"])))
        for case in all_cases
    ]
    result: dict[str, Any] = {
        **collect_benchmark_metadata(torch),
        "checkpoint_manifest": checkpoint_manifest_metadata(args.model),
        "configuration": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "recurrent_state_dtype": args.recurrent_state_dtype,
            "requested_weight_quant_backend": args.weight_quant_backend,
            "qwen35_moe_decode_backend": args.qwen35_moe_decode_backend,
            "prompt_lengths": actual_prompt_lengths,
            "cases_per_length": args.cases_per_length,
            "continuation_len": args.continuation_len,
            "cases_file": str(args.cases_file) if args.cases_file else None,
            "trace_max_events": args.trace_max_events,
            "trace_max_index_values": args.trace_max_index_values,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "partition_threshold": args.partition_threshold,
            "partition_size": args.partition_size,
            "compute_dtype": "model_config_dtype",
            "auto_mode": "BF16/float KV cache",
            "int8_mode": "INT8 KV cache + fused decode",
            "teacher_forcing": True,
            "natural_language_corpus": True,
        },
        "batches": [],
        "case_token_digest": hashlib.sha256(
            json.dumps(
                [
                    {
                        "prompt_ids": case["prompt_ids"],
                        "target_ids": case["target_ids"],
                    }
                    for case in all_cases
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    for index in range(0, len(all_cases), args.cases_per_length):
        cases = all_cases[index : index + args.cases_per_length]
        prompt_label = cases[0].get("prompt_length", len(cases[0]["prompt_ids"]))
        batch_name = f"batch{index // args.cases_per_length}_len{prompt_label}"
        cases_file = result_dir / f"{batch_name}_cases.json"
        cases_file.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
        auto_path = run_worker_process(
            args,
            mode="auto",
            cases_file=cases_file,
            result_dir=result_dir,
            batch_name=batch_name,
        )
        int8_path = run_worker_process(
            args,
            mode="int8",
            cases_file=cases_file,
            result_dir=result_dir,
            batch_name=batch_name,
        )
        auto = torch.load(auto_path, map_location="cpu", weights_only=False)
        int8 = torch.load(int8_path, map_location="cpu", weights_only=False)
        comparison = compare_workers(auto, int8)
        comparison["batch_name"] = batch_name
        comparison["cases_file"] = str(cases_file)
        result["batches"].append(comparison)
    result["summary"] = summarize_batch_comparisons(result["batches"])
    effective_backends = {
        batch["weight_quant_backend"] for batch in result["batches"]
    }
    if None in effective_backends or len(effective_backends) != 1:
        raise RuntimeError("quality batches used inconsistent weight backends")
    result["configuration"]["weight_quant_backend"] = effective_backends.pop()
    result_path = result_dir / f"{args.name}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
