"""Compare BF16-KV and INT8-KV model outputs on identical token prompts.

The script intentionally runs in eager mode and records extra diagnostics:
decode-step logits summaries/top-k, attention-output summaries for every
layer, and optional raw tensors for the first decode steps.  It is a quality
experiment, not a latency measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.benchmark_metadata import collect_benchmark_metadata
from nanovllm.utils.context import get_context


def run_in_worker_process(
    args: argparse.Namespace,
    *,
    mode: str,
    prompts_file: Path,
    result_dir: Path,
    batch_name: str,
) -> Path:
    """Run one KV-cache mode in a fresh process and return its artifact path."""

    output_path = result_dir / "workers" / f"{batch_name}_{mode}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-prompts-file",
        str(prompts_file),
        "--worker-output-file",
        str(output_path),
        "--worker-batch-name",
        batch_name,
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--vocab-size",
        str(args.vocab_size),
        "--seed",
        str(args.seed),
        "--capture-decode-steps",
        str(args.capture_decode_steps),
        "--result-dir",
        str(result_dir),
    ]
    command.append("--save-raw" if args.save_raw else "--no-save-raw")
    subprocess.run(command, check=True)
    if not output_path.is_file():
        raise RuntimeError(f"quality worker did not create {output_path}")
    return output_path


def build_prompts(length: int, count: int, *, seed: int, vocab_size: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(length)]
        for _ in range(count)
    ]


def tensor_summary(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "min": value.min().item(),
        "max": value.max().item(),
        "max_abs": value.abs().max().item(),
        "l2": value.square().sum().sqrt().item(),
    }


def topk_summary(logits: torch.Tensor, k: int = 5) -> dict:
    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    return {
        "topk_ids": indices.cpu().tolist(),
        "topk_logits": values.cpu().tolist(),
    }


def iter_full_attention_modules(layers):
    """Yield only cache-backed attention modules in a hybrid model."""

    for layer_id, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        attention = getattr(self_attn, "attn", None)
        if attention is not None:
            yield layer_id, attention


def run_one_mode(
    *,
    model: str,
    prompts: list[list[int]],
    args: argparse.Namespace,
    kv_cache_dtype: str,
    result_dir: Path,
    batch_name: str,
) -> dict:
    params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    llm = LLM(
        model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=len(prompts),
        kv_cache_dtype=kv_cache_dtype,
        kv_dequant_backend="fused",
        int8_partitioned_decode_threshold=999999,
    )
    runner = llm.model_runner
    state = {
        "decode_step": 0,
        "current_is_prefill": None,
        "logits_records": [],
        "attention_records": [],
        "raw_logits": [],
        "raw_attention": {},
    }

    original_run_model = runner.run_model
    original_compute_logits = runner.model.compute_logits

    def wrapped_run_model(input_ids, positions, is_prefill):
        state["current_is_prefill"] = bool(is_prefill)
        if not is_prefill:
            state["decode_step"] += 1
        return original_run_model(input_ids, positions, is_prefill)

    def wrapped_compute_logits(hidden_states):
        logits = original_compute_logits(hidden_states)
        is_prefill = bool(state["current_is_prefill"])
        step = state["decode_step"]
        if is_prefill or step <= args.capture_decode_steps:
            record = {
                "is_prefill": is_prefill,
                "decode_step": step if not is_prefill else None,
                "summary": tensor_summary(logits),
            }
            if not is_prefill:
                record.update(topk_summary(logits))
                state["raw_logits"].append(logits.detach().float().cpu())
            state["logits_records"].append(record)
        return logits

    runner.run_model = wrapped_run_model
    runner.model.compute_logits = wrapped_compute_logits
    runner.call("reset_execution_stats")
    runner.call("reset_shape_trace")
    hooks = []

    def make_hook(layer_id: int):
        def hook(module, _inputs, output):
            context = get_context()
            if context.is_prefill:
                return
            step = state["decode_step"]
            if step <= args.capture_decode_steps:
                tensor = output.detach()
                record = {
                    "decode_step": step,
                    "layer_id": layer_id,
                    "summary": tensor_summary(tensor),
                }
                state["attention_records"].append(record)
                state["raw_attention"][f"step{step}_layer{layer_id}"] = (
                    tensor.float().cpu()
                )
        return hook

    try:
        attention_layer_ids = []
        for layer_id, attention in iter_full_attention_modules(
            runner.model.model.layers
        ):
            attention_layer_ids.append(layer_id)
            hooks.append(attention.register_forward_hook(make_hook(layer_id)))
        outputs = llm.generate(
            prompts,
            params,
            use_tqdm=False,
        )
        execution_stats = runner.call("get_execution_stats")
        shape_trace = runner.call("get_shape_trace")
        result = {
            "kv_cache_dtype": kv_cache_dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "captured_attention_layer_ids": attention_layer_ids,
            "outputs": [item["token_ids"] for item in outputs],
            "logits_records": state["logits_records"],
            "attention_records": state["attention_records"],
            "execution_stats": execution_stats,
            "shape_trace": shape_trace,
        }
        expected_paths = (
            {"prefill_eager", "decode_eager", "float_flash_prefill", "float_flash_decode"}
            if kv_cache_dtype == "auto"
            else {"prefill_eager", "decode_eager", "int8_prefill", "int8_fused_decode"}
        )
        observed_paths = set(execution_stats["model_path_counts"]) | set(
            execution_stats["attention_path_counts"]
        )
        result["execution_validation"] = {
            "expected_paths": sorted(expected_paths),
            "observed_paths": sorted(observed_paths),
            "missing_paths": sorted(expected_paths - observed_paths),
            "valid": expected_paths <= observed_paths
            and execution_stats.get("dropped_execution_signature_steps", 0) == 0,
        }
        if not result["execution_validation"]["valid"]:
            raise RuntimeError(
                f"{kv_cache_dtype} quality run used unexpected paths: "
                f"{result['execution_validation']}"
            )
        if args.save_raw:
            raw_dir = result_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{batch_name}_{kv_cache_dtype}.pt"
            torch.save(
                {
                    "logits": state["raw_logits"],
                    "attention": state["raw_attention"],
                },
                raw_path,
            )
            result["raw_tensor_file"] = str(raw_path)
        return result
    finally:
        for hook in hooks:
            hook.remove()
        llm.exit()


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model quality comparison")
    prompts_path = Path(args.worker_prompts_file)
    output_path = Path(args.worker_output_file)
    prompts = json.loads(prompts_path.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("worker prompts file must contain a non-empty prompt list")
    result = run_one_mode(
        model=args.model,
        prompts=prompts,
        args=args,
        kv_cache_dtype=args.worker_mode,
        result_dir=Path(args.result_dir),
        batch_name=args.worker_batch_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"Wrote {output_path}")


def compare_logits(bf16: list[torch.Tensor], int8: list[torch.Tensor]) -> dict:
    rows = []
    sequence_lengths_match = len(bf16) == len(int8)
    for step, (left, right) in enumerate(zip(bf16, int8), start=1):
        if left.shape != right.shape:
            rows.append(
                {
                    "decode_step": step,
                    "shape_match": False,
                    "passed": False,
                }
            )
            continue
        left = left.float()
        right = right.float()
        diff = left - right
        left_prob = torch.softmax(left, dim=-1)
        left_log_prob = torch.log_softmax(left, dim=-1)
        right_log_prob = torch.log_softmax(right, dim=-1)
        top1_left = left.argmax(dim=-1)
        top1_right = right.argmax(dim=-1)
        top5_left = torch.topk(left, 5, dim=-1).indices
        top5_right = torch.topk(right, 5, dim=-1).indices
        overlap = []
        for left_row, right_row in zip(top5_left, top5_right):
            overlap.append(
                len(set(left_row.tolist()) & set(right_row.tolist())) / 5.0
            )
        rows.append(
            {
                "decode_step": step,
                "shape_match": True,
                "max_abs": diff.abs().max().item(),
                "mean_abs": diff.abs().mean().item(),
                "rmse": diff.square().mean().sqrt().item(),
                "top1_agreement": (top1_left == top1_right).float().mean().item(),
                "top5_overlap": sum(overlap) / len(overlap),
                "kl_bf16_to_int8": (
                    (left_prob * (left_log_prob - right_log_prob))
                    .sum(dim=-1)
                    .mean()
                    .item()
                ),
            }
        )
    return {
        "steps_compared": len(rows),
        "bf16_steps": len(bf16),
        "int8_steps": len(int8),
        "sequence_lengths_match": sequence_lengths_match,
        "rows": rows,
        "top1_agreement_mean": (
            sum(row["top1_agreement"] for row in rows if row["shape_match"])
            / max(1, sum(row["shape_match"] for row in rows))
        ),
        "top5_overlap_mean": (
            sum(row["top5_overlap"] for row in rows if row["shape_match"])
            / max(1, sum(row["shape_match"] for row in rows))
        ),
        "kl_bf16_to_int8_mean": (
            sum(row["kl_bf16_to_int8"] for row in rows if row["shape_match"])
            / max(1, sum(row["shape_match"] for row in rows))
        ),
    }


def compare_attention(
    bf16: dict[str, torch.Tensor],
    int8: dict[str, torch.Tensor],
) -> dict:
    rows = []
    bf16_keys = set(bf16)
    int8_keys = set(int8)
    for key in sorted(bf16_keys & int8_keys):
        left = bf16[key].float()
        right = int8[key].float()
        if left.shape != right.shape:
            rows.append(
                {
                    "key": key,
                    "shape_match": False,
                    "passed": False,
                }
            )
            continue
        diff = left - right
        rows.append(
            {
                "key": key,
                "shape_match": True,
                "max_abs": diff.abs().max().item(),
                "mean_abs": diff.abs().mean().item(),
                "rmse": diff.square().mean().sqrt().item(),
            }
        )
    return {
        "records_compared": len(rows),
        "bf16_records": len(bf16),
        "int8_records": len(int8),
        "missing_in_int8": sorted(bf16_keys - int8_keys),
        "missing_in_bf16": sorted(int8_keys - bf16_keys),
        "record_keys_match": bf16_keys == int8_keys,
        "rows": rows,
    }


def compare_outputs(
    bf16: dict,
    int8: dict,
) -> dict:
    left_outputs = bf16["outputs"]
    right_outputs = int8["outputs"]
    first_divergence = None
    total = 0
    equal = 0
    request_counts_match = len(left_outputs) == len(right_outputs)
    lengths_match = [len(item) for item in left_outputs] == [
        len(item) for item in right_outputs
    ]
    for request_id, (left, right) in enumerate(zip(left_outputs, right_outputs)):
        for token_index, (left_token, right_token) in enumerate(zip(left, right)):
            total += 1
            equal += int(left_token == right_token)
            if left_token != right_token and first_divergence is None:
                first_divergence = {
                    "request_id": request_id,
                    "token_index": token_index,
                    "bf16_token": left_token,
                    "int8_token": right_token,
                }
    return {
        "request_count": min(len(left_outputs), len(right_outputs)),
        "bf16_request_count": len(left_outputs),
        "int8_request_count": len(right_outputs),
        "request_counts_match": request_counts_match,
        "token_count_compared": total,
        "token_agreement": equal / total if total else 1.0,
        "first_divergence": first_divergence,
        "lengths_match": lengths_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BF16-KV and INT8-KV greedy model quality."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-decode-steps", type=int, default=16)
    parser.add_argument(
        "--prompt-ids-file",
        default=None,
        help="JSON list of fixed token-id prompts for natural-language free generation.",
    )
    parser.add_argument("--trace-max-events", type=int, default=2048)
    parser.add_argument("--trace-max-index-values", type=int, default=64)
    parser.add_argument(
        "--result-dir",
        default="benchmark_results/kv_quality",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--save-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--worker-mode",
        choices=("auto", "int8"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-prompts-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-name", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_mode is not None:
        if not all(
            (
                args.worker_prompts_file,
                args.worker_output_file,
                args.worker_batch_name,
            )
        ):
            parser.error("quality worker arguments are incomplete")
        run_worker(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model quality comparison")
    if args.output_len <= 0 or args.capture_decode_steps <= 0:
        parser.error("output and capture lengths must be positive")
    if args.trace_max_events <= 0 or args.trace_max_index_values <= 0:
        parser.error("trace limits must be positive")
    # Quality collection is intentionally not a latency measurement.  Enable
    # metadata tracing so the saved artifact also contains the actual
    # model-step/index-tensor layouts used by both modes.
    os.environ["NANOVLLM_SHAPE_TRACE"] = "1"
    os.environ["NANOVLLM_SHAPE_TRACE_MAX_EVENTS"] = str(args.trace_max_events)
    os.environ["NANOVLLM_SHAPE_TRACE_MAX_INDEX_VALUES"] = str(
        args.trace_max_index_values
    )

    # By default retain the old random-token smoke matrix.  The formal
    # natural-language run supplies a fixed JSON prompt file generated from an
    # offline corpus, so both KV modes consume exactly the same text tokens.
    prompt_groups = []
    if args.prompt_ids_file is not None:
        prompt_data = json.loads(Path(args.prompt_ids_file).read_text())
        if (
            not isinstance(prompt_data, list)
            or not prompt_data
            or any(
                not isinstance(prompt, list) or not prompt
                for prompt in prompt_data
            )
        ):
            parser.error("--prompt-ids-file must contain a non-empty JSON list of token lists")
        for offset in range(0, len(prompt_data), 4):
            prompts = prompt_data[offset : offset + 4]
            lengths = {len(prompt) for prompt in prompts}
            prompt_groups.append(
                {
                    "length": min(lengths) if len(lengths) == 1 else "mixed",
                    "prompts": prompts,
                }
            )
    else:
        for group_index, (length, count) in enumerate(
            ((128, 8), (1024, 8), (3072, 4))
        ):
            prompts = build_prompts(
                length,
                count,
                seed=args.seed + group_index,
                vocab_size=args.vocab_size,
            )
            for offset in range(0, count, 4):
                prompt_groups.append(
                    {
                        "length": length,
                        "prompts": prompts[offset : offset + 4],
                    }
                )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        **collect_benchmark_metadata(torch),
        "configuration": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "prompt_lengths": [
                len(prompt)
                for group in prompt_groups
                for prompt in group["prompts"]
            ],
            "batch_size": 4,
            "capture_decode_steps": args.capture_decode_steps,
            "same_prompt_seed": args.seed,
            "prompt_ids_file": str(args.prompt_ids_file)
            if args.prompt_ids_file
            else None,
            "prompt_source": (
                "offline fixed token file"
                if args.prompt_ids_file
                else "random token ids"
            ),
            "trace_max_events": args.trace_max_events,
            "trace_max_index_values": args.trace_max_index_values,
        },
        "batches": [],
    }
    for batch_index, group in enumerate(prompt_groups):
        batch_name = f"batch{batch_index}_len{group['length']}"
        prompt_path = result_dir / f"{batch_name}_prompts.json"
        prompt_path.write_text(
            json.dumps(group["prompts"], ensure_ascii=False) + "\n"
        )
        bf16_worker_path = run_in_worker_process(
            args,
            mode="auto",
            prompts_file=prompt_path,
            result_dir=result_dir,
            batch_name=batch_name,
        )
        int8_worker_path = run_in_worker_process(
            args,
            mode="int8",
            prompts_file=prompt_path,
            result_dir=result_dir,
            batch_name=batch_name,
        )
        bf16 = torch.load(bf16_worker_path, map_location="cpu", weights_only=False)
        int8 = torch.load(int8_worker_path, map_location="cpu", weights_only=False)
        # Raw logits are kept in the result-local tensor files only when
        # requested.  For comparison in the same process, rerun the compact
        # summaries through the raw files when available.
        if args.save_raw:
            bf16_raw = torch.load(
                result_dir / "raw" / f"{batch_name}_auto.pt",
                map_location="cpu",
                weights_only=False,
            )
            int8_raw = torch.load(
                result_dir / "raw" / f"{batch_name}_int8.pt",
                map_location="cpu",
                weights_only=False,
            )
            logits_comparison = compare_logits(
                bf16_raw["logits"],
                int8_raw["logits"],
            )
            attention_comparison = compare_attention(
                bf16_raw["attention"],
                int8_raw["attention"],
            )
        else:
            logits_comparison = {
                "steps_compared": 0,
                "note": "raw tensors disabled; only compact summaries were saved",
            }
            attention_comparison = {
                "records_compared": 0,
                "note": "raw tensors disabled",
            }
        batch_result = {
            "batch_index": batch_index,
            "prompt_length": group["length"],
            "prompts_file": str(prompt_path),
            "bf16_worker_artifact": str(bf16_worker_path),
            "int8_worker_artifact": str(int8_worker_path),
            "bf16": bf16,
            "int8": int8,
            "logits_comparison": logits_comparison,
            "attention_comparison": attention_comparison,
            "output_comparison": compare_outputs(bf16, int8),
        }
        result["batches"].append(batch_result)

    all_output = [batch["output_comparison"] for batch in result["batches"]]
    total_tokens = sum(item["token_count_compared"] for item in all_output)
    equal_tokens = sum(
        item["token_count_compared"] * item["token_agreement"]
        for item in all_output
    )
    result["summary"] = {
        "batches": len(result["batches"]),
        "token_count_compared": total_tokens,
        "token_agreement": equal_tokens / total_tokens if total_tokens else 1.0,
        "first_divergences": [
            {
                "batch_index": index,
                **batch["output_comparison"]["first_divergence"],
            }
            for index, batch in enumerate(result["batches"])
            if batch["output_comparison"]["first_divergence"] is not None
        ],
    }
    name = args.name or "kv_quality_bf16_vs_int8"
    path = result_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
