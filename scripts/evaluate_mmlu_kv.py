"""Compare BF16/INT8 first-token MMLU scores on an offline subset.

This is intentionally a subset evaluator. It uses the engine's prefill logits
to score the four answer-label tokens, runs the two KV modes in separate
processes, and records the dataset/model/template provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.benchmark_metadata import collect_benchmark_metadata
from nanovllm.utils.context import get_context


def load_questions(data_dir: Path, subjects: list[str], limit: int) -> list[dict[str, Any]]:
    rows = []
    for subject in subjects:
        path = data_dir / "test" / f"{subject}_test.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 6:
                    continue
                question, a, b, c, d, answer = row
                rows.append(
                    {
                        "subject": subject,
                        "question": question,
                        "choices": [a, b, c, d],
                        "answer": answer,
                    }
                )
                if limit > 0 and sum(item["subject"] == subject for item in rows) >= limit:
                    break
    return rows


def format_prompt(item: dict[str, Any]) -> str:
    labels = ("A", "B", "C", "D")
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, item["choices"])
    )
    return (
        "The following is a multiple-choice question. Choose the correct answer "
        "from A, B, C, or D.\n\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )


def run_worker(
    *,
    model: str,
    prompts_file: Path,
    output_file: Path,
    mode: str,
    max_model_len: int,
    max_num_seqs: int,
    trace_max_events: int,
) -> None:
    prompts = json.loads(prompts_file.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    option_ids = []
    # Qwen3 tokenizes the answer continuation with a leading space.  Scoring
    # bare "A"/"B"/"C"/"D" would measure the wrong vocabulary entries.
    for label in (" A", " B", " C", " D"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"answer label {label!r} is not one token: {ids}")
        option_ids.append(ids[0])

    llm = LLM(
        model,
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len * max_num_seqs,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype="auto" if mode == "auto" else "int8",
        kv_dequant_backend="fused",
        int8_partitioned_decode_threshold=999999,
    )
    logits_rows: list[torch.Tensor] = []
    runner = llm.model_runner
    original_compute_logits = runner.model.compute_logits

    def capture_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        logits = original_compute_logits(hidden_states)
        context = get_context()
        if context.is_prefill:
            logits_rows.append(logits.detach().float().cpu())
        return logits

    runner.model.compute_logits = capture_logits
    runner.call("reset_execution_stats")
    runner.call("reset_shape_trace")
    try:
        for ids in prompt_ids:
            llm.add_request(
                ids,
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
            )
        while not llm.is_finished():
            llm.step()
        if not logits_rows:
            raise RuntimeError("worker produced no prefill logits")
        prefill_logits = torch.cat(logits_rows, dim=0)
        if prefill_logits.size(0) != len(prompt_ids):
            raise RuntimeError(
                "prefill logits/request count mismatch: "
                f"{prefill_logits.size(0)} != {len(prompt_ids)}"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": mode,
                "logits": prefill_logits,
                "prompt_lengths": [len(ids) for ids in prompt_ids],
                "option_ids": option_ids,
                "execution_stats": runner.call("get_execution_stats"),
                "shape_trace": runner.call("get_shape_trace"),
            },
            output_file,
        )
    finally:
        runner.model.compute_logits = original_compute_logits
        llm.exit()


def score_worker(
    questions: list[dict[str, Any]],
    worker: dict[str, Any],
) -> dict[str, Any]:
    logits = worker["logits"].float()
    option_ids = worker["option_ids"]
    scores = logits[:, option_ids]
    predicted = scores.argmax(dim=-1).tolist()
    gold = ["ABCD".index(item["answer"]) for item in questions]
    by_subject: dict[str, dict[str, int]] = {}
    for item, prediction, target in zip(questions, predicted, gold):
        row = by_subject.setdefault(
            item["subject"],
            {"correct": 0, "count": 0},
        )
        row["count"] += 1
        row["correct"] += int(prediction == target)
    for row in by_subject.values():
        row["accuracy"] = row["correct"] / row["count"]
    return {
        "mode": worker["mode"],
        "count": len(questions),
        "accuracy": sum(p == g for p, g in zip(predicted, gold)) / len(gold),
        "predicted_labels": ["ABCD"[p] for p in predicted],
        "gold_labels": ["ABCD"[g] for g in gold],
        "option_score_shape": list(scores.shape),
        "prompt_lengths": worker["prompt_lengths"],
        "by_subject": by_subject,
        "execution_stats": worker["execution_stats"],
        "shape_trace": worker["shape_trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--limit-per-subject", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--worker-mode", choices=("auto", "int8"))
    parser.add_argument("--prompts-file")
    parser.add_argument("--worker-output-file")
    parser.add_argument("--trace-max-events", type=int, default=2048)
    args = parser.parse_args()

    if args.worker_mode:
        if not args.prompts_file or not args.worker_output_file:
            parser.error("worker mode requires prompts and output paths")
        import os

        os.environ["NANOVLLM_SHAPE_TRACE"] = "1"
        os.environ["NANOVLLM_SHAPE_TRACE_MAX_EVENTS"] = str(args.trace_max_events)
        run_worker(
            model=args.model,
            prompts_file=Path(args.prompts_file),
            output_file=Path(args.worker_output_file),
            mode=args.worker_mode,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            trace_max_events=args.trace_max_events,
        )
        return

    subjects = [item.strip() for item in args.subjects.split(",") if item.strip()]
    questions = load_questions(
        Path(args.data_dir),
        subjects,
        args.limit_per_subject,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts = [format_prompt(item) for item in questions]
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    if max(map(len, prompt_ids)) > args.max_model_len:
        parser.error("an MMLU prompt exceeds --max-model-len")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = result_dir / "prompts.json"
    prompts_file.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n")
    workers = {}
    for mode in ("auto", "int8"):
        output_file = result_dir / f"{mode}.pt"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            args.model,
            "--data-dir",
            args.data_dir,
            "--subjects",
            args.subjects,
            "--limit-per-subject",
            str(args.limit_per_subject),
            "--max-model-len",
            str(args.max_model_len),
            "--batch-size",
            str(args.batch_size),
            "--result-dir",
            str(result_dir),
            "--worker-mode",
            mode,
            "--prompts-file",
            str(prompts_file),
            "--worker-output-file",
            str(output_file),
            "--trace-max-events",
            str(args.trace_max_events),
        ]
        subprocess.run(command, check=True)
        workers[mode] = torch.load(output_file, map_location="cpu", weights_only=False)
    summary = {
        **collect_benchmark_metadata(torch),
        "subjects": subjects,
        "limit_per_subject": args.limit_per_subject,
        "count": len(questions),
        "template": (
            "zero-shot question + four choices + Answer:; "
            "score next-token logits for leading-space labels"
        ),
        "evaluation_scope": (
            "offline MMLU test-subset first-token label score, not a full "
            "official MMLU reproduction"
        ),
        "auto": score_worker(questions, workers["auto"]),
        "int8": score_worker(questions, workers["int8"]),
    }
    summary["token_prediction_agreement"] = sum(
        a == b
        for a, b in zip(
            summary["auto"]["predicted_labels"],
            summary["int8"]["predicted_labels"],
        )
    ) / len(questions)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
