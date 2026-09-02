"""Compare teacher-forced quality trajectories across two checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required quality artifact is missing: {path}")
    return json.loads(path.read_text())


def case_path(root: Path, run_id: str, tp_size: int, state_dtype: str) -> Path:
    name = f"{run_id}_qwen35_tp{tp_size}_state-{state_dtype}"
    return root / name / f"{name}.json"


def flatten_trajectory(result: dict, name: str) -> list[float | int]:
    return [
        value
        for batch in result.get("batches", [])
        for step in batch.get("decode_trajectories", {}).get(name, [])
        for value in step
    ]


def compare_case(
    baseline: dict,
    candidate: dict,
    *,
    min_top1_agreement: float,
    max_mean_logprob_diff: float,
    max_ppl_relative_change: float,
) -> dict:
    for field in ("commit", "case_token_digest"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"checkpoint quality {field} differs")
    for field in ("tensor_parallel_size", "recurrent_state_dtype"):
        if baseline.get("configuration", {}).get(field) != candidate.get(
            "configuration", {}
        ).get(field):
            raise ValueError(f"checkpoint quality configuration differs: {field}")
    baseline_top1 = flatten_trajectory(baseline, "bf16_top1_token_ids")
    candidate_top1 = flatten_trajectory(candidate, "bf16_top1_token_ids")
    baseline_logprobs = flatten_trajectory(baseline, "bf16_target_logprobs")
    candidate_logprobs = flatten_trajectory(candidate, "bf16_target_logprobs")
    if (
        not baseline_top1
        or len(baseline_top1) != len(candidate_top1)
        or not baseline_logprobs
        or len(baseline_logprobs) != len(candidate_logprobs)
        or len(baseline_top1) != len(baseline_logprobs)
    ):
        raise ValueError("checkpoint quality trajectories are empty or misaligned")
    logprob_diffs = [
        abs(float(left) - float(right))
        for left, right in zip(baseline_logprobs, candidate_logprobs)
    ]
    if not all(math.isfinite(value) for value in logprob_diffs):
        raise ValueError("checkpoint quality contains non-finite log probabilities")
    baseline_nll = -sum(float(value) for value in baseline_logprobs) / len(
        baseline_logprobs
    )
    candidate_nll = -sum(float(value) for value in candidate_logprobs) / len(
        candidate_logprobs
    )
    baseline_ppl = math.exp(baseline_nll)
    candidate_ppl = math.exp(candidate_nll)
    top1_agreement = sum(
        left == right for left, right in zip(baseline_top1, candidate_top1)
    ) / len(baseline_top1)
    mean_logprob_diff = sum(logprob_diffs) / len(logprob_diffs)
    ppl_relative_change = candidate_ppl / baseline_ppl - 1.0
    return {
        "valid": (
            top1_agreement >= min_top1_agreement
            and mean_logprob_diff <= max_mean_logprob_diff
            and abs(ppl_relative_change) <= max_ppl_relative_change
        ),
        "token_count": len(baseline_top1),
        "top1_agreement": top1_agreement,
        "mean_abs_target_logprob_diff": mean_logprob_diff,
        "max_abs_target_logprob_diff": max(logprob_diffs),
        "baseline_target_nll": baseline_nll,
        "candidate_target_nll": candidate_nll,
        "baseline_target_ppl": baseline_ppl,
        "candidate_target_ppl": candidate_ppl,
        "ppl_relative_change": ppl_relative_change,
    }


def compare_quality_runs(
    baseline_dir: Path,
    baseline_run_id: str,
    candidate_dir: Path,
    candidate_run_id: str,
    tp_sizes: tuple[int, ...],
    *,
    min_top1_agreement: float,
    max_mean_logprob_diff: float,
    max_ppl_relative_change: float,
) -> dict:
    cases = {}
    commits = set()
    baseline_digests = set()
    candidate_digests = set()
    for tp_size in tp_sizes:
        for state_dtype in ("float32", "model"):
            baseline = load_json(
                case_path(baseline_dir, baseline_run_id, tp_size, state_dtype)
            )
            candidate = load_json(
                case_path(candidate_dir, candidate_run_id, tp_size, state_dtype)
            )
            result = compare_case(
                baseline,
                candidate,
                min_top1_agreement=min_top1_agreement,
                max_mean_logprob_diff=max_mean_logprob_diff,
                max_ppl_relative_change=max_ppl_relative_change,
            )
            name = f"tp{tp_size}_state-{state_dtype}"
            cases[name] = result
            commits.update((baseline["commit"], candidate["commit"]))
            baseline_digests.add(baseline["checkpoint_manifest"]["digest"])
            candidate_digests.add(candidate["checkpoint_manifest"]["digest"])
    identity_valid = (
        len(commits) == 1
        and len(baseline_digests) == 1
        and len(candidate_digests) == 1
        and baseline_digests != candidate_digests
    )
    return {
        "valid": identity_valid and all(case["valid"] for case in cases.values()),
        "identity_valid": identity_valid,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "tensor_parallel_sizes": list(tp_sizes),
        "thresholds": {
            "min_top1_agreement": min_top1_agreement,
            "max_mean_abs_target_logprob_diff": max_mean_logprob_diff,
            "max_abs_ppl_relative_change": max_ppl_relative_change,
        },
        "commit": next(iter(commits)) if len(commits) == 1 else None,
        "baseline_checkpoint_digest": (
            next(iter(baseline_digests)) if len(baseline_digests) == 1 else None
        ),
        "candidate_checkpoint_digest": (
            next(iter(candidate_digests)) if len(candidate_digests) == 1 else None
        ),
        "cases": cases,
    }


def parse_tp_sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("TP sizes must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, required=True)
    parser.add_argument("--min-top1-agreement", type=float, default=0.80)
    parser.add_argument("--max-mean-logprob-diff", type=float, default=0.50)
    parser.add_argument("--max-ppl-relative-change", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.min_top1_agreement <= 1:
        parser.error("min-top1-agreement must be in [0, 1]")
    if args.max_mean_logprob_diff < 0 or args.max_ppl_relative_change < 0:
        parser.error("quality difference limits must be non-negative")
    report = compare_quality_runs(
        args.baseline_dir,
        args.baseline_run_id,
        args.candidate_dir,
        args.candidate_run_id,
        args.tp_sizes,
        min_top1_agreement=args.min_top1_agreement,
        max_mean_logprob_diff=args.max_mean_logprob_diff,
        max_ppl_relative_change=args.max_ppl_relative_change,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
