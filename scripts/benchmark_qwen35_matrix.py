import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts" / "benchmark_baseline.py"


@dataclass(frozen=True)
class BenchmarkCase:
    tensor_parallel_size: int
    recurrent_state_dtype: str
    kv_cache_dtype: str

    @property
    def name(self) -> str:
        kv = "bf16" if self.kv_cache_dtype == "auto" else self.kv_cache_dtype
        return f"qwen35_tp{self.tensor_parallel_size}_state-{self.recurrent_state_dtype}_kv-{kv}"


def comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def build_cases(tp_sizes: tuple[int, ...]) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(tp, state_dtype, kv_dtype)
        for tp, state_dtype, kv_dtype in itertools.product(
            tp_sizes,
            ("float32", "model"),
            ("auto", "int8"),
        )
    ]


def command_for_case(
    args: argparse.Namespace,
    case: BenchmarkCase,
    repeat: int = 1,
) -> list[str]:
    repeats = getattr(args, "repeats", 1)
    name = f"{case.name}_r{repeat}" if repeats > 1 else case.name
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(case.tensor_parallel_size),
        "--recurrent-state-dtype",
        case.recurrent_state_dtype,
        "--kv-cache-dtype",
        case.kv_cache_dtype,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--seed",
        str(args.seed),
        "--name",
        name,
        "--result-dir",
        args.result_dir,
        "--enforce-eager",
    ]
    if not args.warmup:
        command.append("--no-warmup")
    return command


def visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return 0 if devices == ["-1"] else len(devices)

    import torch

    return torch.cuda.device_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Qwen3.5 TP4/TP8 benchmark matrix."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-sizes", type=comma_separated_ints, default=(4, 8))
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--result-dir", default="benchmark_results/qwen35_matrix")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    cases = build_cases(args.tp_sizes)
    if not args.dry_run:
        required_gpus = max(case.tensor_parallel_size for case in cases)
        available_gpus = visible_gpu_count()
        if available_gpus < required_gpus:
            raise SystemExit(
                f"benchmark matrix requires {required_gpus} visible GPUs, "
                f"but found {available_gpus}; use --tp-sizes to select a runnable subset"
            )

    runs = [
        (case, repeat)
        for case in cases
        for repeat in range(1, args.repeats + 1)
    ]
    for index, (case, repeat) in enumerate(runs, start=1):
        command = command_for_case(args, case, repeat)
        print(f"[{index}/{len(runs)}] {case.name} repeat {repeat}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
