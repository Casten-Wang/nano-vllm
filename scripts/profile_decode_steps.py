import argparse
import json
import os
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def build_prompts(num_seqs: int, input_len: int, vocab_size: int) -> list[list[int]]:
    return [[(i * 997 + j) % vocab_size for j in range(input_len)] for i in range(num_seqs)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile steady decode steps with torch.profiler.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=3072)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--profile-decode-steps", type=int, default=16)
    parser.add_argument("--skip-decode-steps", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="int8")
    parser.add_argument("--kv-dequant-backend", choices=("fused", "triton", "torch"), default="fused")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--result-dir", default="benchmark_results/profile_decode")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
    )
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
    for prompt in build_prompts(args.num_seqs, args.input_len, 10000):
        llm.add_request(prompt, sampling_params)

    decode_steps = 0
    while not llm.is_finished() and decode_steps < args.skip_decode_steps:
        _output, _num_tokens, _prefill_tokens, step_decode_tokens = llm.step()
        if step_decode_tokens > 0:
            decode_steps += 1
    assert decode_steps >= args.skip_decode_steps, "request finished before profile window"

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        profiled = 0
        while not llm.is_finished() and profiled < args.profile_decode_steps:
            _output, _num_tokens, _prefill_tokens, step_decode_tokens = llm.step()
            if step_decode_tokens > 0:
                profiled += 1
                prof.step()
    assert profiled > 0

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"profile_decode_steps_{args.kv_cache_dtype}_{args.num_seqs}x{args.input_len}"
    table_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    table_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=40)
    text_path = result_dir / f"{name}.txt"
    json_path = result_dir / f"{name}.json"
    trace_path = result_dir / f"{name}.json.gz"

    text_path.write_text(
        "CUDA time sorted\\n"
        "================\\n"
        f"{table_cuda}\\n\\n"
        "CPU time sorted\\n"
        "===============\\n"
        f"{table_cpu}\\n"
    )
    rows = []
    for item in prof.key_averages():
        device_time_total = getattr(item, "cuda_time_total", getattr(item, "device_time_total", 0.0))
        self_device_time_total = getattr(item, "self_cuda_time_total", getattr(item, "self_device_time_total", 0.0))
        rows.append(
            {
                "key": item.key,
                "count": item.count,
                "cpu_time_total_us": item.cpu_time_total,
                "cuda_time_total_us": device_time_total,
                "self_cpu_time_total_us": item.self_cpu_time_total,
                "self_cuda_time_total_us": self_device_time_total,
            }
        )
    rows.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
    json_path.write_text(json.dumps(rows[:80], indent=2, ensure_ascii=False) + "\n")
    prof.export_chrome_trace(str(trace_path))

    print(text_path.read_text())
    print(f"Wrote {text_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
