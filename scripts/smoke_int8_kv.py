import argparse
import os

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test one nano-vLLM KV cache configuration on a CUDA GPU machine.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--recurrent-state-dtype",
        choices=("float32", "model"),
        default="float32",
    )
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--kv-dequant-backend", choices=("fused", "triton", "torch"), default="fused")
    parser.add_argument("--sliding-window-size", type=int, default=None)
    parser.add_argument("--enable-dynamic-chunked-prefill", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        recurrent_state_dtype=args.recurrent_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_dequant_backend=args.kv_dequant_backend,
        sliding_window_size=args.sliding_window_size,
        enable_dynamic_chunked_prefill=args.enable_dynamic_chunked_prefill,
    )
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
    output = llm.generate([args.prompt], params, use_tqdm=False)[0]
    runner = llm.model_runner
    print(f"kv_cache_dtype: {runner.config.kv_cache_dtype}")
    print(f"kv_dequant_backend: {runner.config.kv_dequant_backend}")
    print(f"sliding_window_size: {runner.config.sliding_window_size}")
    print(f"enable_dynamic_chunked_prefill: {runner.config.enable_dynamic_chunked_prefill}")
    print(f"recurrent_state_dtype: {runner.config.recurrent_state_dtype}")
    print(f"recurrent_state_storage: {runner.get_recurrent_state_stats()}")
    print(f"kv_cache tensor dtype: {runner.kv_cache.dtype}")
    print(f"kv_scale tensor dtype: {None if runner.kv_scale is None else runner.kv_scale.dtype}")
    print(f"num_kvcache_blocks: {runner.config.num_kvcache_blocks}")
    print(f"output token ids: {output['token_ids']}")
    print(f"output text: {output['text']!r}")


if __name__ == "__main__":
    main()
