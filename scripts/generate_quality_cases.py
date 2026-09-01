"""Generate fixed token trajectories for offline BF16/INT8 quality tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--continuation-len", type=int, default=128)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.prompt_len, args.continuation_len, args.num_cases) <= 0:
        parser.error("lengths and num-cases must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    text = Path(args.text_file).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    required = args.num_cases * (args.prompt_len + args.continuation_len)
    if len(token_ids) < required:
        parser.error(
            f"text contains {len(token_ids)} tokens; {required} are required"
        )

    cases = []
    for index in range(args.num_cases):
        start = index * (args.prompt_len + args.continuation_len)
        prompt_ids = token_ids[start : start + args.prompt_len]
        target_ids = token_ids[
            start + args.prompt_len : start + args.prompt_len + args.continuation_len
        ]
        cases.append(
            {
                "case_name": f"wikitext2_case{index}",
                "prompt_length": len(prompt_ids),
                "continuation_length": len(target_ids),
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "prompt_text_preview": tokenizer.decode(prompt_ids[:128]),
                "target_text_preview": tokenizer.decode(target_ids[:128]),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cases),
                "prompt_len": args.prompt_len,
                "continuation_len": args.continuation_len,
                "token_count": required,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
