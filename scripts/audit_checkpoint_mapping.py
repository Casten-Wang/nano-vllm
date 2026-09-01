"""Audit checkpoint-to-model parameter names without loading tensor payloads."""

from __future__ import annotations

import argparse
from glob import glob
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.models.model_spec import resolve_model_spec
from nanovllm.models.registry import create_model
from nanovllm.utils.loader import resolve_packed_parameter


def audit_checkpoint_mapping(model: torch.nn.Module, model_path: str | Path) -> dict:
    files = sorted(glob(str(Path(model_path) / "*.safetensors")))
    if not files:
        raise ValueError(f"no safetensors checkpoint files found in {model_path}")

    expected = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    loaded = set()
    skipped = []
    unexpected = []
    source_count = 0

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for source_name in checkpoint.keys():
                source_count += 1
                mapped_name = map_weight_name(source_name)
                if mapped_name is None:
                    skipped.append(source_name)
                    continue
                packed = resolve_packed_parameter(
                    mapped_name,
                    packed_modules_mapping,
                )
                target_name = packed[0] if packed is not None else mapped_name
                if target_name not in expected:
                    unexpected.append(
                        {"source": source_name, "mapped": target_name}
                    )
                    continue
                loaded.add(target_name)

    missing = sorted(expected - loaded)
    return {
        "scope": "parameter names from safetensors headers; tensor values are not read",
        "shard_count": len(files),
        "source_tensor_count": source_count,
        "expected_parameter_count": len(expected),
        "mapped_parameter_count": len(loaded),
        "skipped_tensor_count": len(skipped),
        "missing_parameters": missing,
        "unexpected_weights": unexpected,
        "valid": not missing and not unexpected,
    }


def instantiate_meta_model(model_path: str, tp_size: int) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_path)
    model_spec = resolve_model_spec(config)
    model_config = model_spec.text_config
    if not hasattr(model_config, "dtype"):
        torch_dtype = getattr(model_config, "torch_dtype", None)
        model_config.dtype = (
            torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16
        )
    original_device = torch.get_default_device()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(model_config.dtype)
        with (
            patch("torch.distributed.get_world_size", return_value=tp_size),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            return create_model(model_spec.architecture, model_config)
    finally:
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)


def parse_tp_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("expected comma-separated positive TP sizes")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-sizes", type=parse_tp_sizes, default=(4, 8))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/checkpoint_mapping_audit.json"),
    )
    args = parser.parse_args()

    results = {}
    for tp_size in args.tp_sizes:
        model = instantiate_meta_model(args.model, tp_size)
        results[f"tp{tp_size}"] = audit_checkpoint_mapping(model, args.model)
        del model

    report = {
        "model": str(Path(args.model).expanduser().resolve()),
        "tensor_parallel_sizes": list(args.tp_sizes),
        "results": results,
        "valid": all(result["valid"] for result in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
