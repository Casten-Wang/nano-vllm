"""Audit checkpoint-to-model parameter names without loading tensor payloads."""

from __future__ import annotations

import argparse
from glob import glob
from importlib.util import find_spec
import json
from math import prod
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.models.model_spec import resolve_model_spec
from nanovllm.models.cache_plan import plan_cache_memory
from nanovllm.models.registry import create_model
from nanovllm.utils.loader import resolve_packed_parameter
from nanovllm.benchmark_metadata import checkpoint_manifest_metadata


class MetaTensorSlice:
    """Shape-only safetensors slice that never reads checkpoint payloads."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        self.tensor = torch.empty(shape, device="meta", dtype=dtype)
        self.requested_numel = 0

    def get_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    def __getitem__(self, key):
        result = self.tensor[key]
        self.requested_numel += result.numel()
        return result


SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_storage_bytes(tensor_slice) -> int:
    dtype = tensor_slice.get_dtype()
    try:
        element_size = SAFETENSORS_DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype: {dtype}") from error
    return prod(tensor_slice.get_shape()) * element_size


def validate_weight_shape(model, checkpoint, source_name, target_name, packed):
    parameter = model.get_parameter(target_name)
    checkpoint_slice = checkpoint.get_slice(source_name)
    source_shape = tuple(checkpoint_slice.get_shape())
    source_bytes = tensor_storage_bytes(checkpoint_slice)
    meta_slice = MetaTensorSlice(source_shape, parameter.dtype)
    if packed is not None:
        _, shard_id = packed
        packed_safetensors_loader = getattr(
            parameter,
            "packed_safetensors_loader",
            None,
        )
        if packed_safetensors_loader is not None:
            packed_safetensors_loader(parameter, meta_slice, shard_id)
            return source_bytes, meta_slice.requested_numel * (
                source_bytes // prod(source_shape)
            ), True
        source = torch.empty(source_shape, device="meta", dtype=parameter.dtype)
        getattr(parameter, "weight_loader")(parameter, source, shard_id)
        return source_bytes, source_bytes, False
    safetensors_loader = getattr(parameter, "safetensors_loader", None)
    if safetensors_loader is not None:
        safetensors_loader(parameter, meta_slice)
        return source_bytes, meta_slice.requested_numel * (
            source_bytes // prod(source_shape)
        ), True
    weight_loader = getattr(parameter, "weight_loader", None)
    if weight_loader is not None:
        source = torch.empty(source_shape, device="meta", dtype=parameter.dtype)
        weight_loader(parameter, source)
        return source_bytes, source_bytes, False
    if tuple(parameter.shape) != source_shape:
        raise ValueError(
            f"checkpoint shape {source_shape} does not match parameter "
            f"shape {tuple(parameter.shape)}"
        )
    return source_bytes, source_bytes, False


def audit_checkpoint_mapping(model: torch.nn.Module, model_path: str | Path) -> dict:
    files = sorted(
        filename
        for filename in glob(str(Path(model_path) / "*.safetensors"))
        if Path(filename).is_file()
    )
    index_path = Path(model_path) / "model.safetensors.index.json"
    index_names = None
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        index_names = list(index["weight_map"])
        expected_files = {
            str(filename) for filename in index["weight_map"].values()
        }
        present_files = {Path(filename).name for filename in files}
        if not expected_files <= present_files:
            files = []
    if not files and index_names is None:
        raise ValueError(
            f"no safetensors checkpoint files or index found in {model_path}"
        )

    expected = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    loaded = set()
    skipped = []
    unexpected = []
    shape_errors = []
    source_count = 0
    mapped_checkpoint_bytes = 0
    estimated_local_payload_bytes = 0
    lazy_tensor_count = 0
    full_tensor_count = 0

    def record_source(source_name, checkpoint=None):
        nonlocal source_count, mapped_checkpoint_bytes
        nonlocal estimated_local_payload_bytes
        nonlocal lazy_tensor_count, full_tensor_count
        source_count += 1
        mapped_name = map_weight_name(source_name)
        if mapped_name is None:
            skipped.append(source_name)
            return
        packed = resolve_packed_parameter(
            mapped_name,
            packed_modules_mapping,
        )
        target_name = packed[0] if packed is not None else mapped_name
        if target_name not in expected:
            unexpected.append(
                {"source": source_name, "mapped": target_name}
            )
            return
        if checkpoint is not None:
            try:
                source_bytes, local_bytes, lazy = validate_weight_shape(
                    model,
                    checkpoint,
                    source_name,
                    target_name,
                    packed,
                )
            except (AssertionError, IndexError, RuntimeError, ValueError) as error:
                shape_errors.append(
                    {
                        "source": source_name,
                        "mapped": target_name,
                        "error": str(error),
                    }
                )
                return
            mapped_checkpoint_bytes += source_bytes
            estimated_local_payload_bytes += local_bytes
            if lazy:
                lazy_tensor_count += 1
            else:
                full_tensor_count += 1
        loaded.add(target_name)

    if files:
        for filename in files:
            with safe_open(filename, framework="pt", device="cpu") as checkpoint:
                for source_name in checkpoint.keys():
                    record_source(source_name, checkpoint)
    else:
        for source_name in index_names or ():
            record_source(source_name)

    missing = sorted(expected - loaded)
    return {
        "scope": (
            "parameter names and shapes from safetensors headers; tensor values are not read"
            if files
            else "parameter names from safetensors index; tensor shapes and values are not read"
        ),
        "validation_level": "names_and_shapes" if files else "names_only",
        "shape_validation_complete": bool(files),
        "shard_count": len(files),
        "source_tensor_count": source_count,
        "expected_parameter_count": len(expected),
        "mapped_parameter_count": len(loaded),
        "skipped_tensor_count": len(skipped),
        "missing_parameters": missing,
        "unexpected_weights": unexpected,
        "shape_errors": shape_errors,
        "checkpoint_loading": (
            {
                "mapped_checkpoint_bytes": mapped_checkpoint_bytes,
                "estimated_local_payload_bytes": estimated_local_payload_bytes,
                "estimated_avoided_payload_bytes": (
                    mapped_checkpoint_bytes - estimated_local_payload_bytes
                ),
                "estimated_local_payload_fraction": (
                    estimated_local_payload_bytes / mapped_checkpoint_bytes
                    if mapped_checkpoint_bytes
                    else 0.0
                ),
                "lazy_tensor_count": lazy_tensor_count,
                "full_tensor_count": full_tensor_count,
            }
            if files
            else None
        ),
        "valid": not missing and not unexpected and not shape_errors,
    }


def parameter_storage_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def cache_storage_metadata(model_spec, tp_size: int, model_dtype_bytes: int) -> dict:
    fp32_state = plan_cache_memory(
        model_spec,
        tp_size,
        kv_dtype_bytes=model_dtype_bytes,
        recurrent_dtype_bytes=4,
        convolution_dtype_bytes=model_dtype_bytes,
    )
    model_state = plan_cache_memory(
        model_spec,
        tp_size,
        kv_dtype_bytes=model_dtype_bytes,
        recurrent_dtype_bytes=model_dtype_bytes,
        convolution_dtype_bytes=model_dtype_bytes,
    )
    int8_cache = plan_cache_memory(
        model_spec,
        tp_size,
        kv_dtype_bytes=1,
        recurrent_dtype_bytes=model_dtype_bytes,
        convolution_dtype_bytes=model_dtype_bytes,
    )
    return {
        "model_max_position_embeddings": int(
            model_spec.text_config.max_position_embeddings
        ),
        "kv_bytes_per_token": fp32_state.kv_bytes_per_token,
        "kv_bytes_per_token_by_dtype": {
            "auto": fp32_state.kv_bytes_per_token,
            "int8": (
                int8_cache.kv_bytes_per_token
                + int8_cache.int8_scale_bytes_per_token
            ),
        },
        "state_bytes_per_sequence": {
            "float32": fp32_state.recurrent_bytes_per_sequence
            + fp32_state.convolution_bytes_per_sequence,
            "model": model_state.recurrent_bytes_per_sequence
            + model_state.convolution_bytes_per_sequence,
            "convolution": fp32_state.convolution_bytes_per_sequence,
        },
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
        audit_modules = {}
        if find_spec("flash_attn") is None:
            class AuditOnlyAttention(torch.nn.Module):
                """Parameter-free stand-in for shape-only model audits."""

                def __init__(self, *args, **kwargs) -> None:
                    super().__init__()

                def forward(self, *args, **kwargs):
                    raise RuntimeError("audit-only attention cannot execute")

            attention_module = types.ModuleType("nanovllm.layers.attention")
            attention_module.Attention = AuditOnlyAttention
            audit_modules["nanovllm.layers.attention"] = attention_module
        with (
            patch("torch.distributed.get_world_size", return_value=tp_size),
            patch("torch.distributed.get_rank", return_value=0),
            patch.object(torch, "compile", lambda function: function),
            patch.dict(sys.modules, audit_modules),
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
        "--require-shards",
        action="store_true",
        help=(
            "Require every indexed shard and complete shape validation. "
            "Without this flag, an index-only name audit is allowed."
        ),
    )
    parser.add_argument(
        "--verify-shard-hashes",
        action="store_true",
        help="Stream every local shard once and record its SHA-256 identity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/checkpoint_mapping_audit.json"),
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    model_spec = resolve_model_spec(config)
    model_config = model_spec.text_config
    model_dtype = getattr(model_config, "dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        model_dtype = getattr(model_config, "torch_dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        model_dtype = torch.bfloat16
    model_dtype_bytes = torch.empty((), dtype=model_dtype).element_size()
    initial_checkpoint_manifest = checkpoint_manifest_metadata(
        args.model,
        require_shards=args.require_shards,
    )
    results = {}
    for tp_size in args.tp_sizes:
        model = instantiate_meta_model(args.model, tp_size)
        result = audit_checkpoint_mapping(model, args.model)
        result["local_parameter_bytes"] = parameter_storage_bytes(model)
        result.update(cache_storage_metadata(model_spec, tp_size, model_dtype_bytes))
        results[f"tp{tp_size}"] = result
        del model
    final_checkpoint_manifest = checkpoint_manifest_metadata(
        args.model,
        require_shards=args.require_shards,
        hash_shards=args.verify_shard_hashes,
    )
    stable_fields = ("config_sha256", "index_sha256", "files")
    initial_stability = {
        "config_sha256": initial_checkpoint_manifest["config_sha256"],
        "index_sha256": initial_checkpoint_manifest["index_sha256"],
        "files": [
            {
                key: item[key]
                for key in ("name", "size_bytes", "mtime_ns", "present")
            }
            for item in initial_checkpoint_manifest["files"]
        ],
    }
    final_stability = {
        "config_sha256": final_checkpoint_manifest["config_sha256"],
        "index_sha256": final_checkpoint_manifest["index_sha256"],
        "files": [
            {
                key: item[key]
                for key in ("name", "size_bytes", "mtime_ns", "present")
            }
            for item in final_checkpoint_manifest["files"]
        ],
    }
    if any(initial_stability[key] != final_stability[key] for key in stable_fields):
        raise RuntimeError("checkpoint files changed during the mapping audit")
    checkpoint_manifest = final_checkpoint_manifest

    report = {
        "model": str(Path(args.model).expanduser().resolve()),
        "checkpoint_manifest": checkpoint_manifest,
        "tensor_parallel_sizes": list(args.tp_sizes),
        "results": results,
        "complete": all(
            result["shape_validation_complete"] for result in results.values()
        ),
        "valid": all(result["valid"] for result in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
