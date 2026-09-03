import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.models.qwen35_fp8 import dequantize_fp8_block_weight


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def resolve_packed_parameter(
    weight_name: str,
    packed_modules_mapping: dict,
) -> tuple[str, object] | None:
    """Map an exact module path segment, never a substring of a packed name."""

    path = weight_name.split(".")
    for source_module, (target_module, shard_id) in packed_modules_mapping.items():
        try:
            index = path.index(source_module)
        except ValueError:
            continue
        mapped_path = path.copy()
        mapped_path[index] = target_module
        return ".".join(mapped_path), shard_id
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    loaded_parameters: set[str] = set()
    loaded_packed_shards: dict[str, set[object]] = {}
    loaded_source_weights: set[str] = set()
    quantization = getattr(model, "checkpoint_quantization_spec", None)
    fp8_block_size = (
        quantization.weight_block_size
        if getattr(quantization, "format", None) == "fp8_block"
        else None
    )
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            source_names = set(f.keys())
            for source_weight_name in sorted(source_names):
                if source_weight_name in loaded_source_weights:
                    raise RuntimeError(
                        "checkpoint contains duplicate weight: "
                        f"{source_weight_name}"
                    )
                loaded_source_weights.add(source_weight_name)
                if fp8_block_size is not None and source_weight_name.endswith(
                    ".weight_scale_inv"
                ):
                    continue
                weight_name = map_weight_name(source_weight_name)
                if weight_name is None:
                    continue
                resolve_checkpoint_parameter = getattr(
                    model,
                    "resolve_checkpoint_parameter",
                    None,
                )
                packed_parameter = (
                    resolve_checkpoint_parameter(weight_name)
                    if resolve_checkpoint_parameter is not None
                    else None
                )
                if packed_parameter is None:
                    packed_parameter = resolve_packed_parameter(
                        weight_name,
                        packed_modules_mapping,
                    )
                loaded_tensor = None
                if fp8_block_size is not None and source_weight_name.endswith(
                    ".weight"
                ):
                    source_slice = f.get_slice(source_weight_name)
                    if source_slice.get_dtype().startswith("F8_"):
                        scale_name = f"{source_weight_name}_scale_inv"
                        if scale_name not in source_names:
                            raise RuntimeError(
                                f"FP8 checkpoint weight is missing scale: {scale_name}"
                            )
                        if packed_parameter is not None:
                            param_name, shard_id = packed_parameter
                            param = model.get_parameter(param_name)
                            fp8_loader = getattr(
                                param,
                                "fp8_packed_safetensors_loader",
                                None,
                            )
                            if fp8_loader is not None:
                                shards = loaded_packed_shards.setdefault(
                                    param_name,
                                    set(),
                                )
                                if shard_id in shards:
                                    raise RuntimeError(
                                        f"packed parameter {param_name} contains "
                                        f"duplicate shard {shard_id!r}"
                                    )
                                shards.add(shard_id)
                                fp8_loader(
                                    param,
                                    source_slice,
                                    f.get_slice(scale_name),
                                    shard_id,
                                    fp8_block_size,
                                )
                                loaded_parameters.add(param_name)
                                continue
                        else:
                            param = model.get_parameter(weight_name)
                            fp8_loader = getattr(
                                param,
                                "fp8_safetensors_loader",
                                None,
                            )
                            if fp8_loader is not None:
                                fp8_loader(
                                    param,
                                    source_slice,
                                    f.get_slice(scale_name),
                                    fp8_block_size,
                                )
                                loaded_parameters.add(weight_name)
                                continue
                        loaded_tensor = dequantize_fp8_block_weight(
                            f.get_tensor(source_weight_name),
                            f.get_tensor(scale_name),
                            fp8_block_size,
                            output_dtype=torch.get_default_dtype(),
                        )
                if packed_parameter is not None:
                    param_name, shard_id = packed_parameter
                    shards = loaded_packed_shards.setdefault(param_name, set())
                    if shard_id in shards:
                        raise RuntimeError(
                            f"packed parameter {param_name} contains duplicate "
                            f"shard {shard_id!r}"
                        )
                    shards.add(shard_id)
                    param = model.get_parameter(param_name)
                    packed_safetensors_loader = getattr(
                        param,
                        "packed_safetensors_loader",
                        None,
                    )
                    if packed_safetensors_loader is not None and loaded_tensor is None:
                        packed_safetensors_loader(
                            param,
                            f.get_slice(source_weight_name),
                            shard_id,
                        )
                        loaded_parameters.add(param_name)
                        continue
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(
                        param,
                        loaded_tensor
                        if loaded_tensor is not None
                        else f.get_tensor(source_weight_name),
                        shard_id,
                    )
                    loaded_parameters.add(param_name)
                else:
                    param = model.get_parameter(weight_name)
                    safetensors_loader = getattr(
                        param,
                        "safetensors_loader",
                        None,
                    )
                    if safetensors_loader is not None and loaded_tensor is None:
                        safetensors_loader(
                            param,
                            f.get_slice(source_weight_name),
                        )
                        loaded_parameters.add(weight_name)
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(
                        param,
                        loaded_tensor
                        if loaded_tensor is not None
                        else f.get_tensor(source_weight_name),
                    )
                    loaded_parameters.add(weight_name)
    if getattr(model, "strict_weight_loading", False):
        expected_parameters = {
            name for name, _ in model.named_parameters(remove_duplicate=False)
        }
        missing = sorted(expected_parameters - loaded_parameters)
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise RuntimeError(f"checkpoint is missing model parameters: {preview}{suffix}")
        packed_shards_by_target: dict[str, set[object]] = {}
        for target_module, shard_id in packed_modules_mapping.values():
            packed_shards_by_target.setdefault(target_module, set()).add(shard_id)
        incomplete = []
        for param_name, loaded_shards in loaded_packed_shards.items():
            parameter = model.get_parameter(param_name)
            explicit_required = getattr(parameter, "required_checkpoint_shards", None)
            if explicit_required is not None:
                required_shards = set(explicit_required)
                if loaded_shards != required_shards:
                    incomplete.append(
                        f"{param_name}: loaded {sorted(map(str, loaded_shards))}, "
                        f"expected {sorted(map(str, required_shards))}"
                    )
                continue
            target_modules = set(param_name.split(".")).intersection(
                packed_shards_by_target
            )
            required_shards = set().union(
                *(packed_shards_by_target[name] for name in target_modules)
            )
            if loaded_shards != required_shards:
                incomplete.append(
                    f"{param_name}: loaded {sorted(map(str, loaded_shards))}, "
                    f"expected {sorted(map(str, required_shards))}"
                )
        if incomplete:
            raise RuntimeError(
                "checkpoint has incomplete packed parameters: "
                + "; ".join(incomplete)
            )
