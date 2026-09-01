import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def resolve_packed_parameter(
    weight_name: str,
    packed_modules_mapping: dict,
) -> tuple[str, object] | None:
    """Map an exact module path segment, never a substring of a packed name."""

    for source_module, (target_module, shard_id) in packed_modules_mapping.items():
        marker = f".{source_module}."
        if marker in weight_name:
            target = weight_name.replace(
                marker,
                f".{target_module}.",
                1,
            )
            return target, shard_id
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", lambda name: name)
    loaded_parameters: set[str] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for source_weight_name in f.keys():
                weight_name = map_weight_name(source_weight_name)
                if weight_name is None:
                    continue
                packed_parameter = resolve_packed_parameter(
                    weight_name,
                    packed_modules_mapping,
                )
                if packed_parameter is not None:
                    param_name, shard_id = packed_parameter
                    param = model.get_parameter(param_name)
                    packed_safetensors_loader = getattr(
                        param,
                        "packed_safetensors_loader",
                        None,
                    )
                    if packed_safetensors_loader is not None:
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
                        f.get_tensor(source_weight_name),
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
                    if safetensors_loader is not None:
                        safetensors_loader(
                            param,
                            f.get_slice(source_weight_name),
                        )
                        loaded_parameters.add(weight_name)
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(source_weight_name))
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
