import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


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
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(
                            param,
                            f.get_tensor(source_weight_name),
                            shard_id,
                        )
                        loaded_parameters.add(param_name)
                        break
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
