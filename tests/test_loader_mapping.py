from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import torch
from torch import nn


class FakeSafeFile:
    requested = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return ("external.text.weight", "external.visual.weight")

    def get_tensor(self, name):
        self.requested.append(name)
        return torch.tensor([3.0, 4.0])


safe_module = types.ModuleType("safetensors")
safe_module.safe_open = lambda *args, **kwargs: FakeSafeFile()
original = sys.modules.get("safetensors")
sys.modules["safetensors"] = safe_module
try:
    path = Path(__file__).parents[1] / "nanovllm" / "utils" / "loader.py"
    spec = spec_from_file_location("loader_under_test", path)
    assert spec is not None and spec.loader is not None
    loader = module_from_spec(spec)
    spec.loader.exec_module(loader)
finally:
    if original is None:
        sys.modules.pop("safetensors", None)
    else:
        sys.modules["safetensors"] = original


class MappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.weight = nn.Parameter(torch.zeros(2))

    @staticmethod
    def map_weight_name(name):
        if name == "external.visual.weight":
            return None
        return "model.weight"


def test_loader_maps_text_names_before_loading_and_skips_visual_tensors(tmp_path):
    (tmp_path / "model.safetensors").touch()
    FakeSafeFile.requested.clear()
    model = MappedModel()

    loader.load_model(model, str(tmp_path))

    torch.testing.assert_close(model.model.weight, torch.tensor([3.0, 4.0]))
    assert FakeSafeFile.requested == ["external.text.weight"]
