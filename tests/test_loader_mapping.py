from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from torch import nn


class FakeSafeFile:
    requested = []
    sliced = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return ("external.text.weight", "external.visual.weight")

    def get_tensor(self, name):
        self.requested.append(name)
        return torch.tensor([3.0, 4.0])

    def get_slice(self, name):
        self.sliced.append(name)
        return torch.tensor([5.0, 6.0])


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


class StrictMappedModel(MappedModel):
    strict_weight_loading = True

    def __init__(self):
        super().__init__()
        self.missing = nn.Parameter(torch.zeros(1))


class SlicedMappedModel(MappedModel):
    def __init__(self):
        super().__init__()
        self.model.weight.safetensors_loader = (
            lambda param, source: param.data.copy_(source)
        )


class PackedSlicedMappedModel(nn.Module):
    packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0)}

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.gate_up_proj = nn.Linear(2, 1, bias=False)
        self.model.gate_up_proj.weight.data.zero_()
        self.model.gate_up_proj.weight.packed_safetensors_loader = (
            lambda param, source, shard_id: param.data.copy_(source + shard_id)
        )

    @staticmethod
    def map_weight_name(name):
        if name == "external.visual.weight":
            return None
        return "model.gate_proj.weight"


class StrictPackedModel(nn.Module):
    strict_weight_loading = True
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(2, 2, bias=False)
        self.loaded_shards = []
        self.gate_up_proj.weight.weight_loader = (
            lambda _param, _source, shard_id: self.loaded_shards.append(shard_id)
        )


class StackedCheckpointModel(nn.Module):
    strict_weight_loading = True

    def __init__(self):
        super().__init__()
        self.stacked = nn.Parameter(torch.zeros(2, 2))
        self.stacked.required_checkpoint_shards = frozenset({0, 1})
        self.stacked.packed_safetensors_loader = self.load_shard

    @staticmethod
    def resolve_checkpoint_parameter(name):
        if name.startswith("expert."):
            return "stacked", int(name.split(".")[1])
        return None

    @staticmethod
    def load_shard(param, source, shard_id):
        param.data[shard_id].copy_(source)


def test_loader_maps_text_names_before_loading_and_skips_visual_tensors(tmp_path):
    (tmp_path / "model.safetensors").touch()
    FakeSafeFile.requested.clear()
    model = MappedModel()

    loader.load_model(model, str(tmp_path))

    torch.testing.assert_close(model.model.weight, torch.tensor([3.0, 4.0]))
    assert FakeSafeFile.requested == ["external.text.weight"]


def test_strict_loader_reports_parameters_absent_from_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").touch()

    try:
        loader.load_model(StrictMappedModel(), str(tmp_path))
    except RuntimeError as error:
        assert "checkpoint is missing model parameters: missing" in str(error)
    else:
        raise AssertionError("strict loading accepted a missing parameter")


def test_parameter_specific_loader_uses_lazy_safetensors_slice(tmp_path):
    (tmp_path / "model.safetensors").touch()
    FakeSafeFile.requested.clear()
    FakeSafeFile.sliced.clear()
    model = SlicedMappedModel()

    loader.load_model(model, str(tmp_path))

    torch.testing.assert_close(model.model.weight, torch.tensor([5.0, 6.0]))
    assert FakeSafeFile.sliced == ["external.text.weight"]
    assert FakeSafeFile.requested == []


def test_packed_parameter_loader_uses_lazy_safetensors_slice(tmp_path):
    (tmp_path / "model.safetensors").touch()
    FakeSafeFile.requested.clear()
    FakeSafeFile.sliced.clear()
    model = PackedSlicedMappedModel()

    loader.load_model(model, str(tmp_path))

    torch.testing.assert_close(model.model.gate_up_proj.weight, torch.tensor([[5.0, 6.0]]))
    assert FakeSafeFile.sliced == ["external.text.weight"]
    assert FakeSafeFile.requested == []


def test_packed_mapping_matches_complete_module_segment_only():
    mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    assert loader.resolve_packed_parameter(
        "model.layers.0.mlp.gate_proj.weight",
        mapping,
    ) == ("model.layers.0.mlp.gate_up_proj.weight", 0)
    assert loader.resolve_packed_parameter(
        "model.layers.0.mlp.up_proj.weight",
        mapping,
    ) == ("model.layers.0.mlp.gate_up_proj.weight", 1)
    assert loader.resolve_packed_parameter(
        "gate_proj.weight",
        mapping,
    ) == ("gate_up_proj.weight", 0)
    assert loader.resolve_packed_parameter(
        "model.layers.0.mlp.experts.gate_up_proj",
        mapping,
    ) is None


def test_strict_loader_rejects_incomplete_packed_parameter(tmp_path):
    (tmp_path / "model.safetensors").touch()
    safe_file = FakeSafeFile()
    safe_file.keys = lambda: ("gate_proj.weight",)

    with patch.object(loader, "safe_open", return_value=safe_file):
        try:
            loader.load_model(StrictPackedModel(), str(tmp_path))
        except RuntimeError as error:
            assert "incomplete packed parameters" in str(error)
            assert "expected ['0', '1']" in str(error)
        else:
            raise AssertionError("strict loading accepted a missing packed shard")


def test_strict_loader_accepts_every_packed_shard(tmp_path):
    (tmp_path / "model.safetensors").touch()
    safe_file = FakeSafeFile()
    safe_file.keys = lambda: ("gate_proj.weight", "up_proj.weight")
    model = StrictPackedModel()

    with patch.object(loader, "safe_open", return_value=safe_file):
        loader.load_model(model, str(tmp_path))

    assert model.loaded_shards == [0, 1]


def test_loader_supports_checkpoint_shards_stacked_by_model_hook(tmp_path):
    (tmp_path / "model.safetensors").touch()
    safe_file = FakeSafeFile()
    safe_file.keys = lambda: ("expert.0", "expert.1")
    model = StackedCheckpointModel()

    with patch.object(loader, "safe_open", return_value=safe_file):
        loader.load_model(model, str(tmp_path))

    torch.testing.assert_close(
        model.stacked,
        torch.tensor([[5.0, 6.0], [5.0, 6.0]]),
    )


def test_loader_rejects_missing_model_defined_checkpoint_shard(tmp_path):
    (tmp_path / "model.safetensors").touch()
    safe_file = FakeSafeFile()
    safe_file.keys = lambda: ("expert.0",)

    with patch.object(loader, "safe_open", return_value=safe_file):
        try:
            loader.load_model(StackedCheckpointModel(), str(tmp_path))
        except RuntimeError as error:
            assert "expected ['0', '1']" in str(error)
        else:
            raise AssertionError("strict loading accepted a missing expert shard")
