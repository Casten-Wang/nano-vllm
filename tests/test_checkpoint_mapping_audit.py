from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "audit_checkpoint_mapping",
    ROOT / "scripts" / "audit_checkpoint_mapping.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyMappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text = torch.nn.Linear(2, 2, bias=False)

    @staticmethod
    def map_weight_name(name):
        if name.startswith("visual."):
            return None
        return name.removeprefix("external.")


def write_checkpoint(path, tensors):
    save_file(tensors, str(path / "model.safetensors"))


def test_audit_maps_text_and_skips_non_text_weights(tmp_path):
    write_checkpoint(
        tmp_path,
        {
            "external.text.weight": torch.ones(2, 2),
            "visual.weight": torch.ones(1),
        },
    )

    result = MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)

    assert result["valid"]
    assert result["source_tensor_count"] == 2
    assert result["mapped_parameter_count"] == 1
    assert result["skipped_tensor_count"] == 1


def test_audit_reports_missing_and_unexpected_weights(tmp_path):
    write_checkpoint(tmp_path, {"external.unknown.weight": torch.ones(2, 2)})

    result = MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)

    assert not result["valid"]
    assert result["missing_parameters"] == ["text.weight"]
    assert result["unexpected_weights"] == [
        {"source": "external.unknown.weight", "mapped": "unknown.weight"}
    ]


def test_audit_rejects_directory_without_safetensors(tmp_path):
    with pytest.raises(ValueError, match="no safetensors"):
        MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)


@pytest.mark.parametrize("value", ["", "0", "4,-1", "four"])
def test_invalid_tp_sizes_are_rejected(value):
    with pytest.raises((ValueError, MODULE.argparse.ArgumentTypeError)):
        MODULE.parse_tp_sizes(value)
