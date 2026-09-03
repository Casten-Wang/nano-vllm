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


class TinySlicedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sharded = torch.nn.Parameter(torch.empty(1, 2, device="meta"))

        def load_slice(parameter, source):
            if tuple(source.get_shape()) != (2, 2):
                raise ValueError("expected full checkpoint shape (2, 2)")
            parameter.data.copy_(source[:1, :])

        self.sharded.safetensors_loader = load_slice


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
    assert result["checkpoint_loading"] == {
        "mapped_checkpoint_bytes": 16,
        "estimated_local_payload_bytes": 16,
        "estimated_avoided_payload_bytes": 0,
        "estimated_local_payload_fraction": 1.0,
        "lazy_tensor_count": 0,
        "full_tensor_count": 1,
    }


def test_audit_reports_missing_and_unexpected_weights(tmp_path):
    write_checkpoint(tmp_path, {"external.unknown.weight": torch.ones(2, 2)})

    result = MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)

    assert not result["valid"]
    assert result["missing_parameters"] == ["text.weight"]
    assert result["unexpected_weights"] == [
        {"source": "external.unknown.weight", "mapped": "unknown.weight"}
    ]


def test_audit_reports_shape_mismatch_without_reading_tensor_payload(tmp_path):
    write_checkpoint(
        tmp_path,
        {"external.text.weight": torch.ones(3, 2)},
    )

    result = MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)

    assert not result["valid"]
    assert result["mapped_parameter_count"] == 0
    assert result["shape_errors"][0]["mapped"] == "text.weight"
    assert "does not match parameter shape" in result["shape_errors"][0]["error"]


def test_audit_exercises_shape_only_safetensors_loader(tmp_path):
    write_checkpoint(tmp_path, {"sharded": torch.ones(2, 2)})

    result = MODULE.audit_checkpoint_mapping(TinySlicedModel(), tmp_path)

    assert result["valid"]
    assert result["mapped_parameter_count"] == 1
    assert result["shape_errors"] == []
    assert result["checkpoint_loading"] == {
        "mapped_checkpoint_bytes": 16,
        "estimated_local_payload_bytes": 8,
        "estimated_avoided_payload_bytes": 8,
        "estimated_local_payload_fraction": 0.5,
        "lazy_tensor_count": 1,
        "full_tensor_count": 0,
    }


def test_audit_rejects_directory_without_safetensors(tmp_path):
    with pytest.raises(ValueError, match="no safetensors"):
        MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)


def test_audit_uses_index_names_when_shards_are_not_local(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps(
            {
                "weight_map": {
                    "external.text.weight": "model-00001.safetensors",
                    "visual.weight": "model-00002.safetensors",
                }
            }
        )
    )

    result = MODULE.audit_checkpoint_mapping(TinyMappedModel(), tmp_path)

    assert result["valid"]
    assert result["validation_level"] == "names_only"
    assert not result["shape_validation_complete"]
    assert result["source_tensor_count"] == 2
    assert result["mapped_parameter_count"] == 1
    assert result["checkpoint_loading"] is None


def test_resident_fp8_runtime_storage_counts_shared_workspace_once():
    class ResidentExpert(torch.nn.Module):
        def __init__(self, pool):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.empty(2, 6, 4, device="meta", dtype=torch.float8_e4m3fn)
            )
            self.down_proj = torch.nn.Parameter(
                torch.empty(2, 4, 3, device="meta", dtype=torch.float8_e4m3fn)
            )
            self.resident_weight_buffer_pool = pool

        def resident_fp8_storage_stats(self):
            return {"weight_bytes": 72, "scale_bytes": 12, "total_bytes": 84}

    model = torch.nn.Module()
    pool = object()
    model.first = ResidentExpert(pool)
    model.second = ResidentExpert(pool)

    result = MODULE.resident_fp8_runtime_storage(model, model_dtype_bytes=2)

    assert result == {
        "layer_count": 2,
        "weight_bytes": 144,
        "scale_bytes": 24,
        "total_bytes": 168,
        "dequant_workspace_pool_count": 1,
        "dequant_workspace_bytes": 48,
        "total_runtime_bytes": 216,
    }


@pytest.mark.parametrize("value", ["", "0", "4,-1", "four"])
def test_invalid_tp_sizes_are_rejected(value):
    with pytest.raises((ValueError, MODULE.argparse.ArgumentTypeError)):
        MODULE.parse_tp_sizes(value)
