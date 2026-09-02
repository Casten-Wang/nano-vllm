from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "audit_remote_checkpoint_headers",
    ROOT / "scripts" / "audit_remote_checkpoint_headers.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_header_only_audit_validates_shape_without_payload():
    model = torch.nn.Linear(2, 3, bias=False, device="meta")
    headers = {"weight": {"dtype": "F32", "shape": [3, 2]}}

    result = MODULE.audit_model_headers(model, headers)

    assert result["valid"]
    assert result["mapped_parameter_count"] == 1
    assert result["shape_errors"] == []
    assert result["checkpoint_loading"] == {
        "mapped_checkpoint_bytes": 24,
        "estimated_local_payload_bytes": 24,
        "estimated_avoided_payload_bytes": 0,
        "estimated_local_payload_fraction": 1.0,
        "lazy_tensor_count": 0,
        "full_tensor_count": 1,
    }


def test_header_only_audit_counts_lazy_payload_slice():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(1, 2, device="meta"))

    def load_slice(parameter, source):
        parameter.data.copy_(source[:1, :])

    model.weight.safetensors_loader = load_slice
    headers = {"weight": {"dtype": "F32", "shape": [2, 2]}}

    result = MODULE.audit_model_headers(model, headers)

    assert result["valid"]
    assert result["checkpoint_loading"] == {
        "mapped_checkpoint_bytes": 16,
        "estimated_local_payload_bytes": 8,
        "estimated_avoided_payload_bytes": 8,
        "estimated_local_payload_fraction": 0.5,
        "lazy_tensor_count": 1,
        "full_tensor_count": 0,
    }


def test_header_only_audit_reports_shape_error():
    model = torch.nn.Linear(2, 3, bias=False, device="meta")
    headers = {"weight": {"dtype": "F32", "shape": [4, 2]}}

    result = MODULE.audit_model_headers(model, headers)

    assert not result["valid"]
    assert result["shape_errors"]


def test_header_only_audit_classifies_text_only_skips():
    class TextOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

        @staticmethod
        def map_weight_name(name):
            if name.startswith("model.visual.") or name.startswith("mtp."):
                return None
            return name

    headers = {
        "weight": {"dtype": "F32", "shape": [1]},
        "model.visual.patch.weight": {"dtype": "F32", "shape": [1]},
        "mtp.head.weight": {"dtype": "F32", "shape": [1]},
    }

    result = MODULE.audit_model_headers(TextOnlyModel(), headers)

    assert result["valid"]
    assert result["skipped_by_prefix"] == {"model.visual.": 1, "mtp.": 1}
    assert result["unclassified_skipped_weights"] == []


def test_header_only_audit_rejects_unclassified_skip():
    class UnexpectedSkipModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

        @staticmethod
        def map_weight_name(name):
            return None if name.startswith("audio.") else name

    headers = {
        "weight": {"dtype": "F32", "shape": [1]},
        "audio.weight": {"dtype": "F32", "shape": [1]},
    }

    result = MODULE.audit_model_headers(UnexpectedSkipModel(), headers)

    assert not result["valid"]
    assert result["unclassified_skipped_weights"] == ["audio.weight"]


def test_fetch_header_rejects_oversized_metadata(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "fetch_bytes",
        lambda _url, _range=None: (1024).to_bytes(8, "little"),
    )

    with pytest.raises(ValueError, match="header length"):
        MODULE.fetch_safetensors_header("https://example.invalid/model", 128)


def test_header_document_excludes_metadata(monkeypatch):
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
        }
    ).encode()
    responses = [len(header).to_bytes(8, "little"), header]
    monkeypatch.setattr(MODULE, "fetch_bytes", lambda *_args: responses.pop(0))

    tensors, downloaded = MODULE.fetch_safetensors_header("url", 1024)

    assert list(tensors) == ["weight"]
    assert downloaded == len(header) + 8


def test_official_shard_metadata_requires_lfs_sha256_and_size():
    digest = "a" * 64
    metadata = {
        "siblings": [
            {
                "rfilename": "model-1.safetensors",
                "lfs": {"sha256": digest, "size": 123},
            }
        ]
    }

    result = MODULE.official_shard_metadata(
        metadata,
        ["model-1.safetensors"],
    )

    assert result == [
        {"name": "model-1.safetensors", "size_bytes": 123, "sha256": digest}
    ]


def test_official_shard_metadata_rejects_missing_identity():
    with pytest.raises(ValueError, match="official LFS identity"):
        MODULE.official_shard_metadata(
            {"siblings": [{"rfilename": "model-1.safetensors"}]},
            ["model-1.safetensors"],
        )


def test_fp8_group_audit_reconstructs_logical_weight():
    spec = MODULE.QuantizationSpec(
        format="fp8_block",
        weight_bits=8,
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
    )
    headers = {
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [512, 2048],
        },
        "model.layers.0.mlp.gate_proj.weight_scale_inv": {
            "dtype": "BF16",
            "shape": [4, 16],
        },
        "model.norm.weight": {"dtype": "BF16", "shape": [2048]},
    }

    logical, result = MODULE.audit_quantized_tensor_groups(headers, spec)

    assert result["valid"]
    assert result["quantized_group_count"] == 1
    assert logical["model.layers.0.mlp.gate_proj.weight"] == {
        "dtype": "BF16",
        "shape": [512, 2048],
    }
    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in logical
    assert "model.norm.weight" in logical


def test_fp8_group_audit_rejects_missing_or_orphan_scale():
    spec = MODULE.QuantizationSpec(
        format="fp8_block",
        weight_bits=8,
        weight_block_size=(128, 128),
    )
    headers = {
        "model.layers.0.mlp.up_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [512, 2048],
        },
        "model.layers.1.mlp.up_proj.weight_scale_inv": {
            "dtype": "BF16",
            "shape": [4, 16],
        },
    }

    _, result = MODULE.audit_quantized_tensor_groups(headers, spec)

    assert not result["valid"]
    assert any("missing" in error for error in result["errors"])
    assert any("orphan" in error for error in result["errors"])


def test_gptq_group_audit_reconstructs_logical_weight():
    spec = MODULE.QuantizationSpec(
        format="gptq_int4",
        weight_bits=4,
        group_size=128,
        symmetric=True,
        desc_act=False,
    )
    prefix = "model.layers.0.mlp.experts.0.down_proj"
    headers = {
        f"{prefix}.qweight": {"dtype": "I32", "shape": [64, 2048]},
        f"{prefix}.qzeros": {"dtype": "I32", "shape": [4, 256]},
        f"{prefix}.scales": {"dtype": "F16", "shape": [4, 2048]},
        f"{prefix}.g_idx": {"dtype": "I32", "shape": [512]},
    }

    logical, result = MODULE.audit_quantized_tensor_groups(headers, spec)

    assert result["valid"]
    assert logical[f"{prefix}.weight"] == {
        "dtype": "BF16",
        "shape": [2048, 512],
    }


def test_gptq_group_audit_rejects_inconsistent_packed_shape():
    spec = MODULE.QuantizationSpec(
        format="gptq_int4",
        weight_bits=4,
        group_size=128,
    )
    prefix = "model.layers.0.mlp.experts.0.gate_proj"
    headers = {
        f"{prefix}.qweight": {"dtype": "I32", "shape": [255, 512]},
        f"{prefix}.qzeros": {"dtype": "I32", "shape": [16, 64]},
        f"{prefix}.scales": {"dtype": "F16", "shape": [16, 512]},
        f"{prefix}.g_idx": {"dtype": "I32", "shape": [2048]},
    }

    _, result = MODULE.audit_quantized_tensor_groups(headers, spec)

    assert not result["valid"]
    assert any("qweight" in error and "shape" in error for error in result["errors"])


def test_tp_layout_reports_partial_fp8_blocks_without_claiming_invalid_shape():
    spec = MODULE.QuantizationSpec(
        format="fp8_block",
        weight_bits=8,
        weight_block_size=(128, 128),
    )
    logical = {
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "BF16",
            "shape": [512, 2048],
        },
        "model.layers.0.mlp.down_proj.weight": {
            "dtype": "BF16",
            "shape": [2048, 512],
        },
    }

    tp4 = MODULE.audit_quantized_tp_layout(logical, spec, 4)
    tp8 = MODULE.audit_quantized_tp_layout(logical, spec, 8)

    assert tp4["valid"]
    assert not tp4["requires_partial_unit_loader"]
    assert tp8["valid"]
    assert tp8["requires_partial_unit_loader"]
    assert len(tp8["partial_quantization_units"]) == 2


def test_per_expert_weights_are_coalesced_to_runtime_layout():
    headers = {}
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        headers[f"{prefix}.gate_proj.weight"] = {
            "dtype": "BF16",
            "shape": [4, 8],
        }
        headers[f"{prefix}.up_proj.weight"] = {
            "dtype": "BF16",
            "shape": [4, 8],
        }
        headers[f"{prefix}.down_proj.weight"] = {
            "dtype": "BF16",
            "shape": [8, 4],
        }

    logical, result = MODULE.coalesce_expert_logical_headers(headers)

    assert result["valid"]
    assert result["stacked_parameter_count"] == 2
    assert result["serialized_expert_weight_count"] == 6
    assert logical["model.layers.0.mlp.experts.gate_up_proj"] == {
        "dtype": "BF16",
        "shape": [2, 8, 8],
    }
    assert logical["model.layers.0.mlp.experts.down_proj"] == {
        "dtype": "BF16",
        "shape": [2, 8, 4],
    }
