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
