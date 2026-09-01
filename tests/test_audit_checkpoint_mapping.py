from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "audit_checkpoint_mapping",
    ROOT / "scripts" / "audit_checkpoint_mapping.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parameter_storage_bytes_uses_parameter_dtype_and_shape():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=True, dtype=torch.bfloat16),
        torch.nn.Linear(4, 2, bias=False, dtype=torch.float32),
    )

    expected = (3 * 4 + 4) * 2 + (4 * 2) * 4

    assert MODULE.parameter_storage_bytes(model) == expected
