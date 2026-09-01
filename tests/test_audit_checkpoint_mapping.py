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


def test_recurrent_storage_bytes_include_every_layer_and_convolution():
    class StatefulLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_v_heads = 4
            self.key_head_dim = 3
            self.value_head_dim = 2
            self.local_conv_dim = 10
            self.conv_kernel_size = 4
            self.in_proj_qkv = torch.nn.Linear(1, 1, dtype=torch.bfloat16)

        def allocate_state_cache(self):
            pass

    model = torch.nn.Sequential(StatefulLayer(), StatefulLayer())

    result = MODULE.recurrent_storage_bytes_per_sequence(model)

    recurrent_elements = 2 * 4 * 3 * 2
    convolution_bytes = 2 * 10 * 4 * 2
    assert result == {
        "float32": recurrent_elements * 4 + convolution_bytes,
        "model": recurrent_elements * 2 + convolution_bytes,
        "convolution": convolution_bytes,
    }
