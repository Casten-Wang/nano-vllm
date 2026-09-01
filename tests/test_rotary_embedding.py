from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "nanovllm" / "layers" / "rotary_embedding.py"
)
SPEC = spec_from_file_location("rotary_embedding_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rotary = module_from_spec(SPEC)
sys.modules[SPEC.name] = rotary
SPEC.loader.exec_module(rotary)


def test_partial_rotary_embedding_preserves_unrotated_dimensions():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    cos = torch.zeros(1, 1, 2)
    sin = torch.ones(1, 1, 2)

    actual = rotary.apply_rotary_emb(x, cos, sin)

    torch.testing.assert_close(
        actual,
        torch.tensor([[[-3.0, -4.0, 1.0, 2.0, 5.0, 6.0]]]),
    )


def test_full_rotary_embedding_behavior_is_unchanged():
    x = torch.randn(3, 2, 8)
    cos = torch.randn(3, 1, 4)
    sin = torch.randn(3, 1, 4)
    first, second = x.float().chunk(2, dim=-1)
    expected = torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )

    torch.testing.assert_close(rotary.apply_rotary_emb(x, cos, sin), expected)


@pytest.mark.parametrize("rotary_dim", [0, 5, 10])
def test_invalid_rotary_dimension_is_rejected(rotary_dim):
    with pytest.raises(ValueError, match="rotary_dim"):
        rotary.RotaryEmbedding(8, rotary_dim, 32, 10_000)
