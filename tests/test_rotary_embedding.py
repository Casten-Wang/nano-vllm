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


@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inference_rotary_can_reuse_input_storage(rotary_dim, dtype):
    torch.manual_seed(5)
    x = torch.randn(3, 2, 8, dtype=dtype)
    original = x.clone()
    cos = torch.randn(3, 1, rotary_dim // 2)
    sin = torch.randn(3, 1, rotary_dim // 2)
    expected = rotary.apply_rotary_emb(original, cos, sin)
    storage = x.data_ptr()

    with torch.inference_mode():
        actual = rotary.apply_rotary_emb(
            x,
            cos,
            sin,
            inplace_output=True,
        )

    assert actual.data_ptr() == storage
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., rotary_dim:], original[..., rotary_dim:])


def test_inplace_rotary_request_preserves_autograd_input():
    x = torch.randn(3, 2, 8, requires_grad=True)
    original = x.detach().clone()
    cos = torch.randn(3, 1, 2)
    sin = torch.randn(3, 1, 2)

    output = rotary.apply_rotary_emb(
        x,
        cos,
        sin,
        inplace_output=True,
    )
    output.square().mean().backward()

    assert output.data_ptr() != x.data_ptr()
    assert torch.equal(x, original)
    assert x.grad is not None


@pytest.mark.parametrize("rotary_dim", [0, 5, 10])
def test_invalid_rotary_dimension_is_rejected(rotary_dim):
    with pytest.raises(ValueError, match="rotary_dim"):
        rotary.RotaryEmbedding(8, rotary_dim, 32, 10_000)
