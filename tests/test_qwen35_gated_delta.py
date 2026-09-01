from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


MODULE_PATH = (
    Path(__file__).parents[1]
    / "nanovllm"
    / "models"
    / "qwen35_gated_delta.py"
)
SPEC = spec_from_file_location("qwen35_gated_delta_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qwen35_gated_delta = module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen35_gated_delta
SPEC.loader.exec_module(qwen35_gated_delta)

Qwen35RecurrentStatePool = qwen35_gated_delta.Qwen35RecurrentStatePool
causal_conv1d_scan = qwen35_gated_delta.causal_conv1d_scan
recurrent_gated_delta_rule = qwen35_gated_delta.recurrent_gated_delta_rule


def inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 5, 3, 4, generator=generator)
    k = torch.randn(2, 5, 3, 4, generator=generator)
    v = torch.randn(2, 5, 3, 6, generator=generator)
    decay = -torch.rand(2, 5, 3, generator=generator)
    beta = torch.rand(2, 5, 3, generator=generator)
    return q, k, v, decay, beta


def test_recurrent_rule_chunked_prefill_matches_one_shot():
    q, k, v, decay, beta = inputs()
    expected, expected_state = recurrent_gated_delta_rule(q, k, v, decay, beta)

    first, state = recurrent_gated_delta_rule(
        q[:, :3], k[:, :3], v[:, :3], decay[:, :3], beta[:, :3]
    )
    second, state = recurrent_gated_delta_rule(
        q[:, 3:], k[:, 3:], v[:, 3:], decay[:, 3:], beta[:, 3:], state
    )

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


def test_single_token_decode_matches_last_prefill_token():
    q, k, v, decay, beta = inputs(1)
    prefix, state = recurrent_gated_delta_rule(
        q[:, :-1], k[:, :-1], v[:, :-1], decay[:, :-1], beta[:, :-1]
    )
    decoded, state = recurrent_gated_delta_rule(
        q[:, -1:], k[:, -1:], v[:, -1:], decay[:, -1:], beta[:, -1:], state
    )
    expected, expected_state = recurrent_gated_delta_rule(q, k, v, decay, beta)

    torch.testing.assert_close(torch.cat((prefix, decoded), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


def test_causal_convolution_chunk_and_decode_are_equivalent():
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(2, 6, 7, generator=generator)
    weight = torch.randn(7, 4, generator=generator)
    initial = torch.zeros(2, 7, 4)
    expected, expected_state = causal_conv1d_scan(x, initial, weight)

    first, state = causal_conv1d_scan(x[:, :5], initial, weight)
    last, state = causal_conv1d_scan(x[:, 5:], state, weight)

    torch.testing.assert_close(torch.cat((first, last), dim=1), expected)
    torch.testing.assert_close(state, expected_state)


def test_causal_convolution_matches_grouped_conv1d_reference():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(2, 6, 7, generator=generator)
    weight = torch.randn(7, 4, generator=generator)
    bias = torch.randn(7, generator=generator)
    initial = torch.zeros(2, 7, 4)

    actual, _ = causal_conv1d_scan(x, initial, weight, bias)
    expected = F.silu(
        F.conv1d(
            x.transpose(1, 2),
            weight.unsqueeze(1),
            bias,
            padding=3,
            groups=7,
        )[:, :, : x.shape[1]]
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_state_pool_isolates_updates_and_resets_reused_slots():
    pool = Qwen35RecurrentStatePool(2, 4, 3, 4, 6, 14, 4, device="cpu")
    slots = torch.tensor([3, 1])
    recurrent, convolution = pool.get(0, slots)
    pool.update(0, slots, recurrent + 1, convolution + 2)

    assert torch.count_nonzero(pool.recurrent[0, 3]) > 0
    assert torch.count_nonzero(pool.recurrent[0, 0]) == 0
    pool.reset(torch.tensor([3]))
    assert torch.count_nonzero(pool.recurrent[:, 3]) == 0
    assert torch.count_nonzero(pool.convolution[:, 3]) == 0
    assert torch.count_nonzero(pool.recurrent[0, 1]) > 0
