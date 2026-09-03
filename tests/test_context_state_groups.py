import pytest
import torch

from nanovllm.utils.context import (
    build_state_prefill_groups,
    build_state_reset_slots,
    get_context,
    reset_context,
    set_context,
)


def test_state_prefill_groups_reuse_slots_for_equal_length_sequences():
    slots = torch.tensor([9, 4, 7, 2], dtype=torch.int32)
    ranges = ((2, 5), (5, 6), (6, 9))

    groups = build_state_prefill_groups(ranges, slots, decode_token_count=1)

    assert len(groups) == 2
    length_three, grouped_ranges, grouped_slots, state_span = groups[0]
    assert length_three == 3
    assert grouped_ranges == ((2, 5, 1), (6, 9, 3))
    assert torch.equal(grouped_slots, torch.tensor([4, 2]))
    assert state_span is None
    assert groups[1][0] == 1
    assert torch.equal(groups[1][2], torch.tensor([7]))


def test_contiguous_state_prefill_group_reuses_slot_storage():
    slots = torch.tensor([9, 4, 7], dtype=torch.int64)
    ranges = ((1, 4), (4, 7))

    groups = build_state_prefill_groups(
        ranges,
        slots,
        decode_token_count=1,
    )

    assert len(groups) == 1
    assert torch.equal(groups[0][2], torch.tensor([4, 7]))
    assert groups[0][2].data_ptr() == slots[1:].data_ptr()


def test_state_prefill_groups_preserve_host_derived_contiguous_spans():
    groups = build_state_prefill_groups(
        ((0, 3), (3, 6)),
        torch.tensor([4, 5], dtype=torch.int64),
        decode_token_count=0,
        state_prefill_spans=((4, 2),),
    )

    assert groups[0][3] == (4, 2)


def test_state_prefill_groups_reject_misaligned_spans():
    with pytest.raises(ValueError, match="spans must match length groups"):
        build_state_prefill_groups(
            ((0, 3), (3, 4)),
            torch.tensor([4, 5], dtype=torch.int64),
            decode_token_count=0,
            state_prefill_spans=((4, 1),),
        )


def test_state_prefill_groups_reject_wrong_span_size():
    with pytest.raises(ValueError, match="span size must match its group"):
        build_state_prefill_groups(
            ((0, 3), (3, 6)),
            torch.tensor([4, 5], dtype=torch.int64),
            decode_token_count=0,
            state_prefill_spans=((4, 1),),
        )


def test_interleaved_state_prefill_group_keeps_indexed_fallback():
    slots = torch.tensor([9, 4, 7, 2], dtype=torch.int64)
    ranges = ((1, 4), (4, 5), (5, 8))

    groups = build_state_prefill_groups(
        ranges,
        slots,
        decode_token_count=1,
    )

    assert torch.equal(groups[0][2], torch.tensor([4, 2]))
    assert groups[0][2].untyped_storage().data_ptr() != (
        slots.untyped_storage().data_ptr()
    )


def test_set_context_precomputes_recurrent_prefill_groups():
    try:
        set_context(
            False,
            is_mixed=True,
            decode_token_count=1,
            state_slots=torch.tensor([3, 8], dtype=torch.int32),
            state_reset_mask=torch.tensor([False, True]),
            state_token_ranges=((1, 4),),
            decode_state_span=(3, 1),
        )

        assert get_context().state_prefill_groups[0][0] == 3
        assert torch.equal(
            get_context().state_prefill_groups[0][2],
            torch.tensor([8]),
        )
        assert torch.equal(
            get_context().state_reset_slots,
            torch.tensor([8], dtype=torch.int32),
        )
        assert get_context().decode_state_span == (3, 1)
    finally:
        reset_context()


def test_state_prefill_groups_skip_kv_only_context_without_slots():
    assert build_state_prefill_groups(((0, 1),), None, decode_token_count=0) == ()


def test_state_reset_slots_preserves_empty_selection():
    slots = torch.tensor([3, 8], dtype=torch.int64)
    reset_slots = build_state_reset_slots(
        slots,
        torch.tensor([False, False]),
    )

    assert reset_slots is not None
    assert reset_slots.dtype == torch.int64
    assert reset_slots.numel() == 0


def test_set_context_prefers_precomputed_reset_slots():
    explicit = torch.tensor([7], dtype=torch.int64)
    try:
        set_context(
            False,
            state_slots=torch.tensor([3, 7], dtype=torch.int64),
            state_reset_mask=torch.tensor([True, False]),
            state_reset_slots=explicit,
            state_reset_span=(7, 1),
        )

        assert get_context().state_reset_slots is explicit
        assert get_context().state_reset_span == (7, 1)
    finally:
        reset_context()
