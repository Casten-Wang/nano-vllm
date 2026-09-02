import pytest
import torch

from nanovllm.engine.decode_input_batch import DecodeInputBatch, TokenInputBatch


def make_batch(capacity=4, max_num_blocks=4):
    return DecodeInputBatch(
        capacity,
        max_num_blocks,
        device="cpu",
        pin_memory=False,
    )


def test_decode_input_batch_preserves_values_and_dtypes():
    batch = make_batch()

    input_ids, positions, slots, lengths = batch.update(
        [11, 23],
        [7, 15],
        [260, 519],
        [8, 16],
    )

    assert torch.equal(input_ids, torch.tensor([11, 23], dtype=torch.int64))
    assert torch.equal(positions, torch.tensor([7, 15], dtype=torch.int64))
    assert torch.equal(slots, torch.tensor([260, 519], dtype=torch.int32))
    assert torch.equal(lengths, torch.tensor([8, 16], dtype=torch.int32))


def test_decode_input_batch_reuses_all_host_and_device_storage():
    batch = make_batch()
    host_pointers = {name: value.data_ptr() for name, value in batch.host.items()}
    first = batch.update([1], [2], [3], [4])
    device_pointers = tuple(value.data_ptr() for value in first)

    second = batch.update([5, 6], [7, 8], [9, 10], [11, 12])

    assert host_pointers == {
        name: value.data_ptr() for name, value in batch.host.items()
    }
    assert device_pointers == tuple(value.data_ptr() for value in second)


def test_decode_block_tables_are_padded_and_reuse_storage():
    batch = make_batch()

    first = batch.update_block_tables([[3, 5, 7], [11]])
    storage = first.data_ptr()
    assert torch.equal(
        first,
        torch.tensor([[3, 5, 7], [11, -1, -1]], dtype=torch.int32),
    )

    second = batch.update_block_tables([[13], [17, 19]])

    assert second.data_ptr() == storage
    assert torch.equal(
        second,
        torch.tensor([[13, -1], [17, 19]], dtype=torch.int32),
    )


def test_decode_state_slots_use_independent_reusable_storage():
    batch = make_batch()

    first_states = batch.update_state_slots([7, 2])
    first_resets = batch.update_reset_slots([2])
    state_storage = first_states.data_ptr()
    reset_storage = first_resets.data_ptr()

    second_states = batch.update_state_slots([3, 4, 5])
    second_resets = batch.update_reset_slots([3, 5])

    assert second_states.data_ptr() == state_storage
    assert second_resets.data_ptr() == reset_storage
    assert torch.equal(second_states, torch.tensor([3, 4, 5]))
    assert torch.equal(second_resets, torch.tensor([3, 5]))


@pytest.mark.parametrize(
    "values",
    [
        ([], [], [], []),
        ([1], [2, 3], [4], [5]),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]),
    ],
)
def test_decode_input_batch_rejects_invalid_sizes(values):
    with pytest.raises(ValueError):
        make_batch(capacity=2).update(*values)


@pytest.mark.parametrize("values", [[], [[]], [[1, 2, 3]]])
def test_decode_input_batch_rejects_invalid_block_tables(values):
    with pytest.raises(ValueError):
        make_batch(capacity=2, max_num_blocks=2).update_block_tables(values)


def make_token_batch(
    token_capacity=6,
    sequence_capacity=3,
    max_num_blocks=4,
):
    return TokenInputBatch(
        token_capacity,
        sequence_capacity,
        max_num_blocks,
        device="cpu",
        pin_memory=False,
    )


def test_token_input_batch_preserves_prefill_metadata():
    batch = make_token_batch()

    ids, positions, slots = batch.update_tokens(
        [11, 12, 13],
        [0, 1, 2],
        [20, 21, 22],
    )
    cu_q, cu_k = batch.update_cu_seqlens([0, 2, 3], [0, 4, 7])

    assert torch.equal(ids, torch.tensor([11, 12, 13]))
    assert torch.equal(positions, torch.tensor([0, 1, 2]))
    assert torch.equal(slots, torch.tensor([20, 21, 22], dtype=torch.int32))
    assert torch.equal(cu_q, torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.equal(cu_k, torch.tensor([0, 4, 7], dtype=torch.int32))


def test_token_input_batch_handles_warmup_and_reuses_storage():
    batch = make_token_batch()
    first = batch.update_tokens([1], [0], [])
    pointers = tuple(value.data_ptr() for value in first[:2])

    second = batch.update_tokens([2, 3], [4, 5], [8, 9])
    lengths = batch.update_decode_context_lens([5, 6])
    indices = batch.update_logits_indices([0, 3])

    assert first[2].numel() == 0
    assert pointers == tuple(value.data_ptr() for value in second[:2])
    assert torch.equal(lengths, torch.tensor([5, 6], dtype=torch.int32))
    assert torch.equal(indices, torch.tensor([0, 3], dtype=torch.int64))


def test_token_input_batch_rejects_invalid_sizes():
    batch = make_token_batch(token_capacity=2, sequence_capacity=1)

    with pytest.raises(ValueError):
        batch.update_tokens([1], [], [2])
    with pytest.raises(ValueError):
        batch.update_tokens([1, 2, 3], [1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        batch.update_cu_seqlens([0, 1], [0])
    with pytest.raises(ValueError):
        batch.update_decode_context_lens([1, 2])


def test_packed_block_metadata_preserves_values_and_reuses_storage():
    batch = make_token_batch()

    first_ids, first_tables = batch.update_packed_block_metadata(
        [7, 3, 11],
        [[0, 1, -1], [2, -1, -1]],
    )
    storage = (first_ids.data_ptr(), first_tables.data_ptr())
    assert torch.equal(first_ids, torch.tensor([7, 3, 11], dtype=torch.int32))
    assert torch.equal(
        first_tables,
        torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32),
    )

    second_ids, second_tables = batch.update_packed_block_metadata(
        [5, 13],
        [[1, 0]],
    )

    assert (second_ids.data_ptr(), second_tables.data_ptr()) == storage
    assert torch.equal(second_ids, torch.tensor([5, 13], dtype=torch.int32))
    assert torch.equal(second_tables, torch.tensor([[1, 0]], dtype=torch.int32))


def test_packed_block_metadata_slots_keep_mixed_live_ranges_disjoint():
    batch = make_token_batch()
    decode_ids, decode_tables = batch.update_packed_block_metadata(
        [7, 3],
        [[0, 1]],
        slot=0,
    )
    prefill_ids, prefill_tables = batch.update_packed_block_metadata(
        [11, 13, 17],
        [[0, 1], [2, -1]],
        slot=1,
    )

    assert decode_ids.data_ptr() != prefill_ids.data_ptr()
    assert decode_tables.data_ptr() != prefill_tables.data_ptr()
    assert torch.equal(decode_ids, torch.tensor([7, 3], dtype=torch.int32))
    assert torch.equal(decode_tables, torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.equal(prefill_ids, torch.tensor([11, 13, 17], dtype=torch.int32))
    assert torch.equal(
        prefill_tables,
        torch.tensor([[0, 1], [2, -1]], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("selected", "tables"),
    (
        ([], [[0]]),
        ([1], []),
        ([1], [[]]),
        ([1], [[0], [0, 1]]),
        ([1], [[0, 1, 2, 3, 4]]),
        ([1], [[0], [0], [0], [0]]),
    ),
)
def test_packed_block_metadata_rejects_invalid_sizes(selected, tables):
    batch = make_token_batch()

    with pytest.raises(ValueError):
        batch.update_packed_block_metadata(selected, tables)

    with pytest.raises(ValueError, match="slot"):
        batch.update_packed_block_metadata([1], [[0]], slot=2)
