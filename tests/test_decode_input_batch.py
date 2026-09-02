import pytest
import torch

from nanovllm.engine.decode_input_batch import DecodeInputBatch


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
