import torch

from nanovllm.engine.sampling_input_batch import SamplingInputBatch


def make_batch(capacity=4):
    return SamplingInputBatch(
        capacity,
        device="cpu",
        pin_memory=False,
    )


def test_sampling_input_batch_preserves_values_and_dtypes():
    batch = make_batch()

    temperatures, top_ks, top_ps = batch.update(
        [0.0, 0.75],
        [-1, 16],
        [1.0, 0.9],
    )

    torch.testing.assert_close(temperatures, torch.tensor([0.0, 0.75]))
    assert torch.equal(top_ks, torch.tensor([-1, 16], dtype=torch.int32))
    torch.testing.assert_close(top_ps, torch.tensor([1.0, 0.9]))
    assert temperatures.dtype == torch.float32
    assert top_ps.dtype == torch.float32


def test_sampling_input_batch_reuses_host_and_device_storage():
    batch = make_batch()
    host_pointers = (
        batch.host_temperatures.data_ptr(),
        batch.host_top_ks.data_ptr(),
        batch.host_top_ps.data_ptr(),
    )
    first = batch.update([1.0], [8], [0.95])
    device_pointers = tuple(tensor.data_ptr() for tensor in first)

    second = batch.update([0.5, 0.75], [4, 16], [0.8, 0.9])

    assert host_pointers == (
        batch.host_temperatures.data_ptr(),
        batch.host_top_ks.data_ptr(),
        batch.host_top_ps.data_ptr(),
    )
    assert device_pointers == tuple(tensor.data_ptr() for tensor in second)


def test_sampling_input_batch_rejects_invalid_sizes():
    batch = make_batch(capacity=2)

    for values in (
        ([], [], []),
        ([1.0], [1, 2], [1.0]),
        ([1.0, 1.0, 1.0], [1, 1, 1], [1.0, 1.0, 1.0]),
    ):
        try:
            batch.update(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sampling batch must fail")
