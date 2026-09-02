from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.cache_transfer import (
    CacheTransferPhase,
    CacheTransferSession,
    RankCacheTransfer,
    export_rank_cache,
    import_rank_cache,
)
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import SequenceStatus


def make_float_cache(fill: bool = True):
    values = torch.arange(2 * 2 * 4 * 2 * 1 * 2, dtype=torch.float32)
    cache = values.view(2, 2, 4, 2, 1, 2)
    return cache if fill else torch.zeros_like(cache)


def make_states(fill: bool = True):
    recurrent = tuple(
        torch.full((2, 2, 3), float(layer + 1)) for layer in range(2)
    )
    convolution = tuple(
        torch.full((4, 2), float(layer + 3)) for layer in range(2)
    )
    if fill:
        return recurrent, convolution
    return (
        tuple(torch.zeros_like(tensor) for tensor in recurrent),
        tuple(torch.zeros_like(tensor) for tensor in convolution),
    )


def test_float_rank_cache_round_trip_uses_logical_block_order():
    source = make_float_cache()
    recurrent, convolution = make_states()
    payload = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-1/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        recurrent_states=recurrent,
        convolution_states=convolution,
    )
    destination = make_float_cache(fill=False)
    destination_recurrent, destination_convolution = make_states(fill=False)

    import_rank_cache(
        payload,
        destination,
        None,
        [0, 2],
        transfer_id="request-1/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        recurrent_states=destination_recurrent,
        convolution_states=destination_convolution,
    )

    torch.testing.assert_close(destination[:, :, 0], source[:, :, 3])
    torch.testing.assert_close(
        destination[:, :, 2, :1],
        source[:, :, 1, :1],
    )
    assert torch.count_nonzero(destination[:, :, 2, 1:]) == 0
    assert torch.count_nonzero(destination[:, :, 1]) == 0
    for expected, actual in zip(recurrent, destination_recurrent):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(convolution, destination_convolution):
        torch.testing.assert_close(actual, expected)


def test_int8_rank_cache_round_trip_includes_scales():
    source = torch.arange(2 * 1 * 3 * 2 * 1 * 2, dtype=torch.int8).view(
        2, 1, 3, 2, 1, 2
    )
    scales = torch.arange(2 * 1 * 3 * 2 * 1, dtype=torch.float16).view(
        2, 1, 3, 2, 1
    )
    payload = export_rank_cache(
        source,
        scales,
        [2],
        transfer_id="request-2/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
    )
    destination = torch.zeros_like(source)
    destination_scales = torch.zeros_like(scales)

    import_rank_cache(
        payload,
        destination,
        destination_scales,
        [1],
        transfer_id="request-2/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
    )

    torch.testing.assert_close(destination[:, :, 1], source[:, :, 2])
    torch.testing.assert_close(destination_scales[:, :, 1], scales[:, :, 2])


def test_import_validation_failure_does_not_modify_destination():
    source = make_float_cache()
    payload = export_rank_cache(
        source,
        None,
        [1],
        transfer_id="request-3/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
    )
    invalid = RankCacheTransfer(
        format_version=payload.format_version,
        transfer_id=payload.transfer_id,
        tensor_parallel_rank=payload.tensor_parallel_rank,
        tensor_parallel_size=payload.tensor_parallel_size,
        block_size=payload.block_size,
        cached_tokens=payload.cached_tokens,
        kv_blocks=payload.kv_blocks.to(torch.float16),
        kv_scales=None,
        recurrent_states=(),
        convolution_states=(),
    )
    destination = torch.full_like(source, -1)
    before = destination.clone()

    with pytest.raises(ValueError, match="dtype"):
        import_rank_cache(
            invalid,
            destination,
            None,
            [0],
            transfer_id="request-3/attempt-1",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            block_size=2,
        )

    torch.testing.assert_close(destination, before)


def test_import_rejects_wrong_tp_rank_before_modifying_destination():
    source = make_float_cache()
    payload = export_rank_cache(
        source,
        None,
        [1],
        transfer_id="request-4/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=2,
        block_size=2,
        cached_tokens=2,
    )
    destination = torch.full_like(source, -1)
    before = destination.clone()

    with pytest.raises(ValueError, match="tensor-parallel identity"):
        import_rank_cache(
            replace(payload, tensor_parallel_rank=1),
            destination,
            None,
            [0],
            transfer_id="request-4/attempt-1",
            tensor_parallel_rank=0,
            tensor_parallel_size=2,
            block_size=2,
        )

    torch.testing.assert_close(destination, before)


def make_runner(kv_cache, state_value):
    pools = []
    modules = []
    for layer_idx in (1, 3):
        pool = SimpleNamespace(
            recurrent=torch.full((1, 2, 2, 2, 3), state_value),
            convolution=torch.full((1, 2, 4, 2), state_value + 1),
        )
        pools.append(pool)
        modules.append(SimpleNamespace(layer_idx=layer_idx, state_pool=pool))
    runner = object.__new__(ModelRunner)
    runner.block_size = 2
    runner.rank = 0
    runner.world_size = 1
    runner.kv_cache = kv_cache
    runner.kv_scale = None
    runner.config = SimpleNamespace(
        model_spec=SimpleNamespace(
            is_hybrid=True,
            linear_attention_layers=(1, 3),
        )
    )
    runner.model = SimpleNamespace(modules=lambda: modules)
    return runner, pools


def test_model_runner_exports_and_imports_complete_hybrid_state():
    source, source_pools = make_runner(make_float_cache(), 5.0)
    destination, destination_pools = make_runner(
        make_float_cache(fill=False),
        0.0,
    )
    source_seq = SimpleNamespace(
        block_table=[3, 1],
        state_slot=1,
        num_cached_tokens=3,
    )
    destination_seq = SimpleNamespace(
        block_table=[0, 2],
        state_slot=0,
        status=SequenceStatus.TRANSFERRING,
        num_prompt_tokens=3,
        num_cached_tokens=0,
    )

    payload = source.export_sequence_cache(
        source_seq,
        transfer_id="request-5/attempt-1",
    )
    destination.import_sequence_cache(
        destination_seq,
        payload,
        transfer_id="request-5/attempt-1",
    )

    torch.testing.assert_close(
        destination.kv_cache[:, :, 0],
        source.kv_cache[:, :, 3],
    )
    torch.testing.assert_close(
        destination.kv_cache[:, :, 2, :1],
        source.kv_cache[:, :, 1, :1],
    )
    assert torch.count_nonzero(destination.kv_cache[:, :, 2, 1:]) == 0
    for source_pool, destination_pool in zip(source_pools, destination_pools):
        torch.testing.assert_close(
            destination_pool.recurrent[0, 0],
            source_pool.recurrent[0, 1],
        )
        torch.testing.assert_close(
            destination_pool.convolution[0, 0],
            source_pool.convolution[0, 1],
        )


def test_transfer_session_commits_only_after_every_rank_acknowledges():
    session = CacheTransferSession(
        "request-6/attempt-1",
        4,
        started_at=10.0,
        timeout_s=5.0,
    )
    for rank in (2, 0, 2, 1):
        session.acknowledge(rank, now=11.0)
    assert session.phase is CacheTransferPhase.RECEIVING
    with pytest.raises(RuntimeError, match="not ready"):
        session.commit(now=11.0)

    session.acknowledge(3, now=12.0)
    assert session.phase is CacheTransferPhase.READY
    session.commit(now=12.0)

    assert session.phase is CacheTransferPhase.COMMITTED
    assert not session.fallback_required


def test_transfer_session_timeout_requires_colocated_fallback():
    session = CacheTransferSession(
        "request-7/attempt-1",
        2,
        started_at=10.0,
        timeout_s=2.0,
    )
    session.acknowledge(0, now=11.0)

    assert session.poll(now=12.0) is CacheTransferPhase.TIMED_OUT
    assert session.fallback_required
    with pytest.raises(RuntimeError, match="terminal"):
        session.acknowledge(1, now=12.1)


def test_transfer_session_rank_failure_aborts_all_rank_commit():
    session = CacheTransferSession(
        "request-8/attempt-1",
        2,
        started_at=10.0,
        timeout_s=5.0,
    )
    session.acknowledge(0, now=11.0)
    session.fail(1, "checksum mismatch", now=11.5)

    assert session.phase is CacheTransferPhase.ABORTED
    assert session.failure_reason == "rank 1: checksum mismatch"
    assert session.fallback_required
    with pytest.raises(RuntimeError, match="not ready"):
        session.commit(now=12.0)
