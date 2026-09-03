import socket
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nanovllm.engine.cache_transfer import (
    TRANSFER_FORMAT_VERSION,
    HostStagingBufferPool,
    RankCacheTransfer,
)
from nanovllm.engine.heterogeneous_cache_transfer import (
    assemble_qwen35_peer_cache_fragments,
    build_qwen35_peer_cache_fragments,
    stage_qwen35_peer_cache_fragments,
)
from nanovllm.engine.cache_transfer_wire import (
    receive_peer_cache_fragment,
    receive_rank_cache_transfer,
    send_peer_cache_fragment,
)
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.tp_cache_reshard import (
    apply_tp_transfer_plan,
    build_qwen35_cache_transfer_plan,
)


def make_payload(rank: int, tp_size: int, *, with_scales: bool):
    kv = torch.arange(2 * 2 * 2 * 4 * 1 * 2).reshape(2, 2, 2, 4, 1, 2)
    kv = kv.to(torch.int8 if with_scales else torch.float32)
    scale = torch.arange(2 * 2 * 2 * 4, dtype=torch.float16).reshape(
        2, 2, 2, 4, 1
    )
    recurrent_heads = 32 // tp_size
    convolution_channels = 64 // tp_size
    recurrent = torch.arange(
        recurrent_heads * 2 * 2,
        dtype=torch.float32,
    ).reshape(
        recurrent_heads,
        2,
        2,
    )
    convolution = torch.arange(
        convolution_channels * 3,
        dtype=torch.bfloat16,
    ).reshape(
        convolution_channels,
        3,
    )
    return RankCacheTransfer(
        format_version=TRANSFER_FORMAT_VERSION,
        transfer_id="request/attempt-1",
        tensor_parallel_rank=rank,
        tensor_parallel_size=tp_size,
        block_size=4,
        cached_tokens=7,
        kv_blocks=kv + rank * 10_000,
        kv_scales=scale + rank * 1_000 if with_scales else None,
        recurrent_states=(recurrent + rank * 10_000,),
        convolution_states=(convolution + rank * 10_000,),
    )


def make_plan(*, with_scales: bool):
    return build_qwen35_cache_transfer_plan(
        src_tp_size=4,
        dst_tp_size=8,
        total_kv_heads=2,
        kv_bytes_per_head=(
            2 * 2 * 2 * 4 * 2 * (1 if with_scales else 4)
        ),
        kv_scale_bytes_per_head=(2 * 2 * 2 * 4 * 2 if with_scales else 0),
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )


class MemorySocket:
    def __init__(self):
        self.data = bytearray()
        self.offset = 0

    def sendall(self, data):
        self.data.extend(data)

    def recv_into(self, target):
        size = min(len(target), len(self.data) - self.offset)
        if size <= 0:
            return 0
        target[:size] = self.data[self.offset : self.offset + size]
        self.offset += size
        return size


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4)])
@pytest.mark.parametrize("with_scales", [False, True])
def test_source_rank_fragments_match_peer_capacity_without_copying(
    src_tp,
    dst_tp,
    with_scales,
):
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=src_tp,
        dst_tp_size=dst_tp,
        total_kv_heads=2,
        kv_bytes_per_head=(
            2 * 2 * 2 * 4 * 2 * (1 if with_scales else 4)
        ),
        kv_scale_bytes_per_head=(2 * 2 * 2 * 4 * 2 if with_scales else 0),
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )
    all_fragments = []
    for rank in range(src_tp):
        payload = make_payload(rank, src_tp, with_scales=with_scales)
        fragments = build_qwen35_peer_cache_fragments(payload, plan)
        all_fragments.extend(fragments)
        assert sum(fragment.nbytes for fragment in fragments) == (
            plan.profile.source_bytes[rank]
        )
        for fragment in fragments:
            for item in fragment.slices:
                source = (
                    payload.kv_blocks
                    if item.component == "kv"
                    else payload.kv_scales
                    if item.component == "kv_scale"
                    else payload.recurrent_states[item.layer]
                    if item.component == "recurrent"
                    else payload.convolution_states[item.layer]
                )
                assert item.tensor.untyped_storage().data_ptr() == (
                    source.untyped_storage().data_ptr()
                )

    assert {
        (fragment.src_rank, fragment.dst_rank, fragment.nbytes)
        for fragment in all_fragments
    } == set(plan.profile.peer_bytes)


def test_source_rank_fragments_reject_scale_plan_mismatch():
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=4,
        dst_tp_size=8,
        total_kv_heads=2,
        kv_bytes_per_head=512,
        kv_scale_bytes_per_head=128,
        recurrent_heads=32,
        recurrent_bytes_per_head=32,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=24,
    )

    with pytest.raises(ValueError, match="scale layout"):
        build_qwen35_peer_cache_fragments(
            make_payload(0, 4, with_scales=False),
            plan,
        )


@pytest.mark.parametrize("with_scales", [False, True])
def test_peer_fragment_socket_round_trip(with_scales):
    fragment = build_qwen35_peer_cache_fragments(
        make_payload(0, 4, with_scales=with_scales),
        make_plan(with_scales=with_scales),
    )[0]
    sender, receiver = socket.socketpair()
    failures = []

    def send():
        try:
            send_peer_cache_fragment(sender, fragment)
        except BaseException as exc:
            failures.append(exc)
        finally:
            sender.close()

    thread = Thread(target=send)
    thread.start()
    received = receive_peer_cache_fragment(
        receiver,
        expected_transfer_id=fragment.transfer_id,
        expected_src_rank=fragment.src_rank,
        expected_dst_rank=fragment.dst_rank,
        expected_payload_bytes=fragment.nbytes,
    )
    receiver.close()
    thread.join()

    assert not failures
    assert received.nbytes == fragment.nbytes
    assert len(received.slices) == len(fragment.slices)
    for actual, expected in zip(received.slices, fragment.slices):
        assert (
            actual.component,
            actual.layer,
            actual.dst_start,
        ) == (
            expected.component,
            expected.layer,
            expected.dst_start,
        )
        torch.testing.assert_close(actual.tensor, expected.tensor)
    assert len({
        item.tensor.untyped_storage().data_ptr() for item in received.slices
    }) == 1


def test_peer_fragment_wire_enforces_limit_and_distinct_magic():
    fragment = build_qwen35_peer_cache_fragments(
        make_payload(0, 4, with_scales=False),
        make_plan(with_scales=False),
    )[0]
    wire = MemorySocket()
    send_peer_cache_fragment(wire, fragment)

    with pytest.raises(ValueError, match="configured byte limit"):
        receive_peer_cache_fragment(
            wire,
            max_payload_bytes=fragment.nbytes - 1,
        )

    wire.offset = 0
    with pytest.raises(ValueError, match="unsupported cache transfer wire format"):
        receive_rank_cache_transfer(wire)


def test_peer_fragment_wire_rejects_corrupted_payload():
    fragment = build_qwen35_peer_cache_fragments(
        make_payload(0, 4, with_scales=True),
        make_plan(with_scales=True),
    )[0]
    wire = MemorySocket()
    send_peer_cache_fragment(wire, fragment)
    wire.data[-1] ^= 0xFF

    with pytest.raises(ValueError, match="checksum mismatch"):
        receive_peer_cache_fragment(wire)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4)])
@pytest.mark.parametrize("with_scales", [False, True])
def test_destination_assembly_matches_direct_tensor_routes(
    src_tp,
    dst_tp,
    with_scales,
):
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=src_tp,
        dst_tp_size=dst_tp,
        total_kv_heads=2,
        kv_bytes_per_head=(
            2 * 2 * 2 * 4 * 2 * (1 if with_scales else 4)
        ),
        kv_scale_bytes_per_head=(2 * 2 * 2 * 4 * 2 if with_scales else 0),
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )
    payloads = tuple(
        make_payload(rank, src_tp, with_scales=with_scales)
        for rank in range(src_tp)
    )
    fragments = tuple(
        fragment
        for payload in payloads
        for fragment in build_qwen35_peer_cache_fragments(payload, plan)
    )
    expected_kv = apply_tp_transfer_plan(
        tuple(payload.kv_blocks for payload in payloads),
        plan.kv_slices,
        dst_tp,
        shard_dim=4,
        dst_width=1,
    )
    expected_scales = (
        apply_tp_transfer_plan(
            tuple(payload.kv_scales for payload in payloads),
            plan.kv_scale_slices,
            dst_tp,
            shard_dim=4,
            dst_width=1,
        )
        if with_scales
        else (None,) * dst_tp
    )
    expected_recurrent = apply_tp_transfer_plan(
        tuple(payload.recurrent_states[0] for payload in payloads),
        plan.recurrent_slices,
        dst_tp,
        shard_dim=0,
        dst_width=32 // dst_tp,
    )
    expected_convolution = apply_tp_transfer_plan(
        tuple(payload.convolution_states[0] for payload in payloads),
        plan.convolution_slices,
        dst_tp,
        shard_dim=0,
        dst_width=64 // dst_tp,
    )

    for dst_rank in range(dst_tp):
        destination = assemble_qwen35_peer_cache_fragments(
            tuple(
                fragment
                for fragment in reversed(fragments)
                if fragment.dst_rank == dst_rank
            ),
            plan,
        )
        torch.testing.assert_close(destination.kv_blocks, expected_kv[dst_rank])
        if with_scales:
            torch.testing.assert_close(
                destination.kv_scales,
                expected_scales[dst_rank],
            )
        else:
            assert destination.kv_scales is None
        torch.testing.assert_close(
            destination.recurrent_states[0],
            expected_recurrent[dst_rank],
        )
        torch.testing.assert_close(
            destination.convolution_states[0],
            expected_convolution[dst_rank],
        )


def test_destination_assembly_rejects_missing_source_peer():
    src_tp = 8
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=src_tp,
        dst_tp_size=4,
        total_kv_heads=2,
        kv_bytes_per_head=2 * 2 * 2 * 4 * 2 * 4,
        kv_scale_bytes_per_head=0,
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )
    fragments = tuple(
        fragment
        for rank in range(src_tp)
        for fragment in build_qwen35_peer_cache_fragments(
            make_payload(rank, src_tp, with_scales=False),
            plan,
        )
        if fragment.dst_rank == 0
    )
    assert len(fragments) == 2

    with pytest.raises(ValueError, match="peer bytes"):
        assemble_qwen35_peer_cache_fragments(fragments[:1], plan)


def test_model_runner_installs_assembled_destination_atomically():
    src_tp = 8
    dst_tp = 4
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=src_tp,
        dst_tp_size=dst_tp,
        total_kv_heads=2,
        kv_bytes_per_head=2 * 2 * 2 * 4 * 2 * 4,
        kv_scale_bytes_per_head=0,
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )
    fragments = tuple(
        fragment
        for rank in range(src_tp)
        for fragment in build_qwen35_peer_cache_fragments(
            make_payload(rank, src_tp, with_scales=False),
            plan,
        )
        if fragment.dst_rank == 0
    )
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.world_size = dst_tp
    runner.import_sequence_cache = Mock()
    seq = SimpleNamespace()

    result = runner.import_heterogeneous_sequence_cache(
        seq,
        fragments,
        plan,
        transfer_id="request/attempt-1",
    )

    assert result == {
        "rank": 0,
        "cached_tokens": 7,
        "received_bytes": plan.profile.destination_bytes[0],
        "peer_count": 2,
    }
    installed = runner.import_sequence_cache.call_args.args[1]
    assert installed.tensor_parallel_rank == 0
    assert installed.tensor_parallel_size == dst_tp
    runner.import_sequence_cache.assert_called_once_with(
        seq,
        installed,
        transfer_id="request/attempt-1",
    )


def test_selective_source_staging_reuses_fanout_views_and_pool():
    plan = make_plan(with_scales=True)
    payload = make_payload(0, 4, with_scales=True)
    fragments = build_qwen35_peer_cache_fragments(payload, plan)
    pool = HostStagingBufferPool()

    staged = stage_qwen35_peer_cache_fragments(
        fragments,
        plan,
        host_staging_pool=pool,
    )

    assert staged.staged_bytes == plan.profile.source_staging_bytes[0]
    assert sum(fragment.nbytes for fragment in staged.fragments) == (
        plan.profile.source_bytes[0]
    )
    kv_views = [
        item.tensor
        for fragment in staged.fragments
        for item in fragment.slices
        if item.component == "kv"
    ]
    assert len(kv_views) == 2
    assert kv_views[0].data_ptr() == kv_views[1].data_ptr()
    assert all(
        item.tensor.device.type == "cpu"
        for fragment in staged.fragments
        for item in fragment.slices
    )
    assert pool.storage_stats()["leased"] == 1
    staged.release()
    assert pool.storage_stats()["leased"] == 0

    staged_again = stage_qwen35_peer_cache_fragments(
        fragments,
        plan,
        host_staging_pool=pool,
    )
    assert pool.storage_stats()["reuse_count"] == 1
    staged_again.release()


def test_selective_source_staging_omits_unused_kv_replica():
    src_tp = 8
    plan = build_qwen35_cache_transfer_plan(
        src_tp_size=src_tp,
        dst_tp_size=4,
        total_kv_heads=2,
        kv_bytes_per_head=2 * 2 * 2 * 4 * 2 * 4,
        kv_scale_bytes_per_head=0,
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 4,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 2,
    )
    payload = make_payload(1, src_tp, with_scales=False)
    fragments = build_qwen35_peer_cache_fragments(payload, plan)

    staged = stage_qwen35_peer_cache_fragments(fragments, plan)

    assert all(
        item.component != "kv"
        for fragment in staged.fragments
        for item in fragment.slices
    )
    assert staged.staged_bytes == plan.profile.source_staging_bytes[1]
    assert staged.staged_bytes < payload.nbytes
    staged.release()
