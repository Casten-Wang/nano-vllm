import pytest
import torch

from nanovllm.engine.cache_transfer import (
    TRANSFER_FORMAT_VERSION,
    RankCacheTransfer,
)
from nanovllm.engine.heterogeneous_cache_transfer import (
    build_qwen35_peer_cache_fragments,
)
from nanovllm.engine.tp_cache_reshard import (
    build_qwen35_cache_transfer_plan,
)


def make_payload(rank: int, tp_size: int, *, with_scales: bool):
    kv = torch.arange(2 * 2 * 2 * 4 * 1 * 2).reshape(2, 2, 2, 4, 1, 2)
    scale = torch.arange(2 * 2 * 2 * 4).reshape(2, 2, 2, 4, 1)
    recurrent_heads = 32 // tp_size
    convolution_channels = 64 // tp_size
    recurrent = torch.arange(recurrent_heads * 2 * 2).reshape(
        recurrent_heads,
        2,
        2,
    )
    convolution = torch.arange(convolution_channels * 3).reshape(
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
        kv_bytes_per_head=2 * 2 * 2 * 4 * 2 * 8,
        kv_scale_bytes_per_head=(2 * 2 * 2 * 4 * 8 if with_scales else 0),
        recurrent_heads=32,
        recurrent_bytes_per_head=2 * 2 * 8,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=3 * 8,
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
