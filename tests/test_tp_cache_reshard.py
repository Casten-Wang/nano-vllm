import pytest
import torch

from nanovllm.engine.tp_cache_reshard import (
    TPTransferProfile,
    TPTransferSlice,
    aggregate_tp_transfer_profiles,
    apply_tp_transfer_plan,
    plan_grouped_uniform_reshard,
    plan_kv_head_reshard,
    plan_uniform_reshard,
    profile_tp_transfer_plan,
    reshard_kv_heads,
    reshard_qwen35_convolution_state,
    reshard_uniform_tensor,
)


def make_uniform_shards(
    global_tensor: torch.Tensor,
    tp_size: int,
    *,
    dim: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(chunk.clone() for chunk in global_tensor.chunk(tp_size, dim=dim))


def make_kv_head_shards(
    global_tensor: torch.Tensor,
    tp_size: int,
    *,
    total_kv_heads: int,
    dim: int,
) -> tuple[torch.Tensor, ...]:
    if total_kv_heads >= tp_size:
        return make_uniform_shards(global_tensor, tp_size, dim=dim)
    replicas = tp_size // total_kv_heads
    unique_heads = global_tensor.chunk(total_kv_heads, dim=dim)
    return tuple(
        unique_heads[rank // replicas].clone()
        for rank in range(tp_size)
    )


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_qwen36_kv_cache_reshard_preserves_replicated_heads(src_tp, dst_tp):
    global_kv = torch.arange(2 * 3 * 2 * 4 * 2 * 2).reshape(
        2, 3, 2, 4, 2, 2
    )
    source = make_kv_head_shards(
        global_kv,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )

    destination = reshard_kv_heads(
        source,
        dst_tp,
        total_kv_heads=2,
        head_dim=4,
    )

    expected = make_kv_head_shards(
        global_kv,
        dst_tp,
        total_kv_heads=2,
        dim=4,
    )
    for actual, expected_rank in zip(destination, expected):
        torch.testing.assert_close(actual, expected_rank)
    assert all(shard.data_ptr() != global_kv.data_ptr() for shard in destination)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_direct_kv_transfer_plan_matches_qwen36_oracle(src_tp, dst_tp):
    global_kv = torch.arange(2 * 3 * 2 * 4 * 2 * 2).reshape(
        2, 3, 2, 4, 2, 2
    )
    source = make_kv_head_shards(
        global_kv,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )
    expected = reshard_kv_heads(
        source,
        dst_tp,
        total_kv_heads=2,
        head_dim=4,
    )

    actual = apply_tp_transfer_plan(
        source,
        plan_kv_head_reshard(2, src_tp, dst_tp),
        dst_tp,
        shard_dim=4,
        dst_width=1,
    )

    for actual_rank, expected_rank in zip(actual, expected):
        torch.testing.assert_close(actual_rank, expected_rank)


def test_qwen36_kv_fanout_uses_all_available_source_replicas():
    global_kv = torch.arange(2 * 3 * 2 * 4 * 2 * 2).reshape(
        2, 3, 2, 4, 2, 2
    )
    source = make_kv_head_shards(
        global_kv,
        4,
        total_kv_heads=2,
        dim=4,
    )
    plan = plan_kv_head_reshard(2, 4, 8)

    profile = profile_tp_transfer_plan(
        source,
        plan,
        8,
        shard_dim=4,
        dst_width=1,
    )

    per_head_bytes = global_kv.numel() // 2 * global_kv.element_size()
    assert profile.source_bytes == (per_head_bytes * 2,) * 4
    assert profile.destination_bytes == (per_head_bytes,) * 8
    assert profile.source_peer_counts == (2, 2, 2, 2)
    assert profile.destination_peer_counts == (1,) * 8
    assert profile.wire_bytes == per_head_bytes * 8
    assert sum(byte_count for _, _, byte_count in profile.peer_bytes) == (
        profile.wire_bytes
    )
    assert {(src, dst) for src, dst, _ in profile.peer_bytes} == {
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7),
    }


@pytest.mark.parametrize(
    "src_tp,dst_tp,source_peers,destination_peers",
    [
        (4, 8, (2,) * 4, (1,) * 8),
        (8, 4, (1,) * 8, (2,) * 4),
        (4, 4, (1,) * 4, (1,) * 4),
    ],
)
def test_request_profile_aggregates_all_qwen36_cache_components(
    src_tp,
    dst_tp,
    source_peers,
    destination_peers,
):
    global_kv = torch.arange(2 * 3 * 2 * 4 * 2 * 2).reshape(
        2, 3, 2, 4, 2, 2
    )
    global_scale = torch.arange(2 * 3 * 2 * 4 * 2, dtype=torch.float16).reshape(
        2, 3, 2, 4, 2
    )
    kv_source = make_kv_head_shards(
        global_kv,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )
    scale_source = make_kv_head_shards(
        global_scale,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )
    plan = plan_kv_head_reshard(2, src_tp, dst_tp)
    kv_profile = profile_tp_transfer_plan(
        kv_source,
        plan,
        dst_tp,
        shard_dim=4,
        dst_width=1,
    )
    scale_profile = profile_tp_transfer_plan(
        scale_source,
        plan,
        dst_tp,
        shard_dim=4,
        dst_width=1,
    )
    global_recurrent = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(
        32, 2, 3
    )
    recurrent_source = make_uniform_shards(global_recurrent, src_tp, dim=0)
    recurrent_profile = profile_tp_transfer_plan(
        recurrent_source,
        plan_uniform_reshard(32, src_tp, dst_tp),
        dst_tp,
        shard_dim=0,
        dst_width=32 // dst_tp,
    )
    query = torch.arange(16 * 3).reshape(16, 3)
    key = 1_000 + torch.arange(16 * 3).reshape(16, 3)
    value = 2_000 + torch.arange(32 * 3).reshape(32, 3)
    convolution_source = make_convolution_shards(query, key, value, src_tp)
    convolution_profile = profile_tp_transfer_plan(
        convolution_source,
        plan_grouped_uniform_reshard((16, 16, 32), src_tp, dst_tp),
        dst_tp,
        shard_dim=0,
        dst_width=64 // dst_tp,
    )

    request_profile = aggregate_tp_transfer_profiles(
        (
            kv_profile,
            scale_profile,
            recurrent_profile,
            convolution_profile,
        )
    )

    component_profiles = (
        kv_profile,
        scale_profile,
        recurrent_profile,
        convolution_profile,
    )
    assert request_profile.wire_bytes == sum(
        profile.wire_bytes for profile in component_profiles
    )
    assert request_profile.source_bytes == tuple(
        sum(profile.source_bytes[rank] for profile in component_profiles)
        for rank in range(src_tp)
    )
    assert request_profile.destination_bytes == tuple(
        sum(
            profile.destination_bytes[rank]
            for profile in component_profiles
        )
        for rank in range(dst_tp)
    )
    assert request_profile.source_peer_counts == source_peers
    assert request_profile.destination_peer_counts == destination_peers
    assert request_profile.slice_count == sum(
        profile.slice_count for profile in component_profiles
    )
    assert len(request_profile.peer_bytes) == max(src_tp, dst_tp)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_int8_scale_reshard_uses_kv_head_dimension(src_tp, dst_tp):
    global_scale = torch.arange(2 * 3 * 2 * 4 * 2, dtype=torch.float16).reshape(
        2, 3, 2, 4, 2
    )
    source = make_kv_head_shards(
        global_scale,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )

    destination = reshard_kv_heads(
        source,
        dst_tp,
        total_kv_heads=2,
        head_dim=4,
    )

    expected = make_kv_head_shards(
        global_scale,
        dst_tp,
        total_kv_heads=2,
        dim=4,
    )
    for actual, expected_rank in zip(destination, expected):
        torch.testing.assert_close(actual, expected_rank)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_recurrent_state_reshard_uses_value_head_dimension(src_tp, dst_tp):
    global_state = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(32, 2, 3)
    source = make_uniform_shards(global_state, src_tp, dim=0)

    destination = reshard_uniform_tensor(source, dst_tp, shard_dim=0)

    torch.testing.assert_close(torch.cat(destination, dim=0), global_state)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_direct_recurrent_transfer_plan_matches_oracle(src_tp, dst_tp):
    global_state = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(32, 2, 3)
    source = make_uniform_shards(global_state, src_tp, dim=0)
    expected = reshard_uniform_tensor(source, dst_tp, shard_dim=0)

    actual = apply_tp_transfer_plan(
        source,
        plan_uniform_reshard(32, src_tp, dst_tp),
        dst_tp,
        shard_dim=0,
        dst_width=32 // dst_tp,
    )

    for actual_rank, expected_rank in zip(actual, expected):
        torch.testing.assert_close(actual_rank, expected_rank)


@pytest.mark.parametrize(
    "src_tp,dst_tp,source_peers,destination_peers",
    [
        (4, 8, (2,) * 4, (1,) * 8),
        (8, 4, (1,) * 8, (2,) * 4),
        (4, 4, (1,) * 4, (1,) * 4),
    ],
)
def test_uniform_transfer_profile_reports_fanout_and_fanin(
    src_tp,
    dst_tp,
    source_peers,
    destination_peers,
):
    global_state = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(32, 2, 3)
    source = make_uniform_shards(global_state, src_tp, dim=0)

    profile = profile_tp_transfer_plan(
        source,
        plan_uniform_reshard(32, src_tp, dst_tp),
        dst_tp,
        shard_dim=0,
        dst_width=32 // dst_tp,
    )

    assert profile.source_peer_counts == source_peers
    assert profile.destination_peer_counts == destination_peers
    assert profile.wire_bytes == global_state.numel() * global_state.element_size()
    assert sum(profile.source_bytes) == profile.wire_bytes
    assert sum(profile.destination_bytes) == profile.wire_bytes


def make_convolution_shards(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tp_size: int,
) -> tuple[torch.Tensor, ...]:
    query_shards = query.chunk(tp_size, dim=0)
    key_shards = key.chunk(tp_size, dim=0)
    value_shards = value.chunk(tp_size, dim=0)
    return tuple(
        torch.cat((query_shards[rank], key_shards[rank], value_shards[rank]), dim=0)
        for rank in range(tp_size)
    )


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_qwen35_convolution_reshard_preserves_independent_groups(src_tp, dst_tp):
    query = torch.arange(16 * 3).reshape(16, 3)
    key = 1_000 + torch.arange(16 * 3).reshape(16, 3)
    value = 2_000 + torch.arange(32 * 3).reshape(32, 3)
    source = make_convolution_shards(query, key, value, src_tp)

    destination = reshard_qwen35_convolution_state(
        source,
        dst_tp,
        key_channels_per_src_rank=16 // src_tp,
        value_channels_per_src_rank=32 // src_tp,
    )

    expected = make_convolution_shards(query, key, value, dst_tp)
    for actual, expected_rank in zip(destination, expected):
        torch.testing.assert_close(actual, expected_rank)


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4), (4, 4)])
def test_direct_qwen35_convolution_plan_matches_oracle(src_tp, dst_tp):
    query = torch.arange(16 * 3).reshape(16, 3)
    key = 1_000 + torch.arange(16 * 3).reshape(16, 3)
    value = 2_000 + torch.arange(32 * 3).reshape(32, 3)
    source = make_convolution_shards(query, key, value, src_tp)
    expected = reshard_qwen35_convolution_state(
        source,
        dst_tp,
        key_channels_per_src_rank=16 // src_tp,
        value_channels_per_src_rank=32 // src_tp,
    )

    actual = apply_tp_transfer_plan(
        source,
        plan_grouped_uniform_reshard((16, 16, 32), src_tp, dst_tp),
        dst_tp,
        shard_dim=0,
        dst_width=64 // dst_tp,
    )

    for actual_rank, expected_rank in zip(actual, expected):
        torch.testing.assert_close(actual_rank, expected_rank)


def test_plain_rank_concatenation_is_not_a_valid_qkv_reshard():
    query = torch.arange(8).reshape(8, 1)
    key = 100 + torch.arange(8).reshape(8, 1)
    value = 200 + torch.arange(16).reshape(16, 1)
    source = make_convolution_shards(query, key, value, 4)
    wrong = make_uniform_shards(torch.cat(source, dim=0), 8, dim=0)
    expected = make_convolution_shards(query, key, value, 8)

    assert any(not torch.equal(actual, wanted) for actual, wanted in zip(wrong, expected))


@pytest.mark.parametrize("dst_tp_size", [0, -1, True, 1.5])
def test_reshard_rejects_invalid_destination_tp_size(dst_tp_size):
    with pytest.raises(ValueError, match="positive integer"):
        reshard_uniform_tensor((torch.ones(2),), dst_tp_size, shard_dim=0)


def test_reshard_rejects_non_divisible_global_dimension():
    with pytest.raises(ValueError, match="must divide"):
        reshard_uniform_tensor((torch.ones(3), torch.ones(3)), 4, shard_dim=0)


def test_reshard_rejects_unequal_source_shard_widths():
    with pytest.raises(ValueError, match="equal width"):
        reshard_uniform_tensor((torch.ones(3), torch.ones(2)), 1, shard_dim=0)


def test_reference_reshard_rejects_non_cpu_storage():
    with pytest.raises(ValueError, match="CPU tensors only"):
        reshard_uniform_tensor(
            (torch.empty(2, device="meta"),),
            1,
            shard_dim=0,
        )


@pytest.mark.parametrize("shard_dim", [True, 2, -3])
def test_reshard_rejects_invalid_sharded_dimension(shard_dim):
    with pytest.raises(ValueError, match="sharded dimension"):
        reshard_uniform_tensor((torch.ones(2, 3),), 1, shard_dim=shard_dim)


def test_convolution_reshard_rejects_inconsistent_group_metadata():
    with pytest.raises(ValueError, match="Q/K/V channel layout"):
        reshard_qwen35_convolution_state(
            (torch.ones(7, 3),),
            1,
            key_channels_per_src_rank=2,
            value_channels_per_src_rank=2,
        )


def test_kv_reshard_rejects_divergent_replicas():
    global_kv = torch.arange(2 * 2).reshape(2, 2)
    source = list(
        make_kv_head_shards(global_kv, 4, total_kv_heads=2, dim=0)
    )
    source[1].add_(1)

    with pytest.raises(ValueError, match="different data"):
        reshard_kv_heads(source, 8, total_kv_heads=2, head_dim=0)


@pytest.mark.parametrize(
    "total_kv_heads,src_tp,dst_tp",
    [(3, 2, 1), (2, 3, 1), (2, 1, 3)],
)
def test_kv_reshard_rejects_incompatible_head_topology(
    total_kv_heads,
    src_tp,
    dst_tp,
):
    shards = tuple(torch.ones(1, 2) for _ in range(src_tp))
    with pytest.raises(ValueError, match="divide"):
        reshard_kv_heads(
            shards,
            dst_tp,
            total_kv_heads=total_kv_heads,
            head_dim=0,
        )


def test_transfer_plan_rejects_destination_gaps():
    source = (torch.arange(4),)
    with pytest.raises(ValueError, match="destination gaps"):
        apply_tp_transfer_plan(
            source,
            (TPTransferSlice(0, 0, 0, 0, 3),),
            1,
            shard_dim=0,
            dst_width=4,
        )


def test_transfer_plan_rejects_overlapping_writes():
    source = (torch.arange(4),)
    with pytest.raises(ValueError, match="overlapping writes"):
        apply_tp_transfer_plan(
            source,
            (
                TPTransferSlice(0, 0, 0, 0, 3),
                TPTransferSlice(0, 0, 2, 2, 2),
            ),
            1,
            shard_dim=0,
            dst_width=4,
        )


@pytest.mark.parametrize(
    "entry",
    [
        TPTransferSlice(1, 0, 0, 0, 1),
        TPTransferSlice(0, 1, 0, 0, 1),
        TPTransferSlice(0, 0, 4, 0, 1),
        TPTransferSlice(0, 0, 0, 4, 1),
    ],
)
def test_transfer_plan_rejects_out_of_bounds_entries(entry):
    with pytest.raises(ValueError, match="out of bounds"):
        apply_tp_transfer_plan(
            (torch.arange(4),),
            (entry,),
            1,
            shard_dim=0,
            dst_width=4,
        )


@pytest.mark.parametrize(
    "args",
    [
        (True, 0, 0, 0, 1),
        (0, -1, 0, 0, 1),
        (0, 0, 0, 0, 0),
    ],
)
def test_transfer_slice_rejects_invalid_metadata(args):
    with pytest.raises(ValueError):
        TPTransferSlice(*args)


def test_uniform_plan_rejects_non_divisible_topology():
    with pytest.raises(ValueError, match="divide both"):
        plan_uniform_reshard(6, 4, 2)


def test_grouped_plan_requires_groups():
    with pytest.raises(ValueError, match="at least one"):
        plan_grouped_uniform_reshard((), 4, 8)


def test_request_profile_rejects_mixed_tp_topologies():
    global_state = torch.arange(32 * 2, dtype=torch.float32).reshape(32, 2)
    source4 = make_uniform_shards(global_state, 4, dim=0)
    source8 = make_uniform_shards(global_state, 8, dim=0)
    profile4_to_8 = profile_tp_transfer_plan(
        source4,
        plan_uniform_reshard(32, 4, 8),
        8,
        shard_dim=0,
        dst_width=4,
    )
    profile8_to_4 = profile_tp_transfer_plan(
        source8,
        plan_uniform_reshard(32, 8, 4),
        4,
        shard_dim=0,
        dst_width=8,
    )

    with pytest.raises(ValueError, match="different TP topologies"):
        aggregate_tp_transfer_profiles((profile4_to_8, profile8_to_4))


def test_transfer_profile_rejects_inconsistent_peer_ledger():
    with pytest.raises(ValueError, match="peer bytes do not match"):
        TPTransferProfile(
            wire_bytes=4,
            source_bytes=(4,),
            destination_bytes=(4,),
            source_peer_counts=(1,),
            destination_peer_counts=(1,),
            peer_bytes=((0, 0, 3),),
            slice_count=1,
        )
