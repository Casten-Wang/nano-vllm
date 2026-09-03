import json

import pytest
import torch

from nanovllm.engine.cache_transfer import (
    TRANSFER_FORMAT_VERSION,
    CacheTransferPhase,
    RankCacheTransfer,
)
from nanovllm.engine.tp_cache_reshard import (
    Qwen35CacheTransferPlan,
    TPPeerTransferSession,
    TPTransferProfile,
    TPTransferSlice,
    aggregate_tp_transfer_profiles,
    apply_tp_transfer_plan,
    build_qwen35_cache_transfer_plan,
    plan_grouped_uniform_reshard,
    plan_kv_head_reshard,
    plan_uniform_reshard,
    profile_qwen35_cache_transfer_layout,
    profile_tp_transfer_plan,
    profile_tp_transfer_layout,
    reshard_kv_heads,
    reshard_qwen35_convolution_state,
    reshard_qwen35_rank_cache_transfers,
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


@pytest.mark.parametrize("src_tp,dst_tp", [(4, 8), (8, 4)])
@pytest.mark.parametrize("with_scales", [False, True])
def test_qwen36_rank_cache_payload_reshard_matches_tensor_oracles(
    src_tp,
    dst_tp,
    with_scales,
):
    global_kv = torch.arange(2 * 2 * 2 * 4 * 2 * 2).reshape(
        2, 2, 2, 4, 2, 2
    )
    global_scale = torch.arange(2 * 2 * 2 * 4 * 2).reshape(2, 2, 2, 4, 2)
    global_recurrent = torch.arange(32 * 2 * 2).reshape(32, 2, 2)
    query = torch.arange(16 * 3).reshape(16, 3)
    key = query + 1_000
    value = torch.arange(32 * 3).reshape(32, 3) + 2_000
    source_kv = make_kv_head_shards(
        global_kv,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )
    source_scales = make_kv_head_shards(
        global_scale,
        src_tp,
        total_kv_heads=2,
        dim=4,
    )
    source_recurrent = make_uniform_shards(global_recurrent, src_tp, dim=0)
    source_convolution = tuple(
        torch.cat(parts, dim=0)
        for parts in zip(
            make_uniform_shards(query, src_tp, dim=0),
            make_uniform_shards(key, src_tp, dim=0),
            make_uniform_shards(value, src_tp, dim=0),
        )
    )
    payloads = tuple(
        RankCacheTransfer(
            format_version=TRANSFER_FORMAT_VERSION,
            transfer_id="request/attempt-1",
            tensor_parallel_rank=rank,
            tensor_parallel_size=src_tp,
            block_size=4,
            cached_tokens=7,
            kv_blocks=source_kv[rank],
            kv_scales=source_scales[rank] if with_scales else None,
            recurrent_states=(source_recurrent[rank],),
            convolution_states=(source_convolution[rank],),
        )
        for rank in range(src_tp)
    )

    destination = reshard_qwen35_rank_cache_transfers(
        payloads,
        dst_tp,
        total_kv_heads=2,
        key_channels_per_src_rank=16 // src_tp,
        value_channels_per_src_rank=32 // src_tp,
    )

    expected_kv = make_kv_head_shards(
        global_kv,
        dst_tp,
        total_kv_heads=2,
        dim=4,
    )
    expected_scales = make_kv_head_shards(
        global_scale,
        dst_tp,
        total_kv_heads=2,
        dim=4,
    )
    expected_recurrent = make_uniform_shards(global_recurrent, dst_tp, dim=0)
    expected_convolution = tuple(
        torch.cat(parts, dim=0)
        for parts in zip(
            make_uniform_shards(query, dst_tp, dim=0),
            make_uniform_shards(key, dst_tp, dim=0),
            make_uniform_shards(value, dst_tp, dim=0),
        )
    )
    assert len(destination) == dst_tp
    for rank, payload in enumerate(destination):
        assert payload.tensor_parallel_rank == rank
        assert payload.tensor_parallel_size == dst_tp
        torch.testing.assert_close(payload.kv_blocks, expected_kv[rank])
        if with_scales:
            torch.testing.assert_close(payload.kv_scales, expected_scales[rank])
        else:
            assert payload.kv_scales is None
        torch.testing.assert_close(
            payload.recurrent_states[0],
            expected_recurrent[rank],
        )
        torch.testing.assert_close(
            payload.convolution_states[0],
            expected_convolution[rank],
        )


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
    assert profile.source_staging_bytes == (per_head_bytes,) * 4
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
    estimated_profile = profile_qwen35_cache_transfer_layout(
        src_tp_size=src_tp,
        dst_tp_size=dst_tp,
        total_kv_heads=2,
        kv_bytes_per_head=(
            global_kv.numel() // 2 * global_kv.element_size()
        ),
        kv_scale_bytes_per_head=(
            global_scale.numel() // 2 * global_scale.element_size()
        ),
        recurrent_heads=32,
        recurrent_bytes_per_head=(
            global_recurrent.numel() // 32 * global_recurrent.element_size()
        ),
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=query.shape[1] * query.element_size(),
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
    assert request_profile.source_staging_bytes == tuple(
        sum(
            profile.source_staging_bytes[rank]
            for profile in component_profiles
        )
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
    assert estimated_profile == request_profile


def test_transfer_profile_survives_json_control_plane_round_trip():
    profile = profile_qwen35_cache_transfer_layout(
        src_tp_size=4,
        dst_tp_size=8,
        total_kv_heads=2,
        kv_bytes_per_head=1_024,
        kv_scale_bytes_per_head=32,
        recurrent_heads=32,
        recurrent_bytes_per_head=512,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=64,
    )
    report = json.loads(json.dumps(profile.to_dict()))

    assert TPTransferProfile.from_dict(report) == profile


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_tp_size", 8, "source_tp_size"),
        ("destination_tp_size", True, "destination_tp_size"),
        ("peer_bytes", [[0, 0]], "peer_bytes"),
    ],
)
def test_transfer_profile_rejects_corrupted_serialized_topology(
    field,
    value,
    message,
):
    profile = profile_qwen35_cache_transfer_layout(
        src_tp_size=4,
        dst_tp_size=8,
        total_kv_heads=2,
        kv_bytes_per_head=1_024,
        kv_scale_bytes_per_head=0,
        recurrent_heads=32,
        recurrent_bytes_per_head=512,
        convolution_group_widths=(16, 16, 32),
        convolution_bytes_per_channel=64,
    )
    report = profile.to_dict()
    report[field] = value

    with pytest.raises(ValueError, match=message):
        TPTransferProfile.from_dict(report)


def test_layout_preflight_omits_scale_traffic_for_floating_kv_cache():
    common = {
        "src_tp_size": 4,
        "dst_tp_size": 8,
        "total_kv_heads": 2,
        "kv_bytes_per_head": 1_024,
        "recurrent_heads": 32,
        "recurrent_bytes_per_head": 512,
        "convolution_group_widths": (16, 16, 32),
        "convolution_bytes_per_channel": 64,
    }
    floating = profile_qwen35_cache_transfer_layout(
        **common,
        kv_scale_bytes_per_head=0,
    )
    quantized = profile_qwen35_cache_transfer_layout(
        **common,
        kv_scale_bytes_per_head=32,
    )

    assert quantized.wire_bytes - floating.wire_bytes == 8 * 32
    assert (
        sum(quantized.source_staging_bytes)
        - sum(floating.source_staging_bytes)
        == 4 * 32
    )
    assert all(
        quantized.destination_bytes[rank]
        - floating.destination_bytes[rank]
        == 32
        for rank in range(8)
    )


@pytest.mark.parametrize("kv_scale_bytes_per_head", [0, 32])
def test_qwen36_request_plan_exposes_exact_routes_and_capacity(
    kv_scale_bytes_per_head,
):
    arguments = {
        "src_tp_size": 4,
        "dst_tp_size": 8,
        "total_kv_heads": 2,
        "kv_bytes_per_head": 1_024,
        "kv_scale_bytes_per_head": kv_scale_bytes_per_head,
        "recurrent_heads": 32,
        "recurrent_bytes_per_head": 512,
        "convolution_group_widths": (16, 16, 32),
        "convolution_bytes_per_channel": 64,
    }

    plan = build_qwen35_cache_transfer_plan(**arguments)

    assert isinstance(plan, Qwen35CacheTransferPlan)
    assert plan.profile == profile_qwen35_cache_transfer_layout(**arguments)
    assert bool(plan.kv_scale_slices) == bool(kv_scale_bytes_per_head)
    if kv_scale_bytes_per_head:
        assert plan.kv_scale_slices == plan.kv_slices
    assert plan.profile.slice_count == sum(
        len(routes)
        for routes in (
            plan.kv_slices,
            plan.kv_scale_slices,
            plan.recurrent_slices,
            plan.convolution_slices,
        )
    )
    assert {
        (route.src_rank, route.dst_rank)
        for route in plan.kv_slices
    } <= {
        (src_rank, dst_rank)
        for src_rank, dst_rank, _byte_count in plan.profile.peer_bytes
    }


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("kv_bytes_per_head", True, "kv_bytes_per_head"),
        ("kv_scale_bytes_per_head", -1, "kv_scale_bytes_per_head"),
        ("recurrent_heads", 0, "recurrent_heads"),
        ("convolution_group_widths", [16, 16, 32], "group widths"),
    ],
)
def test_layout_preflight_rejects_invalid_capacity_metadata(field, value, match):
    arguments = {
        "src_tp_size": 4,
        "dst_tp_size": 8,
        "total_kv_heads": 2,
        "kv_bytes_per_head": 1_024,
        "kv_scale_bytes_per_head": 0,
        "recurrent_heads": 32,
        "recurrent_bytes_per_head": 512,
        "convolution_group_widths": (16, 16, 32),
        "convolution_bytes_per_channel": 64,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=match):
        profile_qwen35_cache_transfer_layout(**arguments)


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
            source_staging_bytes=(4,),
            destination_bytes=(4,),
            source_peer_counts=(1,),
            destination_peer_counts=(1,),
            peer_bytes=((0, 0, 3),),
            slice_count=1,
        )


def test_transfer_profile_rejects_per_rank_byte_mismatch():
    with pytest.raises(ValueError, match="source bytes do not match"):
        TPTransferProfile(
            wire_bytes=4,
            source_bytes=(2, 2),
            source_staging_bytes=(2, 2),
            destination_bytes=(2, 2),
            source_peer_counts=(2, 0),
            destination_peer_counts=(1, 1),
            peer_bytes=((0, 0, 2), (0, 1, 2)),
            slice_count=2,
        )


def test_transfer_profile_rejects_staging_larger_than_egress():
    with pytest.raises(ValueError, match="staging bytes"):
        TPTransferProfile(
            wire_bytes=4,
            source_bytes=(4,),
            source_staging_bytes=(5,),
            destination_bytes=(4,),
            source_peer_counts=(1,),
            destination_peer_counts=(1,),
            peer_bytes=((0, 0, 4),),
            slice_count=1,
        )


def make_peer_session(*, started_at=10.0, timeout_s=5.0):
    global_tensor = torch.arange(8, dtype=torch.float32)
    source = make_uniform_shards(global_tensor, 4, dim=0)
    profile = profile_tp_transfer_plan(
        source,
        plan_uniform_reshard(8, 4, 2),
        2,
        shard_dim=0,
        dst_width=4,
    )
    return TPPeerTransferSession(
        "request-1",
        profile,
        started_at=started_at,
        timeout_s=timeout_s,
    )


def test_peer_session_commits_only_after_every_destination_install():
    session = make_peer_session()
    expected = {(src, dst): size for src, dst, size in session.profile.peer_bytes}

    session.acknowledge(0, 0, expected[0, 0], now=11.0)
    assert session.phase is CacheTransferPhase.RECEIVING
    assert session.ready_destination_ranks == ()
    session.acknowledge(1, 0, expected[1, 0], now=11.1)
    assert session.ready_destination_ranks == (0,)
    with pytest.raises(RuntimeError, match="not ready"):
        session.commit(now=11.2)

    session.acknowledge(2, 1, expected[2, 1], now=11.3)
    session.acknowledge(3, 1, expected[3, 1], now=11.4)
    assert session.phase is CacheTransferPhase.READY
    assert session.ready_destination_ranks == (0, 1)
    assert session.pending_peer_bytes == ()
    assert session.acknowledged_bytes == session.profile.wire_bytes

    session.commit(now=11.5)
    assert session.phase is CacheTransferPhase.COMMITTED


def test_peer_session_duplicate_acknowledgement_is_idempotent():
    session = make_peer_session()
    src_rank, dst_rank, byte_count = session.profile.peer_bytes[0]

    session.acknowledge(src_rank, dst_rank, byte_count, now=11.0)
    acknowledged = session.acknowledged_bytes
    session.acknowledge(src_rank, dst_rank, byte_count, now=11.1)

    assert session.acknowledged_bytes == acknowledged


def test_peer_session_rejects_wrong_bytes_without_progress():
    session = make_peer_session()
    src_rank, dst_rank, byte_count = session.profile.peer_bytes[0]

    with pytest.raises(ValueError, match="byte count"):
        session.acknowledge(src_rank, dst_rank, byte_count - 1, now=11.0)

    assert session.acknowledged_bytes == 0
    assert len(session.pending_peer_bytes) == len(session.profile.peer_bytes)


@pytest.mark.parametrize("src_rank,dst_rank", [(True, 0), (0, False), (9, 0)])
def test_peer_session_rejects_invalid_or_unknown_peer(src_rank, dst_rank):
    session = make_peer_session()
    with pytest.raises(ValueError, match="rank|unexpected"):
        session.acknowledge(src_rank, dst_rank, 8, now=11.0)


def test_peer_session_failure_is_terminal_and_requires_fallback():
    session = make_peer_session()
    src_rank, dst_rank, _ = session.profile.peer_bytes[0]

    session.fail(src_rank, dst_rank, "connection reset", now=11.0)

    assert session.phase is CacheTransferPhase.ABORTED
    assert session.fallback_required
    assert "connection reset" in session.failure_reason
    with pytest.raises(RuntimeError, match="terminal"):
        session.acknowledge(src_rank, dst_rank, 8, now=11.1)


def test_peer_session_timeout_prevents_late_commit():
    session = make_peer_session(started_at=10.0, timeout_s=2.0)

    assert session.poll(now=12.0) is CacheTransferPhase.TIMED_OUT
    assert session.fallback_required
    with pytest.raises(RuntimeError, match="not ready"):
        session.commit(now=12.1)


def test_peer_session_ready_state_still_expires_before_commit():
    session = make_peer_session(started_at=10.0, timeout_s=2.0)
    for src_rank, dst_rank, byte_count in session.profile.peer_bytes:
        session.acknowledge(src_rank, dst_rank, byte_count, now=11.0)
    assert session.phase is CacheTransferPhase.READY

    assert session.poll(now=12.0) is CacheTransferPhase.TIMED_OUT
    assert session.fallback_required


@pytest.mark.parametrize(
    "started_at,timeout_s",
    [(True, 1.0), (0.0, False), (float("nan"), 1.0), (0.0, float("inf"))],
)
def test_peer_session_rejects_invalid_time_metadata(started_at, timeout_s):
    with pytest.raises(ValueError, match="started_at|timeout_s"):
        make_peer_session(started_at=started_at, timeout_s=timeout_s)


def test_peer_session_rejects_overflowed_deadline():
    with pytest.raises(ValueError, match="deadline"):
        make_peer_session(started_at=1e308, timeout_s=1e308)


def test_peer_session_rejects_unrepresentable_integer_time():
    with pytest.raises(ValueError, match="started_at"):
        make_peer_session(started_at=10**1_000, timeout_s=1.0)
