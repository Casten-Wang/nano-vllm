import pytest
import torch

from nanovllm.engine.tp_cache_reshard import (
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
