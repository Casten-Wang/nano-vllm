from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.cache_transfer import (
    CacheTransferPhase,
    CacheTransferSession,
    HostStagingBufferPool,
    RankCacheTransfer,
    build_cache_transfer_fingerprint,
    build_token_fingerprint,
    estimate_rank_cache_transfer_bytes,
    export_rank_cache,
    import_rank_cache,
    validate_cache_transfer_payload_limit,
)
from nanovllm.engine.model_runner import (
    ModelRunner,
    build_qwen35_cache_transfer_plan_from_spec,
    plan_qwen35_cache_transfer_capacity,
)
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.models.cache_plan import plan_cache_memory


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


def test_cache_transfer_fingerprint_tracks_identity_and_cache_layout():
    model_config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=4096,
        dtype=torch.bfloat16,
    )
    model_spec = SimpleNamespace(
        architecture="Qwen3_5MoeForCausalLM",
        full_attention_layers=(3, 7),
        linear_attention_layers=(0, 1, 2, 4, 5, 6),
        num_hidden_layers=8,
    )

    def config(model_id, *, block_size=256):
        return SimpleNamespace(
            model="/different/local/path",
            cache_transfer_model_id=model_id,
            hf_config=SimpleNamespace(_commit_hash=None),
            model_spec=model_spec,
            model_config=model_config,
            kv_cache_dtype="auto",
            recurrent_state_dtype="float32",
            kvcache_block_size=block_size,
            sliding_window_size=None,
        )

    baseline = build_cache_transfer_fingerprint(config("qwen36-revision-a"))
    assert baseline == build_cache_transfer_fingerprint(config("qwen36-revision-a"))
    assert baseline != build_cache_transfer_fingerprint(config("qwen36-revision-b"))
    assert baseline != build_cache_transfer_fingerprint(
        config("qwen36-revision-a", block_size=512)
    )


def test_token_fingerprint_tracks_order_and_cached_prefix_only():
    baseline = build_token_fingerprint([11, 22, 33, 44], 3)
    assert baseline == build_token_fingerprint([11, 22, 33, 99], 3)
    assert baseline != build_token_fingerprint([11, 33, 22, 44], 3)


def make_qwen36_model_spec():
    config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        is_hybrid=True,
        text_config=config,
        num_kv_cache_layers=10,
        linear_attention_layers=tuple(range(30)),
    )


@pytest.mark.parametrize(
    "tp_size,kv_dtype,recurrent_dtype",
    [
        (4, torch.bfloat16, torch.float32),
        (8, torch.bfloat16, torch.float32),
        (4, torch.int8, torch.bfloat16),
    ],
)
def test_runtime_transfer_preflight_matches_same_tp_cache_plan(
    tp_size,
    kv_dtype,
    recurrent_dtype,
):
    model_spec = make_qwen36_model_spec()
    num_blocks = 3
    block_size = 256
    profile = plan_qwen35_cache_transfer_capacity(
        model_spec,
        src_tp_size=tp_size,
        dst_tp_size=tp_size,
        num_blocks=num_blocks,
        block_size=block_size,
        kv_dtype=kv_dtype,
        recurrent_dtype=recurrent_dtype,
        convolution_dtype=torch.bfloat16,
    )
    existing = plan_cache_memory(
        model_spec,
        tp_size,
        kv_dtype_bytes=kv_dtype.itemsize,
        recurrent_dtype_bytes=recurrent_dtype.itemsize,
        convolution_dtype_bytes=2,
    )
    expected_rank_bytes = (
        existing.kv_bytes_per_token * num_blocks * block_size
        + (
            existing.int8_scale_bytes_per_token * num_blocks * block_size
            if kv_dtype == torch.int8
            else 0
        )
        + existing.recurrent_bytes_per_sequence
        + existing.convolution_bytes_per_sequence
    )

    assert profile.source_staging_bytes == (expected_rank_bytes,) * tp_size
    assert profile.source_bytes == profile.source_staging_bytes
    assert profile.destination_bytes == profile.source_staging_bytes
    assert profile.wire_bytes == expected_rank_bytes * tp_size


def test_model_runner_exposes_serializable_heterogeneous_transfer_preflight():
    model_spec = make_qwen36_model_spec()
    runner = object.__new__(ModelRunner)
    runner.world_size = 4
    runner.block_size = 256
    runner.config = SimpleNamespace(
        model_spec=model_spec,
        model_config=model_spec.text_config,
        kv_cache_dtype="int8",
        recurrent_state_dtype="model",
    )

    report = runner.estimate_heterogeneous_cache_transfer_for_blocks(3, 8)
    plan = build_qwen35_cache_transfer_plan_from_spec(
        model_spec,
        src_tp_size=4,
        dst_tp_size=8,
        num_blocks=3,
        block_size=256,
        kv_dtype=torch.int8,
        recurrent_dtype=torch.bfloat16,
        convolution_dtype=torch.bfloat16,
    )

    assert report == plan.profile.to_dict()
    assert report["source_tp_size"] == 4
    assert report["destination_tp_size"] == 8
    assert report["wire_bytes"] == sum(report["source_egress_bytes"])
    assert report["wire_bytes"] == sum(report["destination_bytes"])
    assert sum(report["source_staging_bytes"]) < report["wire_bytes"]
    assert report["source_peer_counts"] == (2,) * 4
    assert report["destination_peer_counts"] == (1,) * 8


@pytest.mark.parametrize("src_tp_size,dst_tp_size", [(4, 8), (8, 4)])
@pytest.mark.parametrize(
    "kv_dtype,recurrent_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.int8, torch.bfloat16),
    ],
)
def test_heterogeneous_destination_bytes_match_destination_cache_layout(
    src_tp_size,
    dst_tp_size,
    kv_dtype,
    recurrent_dtype,
):
    model_spec = make_qwen36_model_spec()
    kwargs = {
        "num_blocks": 3,
        "block_size": 256,
        "kv_dtype": kv_dtype,
        "recurrent_dtype": recurrent_dtype,
        "convolution_dtype": torch.bfloat16,
    }

    heterogeneous = plan_qwen35_cache_transfer_capacity(
        model_spec,
        src_tp_size=src_tp_size,
        dst_tp_size=dst_tp_size,
        **kwargs,
    )
    destination_local = plan_qwen35_cache_transfer_capacity(
        model_spec,
        src_tp_size=dst_tp_size,
        dst_tp_size=dst_tp_size,
        **kwargs,
    )

    assert heterogeneous.destination_bytes == destination_local.source_staging_bytes
    assert heterogeneous.wire_bytes == destination_local.wire_bytes


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


def test_import_rejects_wrong_cache_fingerprint_before_modifying_destination():
    source = make_float_cache()
    payload = export_rank_cache(
        source,
        None,
        [1],
        transfer_id="request-fingerprint/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
        cache_fingerprint="source-model",
    )
    destination = make_float_cache(fill=False)

    with pytest.raises(ValueError, match="fingerprint does not match destination"):
        import_rank_cache(
            payload,
            destination,
            None,
            [0],
            transfer_id="request-fingerprint/attempt-1",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            block_size=2,
            cache_fingerprint="destination-model",
        )

    assert torch.count_nonzero(destination) == 0


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


def test_host_export_preserves_order_tail_scales_and_states():
    source = torch.arange(2 * 1 * 3 * 2 * 1 * 2, dtype=torch.int8).view(
        2, 1, 3, 2, 1, 2
    )
    scales = torch.arange(2 * 1 * 3 * 2 * 1, dtype=torch.float16).view(
        2, 1, 3, 2, 1
    )
    recurrent = (torch.arange(6, dtype=torch.float32).view(2, 3),)
    convolution = (torch.arange(8, dtype=torch.float16).view(4, 2),)

    payload = export_rank_cache(
        source,
        scales,
        [2, 0],
        transfer_id="request-host/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        recurrent_states=recurrent,
        convolution_states=convolution,
        to_host=True,
    )

    assert payload.kv_blocks.device.type == "cpu"
    assert payload.kv_scales is not None
    assert payload.kv_scales.device.type == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in payload.recurrent_states)
    assert all(tensor.device.type == "cpu" for tensor in payload.convolution_states)
    torch.testing.assert_close(payload.kv_blocks[:, :, 0], source[:, :, 2])
    torch.testing.assert_close(payload.kv_blocks[:, :, 1, :1], source[:, :, 0, :1])
    assert torch.count_nonzero(payload.kv_blocks[:, :, 1, 1:]) == 0
    torch.testing.assert_close(payload.kv_scales[:, :, 0], scales[:, :, 2])
    torch.testing.assert_close(payload.kv_scales[:, :, 1, :1], scales[:, :, 0, :1])
    assert torch.count_nonzero(payload.kv_scales[:, :, 1, 1:]) == 0
    torch.testing.assert_close(payload.recurrent_states[0], recurrent[0])
    torch.testing.assert_close(payload.convolution_states[0], convolution[0])
    staged_tensors = [
        payload.kv_blocks,
        payload.kv_scales,
        *payload.recurrent_states,
        *payload.convolution_states,
    ]
    assert len(
        {tensor.untyped_storage().data_ptr() for tensor in staged_tensors}
    ) == 1
    assert all(
        tensor.data_ptr() % tensor.element_size() == 0
        for tensor in staged_tensors
    )


def test_host_export_does_not_materialize_unused_block_index(monkeypatch):
    source = make_float_cache()

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host export must not allocate a block-index tensor")
        ),
    )

    payload = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-host-index/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        to_host=True,
    )

    torch.testing.assert_close(payload.kv_blocks[:, :, 0], source[:, :, 3])
    torch.testing.assert_close(
        payload.kv_blocks[:, :, 1, :1],
        source[:, :, 1, :1],
    )


def test_host_export_allocates_one_storage_for_complete_payload(monkeypatch):
    source = make_float_cache()
    recurrent, convolution = make_states()
    original_empty = torch.empty
    allocations = []

    def tracked_empty(*args, **kwargs):
        allocations.append((args, kwargs))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", tracked_empty)
    payload = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-host-storage/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        recurrent_states=recurrent,
        convolution_states=convolution,
        to_host=True,
    )

    assert len(allocations) == 1
    assert allocations[0][1]["dtype"] == torch.uint8
    assert payload.nbytes == 288


def test_host_export_reuses_released_staging_storage():
    source = make_float_cache()
    pool = HostStagingBufferPool()
    first = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-host-pool/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        to_host=True,
        host_staging_pool=pool,
    )
    first_ptr = first.kv_blocks.untyped_storage().data_ptr()
    first.release_host_staging()
    first.release_host_staging()

    second = export_rank_cache(
        source,
        None,
        [2],
        transfer_id="request-host-pool/attempt-2",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
        to_host=True,
        host_staging_pool=pool,
    )

    assert second.kv_blocks.untyped_storage().data_ptr() == first_ptr
    assert pool.storage_stats() == {
        "max_cached_bytes": None,
        "storage_bytes": first.nbytes,
        "allocation_count": 1,
        "reuse_count": 1,
        "transient_allocation_count": 0,
        "leased": 1,
    }
    second.release_host_staging()


def test_host_staging_pool_never_aliases_concurrent_payloads():
    source = make_float_cache()
    pool = HostStagingBufferPool()
    first = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-host-pool/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        to_host=True,
        host_staging_pool=pool,
    )
    second = export_rank_cache(
        source,
        None,
        [2, 0],
        transfer_id="request-host-pool/attempt-2",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        to_host=True,
        host_staging_pool=pool,
    )

    assert (
        first.kv_blocks.untyped_storage().data_ptr()
        != second.kv_blocks.untyped_storage().data_ptr()
    )
    assert pool.storage_stats()["transient_allocation_count"] == 1
    second.release_host_staging()
    first.release_host_staging()


def test_host_staging_pool_does_not_retain_oversized_allocation():
    pool = HostStagingBufferPool(max_cached_bytes=16)

    oversized = pool.acquire(17, pin_memory=False)
    assert pool.storage_stats() == {
        "max_cached_bytes": 16,
        "storage_bytes": 0,
        "allocation_count": 0,
        "reuse_count": 0,
        "transient_allocation_count": 1,
        "leased": 0,
    }
    oversized.release()

    cached = pool.acquire(16, pin_memory=False)
    assert pool.storage_stats()["storage_bytes"] == 16
    assert pool.storage_stats()["leased"] == 1
    cached.release()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_host_staging_pool_rejects_invalid_retention_limit(value):
    with pytest.raises(ValueError, match="max_cached_bytes"):
        HostStagingBufferPool(max_cached_bytes=value)


@pytest.mark.parametrize("block_ids", [[], [0, 0], [-1], [4]])
def test_host_export_validates_block_ids_without_materializing_index(block_ids):
    source = make_float_cache()

    with pytest.raises(ValueError, match="block"):
        export_rank_cache(
            source,
            None,
            block_ids,
            transfer_id="request-host-invalid/attempt-1",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            block_size=2,
            cached_tokens=max(len(block_ids), 1) * 2,
            to_host=True,
        )


def test_transfer_byte_estimate_matches_int8_payload_with_hybrid_state():
    source = torch.arange(2 * 1 * 3 * 2 * 1 * 2, dtype=torch.int8).view(
        2, 1, 3, 2, 1, 2
    )
    scales = torch.ones(2, 1, 3, 2, 1, dtype=torch.float16)
    recurrent = (torch.ones(2, 3, dtype=torch.float32),)
    convolution = (torch.ones(4, 2, dtype=torch.float16),)

    estimated = estimate_rank_cache_transfer_bytes(
        source,
        scales,
        [2, 0],
        recurrent_states=recurrent,
        convolution_states=convolution,
    )
    payload = export_rank_cache(
        source,
        scales,
        [2, 0],
        transfer_id="request-bytes/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        recurrent_states=recurrent,
        convolution_states=convolution,
        to_host=True,
    )

    assert estimated == payload.nbytes
    assert estimated == 72


def test_transfer_byte_estimate_matches_bf16_payload():
    source = torch.ones(2, 2, 4, 2, 1, 2, dtype=torch.bfloat16)

    estimated = estimate_rank_cache_transfer_bytes(source, None, [3])
    payload = export_rank_cache(
        source,
        None,
        [3],
        transfer_id="request-bytes/attempt-2",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
    )

    assert estimated == payload.nbytes == 32


def test_default_export_keeps_source_device_and_owns_storage():
    source = make_float_cache()
    payload = export_rank_cache(
        source,
        None,
        [1],
        transfer_id="request-default/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
    )

    assert payload.kv_blocks.device == source.device
    before = payload.kv_blocks.clone()
    source[:, :, 1].zero_()
    torch.testing.assert_close(payload.kv_blocks, before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transfer_id", 1),
        ("tensor_parallel_rank", False),
        ("tensor_parallel_size", True),
        ("block_size", True),
        ("cached_tokens", True),
    ],
)
def test_export_rejects_noncanonical_transfer_metadata(field, value):
    metadata = {
        "transfer_id": "request-metadata/attempt-1",
        "tensor_parallel_rank": 0,
        "tensor_parallel_size": 1,
        "block_size": 2,
        "cached_tokens": 2,
    }
    metadata[field] = value

    with pytest.raises(ValueError):
        export_rank_cache(
            make_float_cache(),
            None,
            [0],
            **metadata,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", True),
        ("transfer_id", 1),
        ("tensor_parallel_rank", False),
        ("tensor_parallel_size", True),
        ("block_size", True),
        ("cached_tokens", True),
    ],
)
def test_import_rejects_noncanonical_payload_metadata(field, value):
    source = make_float_cache()
    payload = export_rank_cache(
        source,
        None,
        [1],
        transfer_id="request-metadata/attempt-2",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=2,
    )
    destination = torch.full_like(source, -1)
    before = destination.clone()

    with pytest.raises(ValueError):
        import_rank_cache(
            replace(payload, **{field: value}),
            destination,
            None,
            [0],
            transfer_id="request-metadata/attempt-2",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            block_size=2,
        )

    torch.testing.assert_close(destination, before)


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
    source.cache_transfer_fingerprint = "same-model"
    destination.cache_transfer_fingerprint = "same-model"
    source_seq = SimpleNamespace(
        block_table=[3, 1],
        state_slot=1,
        num_cached_tokens=3,
        token_ids=[11, 22, 33, 44],
    )
    destination_seq = SimpleNamespace(
        block_table=[0, 2],
        state_slot=0,
        status=SequenceStatus.TRANSFERRING,
        num_prompt_tokens=3,
        num_cached_tokens=0,
        token_ids=[11, 22, 33],
    )

    payload = source.export_sequence_cache(
        source_seq,
        transfer_id="request-5/attempt-1",
    )
    assert source.estimate_sequence_cache_bytes(source_seq) == {
        "rank": 0,
        "staged_bytes": payload.nbytes,
    }
    estimated_before_reservation = (
        source.estimate_cache_transfer_bytes_for_blocks(2)
    )
    assert estimated_before_reservation == {
        "rank": 0,
        "staged_bytes": payload.nbytes,
    }
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


def test_model_runner_rejects_equal_length_wrong_prompt_before_install():
    source, _ = make_runner(make_float_cache(), 5.0)
    destination, _ = make_runner(make_float_cache(fill=False), 0.0)
    source.cache_transfer_fingerprint = "same-model"
    destination.cache_transfer_fingerprint = "same-model"
    source_seq = SimpleNamespace(
        block_table=[3, 1],
        state_slot=1,
        num_cached_tokens=3,
        token_ids=[11, 22, 33, 44],
    )
    destination_seq = SimpleNamespace(
        block_table=[0, 2],
        state_slot=0,
        status=SequenceStatus.TRANSFERRING,
        num_prompt_tokens=3,
        num_cached_tokens=0,
        token_ids=[11, 22, 34],
    )
    payload = source.export_sequence_cache(
        source_seq,
        transfer_id="request-token-identity/attempt-1",
    )

    with pytest.raises(ValueError, match="token fingerprint"):
        destination.import_sequence_cache(
            destination_seq,
            payload,
            transfer_id="request-token-identity/attempt-1",
        )

    assert torch.count_nonzero(destination.kv_cache) == 0


def test_model_runner_rejects_wrong_model_before_install():
    source, _ = make_runner(make_float_cache(), 5.0)
    destination, _ = make_runner(make_float_cache(fill=False), 0.0)
    source.cache_transfer_fingerprint = "source-model"
    destination.cache_transfer_fingerprint = "destination-model"
    source_seq = SimpleNamespace(
        block_table=[3, 1],
        state_slot=1,
        num_cached_tokens=3,
        token_ids=[11, 22, 33],
    )
    destination_seq = SimpleNamespace(
        block_table=[0, 2],
        state_slot=0,
        status=SequenceStatus.TRANSFERRING,
        num_prompt_tokens=3,
        num_cached_tokens=0,
        token_ids=[11, 22, 33],
    )
    payload = source.export_sequence_cache(
        source_seq,
        transfer_id="request-model-identity/attempt-1",
    )

    with pytest.raises(ValueError, match="fingerprint does not match destination"):
        destination.import_sequence_cache(
            destination_seq,
            payload,
            transfer_id="request-model-identity/attempt-1",
        )

    assert torch.count_nonzero(destination.kv_cache) == 0


def test_import_avoids_full_payload_device_conversion(monkeypatch):
    source = make_float_cache()
    recurrent, convolution = make_states()
    payload = export_rank_cache(
        source,
        None,
        [3, 1],
        transfer_id="request-direct-copy/attempt-1",
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        block_size=2,
        cached_tokens=3,
        recurrent_states=recurrent,
        convolution_states=convolution,
    )
    destination = make_float_cache(fill=False)
    destination_recurrent, destination_convolution = make_states(fill=False)

    with monkeypatch.context() as context:
        context.setattr(
            torch.Tensor,
            "to",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cache import must copy directly into destination")
            ),
        )
        import_rank_cache(
            payload,
            destination,
            None,
            [0, 2],
            transfer_id="request-direct-copy/attempt-1",
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
            block_size=2,
            recurrent_states=destination_recurrent,
            convolution_states=destination_convolution,
        )

    torch.testing.assert_close(destination[:, :, 0], payload.kv_blocks[:, :, 0])
    torch.testing.assert_close(destination[:, :, 2], payload.kv_blocks[:, :, 1])
    for actual, expected in zip(destination_recurrent, recurrent):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(destination_convolution, convolution):
        torch.testing.assert_close(actual, expected)


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


@pytest.mark.parametrize(
    "timeout_s",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
        "1",
        10**1000,
    ],
)
def test_transfer_session_rejects_invalid_timeout(timeout_s):
    with pytest.raises(ValueError, match="timeout must be a finite positive number"):
        CacheTransferSession(
            "request-invalid-timeout/attempt-1",
            2,
            started_at=10.0,
            timeout_s=timeout_s,
        )


@pytest.mark.parametrize("transfer_id", ["", 1, True, None, []])
def test_transfer_session_rejects_invalid_transfer_id(transfer_id):
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        CacheTransferSession(
            transfer_id,
            2,
            started_at=10.0,
            timeout_s=5.0,
        )


@pytest.mark.parametrize(
    "max_payload_bytes",
    [0, -1, 1.5, float("nan"), float("inf"), True, "1"],
)
def test_cache_transfer_rejects_invalid_payload_limit(max_payload_bytes):
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_cache_transfer_payload_limit(max_payload_bytes)


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
