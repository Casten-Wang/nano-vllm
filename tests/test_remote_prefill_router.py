import pytest

from nanovllm.engine.remote_prefill_router import (
    RemotePrefillDemand,
    rank_remote_prefill_destinations,
)


def snapshot(
    *,
    sequence_free=2,
    kv_free=12,
    transfer_free=2,
    staging_limit=1_000,
    staging_free=1_000,
    waiting=0,
    running=0,
):
    return {
        "waiting_requests": waiting,
        "running_requests": running,
        "sequence_slots_total": 4,
        "sequence_slots_free": sequence_free,
        "kv_blocks_total": 16,
        "kv_blocks_free": kv_free,
        "transfer_slots_total": 2,
        "transfer_slots_free": transfer_free,
        "staging_bytes_limit": staging_limit,
        "staging_bytes_free": staging_free,
    }


def test_router_rejects_nodes_that_cannot_admit_the_whole_request():
    candidates = {
        "no-sequence": snapshot(sequence_free=0),
        "no-kv": snapshot(kv_free=3),
        "no-transfer": snapshot(transfer_free=0),
        "no-staging": snapshot(staging_free=199),
        "fits": snapshot(kv_free=4, staging_free=200),
    }

    assert rank_remote_prefill_destinations(
        candidates,
        RemotePrefillDemand(kv_blocks=4, staging_bytes=200),
    ) == ("fits",)


def test_router_uses_post_placement_bottleneck_instead_of_raw_kv_capacity():
    candidates = {
        "kv-rich-transfer-tight": snapshot(kv_free=16, transfer_free=1),
        "balanced": snapshot(kv_free=12, transfer_free=2),
    }

    assert rank_remote_prefill_destinations(
        candidates,
        RemotePrefillDemand(kv_blocks=4, staging_bytes=200),
    ) == ("balanced", "kv-rich-transfer-tight")


def test_router_uses_queue_depth_and_then_input_order_as_tie_breakers():
    candidates = {
        "first": snapshot(waiting=1, running=2),
        "less-loaded": snapshot(waiting=0, running=2),
        "last": snapshot(waiting=1, running=2),
    }

    assert rank_remote_prefill_destinations(
        candidates,
        RemotePrefillDemand(kv_blocks=4, staging_bytes=200),
    ) == ("less-loaded", "first", "last")


def test_router_accepts_unbounded_staging_capacity():
    candidates = {
        "unbounded": snapshot(staging_limit=None, staging_free=None),
    }

    assert rank_remote_prefill_destinations(
        candidates,
        RemotePrefillDemand(kv_blocks=4, staging_bytes=10_000),
    ) == ("unbounded",)


def test_router_accepts_per_destination_demands_for_heterogeneous_nodes():
    candidates = {
        "small-blocks": snapshot(kv_free=4),
        "large-blocks": snapshot(kv_free=4),
    }
    demands = {
        "small-blocks": RemotePrefillDemand(kv_blocks=5, staging_bytes=200),
        "large-blocks": RemotePrefillDemand(kv_blocks=4, staging_bytes=200),
    }

    assert rank_remote_prefill_destinations(candidates, demands) == (
        "large-blocks",
    )


def test_router_requires_a_demand_for_every_candidate():
    with pytest.raises(ValueError, match="missing for 'decode-1'"):
        rank_remote_prefill_destinations(
            {"decode-1": snapshot()},
            {},
        )


@pytest.mark.parametrize(
    ("demand", "message"),
    [
        (dict(kv_blocks=0, staging_bytes=0), "kv_blocks"),
        (dict(kv_blocks=1, staging_bytes=-1), "staging_bytes"),
    ],
)
def test_demand_rejects_invalid_resources(demand, message):
    with pytest.raises(ValueError, match=message):
        RemotePrefillDemand(**demand)


def test_router_rejects_inconsistent_snapshots():
    invalid = snapshot(sequence_free=5)

    with pytest.raises(ValueError, match="sequence_slots_free"):
        rank_remote_prefill_destinations(
            {"broken": invalid},
            RemotePrefillDemand(kv_blocks=1, staging_bytes=0),
        )
