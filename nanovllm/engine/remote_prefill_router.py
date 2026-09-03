"""Capacity-aware destination selection for disaggregated prefill."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction


CapacityValue = int | float | None
CapacitySnapshot = Mapping[str, CapacityValue]


@dataclass(frozen=True, slots=True)
class RemotePrefillDemand:
    """Decode-side resources reserved by one remote-prefill request."""

    kv_blocks: int
    staging_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kv_blocks, int)
            or isinstance(self.kv_blocks, bool)
            or self.kv_blocks <= 0
        ):
            raise ValueError("kv_blocks must be a positive integer")
        if (
            not isinstance(self.staging_bytes, int)
            or isinstance(self.staging_bytes, bool)
            or self.staging_bytes < 0
        ):
            raise ValueError("staging_bytes must be a non-negative integer")


def _capacity(snapshot: CapacitySnapshot, name: str, *, positive: bool) -> int:
    value = snapshot.get(name)
    lower_bound = 1 if positive else 0
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < lower_bound
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _headroom_score(
    snapshot: CapacitySnapshot,
    demand: RemotePrefillDemand,
) -> tuple[Fraction, Fraction, int, int] | None:
    sequence_total = _capacity(snapshot, "sequence_slots_total", positive=True)
    sequence_free = _capacity(snapshot, "sequence_slots_free", positive=False)
    kv_total = _capacity(snapshot, "kv_blocks_total", positive=True)
    kv_free = _capacity(snapshot, "kv_blocks_free", positive=False)
    transfer_total = _capacity(snapshot, "transfer_slots_total", positive=True)
    transfer_free = _capacity(snapshot, "transfer_slots_free", positive=False)
    waiting = _capacity(snapshot, "waiting_requests", positive=False)
    running = _capacity(snapshot, "running_requests", positive=False)

    if sequence_free > sequence_total:
        raise ValueError("sequence_slots_free exceeds sequence_slots_total")
    if kv_free > kv_total:
        raise ValueError("kv_blocks_free exceeds kv_blocks_total")
    if transfer_free > transfer_total:
        raise ValueError("transfer_slots_free exceeds transfer_slots_total")

    staging_limit = snapshot.get("staging_bytes_limit")
    staging_free = snapshot.get("staging_bytes_free")
    if staging_limit is None:
        if staging_free is not None:
            raise ValueError("staging_bytes_free must be None for an unbounded limit")
        staging_headroom = Fraction(1)
    else:
        if (
            not isinstance(staging_limit, int)
            or isinstance(staging_limit, bool)
            or staging_limit <= 0
        ):
            raise ValueError("staging_bytes_limit must be a positive integer or None")
        if (
            not isinstance(staging_free, int)
            or isinstance(staging_free, bool)
            or staging_free < 0
            or staging_free > staging_limit
        ):
            raise ValueError(
                "staging_bytes_free must be between zero and staging_bytes_limit"
            )
        if staging_free < demand.staging_bytes:
            return None
        staging_headroom = Fraction(
            staging_free - demand.staging_bytes,
            staging_limit,
        )

    if sequence_free < 1 or kv_free < demand.kv_blocks or transfer_free < 1:
        return None

    headrooms = (
        Fraction(sequence_free - 1, sequence_total),
        Fraction(kv_free - demand.kv_blocks, kv_total),
        Fraction(transfer_free - 1, transfer_total),
        staging_headroom,
    )
    return min(headrooms), sum(headrooms), -waiting, -running


def rank_remote_prefill_destinations(
    candidates: Mapping[str, CapacitySnapshot],
    demand: RemotePrefillDemand,
) -> tuple[str, ...]:
    """Rank admissible decode nodes by post-placement capacity headroom.

    The first result is preferred. Remaining results are deterministic fallback
    destinations if a concurrent reservation makes an earlier snapshot stale.
    Ties preserve caller order so a controller may rotate input order without
    putting round-robin state in this policy.
    """

    scored: list[tuple[tuple[Fraction, Fraction, int, int], int, str]] = []
    for position, (destination, snapshot) in enumerate(candidates.items()):
        score = _headroom_score(snapshot, demand)
        if score is not None:
            scored.append((score, -position, destination))
    scored.sort(reverse=True)
    return tuple(destination for _score, _position, destination in scored)
