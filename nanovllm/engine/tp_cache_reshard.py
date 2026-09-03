"""CPU-reference tensor-parallel cache re-sharding primitives.

These helpers define layout correctness for heterogeneous-TP prefill/decode
handoff. They intentionally reconstruct a logical tensor before splitting it;
the production transport may later replace that work with direct peer slices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch

from nanovllm.engine.cache_transfer import (
    TRANSFER_FORMAT_VERSION,
    CacheTransferPhase,
    RankCacheTransfer,
)


@dataclass(frozen=True, slots=True)
class TPTransferSlice:
    """One direct copy along a tensor's sharded dimension."""

    src_rank: int
    dst_rank: int
    src_start: int
    dst_start: int
    length: int

    def __post_init__(self) -> None:
        for name, value in (
            ("src_rank", self.src_rank),
            ("dst_rank", self.dst_rank),
            ("src_start", self.src_start),
            ("dst_start", self.dst_start),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.length, int)
            or isinstance(self.length, bool)
            or self.length <= 0
        ):
            raise ValueError("length must be a positive integer")


@dataclass(frozen=True, slots=True)
class TPTransferProfile:
    """Per-rank pressure produced by a direct tensor transfer plan."""

    wire_bytes: int
    source_bytes: tuple[int, ...]
    source_staging_bytes: tuple[int, ...]
    destination_bytes: tuple[int, ...]
    source_peer_counts: tuple[int, ...]
    destination_peer_counts: tuple[int, ...]
    peer_bytes: tuple[tuple[int, int, int], ...]
    slice_count: int

    def __post_init__(self) -> None:
        if not self.source_bytes or not self.destination_bytes:
            raise ValueError("transfer profile must include source and destination ranks")
        integer_fields = (
            self.wire_bytes,
            self.slice_count,
            *self.source_bytes,
            *self.source_staging_bytes,
            *self.destination_bytes,
            *self.source_peer_counts,
            *self.destination_peer_counts,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in integer_fields
        ):
            raise ValueError("transfer profile counters must be non-negative integers")
        if self.wire_bytes <= 0 or self.slice_count <= 0:
            raise ValueError("transfer profile wire bytes and slice count must be positive")
        if (
            len(self.source_bytes) != len(self.source_peer_counts)
            or len(self.source_bytes) != len(self.source_staging_bytes)
            or len(self.destination_bytes) != len(self.destination_peer_counts)
        ):
            raise ValueError("transfer profile rank vectors have inconsistent lengths")
        if (
            sum(self.source_bytes) != self.wire_bytes
            or sum(self.destination_bytes) != self.wire_bytes
        ):
            raise ValueError("transfer profile rank bytes do not match wire bytes")
        if any(byte_count <= 0 for byte_count in self.destination_bytes):
            raise ValueError("every destination rank must receive transfer bytes")
        if any(
            staged < 0 or staged > transmitted
            for staged, transmitted in zip(
                self.source_staging_bytes,
                self.source_bytes,
            )
        ):
            raise ValueError("source staging bytes exceed transmitted bytes")

        peer_map = {}
        source_peers = [set() for _ in self.source_bytes]
        destination_peers = [set() for _ in self.destination_bytes]
        for peer in self.peer_bytes:
            if (
                not isinstance(peer, tuple)
                or len(peer) != 3
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in peer
                )
            ):
                raise ValueError("transfer profile peer bytes are invalid")
            src_rank, dst_rank, byte_count = peer
            if (
                not 0 <= src_rank < len(self.source_bytes)
                or not 0 <= dst_rank < len(self.destination_bytes)
                or byte_count <= 0
                or (src_rank, dst_rank) in peer_map
            ):
                raise ValueError("transfer profile peer bytes are invalid")
            peer_map[src_rank, dst_rank] = byte_count
            source_peers[src_rank].add(dst_rank)
            destination_peers[dst_rank].add(src_rank)
        if sum(peer_map.values()) != self.wire_bytes:
            raise ValueError("transfer profile peer bytes do not match wire bytes")
        source_peer_bytes = [0] * len(self.source_bytes)
        destination_peer_bytes = [0] * len(self.destination_bytes)
        for (src_rank, dst_rank), byte_count in peer_map.items():
            source_peer_bytes[src_rank] += byte_count
            destination_peer_bytes[dst_rank] += byte_count
        if tuple(source_peer_bytes) != self.source_bytes:
            raise ValueError("transfer profile source bytes do not match peers")
        if tuple(destination_peer_bytes) != self.destination_bytes:
            raise ValueError("transfer profile destination bytes do not match peers")
        if tuple(map(len, source_peers)) != self.source_peer_counts:
            raise ValueError("transfer profile source peer counts do not match")
        if tuple(map(len, destination_peers)) != self.destination_peer_counts:
            raise ValueError("transfer profile destination peer counts do not match")

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-safe capacity and topology report."""

        return {
            "source_tp_size": len(self.source_bytes),
            "destination_tp_size": len(self.destination_bytes),
            "wire_bytes": self.wire_bytes,
            "source_egress_bytes": self.source_bytes,
            "source_staging_bytes": self.source_staging_bytes,
            "destination_bytes": self.destination_bytes,
            "source_peer_counts": self.source_peer_counts,
            "destination_peer_counts": self.destination_peer_counts,
            "peer_bytes": self.peer_bytes,
            "slice_count": self.slice_count,
        }

    @classmethod
    def from_dict(cls, report: Mapping[str, object]) -> "TPTransferProfile":
        """Rebuild and fully validate a control-plane transfer report."""

        if not isinstance(report, Mapping):
            raise ValueError("transfer profile report must be a mapping")

        def rank_vector(name: str) -> tuple[int, ...]:
            values = report.get(name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"{name} must be a non-empty rank vector")
            return tuple(values)

        def counter(name: str) -> int:
            value = report.get(name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
            return value

        raw_peers = report.get("peer_bytes")
        if not isinstance(raw_peers, (tuple, list)) or not raw_peers:
            raise ValueError("peer_bytes must be a non-empty peer vector")
        peers = []
        for peer in raw_peers:
            if not isinstance(peer, (tuple, list)) or len(peer) != 3:
                raise ValueError("peer_bytes entries must contain source, destination, bytes")
            peers.append(tuple(peer))

        profile = cls(
            wire_bytes=counter("wire_bytes"),
            source_bytes=rank_vector("source_egress_bytes"),
            source_staging_bytes=rank_vector("source_staging_bytes"),
            destination_bytes=rank_vector("destination_bytes"),
            source_peer_counts=rank_vector("source_peer_counts"),
            destination_peer_counts=rank_vector("destination_peer_counts"),
            peer_bytes=tuple(peers),
            slice_count=counter("slice_count"),
        )
        for name, actual in (
            ("source_tp_size", len(profile.source_bytes)),
            ("destination_tp_size", len(profile.destination_bytes)),
        ):
            declared = report.get(name)
            if (
                not isinstance(declared, int)
                or isinstance(declared, bool)
                or declared != actual
            ):
                raise ValueError(f"{name} does not match the rank vectors")
        return profile


class TPPeerTransferSession:
    """Track destination-installed bytes for one heterogeneous-TP request."""

    def __init__(
        self,
        transfer_id: str,
        profile: TPTransferProfile,
        *,
        started_at: float,
        timeout_s: float,
    ) -> None:
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("peer transfer id must not be empty")
        if not isinstance(profile, TPTransferProfile):
            raise ValueError("peer transfer profile is invalid")
        validated_started_at = self._validated_time(started_at, "started_at")
        validated_timeout_s = self._validated_time(
            timeout_s,
            "timeout_s",
            positive=True,
        )
        self.transfer_id = transfer_id
        self.profile = profile
        self.deadline = validated_started_at + validated_timeout_s
        if not math.isfinite(self.deadline):
            raise ValueError("peer transfer deadline must be finite")
        self.phase = CacheTransferPhase.RECEIVING
        self.failure_reason: str | None = None
        self._expected_peer_bytes = {
            (src_rank, dst_rank): byte_count
            for src_rank, dst_rank, byte_count in profile.peer_bytes
        }
        self._acknowledged_peers: set[tuple[int, int]] = set()

    @staticmethod
    def _validated_time(
        value: float,
        name: str,
        *,
        positive: bool = False,
    ) -> float:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        try:
            numeric = float(value) if valid_type else math.nan
        except (OverflowError, ValueError):
            numeric = math.nan
        if (
            not math.isfinite(numeric)
            or (positive and numeric <= 0)
        ):
            qualifier = "a positive finite number" if positive else "finite"
            raise ValueError(f"{name} must be {qualifier}")
        return numeric

    @property
    def fallback_required(self) -> bool:
        return self.phase in {
            CacheTransferPhase.ABORTED,
            CacheTransferPhase.TIMED_OUT,
        }

    @property
    def acknowledged_bytes(self) -> int:
        return sum(
            self._expected_peer_bytes[peer]
            for peer in self._acknowledged_peers
        )

    @property
    def pending_peer_bytes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (src_rank, dst_rank, byte_count)
            for (src_rank, dst_rank), byte_count in sorted(
                self._expected_peer_bytes.items()
            )
            if (src_rank, dst_rank) not in self._acknowledged_peers
        )

    @property
    def ready_destination_ranks(self) -> tuple[int, ...]:
        ready = []
        for dst_rank in range(len(self.profile.destination_bytes)):
            expected = {
                peer for peer in self._expected_peer_bytes if peer[1] == dst_rank
            }
            if expected and expected <= self._acknowledged_peers:
                ready.append(dst_rank)
        return tuple(ready)

    def _expire(self, now: float) -> None:
        validated_now = self._validated_time(now, "now")
        if (
            self.phase in {CacheTransferPhase.RECEIVING, CacheTransferPhase.READY}
            and validated_now >= self.deadline
        ):
            self.phase = CacheTransferPhase.TIMED_OUT
            self.failure_reason = "peer cache transfer timed out"

    @staticmethod
    def _peer(src_rank: int, dst_rank: int) -> tuple[int, int]:
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in (src_rank, dst_rank)
        ):
            raise ValueError("peer cache transfer rank is invalid")
        return src_rank, dst_rank

    def acknowledge(
        self,
        src_rank: int,
        dst_rank: int,
        byte_count: int,
        *,
        now: float,
    ) -> None:
        self._expire(now)
        if self.phase not in {
            CacheTransferPhase.RECEIVING,
            CacheTransferPhase.READY,
        }:
            raise RuntimeError("peer cache transfer session is already terminal")
        peer = self._peer(src_rank, dst_rank)
        expected = self._expected_peer_bytes.get(peer)
        if expected is None:
            raise ValueError("peer cache transfer acknowledgement is unexpected")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count != expected
        ):
            raise ValueError("peer cache transfer byte count does not match")
        self._acknowledged_peers.add(peer)
        if len(self._acknowledged_peers) == len(self._expected_peer_bytes):
            self.phase = CacheTransferPhase.READY

    def fail(
        self,
        src_rank: int,
        dst_rank: int,
        reason: str,
        *,
        now: float,
    ) -> None:
        self._expire(now)
        if self.phase not in {
            CacheTransferPhase.RECEIVING,
            CacheTransferPhase.READY,
        }:
            raise RuntimeError("peer cache transfer session is already terminal")
        peer = self._peer(src_rank, dst_rank)
        if peer not in self._expected_peer_bytes:
            raise ValueError("peer cache transfer failure is unexpected")
        if not isinstance(reason, str) or not reason:
            raise ValueError("peer cache transfer failure reason must not be empty")
        self.phase = CacheTransferPhase.ABORTED
        self.failure_reason = f"source {src_rank} -> destination {dst_rank}: {reason}"

    def commit(self, *, now: float) -> None:
        self._expire(now)
        if self.phase is not CacheTransferPhase.READY:
            raise RuntimeError("peer cache transfer is not ready to commit")
        self.phase = CacheTransferPhase.COMMITTED

    def poll(self, *, now: float) -> CacheTransferPhase:
        self._expire(now)
        return self.phase


def _validate_tp_size(tp_size: int) -> None:
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size <= 0:
        raise ValueError("destination TP size must be a positive integer")


def _validate_shards(
    shards: Sequence[torch.Tensor],
    *,
    shard_dim: int,
) -> int:
    if not shards:
        raise ValueError("tensor-parallel re-sharding requires source shards")
    first = shards[0]
    if not isinstance(first, torch.Tensor) or first.ndim == 0:
        raise ValueError("tensor-parallel shards must be non-scalar tensors")
    if first.device.type != "cpu":
        raise ValueError("reference re-sharding accepts CPU tensors only")
    if (
        not isinstance(shard_dim, int)
        or isinstance(shard_dim, bool)
        or not -first.ndim <= shard_dim < first.ndim
    ):
        raise ValueError("sharded dimension is invalid for source tensors")
    normalized_dim = shard_dim % first.ndim
    reference_shape = list(first.shape)
    reference_shape[normalized_dim] = -1
    reference_width = first.shape[normalized_dim]
    if reference_width <= 0:
        raise ValueError("tensor-parallel shards must not be empty")
    for shard in shards:
        if not isinstance(shard, torch.Tensor) or shard.ndim != first.ndim:
            raise ValueError("tensor-parallel shards must have equal rank")
        shape = list(shard.shape)
        shape[normalized_dim] = -1
        if shape != reference_shape:
            raise ValueError(
                "tensor-parallel shard shapes differ outside the sharded dimension"
            )
        if shard.shape[normalized_dim] != reference_width:
            raise ValueError("tensor-parallel source shards must be equal width")
        if shard.dtype != first.dtype or shard.device != first.device:
            raise ValueError(
                "tensor-parallel shards must share dtype and device"
            )
    return normalized_dim


def reshard_uniform_tensor(
    shards: Sequence[torch.Tensor],
    dst_tp_size: int,
    *,
    shard_dim: int,
) -> tuple[torch.Tensor, ...]:
    """Reconstruct and evenly split a tensor sharded along one dimension."""

    _validate_tp_size(dst_tp_size)
    normalized_dim = _validate_shards(shards, shard_dim=shard_dim)
    global_size = sum(shard.shape[normalized_dim] for shard in shards)
    if global_size % dst_tp_size:
        raise ValueError(
            "global sharded dimension must divide destination TP size"
        )
    logical = torch.cat(tuple(shards), dim=normalized_dim)
    width = global_size // dst_tp_size
    return tuple(
        logical.narrow(normalized_dim, rank * width, width).clone()
        for rank in range(dst_tp_size)
    )


def _plan_uniform_group(
    *,
    global_width: int,
    src_tp_size: int,
    dst_tp_size: int,
    src_base: int,
    dst_base: int,
) -> list[TPTransferSlice]:
    if global_width % src_tp_size or global_width % dst_tp_size:
        raise ValueError("global group width must divide both TP sizes")
    src_width = global_width // src_tp_size
    dst_width = global_width // dst_tp_size
    slices = []
    for src_rank in range(src_tp_size):
        src_global_start = src_rank * src_width
        src_global_end = src_global_start + src_width
        for dst_rank in range(dst_tp_size):
            dst_global_start = dst_rank * dst_width
            dst_global_end = dst_global_start + dst_width
            overlap_start = max(src_global_start, dst_global_start)
            overlap_end = min(src_global_end, dst_global_end)
            if overlap_start >= overlap_end:
                continue
            slices.append(
                TPTransferSlice(
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    src_start=src_base + overlap_start - src_global_start,
                    dst_start=dst_base + overlap_start - dst_global_start,
                    length=overlap_end - overlap_start,
                )
            )
    return slices


def plan_uniform_reshard(
    global_width: int,
    src_tp_size: int,
    dst_tp_size: int,
) -> tuple[TPTransferSlice, ...]:
    """Plan direct copies for one evenly sharded logical dimension."""

    _validate_tp_size(src_tp_size)
    _validate_tp_size(dst_tp_size)
    if (
        not isinstance(global_width, int)
        or isinstance(global_width, bool)
        or global_width <= 0
    ):
        raise ValueError("global width must be a positive integer")
    return tuple(
        _plan_uniform_group(
            global_width=global_width,
            src_tp_size=src_tp_size,
            dst_tp_size=dst_tp_size,
            src_base=0,
            dst_base=0,
        )
    )


def plan_grouped_uniform_reshard(
    global_group_widths: Sequence[int],
    src_tp_size: int,
    dst_tp_size: int,
) -> tuple[TPTransferSlice, ...]:
    """Plan independently sharded packed groups such as GDN ``Q|K|V``."""

    _validate_tp_size(src_tp_size)
    _validate_tp_size(dst_tp_size)
    if not global_group_widths:
        raise ValueError("at least one packed group is required")
    slices = []
    src_base = 0
    dst_base = 0
    for global_width in global_group_widths:
        if (
            not isinstance(global_width, int)
            or isinstance(global_width, bool)
            or global_width <= 0
        ):
            raise ValueError("global group widths must be positive integers")
        slices.extend(
            _plan_uniform_group(
                global_width=global_width,
                src_tp_size=src_tp_size,
                dst_tp_size=dst_tp_size,
                src_base=src_base,
                dst_base=dst_base,
            )
        )
        src_base += global_width // src_tp_size
        dst_base += global_width // dst_tp_size
    return tuple(slices)


def _kv_head_topology(total_kv_heads: int, tp_size: int) -> tuple[int, int]:
    if (
        not isinstance(total_kv_heads, int)
        or isinstance(total_kv_heads, bool)
        or total_kv_heads <= 0
    ):
        raise ValueError("total KV heads must be a positive integer")
    if total_kv_heads >= tp_size:
        if total_kv_heads % tp_size:
            raise ValueError("total KV heads must divide across TP ranks")
        return total_kv_heads // tp_size, 1
    if tp_size % total_kv_heads:
        raise ValueError("TP ranks must divide across replicated KV heads")
    return 1, tp_size // total_kv_heads


def plan_kv_head_reshard(
    total_kv_heads: int,
    src_tp_size: int,
    dst_tp_size: int,
) -> tuple[TPTransferSlice, ...]:
    """Plan direct KV-head copies while balancing equivalent source replicas."""

    _validate_tp_size(src_tp_size)
    _validate_tp_size(dst_tp_size)
    src_local_heads, src_replication = _kv_head_topology(
        total_kv_heads,
        src_tp_size,
    )
    dst_local_heads, dst_replication = _kv_head_topology(
        total_kv_heads,
        dst_tp_size,
    )
    slices = []
    for head_index in range(total_kv_heads):
        if dst_replication == 1:
            destinations = ((head_index // dst_local_heads, head_index % dst_local_heads),)
        else:
            destinations = tuple(
                (head_index * dst_replication + replica, 0)
                for replica in range(dst_replication)
            )
        for destination_index, (dst_rank, dst_start) in enumerate(destinations):
            if src_replication == 1:
                src_rank = head_index // src_local_heads
                src_start = head_index % src_local_heads
            else:
                # Replicated source heads are interchangeable. Spread fan-out
                # across contiguous destination groups. This both avoids a
                # per-head hotspot and aligns peer edges with uniformly
                # sharded recurrent/QKV state.
                src_rank = (
                    head_index * src_replication
                    + destination_index * src_replication // dst_replication
                )
                src_start = 0
            slices.append(
                TPTransferSlice(
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    src_start=src_start,
                    dst_start=dst_start,
                    length=1,
                )
            )
    return tuple(slices)


def _validate_transfer_layout(
    plan: Sequence[TPTransferSlice],
    source_widths: Sequence[int],
    dst_tp_size: int,
    *,
    dst_width: int,
) -> None:
    _validate_tp_size(dst_tp_size)
    if not source_widths or any(
        not isinstance(width, int) or isinstance(width, bool) or width <= 0
        for width in source_widths
    ):
        raise ValueError("source widths must be positive integers")
    if not isinstance(dst_width, int) or isinstance(dst_width, bool) or dst_width <= 0:
        raise ValueError("destination width must be a positive integer")
    if not plan:
        raise ValueError("tensor-parallel transfer plan must not be empty")

    covered = [[False] * dst_width for _ in range(dst_tp_size)]
    for entry in plan:
        if not isinstance(entry, TPTransferSlice):
            raise ValueError("tensor-parallel transfer plan entry is invalid")
        if entry.src_rank >= len(source_widths) or entry.dst_rank >= dst_tp_size:
            raise ValueError("tensor-parallel transfer rank is out of bounds")
        if entry.src_start + entry.length > source_widths[entry.src_rank]:
            raise ValueError("tensor-parallel source slice is out of bounds")
        if entry.dst_start + entry.length > dst_width:
            raise ValueError("tensor-parallel destination slice is out of bounds")
        destination_coverage = covered[entry.dst_rank]
        if any(destination_coverage[entry.dst_start : entry.dst_start + entry.length]):
            raise ValueError("tensor-parallel transfer plan has overlapping writes")
        destination_coverage[entry.dst_start : entry.dst_start + entry.length] = (
            [True] * entry.length
        )
    if any(not all(destination_coverage) for destination_coverage in covered):
        raise ValueError("tensor-parallel transfer plan leaves destination gaps")


def _validate_transfer_plan(
    shards: Sequence[torch.Tensor],
    plan: Sequence[TPTransferSlice],
    dst_tp_size: int,
    *,
    shard_dim: int,
    dst_width: int,
) -> int:
    normalized_dim = _validate_shards(shards, shard_dim=shard_dim)
    _validate_transfer_layout(
        plan,
        tuple(shard.shape[normalized_dim] for shard in shards),
        dst_tp_size,
        dst_width=dst_width,
    )
    return normalized_dim


def _profile_validated_transfer_layout(
    plan: Sequence[TPTransferSlice],
    source_widths: Sequence[int],
    dst_tp_size: int,
    *,
    bytes_per_dim: int,
) -> TPTransferProfile:
    source_bytes = [0] * len(source_widths)
    destination_bytes = [0] * dst_tp_size
    source_peers = [set() for _ in source_widths]
    destination_peers = [set() for _ in range(dst_tp_size)]
    source_coverage = [[False] * width for width in source_widths]
    peer_bytes = {}
    for entry in plan:
        transfer_bytes = entry.length * bytes_per_dim
        source_bytes[entry.src_rank] += transfer_bytes
        destination_bytes[entry.dst_rank] += transfer_bytes
        source_peers[entry.src_rank].add(entry.dst_rank)
        destination_peers[entry.dst_rank].add(entry.src_rank)
        peer = (entry.src_rank, entry.dst_rank)
        peer_bytes[peer] = peer_bytes.get(peer, 0) + transfer_bytes
        source_coverage[entry.src_rank][
            entry.src_start : entry.src_start + entry.length
        ] = [True] * entry.length
    return TPTransferProfile(
        wire_bytes=sum(source_bytes),
        source_bytes=tuple(source_bytes),
        source_staging_bytes=tuple(
            sum(covered) * bytes_per_dim for covered in source_coverage
        ),
        destination_bytes=tuple(destination_bytes),
        source_peer_counts=tuple(len(peers) for peers in source_peers),
        destination_peer_counts=tuple(len(peers) for peers in destination_peers),
        peer_bytes=tuple(
            (src_rank, dst_rank, byte_count)
            for (src_rank, dst_rank), byte_count in sorted(peer_bytes.items())
        ),
        slice_count=len(plan),
    )


def profile_tp_transfer_layout(
    plan: Sequence[TPTransferSlice],
    src_tp_size: int,
    dst_tp_size: int,
    *,
    src_width: int,
    dst_width: int,
    bytes_per_dim: int,
) -> TPTransferProfile:
    """Estimate exact peer traffic from shapes without allocating tensors."""

    _validate_tp_size(src_tp_size)
    if (
        not isinstance(src_width, int)
        or isinstance(src_width, bool)
        or src_width <= 0
    ):
        raise ValueError("source width must be a positive integer")
    if (
        not isinstance(bytes_per_dim, int)
        or isinstance(bytes_per_dim, bool)
        or bytes_per_dim <= 0
    ):
        raise ValueError("bytes per dimension must be a positive integer")
    _validate_transfer_layout(
        plan,
        (src_width,) * src_tp_size,
        dst_tp_size,
        dst_width=dst_width,
    )
    return _profile_validated_transfer_layout(
        plan,
        (src_width,) * src_tp_size,
        dst_tp_size,
        bytes_per_dim=bytes_per_dim,
    )


def profile_tp_transfer_plan(
    shards: Sequence[torch.Tensor],
    plan: Sequence[TPTransferSlice],
    dst_tp_size: int,
    *,
    shard_dim: int,
    dst_width: int,
) -> TPTransferProfile:
    """Calculate exact tensor bytes and peer fan-out without copying data."""

    normalized_dim = _validate_transfer_plan(
        shards,
        plan,
        dst_tp_size,
        shard_dim=shard_dim,
        dst_width=dst_width,
    )
    bytes_per_dim = (
        shards[0].numel()
        // shards[0].shape[normalized_dim]
        * shards[0].element_size()
    )
    return _profile_validated_transfer_layout(
        plan,
        tuple(shard.shape[normalized_dim] for shard in shards),
        dst_tp_size,
        bytes_per_dim=bytes_per_dim,
    )


def aggregate_tp_transfer_profiles(
    profiles: Sequence[TPTransferProfile],
) -> TPTransferProfile:
    """Combine tensor profiles into one request-level capacity ledger."""

    if not profiles:
        raise ValueError("at least one tensor transfer profile is required")
    if any(not isinstance(profile, TPTransferProfile) for profile in profiles):
        raise ValueError("tensor transfer profile is invalid")
    src_tp_size = len(profiles[0].source_bytes)
    dst_tp_size = len(profiles[0].destination_bytes)
    if any(
        len(profile.source_bytes) != src_tp_size
        or len(profile.destination_bytes) != dst_tp_size
        for profile in profiles
    ):
        raise ValueError("tensor transfer profiles use different TP topologies")

    source_bytes = [0] * src_tp_size
    source_staging_bytes = [0] * src_tp_size
    destination_bytes = [0] * dst_tp_size
    peer_bytes = {}
    for profile in profiles:
        for rank, byte_count in enumerate(profile.source_bytes):
            source_bytes[rank] += byte_count
        for rank, byte_count in enumerate(profile.source_staging_bytes):
            source_staging_bytes[rank] += byte_count
        for rank, byte_count in enumerate(profile.destination_bytes):
            destination_bytes[rank] += byte_count
        for src_rank, dst_rank, byte_count in profile.peer_bytes:
            peer = (src_rank, dst_rank)
            peer_bytes[peer] = peer_bytes.get(peer, 0) + byte_count

    source_peers = [set() for _ in range(src_tp_size)]
    destination_peers = [set() for _ in range(dst_tp_size)]
    for src_rank, dst_rank in peer_bytes:
        source_peers[src_rank].add(dst_rank)
        destination_peers[dst_rank].add(src_rank)
    return TPTransferProfile(
        wire_bytes=sum(profile.wire_bytes for profile in profiles),
        source_bytes=tuple(source_bytes),
        source_staging_bytes=tuple(source_staging_bytes),
        destination_bytes=tuple(destination_bytes),
        source_peer_counts=tuple(len(peers) for peers in source_peers),
        destination_peer_counts=tuple(len(peers) for peers in destination_peers),
        peer_bytes=tuple(
            (src_rank, dst_rank, byte_count)
            for (src_rank, dst_rank), byte_count in sorted(peer_bytes.items())
        ),
        slice_count=sum(profile.slice_count for profile in profiles),
    )


def profile_qwen35_cache_transfer_layout(
    *,
    src_tp_size: int,
    dst_tp_size: int,
    total_kv_heads: int,
    kv_bytes_per_head: int,
    kv_scale_bytes_per_head: int,
    recurrent_heads: int,
    recurrent_bytes_per_head: int,
    convolution_group_widths: tuple[int, int, int],
    convolution_bytes_per_channel: int,
) -> TPTransferProfile:
    """Preflight a complete Qwen3.6 hybrid-cache transfer without tensors."""

    _validate_tp_size(src_tp_size)
    _validate_tp_size(dst_tp_size)
    positive_values = (
        ("kv_bytes_per_head", kv_bytes_per_head),
        ("recurrent_heads", recurrent_heads),
        ("recurrent_bytes_per_head", recurrent_bytes_per_head),
        ("convolution_bytes_per_channel", convolution_bytes_per_channel),
    )
    for name, value in positive_values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(kv_scale_bytes_per_head, int)
        or isinstance(kv_scale_bytes_per_head, bool)
        or kv_scale_bytes_per_head < 0
    ):
        raise ValueError("kv_scale_bytes_per_head must be a non-negative integer")
    if (
        not isinstance(convolution_group_widths, tuple)
        or len(convolution_group_widths) != 3
    ):
        raise ValueError("convolution group widths must contain Q, K, and V")

    src_kv_heads, _ = _kv_head_topology(total_kv_heads, src_tp_size)
    dst_kv_heads, _ = _kv_head_topology(total_kv_heads, dst_tp_size)
    kv_plan = plan_kv_head_reshard(
        total_kv_heads,
        src_tp_size,
        dst_tp_size,
    )
    profiles = [
        profile_tp_transfer_layout(
            kv_plan,
            src_tp_size,
            dst_tp_size,
            src_width=src_kv_heads,
            dst_width=dst_kv_heads,
            bytes_per_dim=kv_bytes_per_head,
        )
    ]
    if kv_scale_bytes_per_head:
        profiles.append(
            profile_tp_transfer_layout(
                kv_plan,
                src_tp_size,
                dst_tp_size,
                src_width=src_kv_heads,
                dst_width=dst_kv_heads,
                bytes_per_dim=kv_scale_bytes_per_head,
            )
        )

    recurrent_plan = plan_uniform_reshard(
        recurrent_heads,
        src_tp_size,
        dst_tp_size,
    )
    profiles.append(
        profile_tp_transfer_layout(
            recurrent_plan,
            src_tp_size,
            dst_tp_size,
            src_width=recurrent_heads // src_tp_size,
            dst_width=recurrent_heads // dst_tp_size,
            bytes_per_dim=recurrent_bytes_per_head,
        )
    )

    convolution_plan = plan_grouped_uniform_reshard(
        convolution_group_widths,
        src_tp_size,
        dst_tp_size,
    )
    profiles.append(
        profile_tp_transfer_layout(
            convolution_plan,
            src_tp_size,
            dst_tp_size,
            src_width=sum(convolution_group_widths) // src_tp_size,
            dst_width=sum(convolution_group_widths) // dst_tp_size,
            bytes_per_dim=convolution_bytes_per_channel,
        )
    )
    return aggregate_tp_transfer_profiles(profiles)


def apply_tp_transfer_plan(
    shards: Sequence[torch.Tensor],
    plan: Sequence[TPTransferSlice],
    dst_tp_size: int,
    *,
    shard_dim: int,
    dst_width: int,
) -> tuple[torch.Tensor, ...]:
    """Apply a direct-copy plan on CPU and reject gaps or overlapping writes."""

    normalized_dim = _validate_transfer_plan(
        shards,
        plan,
        dst_tp_size,
        shard_dim=shard_dim,
        dst_width=dst_width,
    )

    destination_shape = list(shards[0].shape)
    destination_shape[normalized_dim] = dst_width
    destinations = tuple(
        shards[0].new_empty(destination_shape) for _ in range(dst_tp_size)
    )
    for entry in plan:
        source = shards[entry.src_rank]
        destination = destinations[entry.dst_rank]
        destination.narrow(
            normalized_dim,
            entry.dst_start,
            entry.length,
        ).copy_(
            source.narrow(
                normalized_dim,
                entry.src_start,
                entry.length,
            )
        )
    return destinations


def reshard_kv_heads(
    shards: Sequence[torch.Tensor],
    dst_tp_size: int,
    *,
    total_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, ...]:
    """Re-shard or replicate KV-cache heads using QKVParallelLinear topology."""

    _validate_tp_size(dst_tp_size)
    normalized_dim = _validate_shards(shards, shard_dim=head_dim)
    src_tp_size = len(shards)
    src_local_heads, src_replication = _kv_head_topology(
        total_kv_heads,
        src_tp_size,
    )
    dst_local_heads, dst_replication = _kv_head_topology(
        total_kv_heads,
        dst_tp_size,
    )
    if any(shard.shape[normalized_dim] != src_local_heads for shard in shards):
        raise ValueError("source KV-head layout does not match its TP topology")

    if src_replication == 1:
        logical = torch.cat(tuple(shards), dim=normalized_dim)
    else:
        unique_heads = []
        for head_index in range(total_kv_heads):
            first_rank = head_index * src_replication
            canonical = shards[first_rank]
            replicas = shards[first_rank : first_rank + src_replication]
            if any(not torch.equal(canonical, replica) for replica in replicas[1:]):
                raise ValueError("replicated source KV heads contain different data")
            unique_heads.append(canonical)
        logical = torch.cat(tuple(unique_heads), dim=normalized_dim)

    if dst_replication == 1:
        return tuple(
            logical.narrow(
                normalized_dim,
                rank * dst_local_heads,
                dst_local_heads,
            ).clone()
            for rank in range(dst_tp_size)
        )

    unique_heads = tuple(
        logical.narrow(normalized_dim, head_index, 1)
        for head_index in range(total_kv_heads)
    )
    return tuple(
        unique_heads[rank // dst_replication].clone()
        for rank in range(dst_tp_size)
    )


def reshard_qwen35_convolution_state(
    shards: Sequence[torch.Tensor],
    dst_tp_size: int,
    *,
    key_channels_per_src_rank: int,
    value_channels_per_src_rank: int,
) -> tuple[torch.Tensor, ...]:
    """Re-shard Qwen3.6 GDN ``[query | key | value]`` convolution state.

    Each packed group is independently head-sharded. Concatenating complete
    rank-local tensors would interleave the three groups and is therefore
    incorrect whenever source and destination TP sizes differ.
    """

    _validate_tp_size(dst_tp_size)
    channel_dim = _validate_shards(shards, shard_dim=0)
    if channel_dim != 0:
        raise AssertionError("convolution channel dimension must normalize to zero")
    for name, value in (
        ("key_channels_per_src_rank", key_channels_per_src_rank),
        ("value_channels_per_src_rank", value_channels_per_src_rank),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    expected_channels = 2 * key_channels_per_src_rank + value_channels_per_src_rank
    if any(shard.shape[0] != expected_channels for shard in shards):
        raise ValueError(
            "Qwen3.6 convolution shard does not match its Q/K/V channel layout"
        )

    query_shards = []
    key_shards = []
    value_shards = []
    for shard in shards:
        query, key, value = torch.split(
            shard,
            (
                key_channels_per_src_rank,
                key_channels_per_src_rank,
                value_channels_per_src_rank,
            ),
            dim=0,
        )
        query_shards.append(query)
        key_shards.append(key)
        value_shards.append(value)

    query_dst = reshard_uniform_tensor(query_shards, dst_tp_size, shard_dim=0)
    key_dst = reshard_uniform_tensor(key_shards, dst_tp_size, shard_dim=0)
    value_dst = reshard_uniform_tensor(value_shards, dst_tp_size, shard_dim=0)
    return tuple(
        torch.cat((query_dst[rank], key_dst[rank], value_dst[rank]), dim=0)
        for rank in range(dst_tp_size)
    )


def reshard_qwen35_rank_cache_transfers(
    payloads: Sequence[RankCacheTransfer],
    dst_tp_size: int,
    *,
    total_kv_heads: int,
    key_channels_per_src_rank: int,
    value_channels_per_src_rank: int,
) -> tuple[RankCacheTransfer, ...]:
    """Build destination-rank payloads using the CPU correctness oracle."""

    _validate_tp_size(dst_tp_size)
    if not payloads or any(
        not isinstance(payload, RankCacheTransfer) for payload in payloads
    ):
        raise ValueError("cache re-sharding requires source rank payloads")
    src_tp_size = len(payloads)
    first = payloads[0]
    if (
        not isinstance(first.transfer_id, str)
        or not first.transfer_id
        or not isinstance(first.block_size, int)
        or isinstance(first.block_size, bool)
        or first.block_size <= 0
        or not isinstance(first.cached_tokens, int)
        or isinstance(first.cached_tokens, bool)
        or first.cached_tokens <= 0
        or first.num_blocks
        != (first.cached_tokens + first.block_size - 1) // first.block_size
        or len(first.recurrent_states) != len(first.convolution_states)
    ):
        raise ValueError("source rank cache payload metadata is invalid")
    common = (
        first.transfer_id,
        first.block_size,
        first.cached_tokens,
        first.num_blocks,
        len(first.recurrent_states),
        len(first.convolution_states),
        first.kv_scales is not None,
    )
    for rank, payload in enumerate(payloads):
        if (
            not isinstance(payload.format_version, int)
            or isinstance(payload.format_version, bool)
            or payload.format_version != TRANSFER_FORMAT_VERSION
            or not isinstance(payload.tensor_parallel_rank, int)
            or isinstance(payload.tensor_parallel_rank, bool)
            or payload.tensor_parallel_rank != rank
            or not isinstance(payload.tensor_parallel_size, int)
            or isinstance(payload.tensor_parallel_size, bool)
            or payload.tensor_parallel_size != src_tp_size
            or (
                payload.transfer_id,
                payload.block_size,
                payload.cached_tokens,
                payload.num_blocks,
                len(payload.recurrent_states),
                len(payload.convolution_states),
                payload.kv_scales is not None,
            )
            != common
        ):
            raise ValueError("source rank cache payload metadata is inconsistent")
        if payload.host_staging_lease is not None:
            raise ValueError("cache re-sharding does not transfer staging leases")

    destination_kv = reshard_kv_heads(
        tuple(payload.kv_blocks for payload in payloads),
        dst_tp_size,
        total_kv_heads=total_kv_heads,
        head_dim=4,
    )
    destination_scales = (
        reshard_kv_heads(
            tuple(payload.kv_scales for payload in payloads),
            dst_tp_size,
            total_kv_heads=total_kv_heads,
            head_dim=4,
        )
        if first.kv_scales is not None
        else (None,) * dst_tp_size
    )
    recurrent_by_layer = tuple(
        reshard_uniform_tensor(
            tuple(payload.recurrent_states[layer] for payload in payloads),
            dst_tp_size,
            shard_dim=0,
        )
        for layer in range(len(first.recurrent_states))
    )
    convolution_by_layer = tuple(
        reshard_qwen35_convolution_state(
            tuple(payload.convolution_states[layer] for payload in payloads),
            dst_tp_size,
            key_channels_per_src_rank=key_channels_per_src_rank,
            value_channels_per_src_rank=value_channels_per_src_rank,
        )
        for layer in range(len(first.convolution_states))
    )
    return tuple(
        RankCacheTransfer(
            format_version=TRANSFER_FORMAT_VERSION,
            transfer_id=first.transfer_id,
            tensor_parallel_rank=rank,
            tensor_parallel_size=dst_tp_size,
            block_size=first.block_size,
            cached_tokens=first.cached_tokens,
            kv_blocks=destination_kv[rank],
            kv_scales=destination_scales[rank],
            recurrent_states=tuple(
                layer[rank] for layer in recurrent_by_layer
            ),
            convolution_states=tuple(
                layer[rank] for layer in convolution_by_layer
            ),
        )
        for rank in range(dst_tp_size)
    )
