"""CPU-reference tensor-parallel cache re-sharding primitives.

These helpers define layout correctness for heterogeneous-TP prefill/decode
handoff. They intentionally reconstruct a logical tensor before splitting it;
the production transport may later replace that work with direct peer slices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


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
    """Plan direct KV-head copies, selecting one canonical source replica."""

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
        if src_replication == 1:
            src_rank = head_index // src_local_heads
            src_start = head_index % src_local_heads
        else:
            src_rank = head_index * src_replication
            src_start = 0
        if dst_replication == 1:
            destinations = ((head_index // dst_local_heads, head_index % dst_local_heads),)
        else:
            destinations = tuple(
                (head_index * dst_replication + replica, 0)
                for replica in range(dst_replication)
            )
        slices.extend(
            TPTransferSlice(
                src_rank=src_rank,
                dst_rank=dst_rank,
                src_start=src_start,
                dst_start=dst_start,
                length=1,
            )
            for dst_rank, dst_start in destinations
        )
    return tuple(slices)


def apply_tp_transfer_plan(
    shards: Sequence[torch.Tensor],
    plan: Sequence[TPTransferSlice],
    dst_tp_size: int,
    *,
    shard_dim: int,
    dst_width: int,
) -> tuple[torch.Tensor, ...]:
    """Apply a direct-copy plan on CPU and reject gaps or overlapping writes."""

    _validate_tp_size(dst_tp_size)
    normalized_dim = _validate_shards(shards, shard_dim=shard_dim)
    if not isinstance(dst_width, int) or isinstance(dst_width, bool) or dst_width <= 0:
        raise ValueError("destination width must be a positive integer")
    if not plan:
        raise ValueError("tensor-parallel transfer plan must not be empty")

    destination_shape = list(shards[0].shape)
    destination_shape[normalized_dim] = dst_width
    destinations = tuple(
        shards[0].new_empty(destination_shape) for _ in range(dst_tp_size)
    )
    covered = [[False] * dst_width for _ in range(dst_tp_size)]
    for entry in plan:
        if not isinstance(entry, TPTransferSlice):
            raise ValueError("tensor-parallel transfer plan entry is invalid")
        if entry.src_rank >= len(shards) or entry.dst_rank >= dst_tp_size:
            raise ValueError("tensor-parallel transfer rank is out of bounds")
        source = shards[entry.src_rank]
        destination = destinations[entry.dst_rank]
        if entry.src_start + entry.length > source.shape[normalized_dim]:
            raise ValueError("tensor-parallel source slice is out of bounds")
        if entry.dst_start + entry.length > dst_width:
            raise ValueError("tensor-parallel destination slice is out of bounds")
        destination_coverage = covered[entry.dst_rank]
        if any(destination_coverage[entry.dst_start : entry.dst_start + entry.length]):
            raise ValueError("tensor-parallel transfer plan has overlapping writes")
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
        destination_coverage[entry.dst_start : entry.dst_start + entry.length] = (
            [True] * entry.length
        )
    if any(not all(destination_coverage) for destination_coverage in covered):
        raise ValueError("tensor-parallel transfer plan leaves destination gaps")
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
