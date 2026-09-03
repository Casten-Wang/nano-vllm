"""Peer-local payload views for heterogeneous tensor-parallel cache transfer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.engine.cache_transfer import (
    TRANSFER_FORMAT_VERSION,
    RankCacheTransfer,
)
from nanovllm.engine.tp_cache_reshard import (
    Qwen35CacheTransferPlan,
    TPTransferSlice,
)


_COMPONENT_DIMS = {
    "kv": 4,
    "kv_scale": 4,
    "recurrent": 0,
    "convolution": 0,
}


@dataclass(frozen=True, slots=True)
class PeerTensorSlice:
    """One source tensor view and its destination offset."""

    component: str
    layer: int
    dst_start: int
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        if self.component not in _COMPONENT_DIMS:
            raise ValueError("peer tensor slice component is invalid")
        if (
            not isinstance(self.layer, int)
            or isinstance(self.layer, bool)
            or self.layer < -1
            or (self.component in {"kv", "kv_scale"}) != (self.layer == -1)
        ):
            raise ValueError("peer tensor slice layer is invalid")
        if (
            not isinstance(self.dst_start, int)
            or isinstance(self.dst_start, bool)
            or self.dst_start < 0
        ):
            raise ValueError("peer tensor slice destination offset is invalid")
        if not isinstance(self.tensor, torch.Tensor) or self.tensor.ndim == 0:
            raise ValueError("peer tensor slice must contain a tensor view")

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(frozen=True, slots=True)
class PeerCacheFragment:
    """All tensor views sent over one source-rank to destination-rank edge."""

    transfer_id: str
    src_rank: int
    dst_rank: int
    src_tp_size: int
    dst_tp_size: int
    block_size: int
    cached_tokens: int
    slices: tuple[PeerTensorSlice, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise ValueError("peer cache fragment transfer id must not be empty")
        for name, value in (
            ("src_rank", self.src_rank),
            ("dst_rank", self.dst_rank),
            ("src_tp_size", self.src_tp_size),
            ("dst_tp_size", self.dst_tp_size),
            ("block_size", self.block_size),
            ("cached_tokens", self.cached_tokens),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if name.endswith("rank") else 1)
            ):
                raise ValueError(f"peer cache fragment {name} is invalid")
        if self.src_rank >= self.src_tp_size or self.dst_rank >= self.dst_tp_size:
            raise ValueError("peer cache fragment rank is out of bounds")
        if not self.slices or any(
            not isinstance(item, PeerTensorSlice) for item in self.slices
        ):
            raise ValueError("peer cache fragment must contain tensor slices")
        keys = tuple(
            (item.component, item.layer, item.dst_start) for item in self.slices
        )
        if len(keys) != len(set(keys)):
            raise ValueError("peer cache fragment contains duplicate tensor slices")

    @property
    def nbytes(self) -> int:
        return sum(item.nbytes for item in self.slices)


def _source_views(
    tensor: torch.Tensor,
    routes: tuple[TPTransferSlice, ...],
    *,
    src_rank: int,
    component: str,
    layer: int,
) -> tuple[tuple[int, PeerTensorSlice], ...]:
    shard_dim = _COMPONENT_DIMS[component]
    return tuple(
        (
            route.dst_rank,
            PeerTensorSlice(
                component=component,
                layer=layer,
                dst_start=route.dst_start,
                tensor=tensor.narrow(
                    shard_dim,
                    route.src_start,
                    route.length,
                ),
            ),
        )
        for route in routes
        if route.src_rank == src_rank
    )


def build_qwen35_peer_cache_fragments(
    payload: RankCacheTransfer,
    plan: Qwen35CacheTransferPlan,
) -> tuple[PeerCacheFragment, ...]:
    """Group one source rank's staged tensor views by destination peer."""

    if not isinstance(payload, RankCacheTransfer):
        raise ValueError("source cache payload is invalid")
    if not isinstance(plan, Qwen35CacheTransferPlan):
        raise ValueError("heterogeneous cache transfer plan is invalid")
    src_tp_size = len(plan.profile.source_bytes)
    dst_tp_size = len(plan.profile.destination_bytes)
    if (
        not isinstance(payload.format_version, int)
        or isinstance(payload.format_version, bool)
        or payload.format_version != TRANSFER_FORMAT_VERSION
        or not isinstance(payload.transfer_id, str)
        or not payload.transfer_id
        or not isinstance(payload.block_size, int)
        or isinstance(payload.block_size, bool)
        or payload.block_size <= 0
        or not isinstance(payload.cached_tokens, int)
        or isinstance(payload.cached_tokens, bool)
        or payload.cached_tokens <= 0
        or payload.num_blocks
        != (payload.cached_tokens + payload.block_size - 1) // payload.block_size
        or payload.tensor_parallel_size != src_tp_size
        or not 0 <= payload.tensor_parallel_rank < src_tp_size
    ):
        raise ValueError("source cache payload does not match transfer plan")
    if bool(payload.kv_scales is not None) != bool(plan.kv_scale_slices):
        raise ValueError("source cache scale layout does not match transfer plan")
    if len(payload.recurrent_states) != len(payload.convolution_states):
        raise ValueError("source cache state layer counts do not match")
    tensors = (
        payload.kv_blocks,
        *((payload.kv_scales,) if payload.kv_scales is not None else ()),
        *payload.recurrent_states,
        *payload.convolution_states,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("peer cache fragments require host-staged tensors")

    grouped: dict[int, list[PeerTensorSlice]] = {}

    def add(items: tuple[tuple[int, PeerTensorSlice], ...]) -> None:
        for dst_rank, item in items:
            grouped.setdefault(dst_rank, []).append(item)

    add(
        _source_views(
            payload.kv_blocks,
            plan.kv_slices,
            src_rank=payload.tensor_parallel_rank,
            component="kv",
            layer=-1,
        )
    )
    if payload.kv_scales is not None:
        add(
            _source_views(
                payload.kv_scales,
                plan.kv_scale_slices,
                src_rank=payload.tensor_parallel_rank,
                component="kv_scale",
                layer=-1,
            )
        )
    for layer, tensor in enumerate(payload.recurrent_states):
        add(
            _source_views(
                tensor,
                plan.recurrent_slices,
                src_rank=payload.tensor_parallel_rank,
                component="recurrent",
                layer=layer,
            )
        )
    for layer, tensor in enumerate(payload.convolution_states):
        add(
            _source_views(
                tensor,
                plan.convolution_slices,
                src_rank=payload.tensor_parallel_rank,
                component="convolution",
                layer=layer,
            )
        )

    fragments = tuple(
        PeerCacheFragment(
            transfer_id=payload.transfer_id,
            src_rank=payload.tensor_parallel_rank,
            dst_rank=dst_rank,
            src_tp_size=src_tp_size,
            dst_tp_size=dst_tp_size,
            block_size=payload.block_size,
            cached_tokens=payload.cached_tokens,
            slices=tuple(slices),
        )
        for dst_rank, slices in sorted(grouped.items())
    )
    expected = {
        dst_rank: byte_count
        for src_rank, dst_rank, byte_count in plan.profile.peer_bytes
        if src_rank == payload.tensor_parallel_rank
    }
    actual = {fragment.dst_rank: fragment.nbytes for fragment in fragments}
    if actual != expected:
        raise RuntimeError("peer cache fragments do not match capacity preflight")
    return fragments
