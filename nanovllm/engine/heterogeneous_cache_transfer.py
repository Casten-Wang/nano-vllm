"""Peer-local payload views for heterogeneous tensor-parallel cache transfer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.engine.cache_transfer import (
    LEGACY_CACHE_FINGERPRINT,
    TRANSFER_FORMAT_VERSION,
    HostStagingBufferPool,
    HostStagingLease,
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
    cache_fingerprint: str = LEGACY_CACHE_FINGERPRINT

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise ValueError("peer cache fragment transfer id must not be empty")
        if not isinstance(self.cache_fingerprint, str) or not self.cache_fingerprint:
            raise ValueError("peer cache fragment fingerprint must not be empty")
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


@dataclass(frozen=True, slots=True)
class StagedPeerCacheFragments:
    """Peer views backed by one exclusively owned host-staging lease."""

    fragments: tuple[PeerCacheFragment, ...]
    lease: HostStagingLease
    staged_bytes: int

    def __post_init__(self) -> None:
        if not self.fragments:
            raise ValueError("staged peer cache fragments must not be empty")
        if not isinstance(self.lease, HostStagingLease):
            raise ValueError("staged peer cache fragments require a host lease")
        if (
            not isinstance(self.staged_bytes, int)
            or isinstance(self.staged_bytes, bool)
            or self.staged_bytes <= 0
        ):
            raise ValueError("staged peer cache byte count must be positive")

    def release(self) -> None:
        self.lease.release()


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
            cache_fingerprint=payload.cache_fingerprint,
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


def stage_qwen35_peer_cache_fragments(
    fragments: tuple[PeerCacheFragment, ...],
    plan: Qwen35CacheTransferPlan,
    *,
    host_staging_pool: HostStagingBufferPool | None = None,
) -> StagedPeerCacheFragments:
    """Copy unique source slices once into one reusable host arena."""

    if not fragments or any(
        not isinstance(fragment, PeerCacheFragment) for fragment in fragments
    ):
        raise ValueError("peer cache staging requires source fragments")
    if not isinstance(plan, Qwen35CacheTransferPlan):
        raise ValueError("heterogeneous cache transfer plan is invalid")
    first = fragments[0]
    common = (
        first.transfer_id,
        first.src_rank,
        first.src_tp_size,
        first.dst_tp_size,
        first.block_size,
        first.cached_tokens,
    )
    if any(
        (
            fragment.transfer_id,
            fragment.src_rank,
            fragment.src_tp_size,
            fragment.dst_tp_size,
            fragment.block_size,
            fragment.cached_tokens,
        )
        != common
        for fragment in fragments
    ):
        raise ValueError("peer cache staging fragments are inconsistent")
    if first.src_tp_size != len(plan.profile.source_staging_bytes):
        raise ValueError("peer cache staging does not match transfer plan")

    unique: dict[tuple[object, ...], torch.Tensor] = {}
    item_keys: dict[int, tuple[object, ...]] = {}
    for fragment in fragments:
        for item in fragment.slices:
            tensor = item.tensor
            key = (
                tensor.device,
                tensor.dtype,
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            unique.setdefault(key, tensor)
            item_keys[id(item)] = key

    offsets = {}
    end = 0
    for key, tensor in unique.items():
        alignment = tensor.element_size()
        end = (end + alignment - 1) // alignment * alignment
        offsets[key] = end
        end += tensor.numel() * tensor.element_size()
    staged_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in unique.values()
    )
    expected_staged_bytes = plan.profile.source_staging_bytes[first.src_rank]
    if staged_bytes != expected_staged_bytes:
        raise RuntimeError("unique source slices do not match staging preflight")
    pin_memory = any(tensor.device.type == "cuda" for tensor in unique.values())
    lease = (
        host_staging_pool.acquire(end, pin_memory=pin_memory)
        if host_staging_pool is not None
        else HostStagingLease(
            torch.empty(
                end,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            ),
            None,
        )
    )
    storage = lease.storage
    if storage is None:
        raise RuntimeError("host staging lease was released before use")
    staged = {}
    try:
        cuda_devices = set()
        for key, source in unique.items():
            nbytes = source.numel() * source.element_size()
            target = storage[offsets[key] : offsets[key] + nbytes]
            target = target.view(source.dtype).reshape(source.shape)
            target.copy_(
                source,
                non_blocking=source.device.type == "cuda",
            )
            staged[key] = target
            if source.device.type == "cuda":
                cuda_devices.add(source.device)
        for device in cuda_devices:
            torch.cuda.current_stream(device).synchronize()
        staged_fragments = tuple(
            PeerCacheFragment(
                transfer_id=fragment.transfer_id,
                src_rank=fragment.src_rank,
                dst_rank=fragment.dst_rank,
                src_tp_size=fragment.src_tp_size,
                dst_tp_size=fragment.dst_tp_size,
                block_size=fragment.block_size,
                cached_tokens=fragment.cached_tokens,
                cache_fingerprint=fragment.cache_fingerprint,
                slices=tuple(
                    PeerTensorSlice(
                        component=item.component,
                        layer=item.layer,
                        dst_start=item.dst_start,
                        tensor=staged[item_keys[id(item)]],
                    )
                    for item in fragment.slices
                ),
            )
            for fragment in fragments
        )
        return StagedPeerCacheFragments(
            fragments=staged_fragments,
            lease=lease,
            staged_bytes=staged_bytes,
        )
    except BaseException:
        lease.release()
        raise


def stage_qwen35_sequence_cache_for_peers(
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
    block_ids: list[int],
    *,
    recurrent_states: tuple[torch.Tensor, ...],
    convolution_states: tuple[torch.Tensor, ...],
    transfer_id: str,
    src_rank: int,
    block_size: int,
    cached_tokens: int,
    cache_fingerprint: str = LEGACY_CACHE_FINGERPRINT,
    plan: Qwen35CacheTransferPlan,
    host_staging_pool: HostStagingBufferPool | None = None,
) -> StagedPeerCacheFragments:
    """Stage routed peer slices directly from live rank-local cache storage."""

    src_tp_size = len(plan.profile.source_bytes)
    dst_tp_size = len(plan.profile.destination_bytes)
    if (
        not isinstance(transfer_id, str)
        or not transfer_id
        or not isinstance(src_rank, int)
        or isinstance(src_rank, bool)
        or not 0 <= src_rank < src_tp_size
        or not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
        or not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or cached_tokens <= 0
        or not isinstance(cache_fingerprint, str)
        or not cache_fingerprint
    ):
        raise ValueError("live peer cache staging metadata is invalid")
    num_blocks = (cached_tokens + block_size - 1) // block_size
    if (
        kv_cache.ndim != 6
        or kv_cache.shape[0] != 2
        or kv_cache.shape[3] != block_size
        or len(block_ids) != num_blocks
        or len(block_ids) != len(set(block_ids))
        or not block_ids
        or min(block_ids) < 0
        or max(block_ids) >= kv_cache.shape[2]
    ):
        raise ValueError("live peer cache KV layout is invalid")
    if bool(kv_scale is not None) != bool(plan.kv_scale_slices):
        raise ValueError("live peer cache scale layout does not match plan")
    if kv_scale is not None and (
        kv_scale.shape != kv_cache.shape[:-1]
        or kv_cache.dtype != torch.int8
        or kv_scale.dtype != torch.float16
    ):
        raise ValueError("live peer cache INT8 scale layout is invalid")
    if len(recurrent_states) != len(convolution_states) or not recurrent_states:
        raise ValueError("live peer cache state layers are invalid")

    routes_by_component = {
        "kv": plan.kv_slices,
        "kv_scale": plan.kv_scale_slices,
        "recurrent": plan.recurrent_slices,
        "convolution": plan.convolution_slices,
    }
    sources = {
        ("kv", -1): kv_cache,
        **(
            {("kv_scale", -1): kv_scale}
            if kv_scale is not None
            else {}
        ),
        **{
            ("recurrent", layer): tensor
            for layer, tensor in enumerate(recurrent_states)
        },
        **{
            ("convolution", layer): tensor
            for layer, tensor in enumerate(convolution_states)
        },
    }
    specs: dict[tuple[str, int, int, int], tuple[torch.Tensor, tuple[int, ...]]] = {}
    route_keys: dict[tuple[str, int, int, int, int], tuple[str, int, int, int]] = {}
    for (component, layer), source in sources.items():
        shard_dim = _COMPONENT_DIMS[component]
        routes = routes_by_component[component]
        for route in routes:
            if route.src_rank != src_rank:
                continue
            if route.src_start + route.length > source.shape[shard_dim]:
                raise ValueError("live peer cache route exceeds source tensor")
            key = (component, layer, route.src_start, route.length)
            if component in {"kv", "kv_scale"}:
                shape = list(source.shape)
                shape[2] = num_blocks
                shape[shard_dim] = route.length
            else:
                shape = list(source.shape)
                shape[shard_dim] = route.length
            specs.setdefault(key, (source, tuple(shape)))
            route_keys[
                (component, layer, route.dst_rank, route.dst_start, route.length)
            ] = key

    offsets = {}
    end = 0
    for key, (source, shape) in specs.items():
        alignment = source.element_size()
        end = (end + alignment - 1) // alignment * alignment
        offsets[key] = end
        elements = 1
        for dimension in shape:
            elements *= dimension
        end += elements * source.element_size()
    staged_bytes = sum(
        source.element_size() * torch.Size(shape).numel()
        for source, shape in specs.values()
    )
    if staged_bytes != plan.profile.source_staging_bytes[src_rank]:
        raise RuntimeError("live source slices do not match staging preflight")
    pin_memory = any(source.device.type == "cuda" for source, _ in specs.values())
    lease = (
        host_staging_pool.acquire(end, pin_memory=pin_memory)
        if host_staging_pool is not None
        else HostStagingLease(
            torch.empty(
                end,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pin_memory,
            ),
            None,
        )
    )
    storage = lease.storage
    if storage is None:
        raise RuntimeError("host staging lease was released before use")
    staged = {}
    try:
        cuda_devices = set()
        valid_last_block_tokens = cached_tokens % block_size
        for key, (source, shape) in specs.items():
            component, _layer, src_start, length = key
            nbytes = source.element_size() * torch.Size(shape).numel()
            target = storage[offsets[key] : offsets[key] + nbytes]
            target = target.view(source.dtype).reshape(shape)
            shard_dim = _COMPONENT_DIMS[component]
            if component in {"kv", "kv_scale"}:
                for logical_id, physical_id in enumerate(block_ids):
                    source_block = source[:, :, physical_id]
                    target_block = target[:, :, logical_id]
                    source_head_dim = shard_dim - 1
                    source_slice = source_block.narrow(
                        source_head_dim,
                        src_start,
                        length,
                    )
                    if logical_id == num_blocks - 1 and valid_last_block_tokens:
                        target_block.zero_()
                        target_block[:, :, :valid_last_block_tokens].copy_(
                            source_slice[:, :, :valid_last_block_tokens],
                            non_blocking=source.device.type == "cuda",
                        )
                    else:
                        target_block.copy_(
                            source_slice,
                            non_blocking=source.device.type == "cuda",
                        )
            else:
                target.copy_(
                    source.narrow(shard_dim, src_start, length),
                    non_blocking=source.device.type == "cuda",
                )
            staged[key] = target
            if source.device.type == "cuda":
                cuda_devices.add(source.device)
        for device in cuda_devices:
            torch.cuda.current_stream(device).synchronize()

        grouped: dict[int, list[PeerTensorSlice]] = {}
        for (component, layer), _source in sources.items():
            for route in routes_by_component[component]:
                if route.src_rank != src_rank:
                    continue
                key = route_keys[
                    (
                        component,
                        layer,
                        route.dst_rank,
                        route.dst_start,
                        route.length,
                    )
                ]
                grouped.setdefault(route.dst_rank, []).append(
                    PeerTensorSlice(
                        component=component,
                        layer=layer,
                        dst_start=route.dst_start,
                        tensor=staged[key],
                    )
                )
        fragments = tuple(
            PeerCacheFragment(
                transfer_id=transfer_id,
                src_rank=src_rank,
                dst_rank=dst_rank,
                src_tp_size=src_tp_size,
                dst_tp_size=dst_tp_size,
                block_size=block_size,
                cached_tokens=cached_tokens,
                cache_fingerprint=cache_fingerprint,
                slices=tuple(slices),
            )
            for dst_rank, slices in sorted(grouped.items())
        )
        expected = {
            dst_rank: byte_count
            for source_rank, dst_rank, byte_count in plan.profile.peer_bytes
            if source_rank == src_rank
        }
        if {item.dst_rank: item.nbytes for item in fragments} != expected:
            raise RuntimeError("live peer fragments do not match capacity preflight")
        return StagedPeerCacheFragments(
            fragments=fragments,
            lease=lease,
            staged_bytes=staged_bytes,
        )
    except BaseException:
        lease.release()
        raise


def assemble_qwen35_peer_cache_fragments(
    fragments: tuple[PeerCacheFragment, ...],
    plan: Qwen35CacheTransferPlan,
) -> RankCacheTransfer:
    """Validate all destination peers, then assemble one atomic rank payload."""

    if not fragments or any(
        not isinstance(fragment, PeerCacheFragment) for fragment in fragments
    ):
        raise ValueError("destination cache assembly requires peer fragments")
    if not isinstance(plan, Qwen35CacheTransferPlan):
        raise ValueError("heterogeneous cache transfer plan is invalid")
    first = fragments[0]
    src_tp_size = len(plan.profile.source_bytes)
    dst_tp_size = len(plan.profile.destination_bytes)
    common = (
        first.transfer_id,
        first.dst_rank,
        first.src_tp_size,
        first.dst_tp_size,
        first.block_size,
        first.cached_tokens,
        first.cache_fingerprint,
    )
    if (
        first.src_tp_size != src_tp_size
        or first.dst_tp_size != dst_tp_size
        or not 0 <= first.dst_rank < dst_tp_size
    ):
        raise ValueError("destination cache fragments do not match transfer plan")
    expected_peer_bytes = {
        src_rank: byte_count
        for src_rank, dst_rank, byte_count in plan.profile.peer_bytes
        if dst_rank == first.dst_rank
    }
    actual_peer_bytes = {}
    for fragment in fragments:
        if (
            (
                fragment.transfer_id,
                fragment.dst_rank,
                fragment.src_tp_size,
                fragment.dst_tp_size,
                fragment.block_size,
                fragment.cached_tokens,
                fragment.cache_fingerprint,
            )
            != common
            or fragment.src_rank in actual_peer_bytes
        ):
            raise ValueError("destination cache fragment metadata is inconsistent")
        actual_peer_bytes[fragment.src_rank] = fragment.nbytes
    if actual_peer_bytes != expected_peer_bytes:
        raise ValueError("destination cache peer bytes do not match preflight")

    routes_by_component = {
        "kv": plan.kv_slices,
        "kv_scale": plan.kv_scale_slices,
        "recurrent": plan.recurrent_slices,
        "convolution": plan.convolution_slices,
    }
    grouped: dict[tuple[str, int], list[tuple[int, PeerTensorSlice]]] = {}
    for fragment in fragments:
        for item in fragment.slices:
            grouped.setdefault((item.component, item.layer), []).append(
                (fragment.src_rank, item)
            )
    if ("kv", -1) not in grouped:
        raise ValueError("destination cache fragments are missing KV state")
    if bool(("kv_scale", -1) in grouped) != bool(plan.kv_scale_slices):
        raise ValueError("destination cache scale fragments do not match plan")
    recurrent_layers = sorted(
        layer for component, layer in grouped if component == "recurrent"
    )
    convolution_layers = sorted(
        layer for component, layer in grouped if component == "convolution"
    )
    if (
        not recurrent_layers
        or recurrent_layers != convolution_layers
        or recurrent_layers != list(range(len(recurrent_layers)))
    ):
        raise ValueError("destination cache state layer fragments are incomplete")

    assembled: dict[tuple[str, int], torch.Tensor] = {}
    for key, items in grouped.items():
        component, _layer = key
        routes = tuple(
            route
            for route in routes_by_component[component]
            if route.dst_rank == first.dst_rank
        )
        expected_routes = {
            (route.src_rank, route.dst_start, route.length) for route in routes
        }
        shard_dim = _COMPONENT_DIMS[component]
        actual_routes = {
            (src_rank, item.dst_start, item.tensor.shape[shard_dim])
            for src_rank, item in items
        }
        if len(actual_routes) != len(items) or actual_routes != expected_routes:
            raise ValueError("destination cache tensor routes do not match plan")
        first_tensor = items[0][1].tensor
        shape = list(first_tensor.shape)
        destination_width = max(
            route.dst_start + route.length for route in routes
        )
        shape[shard_dim] = destination_width
        destination = first_tensor.new_empty(shape)
        coverage = [False] * destination_width
        for _src_rank, item in items:
            tensor = item.tensor
            reference = list(shape)
            reference[shard_dim] = tensor.shape[shard_dim]
            if (
                tensor.dtype != first_tensor.dtype
                or tensor.device.type != "cpu"
                or list(tensor.shape) != reference
            ):
                raise ValueError("destination cache tensor slice layout is inconsistent")
            end = item.dst_start + tensor.shape[shard_dim]
            if end > destination_width or any(coverage[item.dst_start:end]):
                raise ValueError("destination cache tensor slices overlap")
            coverage[item.dst_start:end] = [True] * (end - item.dst_start)
            destination.narrow(
                shard_dim,
                item.dst_start,
                tensor.shape[shard_dim],
            ).copy_(tensor)
        if not all(coverage):
            raise ValueError("destination cache tensor slices leave gaps")
        assembled[key] = destination

    payload = RankCacheTransfer(
        format_version=TRANSFER_FORMAT_VERSION,
        transfer_id=first.transfer_id,
        tensor_parallel_rank=first.dst_rank,
        tensor_parallel_size=dst_tp_size,
        block_size=first.block_size,
        cached_tokens=first.cached_tokens,
        cache_fingerprint=first.cache_fingerprint,
        kv_blocks=assembled.pop(("kv", -1)),
        kv_scales=assembled.pop(("kv_scale", -1), None),
        recurrent_states=tuple(
            assembled.pop(("recurrent", layer))
            for layer in recurrent_layers
        ),
        convolution_states=tuple(
            assembled.pop(("convolution", layer))
            for layer in convolution_layers
        ),
    )
    if assembled:
        raise ValueError("destination cache fragments contain unexpected tensors")
    if payload.nbytes != plan.profile.destination_bytes[first.dst_rank]:
        raise RuntimeError("assembled cache payload does not match capacity preflight")
    return payload
