"""Validated per-rank cache payloads for prefill/decode handoff."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

import torch


TRANSFER_FORMAT_VERSION = 1
LEGACY_CACHE_FINGERPRINT = "legacy-unscoped"


def build_cache_transfer_fingerprint(config: object) -> str:
    """Identify model semantics and cache layout across PD workers."""

    model_spec = getattr(config, "model_spec", None)
    model_config = getattr(config, "model_config", None)
    if model_spec is None or model_config is None:
        raise ValueError("cache transfer requires an initialized model config")
    explicit_id = getattr(config, "cache_transfer_model_id", None)
    if explicit_id is not None:
        if not isinstance(explicit_id, str) or not explicit_id:
            raise ValueError("cache transfer model id must be a non-empty string")
        model_identity = f"explicit:{explicit_id}"
    else:
        hf_config = getattr(config, "hf_config", None)
        revision = getattr(hf_config, "_commit_hash", None)
        model_identity = (
            f"revision:{revision}"
            if isinstance(revision, str) and revision
            else f"local:{Path(getattr(config, 'model')).resolve()}"
        )
    factors = {
        "schema": 1,
        "model_identity": model_identity,
        "architecture": model_spec.architecture,
        "full_attention_layers": model_spec.full_attention_layers,
        "linear_attention_layers": model_spec.linear_attention_layers,
        "num_hidden_layers": model_spec.num_hidden_layers,
        "num_attention_heads": getattr(model_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", None),
        "head_dim": getattr(model_config, "head_dim", None),
        "hidden_size": getattr(model_config, "hidden_size", None),
        "model_dtype": str(getattr(model_config, "dtype", None)),
        "kv_cache_dtype": getattr(config, "kv_cache_dtype", None),
        "recurrent_state_dtype": getattr(config, "recurrent_state_dtype", None),
        "block_size": getattr(config, "kvcache_block_size", None),
        "sliding_window_size": getattr(config, "sliding_window_size", None),
    }
    encoded = json.dumps(
        factors,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_plain_int(value: object) -> bool:
    """Reject bools at transfer boundaries even though bool subclasses int."""

    return isinstance(value, int) and not isinstance(value, bool)


def validate_cache_transfer_timeout(timeout_s: object) -> float:
    """Return one finite positive timeout for controller and worker paths."""

    valid_type = (
        isinstance(timeout_s, (int, float))
        and not isinstance(timeout_s, bool)
    )
    try:
        validated_timeout_s = float(timeout_s) if valid_type else math.nan
    except (OverflowError, TypeError, ValueError):
        validated_timeout_s = math.nan
    if not math.isfinite(validated_timeout_s) or validated_timeout_s <= 0:
        raise ValueError(
            "cache transfer timeout must be a finite positive number"
        )
    return validated_timeout_s


def validate_cache_transfer_payload_limit(max_payload_bytes: object) -> int:
    """Return one positive plain-integer wire allocation limit."""

    if (
        not isinstance(max_payload_bytes, int)
        or isinstance(max_payload_bytes, bool)
        or max_payload_bytes <= 0
    ):
        raise ValueError(
            "cache transfer max_payload_bytes must be a positive integer"
        )
    return max_payload_bytes


def validate_cache_transfer_id(transfer_id: object) -> str:
    """Return one non-empty string suitable for every ownership map."""

    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("cache transfer id must be a non-empty string")
    return transfer_id


class HostStagingLease:
    """Exclusive ownership of one contiguous host staging allocation."""

    def __init__(
        self,
        storage: torch.Tensor,
        pool: "HostStagingBufferPool | None",
    ) -> None:
        self.storage: torch.Tensor | None = storage
        self._pool = pool

    def release(self) -> None:
        storage = self.storage
        if storage is None:
            return
        self.storage = None
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool._release(storage)


class HostStagingBufferPool:
    """Retain one largest-fit host buffer without aliasing concurrent sends."""

    def __init__(self, max_cached_bytes: int | None = None) -> None:
        if (
            max_cached_bytes is not None
            and (
                not isinstance(max_cached_bytes, int)
                or isinstance(max_cached_bytes, bool)
                or max_cached_bytes < 0
            )
        ):
            raise ValueError("max_cached_bytes must be a non-negative integer")
        self._lock = Lock()
        self._max_cached_bytes = max_cached_bytes
        self._storage: torch.Tensor | None = None
        self._leased = False
        self.allocation_count = 0
        self.reuse_count = 0
        self.transient_allocation_count = 0

    def acquire(self, size: int, *, pin_memory: bool) -> HostStagingLease:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("host staging size must be a positive integer")
        with self._lock:
            storage = self._storage
            cacheable = (
                self._max_cached_bytes is None
                or size <= self._max_cached_bytes
            )
            compatible = (
                storage is not None
                and storage.numel() >= size
                and storage.is_pinned() == pin_memory
            )
            if not self._leased and cacheable:
                if not compatible:
                    storage = torch.empty(
                        size,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    self._storage = storage
                    self.allocation_count += 1
                else:
                    self.reuse_count += 1
                self._leased = True
                return HostStagingLease(storage, self)

            self.transient_allocation_count += 1
        storage = torch.empty(
            size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        return HostStagingLease(storage, None)

    def _release(self, storage: torch.Tensor) -> None:
        with self._lock:
            if not self._leased or storage is not self._storage:
                raise RuntimeError("host staging pool lease is invalid")
            self._leased = False

    def storage_stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "max_cached_bytes": self._max_cached_bytes,
                "storage_bytes": (
                    0 if self._storage is None else self._storage.numel()
                ),
                "allocation_count": self.allocation_count,
                "reuse_count": self.reuse_count,
                "transient_allocation_count": self.transient_allocation_count,
                "leased": int(self._leased),
            }


@dataclass(frozen=True, slots=True)
class RankCacheTransfer:
    """Logical request state produced by one tensor-parallel rank."""

    format_version: int
    transfer_id: str
    tensor_parallel_rank: int
    tensor_parallel_size: int
    block_size: int
    cached_tokens: int
    kv_blocks: torch.Tensor
    kv_scales: torch.Tensor | None
    recurrent_states: tuple[torch.Tensor, ...]
    convolution_states: tuple[torch.Tensor, ...]
    cache_fingerprint: str = LEGACY_CACHE_FINGERPRINT
    host_staging_lease: HostStagingLease | None = None

    @property
    def num_blocks(self) -> int:
        return self.kv_blocks.shape[2]

    @property
    def nbytes(self) -> int:
        tensors = (
            self.kv_blocks,
            *((self.kv_scales,) if self.kv_scales is not None else ()),
            *self.recurrent_states,
            *self.convolution_states,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def release_host_staging(self) -> None:
        if self.host_staging_lease is not None:
            self.host_staging_lease.release()


class CacheTransferPhase(Enum):
    RECEIVING = auto()
    READY = auto()
    COMMITTED = auto()
    ABORTED = auto()
    TIMED_OUT = auto()


class CacheTransferSession:
    """Coordinate all-rank installation before a request may decode."""

    def __init__(
        self,
        transfer_id: str,
        tensor_parallel_size: int,
        *,
        started_at: float,
        timeout_s: float,
    ) -> None:
        transfer_id = validate_cache_transfer_id(transfer_id)
        if tensor_parallel_size <= 0:
            raise ValueError("cache transfer TP size must be positive")
        validated_timeout_s = validate_cache_transfer_timeout(timeout_s)
        self.transfer_id = transfer_id
        self.tensor_parallel_size = tensor_parallel_size
        self.deadline = started_at + validated_timeout_s
        self.phase = CacheTransferPhase.RECEIVING
        self.acknowledged_ranks: set[int] = set()
        self.failure_reason: str | None = None

    @property
    def fallback_required(self) -> bool:
        return self.phase in {
            CacheTransferPhase.ABORTED,
            CacheTransferPhase.TIMED_OUT,
        }

    def _expire(self, now: float) -> None:
        if (
            self.phase in {CacheTransferPhase.RECEIVING, CacheTransferPhase.READY}
            and now >= self.deadline
        ):
            self.phase = CacheTransferPhase.TIMED_OUT
            self.failure_reason = "cache transfer timed out"

    def acknowledge(self, rank: int, *, now: float) -> None:
        self._expire(now)
        if self.phase not in {
            CacheTransferPhase.RECEIVING,
            CacheTransferPhase.READY,
        }:
            raise RuntimeError("cache transfer session is already terminal")
        if not 0 <= rank < self.tensor_parallel_size:
            raise ValueError("cache transfer acknowledgement rank is invalid")
        self.acknowledged_ranks.add(rank)
        if len(self.acknowledged_ranks) == self.tensor_parallel_size:
            self.phase = CacheTransferPhase.READY

    def fail(self, rank: int, reason: str, *, now: float) -> None:
        self._expire(now)
        if self.phase not in {
            CacheTransferPhase.RECEIVING,
            CacheTransferPhase.READY,
        }:
            raise RuntimeError("cache transfer session is already terminal")
        if not 0 <= rank < self.tensor_parallel_size:
            raise ValueError("cache transfer failure rank is invalid")
        if not reason:
            raise ValueError("cache transfer failure reason must not be empty")
        self.phase = CacheTransferPhase.ABORTED
        self.failure_reason = f"rank {rank}: {reason}"

    def commit(self, *, now: float) -> None:
        self._expire(now)
        if self.phase is not CacheTransferPhase.READY:
            raise RuntimeError("cache transfer is not ready to commit")
        self.phase = CacheTransferPhase.COMMITTED

    def poll(self, *, now: float) -> CacheTransferPhase:
        self._expire(now)
        return self.phase


def _validate_cache_layout(
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
) -> None:
    if kv_cache.ndim != 6 or kv_cache.shape[0] != 2:
        raise ValueError("KV cache must have shape [2, layers, blocks, ...]")
    if kv_cache.dtype == torch.int8:
        if kv_scale is None:
            raise ValueError("INT8 KV cache requires scale storage")
        expected = kv_cache.shape[:-1]
        if kv_scale.shape != expected or kv_scale.dtype != torch.float16:
            raise ValueError(
                "INT8 KV scales must be FP16 with shape [2, layers, blocks, tokens, heads]"
            )
    elif kv_scale is not None:
        raise ValueError("floating-point KV cache must not include INT8 scales")


def _validate_block_ids(
    block_ids: list[int],
    *,
    total_blocks: int,
) -> None:
    if not block_ids:
        raise ValueError("cache transfer requires at least one KV block")
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("cache transfer block ids must be unique")
    if min(block_ids) < 0 or max(block_ids) >= total_blocks:
        raise ValueError("cache transfer block id is out of bounds")


def _block_index(
    block_ids: list[int],
    *,
    total_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    _validate_block_ids(block_ids, total_blocks=total_blocks)
    return torch.tensor(block_ids, dtype=torch.int64, device=device)


def estimate_rank_cache_transfer_bytes(
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
    block_ids: list[int],
    *,
    recurrent_states: tuple[torch.Tensor, ...] = (),
    convolution_states: tuple[torch.Tensor, ...] = (),
) -> int:
    """Return exact staged tensor bytes without allocating transfer storage."""

    _validate_cache_layout(kv_cache, kv_scale)
    _validate_state_pairs(recurrent_states, convolution_states)
    if not block_ids:
        raise ValueError("cache transfer requires at least one KV block")
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("cache transfer block ids must be unique")
    if min(block_ids) < 0 or max(block_ids) >= kv_cache.shape[2]:
        raise ValueError("cache transfer block id is out of bounds")
    return estimate_rank_cache_transfer_bytes_for_blocks(
        kv_cache,
        kv_scale,
        len(block_ids),
        recurrent_states=recurrent_states,
        convolution_states=convolution_states,
    )


def estimate_rank_cache_transfer_bytes_for_blocks(
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
    num_blocks: int,
    *,
    recurrent_states: tuple[torch.Tensor, ...] = (),
    convolution_states: tuple[torch.Tensor, ...] = (),
) -> int:
    """Estimate rank-local transfer bytes before physical blocks are reserved."""

    _validate_cache_layout(kv_cache, kv_scale)
    _validate_state_pairs(recurrent_states, convolution_states)
    if (
        not isinstance(num_blocks, int)
        or isinstance(num_blocks, bool)
        or num_blocks <= 0
    ):
        raise ValueError("cache transfer block count must be a positive integer")
    kv_elements_per_block = kv_cache.numel() // kv_cache.shape[2]
    total = num_blocks * kv_elements_per_block * kv_cache.element_size()
    if kv_scale is not None:
        scale_elements_per_block = kv_scale.numel() // kv_scale.shape[2]
        total += num_blocks * scale_elements_per_block * kv_scale.element_size()
    total += sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*recurrent_states, *convolution_states)
    )
    return total


def _validate_state_pairs(
    recurrent_states: tuple[torch.Tensor, ...],
    convolution_states: tuple[torch.Tensor, ...],
) -> None:
    if len(recurrent_states) != len(convolution_states):
        raise ValueError(
            "recurrent and convolution state layer counts must match"
        )
    if any(tensor.ndim == 0 for tensor in (*recurrent_states, *convolution_states)):
        raise ValueError("cache transfer states must not be scalars")


def _allocate_host_staging_views(
    specs: list[tuple[torch.Tensor, tuple[int, ...]]],
    pool: HostStagingBufferPool | None = None,
) -> tuple[list[torch.Tensor], HostStagingLease]:
    """Allocate one aligned host storage for heterogeneous D2H tensors."""

    offsets = []
    end = 0
    for tensor, shape in specs:
        alignment = tensor.dtype.itemsize
        end = (end + alignment - 1) // alignment * alignment
        offsets.append(end)
        elements = 1
        for dimension in shape:
            elements *= dimension
        end += elements * tensor.element_size()
    pin_memory = any(tensor.device.type == "cuda" for tensor, _ in specs)
    lease = (
        pool.acquire(end, pin_memory=pin_memory)
        if pool is not None
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
    views = []
    for (tensor, shape), offset in zip(specs, offsets):
        nbytes = tensor.element_size()
        for dimension in shape:
            nbytes *= dimension
        views.append(storage[offset : offset + nbytes].view(tensor.dtype).view(shape))
    return views, lease


def _export_blocks_to_host(
    storage: torch.Tensor,
    block_ids: list[int],
    *,
    valid_last_block_tokens: int,
    staging: torch.Tensor,
) -> torch.Tensor:
    """Copy physical blocks directly into logical-order host staging."""

    for logical_id, physical_id in enumerate(block_ids):
        source = storage[:, :, physical_id]
        target = staging[:, :, logical_id]
        if logical_id == len(block_ids) - 1 and valid_last_block_tokens:
            target.zero_()
            target[:, :, :valid_last_block_tokens].copy_(
                source[:, :, :valid_last_block_tokens],
                non_blocking=storage.device.type == "cuda",
            )
        else:
            target.copy_(
                source,
                non_blocking=storage.device.type == "cuda",
            )
    return staging


def _export_states_to_host(
    states: tuple[torch.Tensor, ...],
    staging: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    for source, target in zip(states, staging):
        target.copy_(source, non_blocking=source.device.type == "cuda")
    return staging


def _install_cache_blocks(
    destination: torch.Tensor,
    source: torch.Tensor,
    block_ids: list[int],
    index: torch.Tensor,
) -> None:
    """Install logical blocks without a full-size cross-device temporary."""

    if source.device == destination.device:
        destination.index_copy_(2, index, source)
        return
    non_blocking = source.device.type == "cpu" and source.is_pinned()
    for logical_id, physical_id in enumerate(block_ids):
        destination[:, :, physical_id].copy_(
            source[:, :, logical_id],
            non_blocking=non_blocking,
        )


def export_rank_cache(
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
    block_ids: list[int],
    *,
    transfer_id: str,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    block_size: int,
    cached_tokens: int,
    cache_fingerprint: str = LEGACY_CACHE_FINGERPRINT,
    recurrent_states: tuple[torch.Tensor, ...] = (),
    convolution_states: tuple[torch.Tensor, ...] = (),
    to_host: bool = False,
    host_staging_pool: HostStagingBufferPool | None = None,
) -> RankCacheTransfer:
    """Copy one request's rank-local state in logical block order."""

    _validate_cache_layout(kv_cache, kv_scale)
    _validate_state_pairs(recurrent_states, convolution_states)
    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("cache transfer id must not be empty")
    if (
        not _is_plain_int(tensor_parallel_rank)
        or not _is_plain_int(tensor_parallel_size)
        or tensor_parallel_size <= 0
        or not 0 <= tensor_parallel_rank < tensor_parallel_size
    ):
        raise ValueError("cache transfer tensor-parallel identity is invalid")
    if not _is_plain_int(block_size) or block_size <= 0:
        raise ValueError("cache transfer block size must be positive")
    if not _is_plain_int(cached_tokens) or cached_tokens <= 0:
        raise ValueError("cache transfer cached token count must be a positive integer")
    if not isinstance(cache_fingerprint, str) or not cache_fingerprint:
        raise ValueError("cache transfer fingerprint must be a non-empty string")
    expected_blocks = (cached_tokens + block_size - 1) // block_size
    if len(block_ids) != expected_blocks:
        raise ValueError(
            "cache transfer block count does not match cached token count"
        )
    _validate_block_ids(block_ids, total_blocks=kv_cache.shape[2])
    valid_last_block_tokens = cached_tokens % block_size
    staging_lease = None
    if to_host:
        kv_shape = (*kv_cache.shape[:2], len(block_ids), *kv_cache.shape[3:])
        specs = [(kv_cache, tuple(kv_shape))]
        if kv_scale is not None:
            scale_shape = (
                *kv_scale.shape[:2],
                len(block_ids),
                *kv_scale.shape[3:],
            )
            specs.append((kv_scale, tuple(scale_shape)))
        specs.extend((tensor, tuple(tensor.shape)) for tensor in recurrent_states)
        specs.extend((tensor, tuple(tensor.shape)) for tensor in convolution_states)
        allocated_views, staging_lease = _allocate_host_staging_views(
            specs,
            host_staging_pool,
        )
        staging_views = iter(allocated_views)
        try:
            kv_blocks = _export_blocks_to_host(
                kv_cache,
                block_ids,
                valid_last_block_tokens=valid_last_block_tokens,
                staging=next(staging_views),
            )
            kv_scales = (
                _export_blocks_to_host(
                    kv_scale,
                    block_ids,
                    valid_last_block_tokens=valid_last_block_tokens,
                    staging=next(staging_views),
                )
                if kv_scale is not None
                else None
            )
            exported_recurrent = _export_states_to_host(
                recurrent_states,
                tuple(next(staging_views) for _ in recurrent_states),
            )
            exported_convolution = _export_states_to_host(
                convolution_states,
                tuple(next(staging_views) for _ in convolution_states),
            )
            copied_tensors = [kv_cache, *recurrent_states, *convolution_states]
            if kv_scale is not None:
                copied_tensors.append(kv_scale)
            cuda_devices = {
                tensor.device
                for tensor in copied_tensors
                if tensor.device.type == "cuda"
            }
            for device in cuda_devices:
                torch.cuda.current_stream(device).synchronize()
        except BaseException:
            staging_lease.release()
            raise
    else:
        index = torch.tensor(
            block_ids,
            dtype=torch.int64,
            device=kv_cache.device,
        )
        kv_blocks = kv_cache.index_select(2, index).clone()
        kv_scales = (
            kv_scale.index_select(2, index).clone()
            if kv_scale is not None
            else None
        )
        exported_recurrent = tuple(tensor.clone() for tensor in recurrent_states)
        exported_convolution = tuple(
            tensor.clone() for tensor in convolution_states
        )
    if valid_last_block_tokens and not to_host:
        # Physical tail slots can contain data from a previous block owner.
        # They are not semantically part of this request and must not cross a
        # process or host boundary.
        kv_blocks[:, :, -1, valid_last_block_tokens:].zero_()
        if kv_scales is not None:
            kv_scales[:, :, -1, valid_last_block_tokens:].zero_()
    return RankCacheTransfer(
        format_version=TRANSFER_FORMAT_VERSION,
        transfer_id=transfer_id,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        block_size=block_size,
        cached_tokens=cached_tokens,
        cache_fingerprint=cache_fingerprint,
        kv_blocks=kv_blocks,
        kv_scales=kv_scales,
        recurrent_states=exported_recurrent,
        convolution_states=exported_convolution,
        host_staging_lease=staging_lease,
    )


def import_rank_cache(
    payload: RankCacheTransfer,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor | None,
    block_ids: list[int],
    *,
    transfer_id: str,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    block_size: int,
    cache_fingerprint: str | None = None,
    recurrent_states: tuple[torch.Tensor, ...] = (),
    convolution_states: tuple[torch.Tensor, ...] = (),
) -> None:
    """Validate completely, then install one request into destination slots."""

    _validate_cache_layout(kv_cache, kv_scale)
    _validate_state_pairs(recurrent_states, convolution_states)
    _validate_state_pairs(
        payload.recurrent_states,
        payload.convolution_states,
    )
    if (
        not _is_plain_int(tensor_parallel_rank)
        or not _is_plain_int(tensor_parallel_size)
        or tensor_parallel_size <= 0
        or not 0 <= tensor_parallel_rank < tensor_parallel_size
    ):
        raise ValueError("cache transfer tensor-parallel identity is invalid")
    if not _is_plain_int(block_size) or block_size <= 0:
        raise ValueError("cache transfer block size must be positive")
    if (
        cache_fingerprint is not None
        and (
            not isinstance(cache_fingerprint, str)
            or not cache_fingerprint
            or payload.cache_fingerprint != cache_fingerprint
        )
    ):
        raise ValueError("cache transfer fingerprint does not match destination")
    if (
        not _is_plain_int(payload.format_version)
        or payload.format_version != TRANSFER_FORMAT_VERSION
    ):
        raise ValueError("unsupported cache transfer format version")
    if (
        not isinstance(payload.transfer_id, str)
        or not payload.transfer_id
        or not isinstance(transfer_id, str)
        or not transfer_id
        or payload.transfer_id != transfer_id
    ):
        raise ValueError("cache transfer id does not match")
    if (
        not _is_plain_int(payload.tensor_parallel_rank)
        or not _is_plain_int(payload.tensor_parallel_size)
        or payload.tensor_parallel_size <= 0
        or payload.tensor_parallel_rank != tensor_parallel_rank
        or payload.tensor_parallel_size != tensor_parallel_size
    ):
        raise ValueError("cache transfer tensor-parallel identity does not match")
    if (
        not _is_plain_int(payload.block_size)
        or payload.block_size <= 0
        or payload.block_size != block_size
    ):
        raise ValueError("source and destination KV block sizes differ")
    if not _is_plain_int(payload.cached_tokens):
        raise ValueError("cache transfer cached token count must be an integer")
    expected_blocks = (
        payload.cached_tokens + payload.block_size - 1
    ) // payload.block_size
    if payload.cached_tokens <= 0 or payload.num_blocks != expected_blocks:
        raise ValueError("cache transfer payload has an invalid token/block count")
    if len(block_ids) != payload.num_blocks:
        raise ValueError("destination KV block count does not match payload")
    index = _block_index(
        block_ids,
        total_blocks=kv_cache.shape[2],
        device=kv_cache.device,
    )
    expected_kv_shape = (
        kv_cache.shape[0],
        kv_cache.shape[1],
        len(block_ids),
        *kv_cache.shape[3:],
    )
    if payload.kv_blocks.shape != expected_kv_shape:
        raise ValueError("cache transfer KV shape does not match destination")
    if payload.kv_blocks.dtype != kv_cache.dtype:
        raise ValueError("cache transfer KV dtype does not match destination")
    if (payload.kv_scales is None) != (kv_scale is None):
        raise ValueError("cache transfer scale presence does not match destination")
    if kv_scale is not None and (
        payload.kv_scales.shape != (
            kv_scale.shape[0],
            kv_scale.shape[1],
            len(block_ids),
            *kv_scale.shape[3:],
        )
        or payload.kv_scales.dtype != kv_scale.dtype
    ):
        raise ValueError("cache transfer scale layout does not match destination")
    if len(payload.recurrent_states) != len(recurrent_states):
        raise ValueError("cache transfer recurrent layer count does not match")
    if len(payload.convolution_states) != len(convolution_states):
        raise ValueError("cache transfer convolution layer count does not match")
    for source, destination in zip(
        (*payload.recurrent_states, *payload.convolution_states),
        (*recurrent_states, *convolution_states),
    ):
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("cache transfer state layout does not match destination")

    _install_cache_blocks(
        kv_cache,
        payload.kv_blocks,
        block_ids,
        index,
    )
    if kv_scale is not None:
        assert payload.kv_scales is not None
        _install_cache_blocks(
            kv_scale,
            payload.kv_scales,
            block_ids,
            index,
        )
    for source, destination in zip(
        (*payload.recurrent_states, *payload.convolution_states),
        (*recurrent_states, *convolution_states),
    ):
        non_blocking = source.device.type == "cpu" and source.is_pinned()
        destination.copy_(source, non_blocking=non_blocking)
