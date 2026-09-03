"""Validated per-rank cache payloads for prefill/decode handoff."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch


TRANSFER_FORMAT_VERSION = 1


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
        if not transfer_id:
            raise ValueError("cache transfer id must not be empty")
        if tensor_parallel_size <= 0:
            raise ValueError("cache transfer TP size must be positive")
        if timeout_s <= 0:
            raise ValueError("cache transfer timeout must be positive")
        self.transfer_id = transfer_id
        self.tensor_parallel_size = tensor_parallel_size
        self.deadline = started_at + timeout_s
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


def _block_index(
    block_ids: list[int],
    *,
    total_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    if not block_ids:
        raise ValueError("cache transfer requires at least one KV block")
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("cache transfer block ids must be unique")
    if min(block_ids) < 0 or max(block_ids) >= total_blocks:
        raise ValueError("cache transfer block id is out of bounds")
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


def _host_staging_like(
    tensor: torch.Tensor,
    shape: tuple[int, ...] | torch.Size | None = None,
) -> torch.Tensor:
    """Allocate host storage suitable for a direct device-to-host copy."""

    return torch.empty(
        tuple(tensor.shape if shape is None else shape),
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=tensor.device.type == "cuda",
    )


def _export_blocks_to_host(
    storage: torch.Tensor,
    block_ids: list[int],
    *,
    valid_last_block_tokens: int,
) -> torch.Tensor:
    """Copy physical blocks directly into logical-order host staging."""

    shape = (*storage.shape[:2], len(block_ids), *storage.shape[3:])
    staging = _host_staging_like(storage, shape)
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
) -> tuple[torch.Tensor, ...]:
    staging = tuple(_host_staging_like(tensor) for tensor in states)
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
    recurrent_states: tuple[torch.Tensor, ...] = (),
    convolution_states: tuple[torch.Tensor, ...] = (),
    to_host: bool = False,
) -> RankCacheTransfer:
    """Copy one request's rank-local state in logical block order."""

    _validate_cache_layout(kv_cache, kv_scale)
    _validate_state_pairs(recurrent_states, convolution_states)
    if not transfer_id:
        raise ValueError("cache transfer id must not be empty")
    if not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("cache transfer tensor-parallel identity is invalid")
    if block_size <= 0:
        raise ValueError("cache transfer block size must be positive")
    expected_blocks = (cached_tokens + block_size - 1) // block_size
    if cached_tokens <= 0 or len(block_ids) != expected_blocks:
        raise ValueError(
            "cache transfer block count does not match cached token count"
        )
    index = _block_index(
        block_ids,
        total_blocks=kv_cache.shape[2],
        device=kv_cache.device,
    )
    valid_last_block_tokens = cached_tokens % block_size
    if to_host:
        kv_blocks = _export_blocks_to_host(
            kv_cache,
            block_ids,
            valid_last_block_tokens=valid_last_block_tokens,
        )
        kv_scales = (
            _export_blocks_to_host(
                kv_scale,
                block_ids,
                valid_last_block_tokens=valid_last_block_tokens,
            )
            if kv_scale is not None
            else None
        )
        exported_recurrent = _export_states_to_host(recurrent_states)
        exported_convolution = _export_states_to_host(convolution_states)
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
    else:
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
        kv_blocks=kv_blocks,
        kv_scales=kv_scales,
        recurrent_states=exported_recurrent,
        convolution_states=exported_convolution,
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
    if not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("cache transfer tensor-parallel identity is invalid")
    if block_size <= 0:
        raise ValueError("cache transfer block size must be positive")
    if payload.format_version != TRANSFER_FORMAT_VERSION:
        raise ValueError("unsupported cache transfer format version")
    if payload.transfer_id != transfer_id:
        raise ValueError("cache transfer id does not match")
    if (
        payload.tensor_parallel_rank != tensor_parallel_rank
        or payload.tensor_parallel_size != tensor_parallel_size
    ):
        raise ValueError("cache transfer tensor-parallel identity does not match")
    if payload.block_size != block_size:
        raise ValueError("source and destination KV block sizes differ")
    if not isinstance(payload.cached_tokens, int):
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
