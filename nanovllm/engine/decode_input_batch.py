from __future__ import annotations

import torch


class DecodeInputBatch:
    """Persistent host/device buffers for per-sequence decode metadata."""

    def __init__(
        self,
        capacity: int,
        *,
        device: torch.device | str = "cuda",
        pin_memory: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("decode input capacity must be positive")
        self.capacity = capacity
        specs = {
            "input_ids": torch.int64,
            "positions": torch.int64,
            "slot_mapping": torch.int32,
            "context_lens": torch.int32,
        }
        self.host = {
            name: torch.empty(
                capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for name, dtype in specs.items()
        }
        self.device = {
            name: torch.empty(capacity, dtype=dtype, device=device)
            for name, dtype in specs.items()
        }
        self._arrays = {
            name: tensor.numpy() for name, tensor in self.host.items()
        }

    def update(
        self,
        input_ids: list[int],
        positions: list[int],
        slot_mapping: list[int],
        context_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = len(input_ids)
        if any(
            len(values) != size
            for values in (positions, slot_mapping, context_lens)
        ):
            raise ValueError("decode input batch sizes must match")
        if not 0 < size <= self.capacity:
            raise ValueError(
                f"decode batch size must be in [1, {self.capacity}]"
            )
        values_by_name = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
        }
        for name, values in values_by_name.items():
            self._arrays[name][:size] = values
            self.device[name][:size].copy_(
                self.host[name][:size],
                non_blocking=True,
            )
        return tuple(
            self.device[name][:size]
            for name in ("input_ids", "positions", "slot_mapping", "context_lens")
        )
