from __future__ import annotations

import torch


class DecodeInputBatch:
    """Persistent host/device buffers for per-sequence decode metadata."""

    def __init__(
        self,
        capacity: int,
        max_num_blocks: int = 1,
        *,
        device: torch.device | str = "cuda",
        pin_memory: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("decode input capacity must be positive")
        if max_num_blocks <= 0:
            raise ValueError("decode block-table capacity must be positive")
        self.capacity = capacity
        self.max_num_blocks = max_num_blocks
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
        self.host_block_tables = torch.empty(
            capacity,
            max_num_blocks,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.device_block_tables = torch.empty(
            capacity,
            max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        self._block_table_array = self.host_block_tables.numpy()
        self.host_state_slots = torch.empty(
            capacity,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.device_state_slots = torch.empty(
            capacity,
            dtype=torch.int64,
            device=device,
        )
        self.host_reset_slots = torch.empty_like(
            self.host_state_slots,
            pin_memory=pin_memory,
        )
        self.device_reset_slots = torch.empty_like(self.device_state_slots)
        self._state_slot_array = self.host_state_slots.numpy()
        self._reset_slot_array = self.host_reset_slots.numpy()

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

    def update_block_tables(self, block_tables: list[list[int]]) -> torch.Tensor:
        size = len(block_tables)
        if not 0 < size <= self.capacity:
            raise ValueError(
                f"decode batch size must be in [1, {self.capacity}]"
            )
        width = max((len(row) for row in block_tables), default=0)
        if not 0 < width <= self.max_num_blocks:
            raise ValueError(
                "decode block-table width must be in "
                f"[1, {self.max_num_blocks}]"
            )
        host = self.host_block_tables[:size, :width]
        host.fill_(-1)
        for row_index, row in enumerate(block_tables):
            self._block_table_array[row_index, : len(row)] = row
        device = self.device_block_tables[:size, :width]
        device.copy_(host, non_blocking=True)
        return device

    def _update_slots(
        self,
        values: list[int],
        *,
        reset: bool,
    ) -> torch.Tensor:
        size = len(values)
        if not 0 < size <= self.capacity:
            raise ValueError(
                f"decode slot count must be in [1, {self.capacity}]"
            )
        host = self.host_reset_slots if reset else self.host_state_slots
        device = self.device_reset_slots if reset else self.device_state_slots
        array = self._reset_slot_array if reset else self._state_slot_array
        array[:size] = values
        device[:size].copy_(host[:size], non_blocking=True)
        return device[:size]

    def update_state_slots(self, values: list[int]) -> torch.Tensor:
        return self._update_slots(values, reset=False)

    def update_reset_slots(self, values: list[int]) -> torch.Tensor:
        return self._update_slots(values, reset=True)
