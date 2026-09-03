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


class TokenInputBatch:
    """Persistent transfer buffers for prefill and mixed token batches."""

    def __init__(
        self,
        token_capacity: int,
        sequence_capacity: int,
        max_num_blocks: int = 1,
        *,
        device: torch.device | str = "cuda",
        pin_memory: bool = True,
    ) -> None:
        if token_capacity <= 0 or sequence_capacity <= 0:
            raise ValueError("token input capacities must be positive")
        if max_num_blocks <= 0:
            raise ValueError("packed block-table capacity must be positive")
        self.token_capacity = token_capacity
        self.sequence_capacity = sequence_capacity
        self.max_num_blocks = max_num_blocks
        specs = {
            "input_ids": (token_capacity, torch.int64),
            "positions": (token_capacity, torch.int64),
            "slot_mapping": (token_capacity, torch.int32),
            "cu_seqlens_q": (sequence_capacity + 1, torch.int32),
            "cu_seqlens_k": (sequence_capacity + 1, torch.int32),
            "decode_context_lens": (sequence_capacity, torch.int32),
            "logits_indices": (sequence_capacity, torch.int64),
        }
        self.host = {
            name: torch.empty(
                capacity,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            for name, (capacity, dtype) in specs.items()
        }
        self.device = {
            name: torch.empty(capacity, dtype=dtype, device=device)
            for name, (capacity, dtype) in specs.items()
        }
        self._arrays = {
            name: tensor.numpy() for name, tensor in self.host.items()
        }
        self.host_block_tables = torch.empty(
            sequence_capacity,
            max_num_blocks,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.device_block_tables = torch.empty(
            sequence_capacity,
            max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        self._block_table_array = self.host_block_tables.numpy()
        packed_block_capacity = sequence_capacity * max_num_blocks
        # Mixed batches retain decode and prefill metadata at the same time.
        # Two banks keep those live ranges disjoint while still avoiding
        # per-step pinned-host and device allocations.
        self.host_selected_block_ids = tuple(
            torch.empty(
                packed_block_capacity,
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory,
            )
            for _ in range(2)
        )
        self.device_selected_block_ids = tuple(
            torch.empty(
                packed_block_capacity,
                dtype=torch.int32,
                device=device,
            )
            for _ in range(2)
        )
        self.host_packed_block_tables = tuple(
            torch.empty(
                sequence_capacity,
                max_num_blocks,
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory,
            )
            for _ in range(2)
        )
        self.device_packed_block_tables = tuple(
            torch.empty(
                sequence_capacity,
                max_num_blocks,
                dtype=torch.int32,
                device=device,
            )
            for _ in range(2)
        )
        self._selected_block_ids_arrays = tuple(
            tensor.numpy() for tensor in self.host_selected_block_ids
        )
        self._packed_block_tables_arrays = tuple(
            tensor.numpy() for tensor in self.host_packed_block_tables
        )

    def _update(self, name: str, values: list[int]) -> torch.Tensor:
        size = len(values)
        capacity = self.host[name].numel()
        if not 0 < size <= capacity:
            raise ValueError(f"{name} size must be in [1, {capacity}]")
        self._arrays[name][:size] = values
        self.device[name][:size].copy_(
            self.host[name][:size],
            non_blocking=True,
        )
        return self.device[name][:size]

    def update_tokens(
        self,
        input_ids: list[int],
        positions: list[int],
        slot_mapping: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = len(input_ids)
        if len(positions) != size or len(slot_mapping) not in (0, size):
            raise ValueError("token input batch sizes must match")
        ids = self._update("input_ids", input_ids)
        position_tensor = self._update("positions", positions)
        slots = (
            self._update("slot_mapping", slot_mapping)
            if slot_mapping
            else self.device["slot_mapping"][:0]
        )
        return ids, position_tensor, slots

    def update_cu_seqlens(
        self,
        cu_seqlens_q: list[int],
        cu_seqlens_k: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(cu_seqlens_q) != len(cu_seqlens_k):
            raise ValueError("query and key sequence counts must match")
        return (
            self._update("cu_seqlens_q", cu_seqlens_q),
            self._update("cu_seqlens_k", cu_seqlens_k),
        )

    def update_decode_context_lens(self, values: list[int]) -> torch.Tensor:
        return self._update("decode_context_lens", values)

    def update_logits_indices(self, values: list[int]) -> torch.Tensor:
        return self._update("logits_indices", values)

    def update_block_tables(self, block_tables: list[list[int]]) -> torch.Tensor:
        """Stage dense prefill block tables in persistent host/device storage."""

        size = len(block_tables)
        if not 0 < size <= self.sequence_capacity:
            raise ValueError(
                f"prefill batch size must be in [1, {self.sequence_capacity}]"
            )
        width = max((len(row) for row in block_tables), default=0)
        if not 0 < width <= self.max_num_blocks:
            raise ValueError(
                "prefill block-table width must be in "
                f"[1, {self.max_num_blocks}]"
            )
        host = self.host_block_tables[:size, :width]
        host.fill_(-1)
        for row_index, row in enumerate(block_tables):
            self._block_table_array[row_index, : len(row)] = row
        device = self.device_block_tables[:size, :width]
        device.copy_(host, non_blocking=True)
        return device

    def update_packed_block_metadata(
        self,
        selected_block_ids: list[int],
        packed_block_tables: list[list[int]],
        *,
        slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slot not in (0, 1):
            raise ValueError("packed metadata slot must be 0 or 1")
        host_ids = self.host_selected_block_ids[slot]
        device_ids = self.device_selected_block_ids[slot]
        host_tables = self.host_packed_block_tables[slot]
        device_tables = self.device_packed_block_tables[slot]
        selected_count = len(selected_block_ids)
        sequence_count = len(packed_block_tables)
        if not 0 < selected_count <= host_ids.numel():
            raise ValueError(
                "selected block count must be in "
                f"[1, {host_ids.numel()}]"
            )
        if not 0 < sequence_count <= self.sequence_capacity:
            raise ValueError(
                "packed block-table row count must be in "
                f"[1, {self.sequence_capacity}]"
            )
        width = len(packed_block_tables[0])
        if not 0 < width <= self.max_num_blocks:
            raise ValueError(
                "packed block-table width must be in "
                f"[1, {self.max_num_blocks}]"
            )
        if any(len(row) != width for row in packed_block_tables):
            raise ValueError("packed block-table rows must have equal width")

        self._selected_block_ids_arrays[slot][:selected_count] = (
            selected_block_ids
        )
        selected_device = device_ids[:selected_count]
        selected_device.copy_(
            host_ids[:selected_count],
            non_blocking=True,
        )

        self._packed_block_tables_arrays[slot][:sequence_count, :width] = (
            packed_block_tables
        )
        tables_device = device_tables[
            :sequence_count,
            :width,
        ]
        tables_device.copy_(
            host_tables[:sequence_count, :width],
            non_blocking=True,
        )
        return selected_device, tables_device
