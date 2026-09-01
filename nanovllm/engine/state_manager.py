"""Lifecycle management for per-request recurrent model state slots."""

from __future__ import annotations

from collections import deque


class StateSlotManager:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("state slot capacity must be positive")
        self.capacity = capacity
        self._free_slots = deque(range(capacity))
        self._slots_by_sequence: dict[int, int] = {}

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def num_used_slots(self) -> int:
        return len(self._slots_by_sequence)

    def acquire(self, sequence_id: int) -> int:
        existing = self._slots_by_sequence.get(sequence_id)
        if existing is not None:
            return existing
        if not self._free_slots:
            raise RuntimeError("no recurrent state slots are available")
        slot = self._free_slots.popleft()
        self._slots_by_sequence[sequence_id] = slot
        return slot

    def release(self, sequence_id: int) -> int | None:
        slot = self._slots_by_sequence.pop(sequence_id, None)
        if slot is not None:
            self._free_slots.append(slot)
        return slot

    def owns(self, sequence_id: int, slot: int) -> bool:
        return self._slots_by_sequence.get(sequence_id) == slot
