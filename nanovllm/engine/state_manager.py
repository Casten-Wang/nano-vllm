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

    def acquire_many(self, sequence_ids) -> list[int]:
        """Acquire a batch atomically, preferring one contiguous slot span."""

        sequence_ids = list(sequence_ids)
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("sequence ids must be unique")
        existing = [self._slots_by_sequence.get(seq_id) for seq_id in sequence_ids]
        new_count = sum(slot is None for slot in existing)
        if new_count > len(self._free_slots):
            raise RuntimeError("no recurrent state slots are available")
        if not new_count:
            return existing

        free = set(self._free_slots)
        selected: list[int] | None = None
        known_starts = {
            slot - index
            for index, slot in enumerate(existing)
            if slot is not None
        }
        if len(known_starts) == 1:
            start = known_starts.pop()
            if 0 <= start and start + len(sequence_ids) <= self.capacity:
                candidate = list(range(start, start + len(sequence_ids)))
                if all(
                    slot == candidate[index]
                    if slot is not None
                    else candidate[index] in free
                    for index, slot in enumerate(existing)
                ):
                    selected = [
                        candidate[index]
                        for index, slot in enumerate(existing)
                        if slot is None
                    ]
        if selected is None and new_count > 1:
            sorted_free = sorted(free)
            for index in range(len(sorted_free) - new_count + 1):
                candidate = sorted_free[index : index + new_count]
                if candidate[-1] - candidate[0] + 1 == new_count:
                    selected = candidate
                    break

        if selected is None:
            return [self.acquire(seq_id) for seq_id in sequence_ids]
        selected_iter = iter(selected)
        result = []
        for seq_id, slot in zip(sequence_ids, existing):
            if slot is None:
                slot = next(selected_iter)
                self._free_slots.remove(slot)
                self._slots_by_sequence[seq_id] = slot
            result.append(slot)
        return result

    def release(self, sequence_id: int) -> int | None:
        slot = self._slots_by_sequence.pop(sequence_id, None)
        if slot is not None:
            self._free_slots.append(slot)
        return slot

    def owns(self, sequence_id: int, slot: int) -> bool:
        return self._slots_by_sequence.get(sequence_id) == slot
