from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).parents[1] / "nanovllm" / "engine" / "state_manager.py"
SPEC = spec_from_file_location("state_manager_under_test", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
StateSlotManager = MODULE.StateSlotManager


def test_state_slots_are_stable_and_reused_after_release():
    manager = StateSlotManager(2)

    first = manager.acquire(100)
    second = manager.acquire(200)

    assert first == 0
    assert second == 1
    assert manager.acquire(100) == first
    assert manager.num_used_slots == 2
    assert manager.num_free_slots == 0
    assert manager.release(100) == first
    assert manager.acquire(300) == first
    assert manager.owns(300, first)


def test_state_slot_capacity_is_enforced():
    manager = StateSlotManager(1)
    manager.acquire(100)

    with pytest.raises(RuntimeError, match="no recurrent state slots"):
        manager.acquire(200)


def test_releasing_unknown_sequence_is_idempotent():
    manager = StateSlotManager(1)

    assert manager.release(999) is None
    assert manager.num_free_slots == 1


def test_batch_acquire_prefers_contiguous_slots_after_fragmented_release():
    manager = StateSlotManager(6)
    slots = manager.acquire_many(range(6))
    assert slots == list(range(6))
    for sequence_id in (4, 2, 3):
        manager.release(sequence_id)

    assert manager.acquire_many((10, 11, 12)) == [2, 3, 4]


def test_batch_acquire_extends_existing_contiguous_prefix():
    manager = StateSlotManager(4)
    assert manager.acquire_many((10, 11)) == [0, 1]

    assert manager.acquire_many((10, 11, 12, 13)) == [0, 1, 2, 3]


def test_batch_acquire_is_atomic_when_capacity_is_insufficient():
    manager = StateSlotManager(2)

    with pytest.raises(RuntimeError, match="no recurrent state slots"):
        manager.acquire_many((10, 11, 12))

    assert manager.num_free_slots == 2
    assert manager.num_used_slots == 0


def test_batch_acquire_rejects_duplicate_sequence_ids():
    manager = StateSlotManager(2)

    with pytest.raises(ValueError, match="unique"):
        manager.acquire_many((10, 10))
