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
