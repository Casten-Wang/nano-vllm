import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeXXH64:
    def __init__(self):
        self._parts = []

    def update(self, value):
        self._parts.append(bytes(value))

    def intdigest(self):
        digest = sha256(b"".join(self._parts)).digest()
        return int.from_bytes(digest[:8], "little")


class FakeSamplingParams:
    temperature = 0.0
    top_k = -1
    top_p = 1.0
    max_tokens = 8
    ignore_eos = True


@dataclass
class FakeConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8
    eos: int | tuple[int, ...] = -1
    kvcache_block_size: int = 4
    num_kvcache_blocks: int = 8
    enable_dynamic_chunked_prefill: bool = True
    model_spec: object | None = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_lifecycle_modules():
    managed_names = (
        "nanovllm",
        "nanovllm.engine",
        "nanovllm.config",
        "nanovllm.sampling_params",
        "nanovllm.engine.sequence",
        "nanovllm.engine.block_manager",
        "nanovllm.engine.state_manager",
        "nanovllm.engine.scheduler",
        "xxhash",
    )
    saved = {name: sys.modules.get(name) for name in managed_names}
    try:
        nanovllm_pkg = types.ModuleType("nanovllm")
        engine_pkg = types.ModuleType("nanovllm.engine")
        config_module = types.ModuleType("nanovllm.config")
        sampling_module = types.ModuleType("nanovllm.sampling_params")
        xxhash_module = types.ModuleType("xxhash")
        config_module.Config = FakeConfig
        sampling_module.SamplingParams = FakeSamplingParams
        xxhash_module.xxh64 = FakeXXH64
        sys.modules.update(
            {
                "nanovllm": nanovllm_pkg,
                "nanovllm.engine": engine_pkg,
                "nanovllm.config": config_module,
                "nanovllm.sampling_params": sampling_module,
                "xxhash": xxhash_module,
            }
        )
        sequence = load_module(
            "nanovllm.engine.sequence",
            ROOT / "nanovllm" / "engine" / "sequence.py",
        )
        block_manager = load_module(
            "nanovllm.engine.block_manager",
            ROOT / "nanovllm" / "engine" / "block_manager.py",
        )
        state_manager = load_module(
            "nanovllm.engine.state_manager",
            ROOT / "nanovllm" / "engine" / "state_manager.py",
        )
        scheduler = load_module(
            "nanovllm.engine.scheduler",
            ROOT / "nanovllm" / "engine" / "scheduler.py",
        )
        return sequence, block_manager, state_manager, scheduler
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


(
    sequence_module,
    block_manager_module,
    state_manager_module,
    scheduler_module,
) = load_lifecycle_modules()
Sequence = sequence_module.Sequence
SequenceStatus = sequence_module.SequenceStatus
BlockManager = block_manager_module.BlockManager
Scheduler = scheduler_module.Scheduler
StateSlotManager = state_manager_module.StateSlotManager


def assert_block_conservation(testcase, manager):
    testcase.assertEqual(
        manager.num_used_blocks + manager.num_free_blocks,
        manager.num_total_blocks,
    )
    testcase.assertEqual(
        len(set(manager.free_block_ids)),
        manager.num_free_blocks,
    )
    testcase.assertTrue(
        manager.used_block_ids.isdisjoint(set(manager.free_block_ids))
    )


class BlockManagerLifecycleTest(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 4

    def test_allocate_decode_append_and_release(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence([1, 2, 3, 4])

        manager.allocate(seq, num_cached_blocks=0)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(manager.blocks[seq.block_table[0]].ref_count, 1)
        assert_block_conservation(self, manager)

        seq.append_token(5)
        self.assertTrue(manager.can_append(seq))
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)
        assert_block_conservation(self, manager)

        manager.deallocate(seq)
        self.assertEqual(seq.block_table, [])
        self.assertEqual(manager.num_used_blocks, 0)
        assert_block_conservation(self, manager)

    def test_scheduler_stops_on_any_configured_eos_token(self):
        for eos_token in (248046, 248044):
            with self.subTest(eos_token=eos_token):
                scheduler = Scheduler(FakeConfig(eos=(248046, 248044)))
                params = types.SimpleNamespace(
                    temperature=0.0,
                    top_k=-1,
                    top_p=1.0,
                    max_tokens=8,
                    ignore_eos=False,
                )
                seq = Sequence([1], params)
                scheduler.block_manager.allocate(seq, num_cached_blocks=0)
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                seq.num_scheduled_tokens = 1
                scheduler.running.append(seq)

                scheduler.postprocess_one(seq, eos_token, is_prefill=False)

                self.assertTrue(seq.is_finished)
                self.assertEqual(seq.completion_token_ids, [eos_token])
                self.assertNotIn(seq, scheduler.running)

    def test_hybrid_scheduler_disables_kv_only_prefix_reuse(self):
        config = FakeConfig(
            model_spec=types.SimpleNamespace(is_hybrid=True),
        )
        scheduler = Scheduler(config)
        original = Sequence(list(range(8)))
        scheduler.block_manager.allocate(original, num_cached_blocks=0)
        original.num_scheduled_tokens = len(original)
        scheduler.block_manager.hash_blocks(original)
        scheduler.block_manager.deallocate(original)
        replacement = Sequence(list(range(8)))
        scheduler.add(replacement)

        result = scheduler.schedule_dynamic_chunked_prefill()

        self.assertEqual(replacement.num_cached_tokens, 0)
        self.assertEqual(result.prefill_seqs, [replacement])

    def test_prefix_hit_shares_refcount_until_both_requests_release(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        tokens = list(range(8))
        first = Sequence(tokens)
        manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = len(first)
        manager.hash_blocks(first)
        first.num_cached_tokens = len(first)

        second = Sequence(tokens)
        cached_blocks = manager.can_allocate(second)
        self.assertEqual(cached_blocks, 1)
        manager.allocate(second, cached_blocks)
        shared_block = first.block_table[0]
        self.assertEqual(second.block_table[0], shared_block)
        self.assertEqual(manager.blocks[shared_block].ref_count, 2)
        self.assertGreater(manager.prefix_cache_hit_blocks, 0)

        manager.deallocate(first)
        self.assertEqual(manager.blocks[shared_block].ref_count, 1)
        self.assertIn(shared_block, manager.used_block_ids)
        manager.deallocate(second)
        self.assertEqual(manager.blocks[shared_block].ref_count, 0)
        self.assertNotIn(shared_block, manager.used_block_ids)
        assert_block_conservation(self, manager)

    def test_released_prefix_cache_block_still_consumes_a_free_slot(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        tokens = list(range(8))
        original = Sequence(tokens)
        manager.allocate(original, num_cached_blocks=0)
        original.num_scheduled_tokens = len(original)
        manager.hash_blocks(original)
        cached_block = original.block_table[0]
        manager.deallocate(original)

        self.assertIn(cached_block, manager.free_block_ids)
        replacement = Sequence(tokens)
        cached_blocks = manager.get_num_cached_blocks(replacement)
        self.assertEqual(cached_blocks, 1)
        self.assertEqual(
            manager.can_allocate(
                replacement,
                num_blocks=2,
                num_cached_blocks=cached_blocks,
            ),
            1,
        )
        manager.allocate(
            replacement,
            cached_blocks,
            num_blocks=2,
        )

        self.assertEqual(replacement.block_table[0], cached_block)
        self.assertEqual(manager.num_free_blocks, 0)
        assert_block_conservation(self, manager)

    def test_released_prefix_cache_block_is_counted_during_admission(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        tokens = list(range(8))
        original = Sequence(tokens)
        manager.allocate(original, num_cached_blocks=0)
        original.num_scheduled_tokens = len(original)
        manager.hash_blocks(original)
        manager.deallocate(original)

        occupied = Sequence([100, 101, 102, 103])
        manager.allocate(occupied, num_cached_blocks=0)
        replacement = Sequence(tokens)
        cached_blocks = manager.get_num_cached_blocks(replacement)

        self.assertEqual(cached_blocks, 1)
        self.assertEqual(
            manager.can_allocate(
                replacement,
                num_blocks=2,
                num_cached_blocks=cached_blocks,
            ),
            -1,
        )
        self.assertEqual(replacement.block_table, [])
        assert_block_conservation(self, manager)

    def test_reuse_removes_stale_hash_mapping(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        original = Sequence(list(range(8)))
        manager.allocate(original, num_cached_blocks=0)
        original.num_scheduled_tokens = len(original)
        manager.hash_blocks(original)
        old_hashes = {
            manager.blocks[block_id].hash for block_id in original.block_table
        }
        self.assertTrue(all(value in manager.hash_to_block_id for value in old_hashes))
        manager.deallocate(original)

        replacement = Sequence(list(range(100, 108)))
        manager.allocate(replacement, num_cached_blocks=0)
        self.assertEqual(set(replacement.block_table), {0, 1})
        self.assertTrue(
            all(value not in manager.hash_to_block_id for value in old_hashes)
        )
        self.assertTrue(
            all(manager.blocks[block_id].hash == -1 for block_id in replacement.block_table)
        )
        assert_block_conservation(self, manager)

    def test_reset_cache_stats_removes_warmup_counters(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence(list(range(8)))

        manager.can_allocate(seq)
        self.assertEqual(manager.prefix_cache_queries, 1)
        self.assertEqual(manager.prefix_cache_checked_blocks, 1)

        manager.reset_cache_stats()

        self.assertEqual(
            manager.cache_stats(),
            {
                "prefix_cache_queries": 0,
                "prefix_cache_checked_blocks": 0,
                "prefix_cache_hit_blocks": 0,
                "prefix_cache_hit_rate": 0.0,
            },
        )

    def test_scheduler_preempt_returns_blocks_and_request_to_waiting(self):
        scheduler = Scheduler(
            FakeConfig(
                num_kvcache_blocks=3,
                kvcache_block_size=4,
            )
        )
        seq = Sequence([1, 2, 3, 4])
        scheduler.block_manager.allocate(seq, num_cached_blocks=0)
        seq.num_cached_tokens = len(seq)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)

        scheduler.running.remove(seq)
        scheduler.preempt(seq)

        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assertTrue(seq.is_prefill)
        self.assertEqual(seq.block_table, [])
        self.assertIs(scheduler.waiting[0], seq)
        self.assertEqual(scheduler.block_manager.num_used_blocks, 0)
        self.assertEqual(scheduler.preemption_count, 1)
        self.assertEqual(scheduler.preempted_token_progress, 4)
        self.assertEqual(scheduler.max_preempted_token_progress, 4)
        self.assertEqual(scheduler.reclaimed_kv_blocks, 1)
        assert_block_conservation(self, scheduler.block_manager)

    def test_legacy_scheduler_does_not_admit_prefill_past_sequence_limit(self):
        scheduler = Scheduler(
            FakeConfig(
                max_num_seqs=1,
                max_num_batched_tokens=8,
                num_kvcache_blocks=4,
                kvcache_block_size=4,
                enable_dynamic_chunked_prefill=False,
            )
        )
        running = Sequence([1, 2, 3, 4])
        scheduler.block_manager.allocate(running, num_cached_blocks=0)
        running.num_cached_tokens = len(running)
        running.status = SequenceStatus.RUNNING
        running.is_prefill = False
        scheduler.running.append(running)
        waiting = Sequence([5, 6, 7, 8])
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])
        self.assertEqual(list(scheduler.waiting), [waiting])
        self.assertEqual(waiting.block_table, [])
        self.assertEqual(len(scheduler.running), scheduler.max_num_seqs)

    def test_scheduler_releases_recurrent_state_slot_on_preemption(self):
        scheduler = Scheduler(
            FakeConfig(
                num_kvcache_blocks=3,
                kvcache_block_size=4,
            )
        )
        scheduler.state_manager = StateSlotManager(1)
        seq = Sequence([1, 2, 3, 4])
        scheduler.waiting.append(seq)

        result = scheduler.schedule()

        self.assertEqual(result.prefill_seqs, [seq])
        self.assertEqual(seq.state_slot, 0)
        self.assertEqual(scheduler.state_manager.num_used_slots, 1)

        scheduler.running.remove(seq)
        scheduler.preempt(seq)

        self.assertIsNone(seq.state_slot)
        self.assertEqual(scheduler.state_manager.num_used_slots, 0)
        self.assertEqual(scheduler.state_manager.num_free_slots, 1)

    def test_preempted_hybrid_request_replays_from_a_reset_state_slot(self):
        scheduler = Scheduler(
            FakeConfig(
                num_kvcache_blocks=4,
                kvcache_block_size=4,
                model_spec=types.SimpleNamespace(is_hybrid=True),
            )
        )
        seq = Sequence([1, 2, 3, 4])
        scheduler.add(seq)

        first = scheduler.schedule()
        self.assertEqual(first.prefill_seqs, [seq])
        scheduler.postprocess_mixed(first, [5])
        self.assertEqual(seq.num_cached_tokens, 4)
        first_slot = seq.state_slot
        self.assertIsNotNone(first_slot)

        scheduler.running.remove(seq)
        scheduler.preempt(seq)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertIsNone(seq.state_slot)

        replay = scheduler.schedule()

        self.assertEqual(replay.prefill_seqs, [seq])
        self.assertEqual(seq.num_scheduled_tokens, len(seq))
        self.assertEqual(seq.num_cached_tokens, 0)
        replay_slot = seq.state_slot
        self.assertIsNotNone(replay_slot)
        self.assertNotEqual(replay_slot, first_slot)
        self.assertTrue(scheduler.state_manager.owns(seq.seq_id, replay_slot))
        self.assertEqual(scheduler.state_manager.num_used_slots, 1)
        assert_block_conservation(self, scheduler.block_manager)

    def test_dynamic_allocation_can_grow_a_sequence(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        seq = Sequence(list(range(12)))

        manager.allocate(seq, num_cached_blocks=0, num_blocks=1)
        self.assertEqual(len(seq.block_table), 1)
        self.assertTrue(manager.can_grow(seq, 2))
        manager.grow(seq, 2)
        self.assertEqual(len(seq.block_table), 2)
        assert_block_conservation(self, manager)


if __name__ == "__main__":
    unittest.main()
