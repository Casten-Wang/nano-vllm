import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8
    eos: int = -1
    kvcache_block_size: int = 4
    num_kvcache_blocks: int = 32
    enable_dynamic_chunked_prefill: bool = True


MANAGED_MODULES = (
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
SAVED_MODULES = {name: sys.modules.get(name) for name in MANAGED_MODULES}

nanovllm_pkg = types.ModuleType("nanovllm")
engine_pkg = types.ModuleType("nanovllm.engine")
sys.modules["nanovllm"] = nanovllm_pkg
sys.modules["nanovllm.engine"] = engine_pkg


config_mod = types.ModuleType("nanovllm.config")
config_mod.Config = FakeConfig
sys.modules["nanovllm.config"] = config_mod

sampling_mod = types.ModuleType("nanovllm.sampling_params")


class FakeSamplingParams:
    temperature = 0.0
    top_k = -1
    top_p = 1.0
    max_tokens = 8
    ignore_eos = True


sampling_mod.SamplingParams = FakeSamplingParams
sys.modules["nanovllm.sampling_params"] = sampling_mod

xxhash_mod = types.ModuleType("xxhash")


class FakeXXH64:
    def __init__(self):
        self.parts = []

    def update(self, value):
        self.parts.append(bytes(value))

    def intdigest(self):
        return hash(tuple(self.parts)) & ((1 << 64) - 1)


xxhash_mod.xxh64 = FakeXXH64
sys.modules["xxhash"] = xxhash_mod

sequence_mod = load_module("nanovllm.engine.sequence", ROOT / "nanovllm" / "engine" / "sequence.py")
block_manager_mod = load_module("nanovllm.engine.block_manager", ROOT / "nanovllm" / "engine" / "block_manager.py")
load_module("nanovllm.engine.state_manager", ROOT / "nanovllm" / "engine" / "state_manager.py")
scheduler_mod = load_module("nanovllm.engine.scheduler", ROOT / "nanovllm" / "engine" / "scheduler.py")

Sequence = sequence_mod.Sequence
SequenceStatus = sequence_mod.SequenceStatus
Scheduler = scheduler_mod.Scheduler
ScheduleResult = scheduler_mod.ScheduleResult

for name, module in SAVED_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def make_scheduler(max_tokens=8, max_seqs=8, block_size=4, num_blocks=32):
    Sequence.block_size = block_size
    return Scheduler(
        FakeConfig(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=max_tokens,
            kvcache_block_size=block_size,
            num_kvcache_blocks=num_blocks,
        )
    )


def test_dynamic_schedule_decodes_first_and_uses_remaining_budget_for_prefill():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    running = Sequence([1, 2, 3, 4])
    waiting = Sequence([10] * 20)
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert isinstance(result, ScheduleResult)
    assert result.decode_seqs == [running]
    assert result.prefill_seqs == [waiting]
    assert running.num_scheduled_tokens == 1
    assert waiting.num_scheduled_tokens == 7
    assert waiting.num_cached_tokens == 0
    assert waiting.block_table


def test_dynamic_schedule_prefill_waits_when_decode_fills_token_budget():
    scheduler = make_scheduler(max_tokens=2, max_seqs=8, block_size=4)
    running_a = Sequence([1, 2, 3, 4])
    running_b = Sequence([5, 6, 7, 8])
    waiting = Sequence([10] * 20)
    for seq in (running_a, running_b):
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert isinstance(result, ScheduleResult)
    assert result.decode_seqs == [running_a, running_b]
    assert result.prefill_seqs == []
    assert len(scheduler.waiting) == 1
    assert waiting.block_table == []
    assert scheduler.prefill_starved_steps == 1
    assert scheduler.current_prefill_starvation_steps == 1
    assert scheduler.max_prefill_starvation_steps == 1


def test_dynamic_prefill_continues_partially_cached_sequence():
    scheduler = make_scheduler(max_tokens=6, max_seqs=8, block_size=4)
    seq = Sequence([10] * 13)
    scheduler.waiting.append(seq)

    first = scheduler.schedule()
    assert first.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 6
    scheduler.postprocess_mixed(first, [100])

    second = scheduler.schedule()
    assert second.prefill_seqs == [seq]
    assert seq.num_cached_tokens == 6
    assert seq.num_scheduled_tokens == 6


def test_dynamic_prefill_allocates_blocks_incrementally():
    scheduler = make_scheduler(max_tokens=4, max_seqs=8, block_size=4)
    seq = Sequence([10] * 20)
    scheduler.waiting.append(seq)

    first = scheduler.schedule()
    assert first.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 4
    assert len(seq.block_table) == 1

    scheduler.postprocess_mixed(first, [100])
    assert seq.num_cached_tokens == 4

    second = scheduler.schedule()
    assert second.prefill_seqs == [seq]
    assert seq.num_scheduled_tokens == 4
    assert len(seq.block_table) == 2


def test_dynamic_prefix_hit_binds_cached_blocks_and_prefills_the_tail():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    original = Sequence(list(range(8)))
    scheduler.block_manager.allocate(original, 0)
    original.num_scheduled_tokens = len(original)
    scheduler.block_manager.hash_blocks(original)
    cached_block = original.block_table[0]
    scheduler.block_manager.deallocate(original)

    repeated = Sequence(list(range(8)))
    scheduler.waiting.append(repeated)
    result = scheduler.schedule()

    assert result.prefill_seqs == [repeated]
    assert repeated.num_cached_tokens == 4
    assert repeated.num_scheduled_tokens == 4
    assert len(repeated.block_table) == 2
    assert repeated.block_table[0] == cached_block
    assert repeated.status == SequenceStatus.RUNNING


def test_dynamic_decode_rotates_running_requests_when_budget_is_small():
    scheduler = make_scheduler(max_tokens=1, max_seqs=8, block_size=4)
    running = []
    for token in (1, 5, 9):
        seq = Sequence([token] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)

    observed = []
    for token_id in (101, 102, 103):
        result = scheduler.schedule()
        assert result.decode_seqs
        observed.append(result.decode_seqs[0])
        scheduler.postprocess_mixed(result, [token_id])

    assert observed == running


def test_dynamic_schedule_counts_unscheduled_running_requests_in_slot_budget():
    scheduler = make_scheduler(max_tokens=2, max_seqs=3, block_size=4)
    running = []
    for token in (1, 5, 9):
        seq = Sequence([token] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([10] * 4)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert result.decode_seqs == running[:2]
    assert result.prefill_seqs == []
    assert list(scheduler.running) == [running[2], *running[:2]]
    assert list(scheduler.waiting) == [waiting]
    assert waiting.block_table == []


def test_dynamic_scheduler_tracks_and_resets_consecutive_prefill_starvation():
    scheduler = make_scheduler(max_tokens=1, max_seqs=3, block_size=4)
    running = Sequence([1, 2, 3, 4])
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)
    waiting = Sequence([10] * 4)
    scheduler.waiting.append(waiting)

    for token_id in (101, 102):
        result = scheduler.schedule()
        assert result.prefill_seqs == []
        scheduler.postprocess_mixed(result, [token_id])

    assert scheduler.prefill_starved_steps == 2
    assert scheduler.current_prefill_starvation_steps == 2
    assert scheduler.max_prefill_starvation_steps == 2

    scheduler.max_num_batched_tokens = 2
    result = scheduler.schedule()

    assert result.prefill_seqs == [waiting]
    assert scheduler.prefill_starved_steps == 2
    assert scheduler.current_prefill_starvation_steps == 0
    assert scheduler.max_prefill_starvation_steps == 2


def test_dynamic_scheduler_reserves_prefill_after_starvation_threshold():
    scheduler = make_scheduler(max_tokens=2, max_seqs=3, block_size=4)
    scheduler.prefill_starvation_threshold = 2
    running = []
    for token in (1, 5):
        seq = Sequence([token] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([10] * 4)
    scheduler.waiting.append(waiting)

    for token_ids in ((101, 102), (103, 104)):
        result = scheduler.schedule()
        assert result.prefill_seqs == []
        scheduler.postprocess_mixed(result, list(token_ids))

    result = scheduler.schedule()

    assert result.decode_seqs == [running[0]]
    assert result.prefill_seqs == [waiting]
    assert waiting.num_scheduled_tokens == 1
    assert scheduler.current_prefill_starvation_steps == 0
    assert scheduler.max_prefill_starvation_steps == 2


def test_kv_pressure_workload_preempts_and_eventually_completes():
    scheduler = make_scheduler(
        max_tokens=2048,
        max_seqs=8,
        block_size=256,
        num_blocks=12,
    )
    initial = [Sequence([token] * 256) for token in range(1, 5)]
    for seq in initial:
        seq.max_tokens = 16
        scheduler.add(seq)
    all_sequences = list(initial)
    injected = False
    decode_steps = 0

    for _ in range(100):
        if scheduler.is_finished():
            break
        result = scheduler.schedule()
        scheduler.postprocess_mixed(result, [100] * len(result.seqs))
        if result.decode_seqs and not injected:
            decode_steps += 1
            if decode_steps == 1:
                added = [Sequence([token] * 1024) for token in range(5, 9)]
                for seq in added:
                    seq.max_tokens = 16
                    scheduler.add(seq)
                all_sequences.extend(added)
                injected = True
    else:
        raise AssertionError("KV-pressure workload did not terminate")

    assert injected
    assert scheduler.is_finished()
    assert all(seq.is_finished for seq in all_sequences)
    assert scheduler.preemption_count > 0
    assert scheduler.preempted_token_progress > 0
    assert scheduler.reclaimed_kv_blocks > 0
    assert scheduler.block_manager.num_used_blocks == 0


def test_dynamic_prefill_only_result_uses_prefill_mode():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    waiting = Sequence([10] * 6)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert isinstance(result, ScheduleResult)
    assert result.is_prefill
    assert not result.is_mixed
    assert result.seqs == [waiting]
    assert waiting.is_prefill
    assert waiting.num_scheduled_tokens == 6
    assert scheduler.prefill_starved_steps == 0
    assert scheduler.current_prefill_starvation_steps == 0


def test_dynamic_decode_only_result_uses_decode_mode():
    scheduler = make_scheduler(max_tokens=1, max_seqs=8, block_size=4)
    running = Sequence([1, 2, 3, 4])
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)

    result = scheduler.schedule()

    assert isinstance(result, ScheduleResult)
    assert result.is_decode
    assert not result.is_mixed
    assert result.seqs == [running]
    assert not running.is_prefill
    assert running.num_scheduled_tokens == 1


def test_mixed_postprocess_keeps_decode_then_prefill_token_order():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    running = Sequence([1, 2, 3, 4])
    waiting = Sequence([10, 11, 12, 13])
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()
    assert result.is_mixed

    scheduler.postprocess_mixed(result, [101, 202])

    assert running.last_token == 101
    assert waiting.last_token == 202
    assert running.num_cached_tokens == 5
    assert waiting.num_cached_tokens == 4


def test_mixed_postprocess_rejects_sample_count_before_mutating_sequences():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    running = Sequence([1, 2, 3, 4])
    waiting = Sequence([10, 11, 12, 13])
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)
    scheduler.waiting.append(waiting)
    result = scheduler.schedule()
    before = [
        (seq.num_tokens, seq.num_cached_tokens, seq.last_token)
        for seq in result.seqs
    ]

    for token_ids in ([101], [101, 202, 303]):
        try:
            scheduler.postprocess_mixed(result, token_ids)
        except RuntimeError as error:
            assert "expected 2" in str(error)
        else:
            raise AssertionError("mismatched sampler output was accepted")
        assert [
            (seq.num_tokens, seq.num_cached_tokens, seq.last_token)
            for seq in result.seqs
        ] == before


if __name__ == "__main__":
    test_dynamic_schedule_decodes_first_and_uses_remaining_budget_for_prefill()
    test_dynamic_schedule_prefill_waits_when_decode_fills_token_budget()
    test_dynamic_prefill_continues_partially_cached_sequence()
    test_dynamic_prefill_allocates_blocks_incrementally()
    test_dynamic_prefix_hit_binds_cached_blocks_and_prefills_the_tail()
    test_dynamic_decode_rotates_running_requests_when_budget_is_small()
    test_dynamic_prefill_only_result_uses_prefill_mode()
    test_dynamic_decode_only_result_uses_decode_mode()
    test_mixed_postprocess_keeps_decode_then_prefill_token_order()
