import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    prefill_starvation_token_budget: int = 1
    preemption_policy: str = "fcfs"


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
cache_transfer_mod = load_module(
    "cache_transfer_under_test",
    ROOT / "nanovllm" / "engine" / "cache_transfer.py",
)

Sequence = sequence_mod.Sequence
SequenceStatus = sequence_mod.SequenceStatus
Scheduler = scheduler_mod.Scheduler
ScheduleResult = scheduler_mod.ScheduleResult
CacheTransferPhase = cache_transfer_mod.CacheTransferPhase
CacheTransferSession = cache_transfer_mod.CacheTransferSession

for name, module in SAVED_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def make_scheduler(
    max_tokens=8,
    max_seqs=8,
    block_size=4,
    num_blocks=32,
    preemption_policy="fcfs",
    starvation_token_budget=1,
    hybrid=False,
):
    Sequence.block_size = block_size
    config = FakeConfig(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=max_tokens,
            kvcache_block_size=block_size,
            num_kvcache_blocks=num_blocks,
            preemption_policy=preemption_policy,
            prefill_starvation_token_budget=starvation_token_budget,
        )
    if hybrid:
        config.model_spec = SimpleNamespace(is_hybrid=True)
    return Scheduler(config)


def make_transfer(transfer_id="request/attempt-1", tp_size=2):
    return CacheTransferSession(
        transfer_id,
        tp_size,
        started_at=10.0,
        timeout_s=5.0,
    )


def test_remote_prefill_reserves_resources_until_all_ranks_commit():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=2,
        block_size=4,
        num_blocks=4,
        hybrid=True,
    )
    seq = Sequence([1, 2, 3, 4, 5])
    scheduler.add(seq)
    session = make_transfer()

    scheduler.reserve_remote_prefill(seq, session)

    assert seq.status is SequenceStatus.TRANSFERRING
    assert seq not in scheduler.waiting
    assert seq not in scheduler.running
    assert len(seq.block_table) == 2
    assert seq.state_slot is not None
    assert scheduler.block_manager.num_used_blocks == 2
    assert not scheduler.is_finished()
    idle = scheduler.schedule()
    assert idle == ScheduleResult(prefill_seqs=[], decode_seqs=[])

    session.acknowledge(0, now=11.0)
    session.acknowledge(1, now=11.0)
    scheduler.commit_remote_prefill(
        session.transfer_id,
        first_token_id=9,
        now=11.0,
    )

    assert session.phase is CacheTransferPhase.COMMITTED
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_cached_tokens == seq.num_prompt_tokens
    assert seq.completion_token_ids == [9]
    assert list(scheduler.running) == [seq]
    assert scheduler.block_manager.num_used_blocks == 2
    assert scheduler.state_manager.num_used_slots == 1


def test_legacy_scheduler_can_idle_while_remote_prefill_is_pending():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=1,
        block_size=4,
        num_blocks=2,
        hybrid=True,
    )
    scheduler.enable_dynamic_chunked_prefill = False
    seq = Sequence([1, 2, 3, 4])
    scheduler.add(seq)
    scheduler.reserve_remote_prefill(seq, make_transfer(tp_size=1))

    assert scheduler.schedule() == ([], False)


def test_failed_remote_prefill_releases_resources_and_requeues_locally():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=2,
        block_size=4,
        num_blocks=4,
        hybrid=True,
    )
    seq = Sequence([1, 2, 3, 4, 5])
    scheduler.add(seq)
    session = make_transfer()
    scheduler.reserve_remote_prefill(seq, session)

    fallback = scheduler.fail_remote_prefill(
        session.transfer_id,
        rank=1,
        reason="payload validation failed",
        now=11.0,
    )

    assert fallback is seq
    assert session.phase is CacheTransferPhase.ABORTED
    assert seq.status is SequenceStatus.WAITING
    assert seq.state_slot is None
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert list(scheduler.waiting) == [seq]
    assert scheduler.block_manager.num_used_blocks == 0
    assert scheduler.state_manager.num_used_slots == 0


def test_remote_prefill_timeout_requeues_and_preserves_capacity():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=1,
        block_size=4,
        num_blocks=2,
        hybrid=True,
    )
    seq = Sequence([1, 2, 3, 4])
    scheduler.add(seq)
    session = make_transfer(tp_size=1)
    scheduler.reserve_remote_prefill(seq, session)

    fallback = scheduler.poll_remote_prefills(now=15.0)

    assert fallback == [seq]
    assert session.phase is CacheTransferPhase.TIMED_OUT
    assert scheduler.block_manager.num_free_blocks == 2
    assert scheduler.state_manager.num_free_slots == 1
    result = scheduler.schedule()
    assert result.prefill_seqs == [seq]


def test_remote_prefill_reservation_counts_against_sequence_capacity():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=1,
        block_size=4,
        num_blocks=4,
        hybrid=True,
    )
    first = Sequence([1, 2, 3, 4])
    second = Sequence([5, 6, 7, 8])
    scheduler.add(first)
    scheduler.add(second)
    scheduler.reserve_remote_prefill(first, make_transfer("first", tp_size=1))

    with pytest.raises(RuntimeError, match="no sequence slot"):
        scheduler.reserve_remote_prefill(
            second,
            make_transfer("second", tp_size=1),
        )

    assert second.status is SequenceStatus.WAITING
    assert second.block_table == []
    assert second.state_slot is None
    assert scheduler.block_manager.num_used_blocks == 1
    assert scheduler.state_manager.num_used_slots == 1
    assert list(scheduler.waiting) == [second]


@pytest.mark.parametrize(
    ("sampling_params", "first_token_id"),
    [
        (SimpleNamespace(temperature=0.0, top_k=-1, top_p=1.0,
                         max_tokens=1, ignore_eos=True), 9),
        (SimpleNamespace(temperature=0.0, top_k=-1, top_p=1.0,
                         max_tokens=8, ignore_eos=False), -1),
    ],
)
def test_remote_prefill_terminal_first_token_releases_resources(
    sampling_params,
    first_token_id,
):
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=1,
        block_size=4,
        num_blocks=2,
        hybrid=True,
    )
    seq = Sequence([1, 2, 3, 4], sampling_params)
    scheduler.add(seq)
    session = make_transfer(tp_size=1)
    scheduler.reserve_remote_prefill(seq, session)
    session.acknowledge(0, now=11.0)

    scheduler.commit_remote_prefill(
        session.transfer_id,
        first_token_id,
        now=11.0,
    )

    assert seq.status is SequenceStatus.FINISHED
    assert seq.completion_token_ids == [first_token_id]
    assert not scheduler.running
    assert scheduler.block_manager.num_used_blocks == 0
    assert scheduler.state_manager.num_used_slots == 0
    assert scheduler.is_finished()


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
    assert list(scheduler.running) == [running]


def test_newly_admitted_prefill_does_not_jump_ahead_of_existing_decode():
    scheduler = make_scheduler(max_tokens=8, max_seqs=8, block_size=4)
    running = Sequence([1, 2, 3, 4])
    waiting = Sequence([10] * 4)
    scheduler.block_manager.allocate(running, 0)
    running.status = SequenceStatus.RUNNING
    running.is_prefill = False
    running.num_cached_tokens = len(running)
    scheduler.running.append(running)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert result.decode_seqs == [running]
    assert result.prefill_seqs == [waiting]
    assert list(scheduler.running) == [running, waiting]


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


def test_dynamic_scheduler_reserves_useful_prefill_chunk_after_starvation():
    scheduler = make_scheduler(
        max_tokens=16,
        max_seqs=20,
        block_size=4,
        starvation_token_budget=6,
    )
    scheduler.prefill_starvation_threshold = 1
    running = []
    for token in range(16):
        seq = Sequence([token + 1] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([99] * 20)
    scheduler.waiting.append(waiting)

    first = scheduler.schedule()
    assert first.decode_seqs == running
    assert first.prefill_seqs == []
    scheduler.postprocess_mixed(first, [100] * len(first.seqs))

    second = scheduler.schedule()

    assert len(second.decode_seqs) == 10
    assert second.prefill_seqs == [waiting]
    assert waiting.num_scheduled_tokens == 6
    assert scheduler.current_prefill_starvation_steps == 0


def test_fairness_reserve_keeps_half_of_multi_token_budget_for_decode():
    scheduler = make_scheduler(
        max_tokens=4,
        max_seqs=8,
        block_size=4,
        starvation_token_budget=256,
    )
    scheduler.prefill_starvation_threshold = 1
    running = []
    for token in range(4):
        seq = Sequence([token + 1] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([99] * 8)
    scheduler.waiting.append(waiting)

    first = scheduler.schedule()
    scheduler.postprocess_mixed(first, [100] * len(first.seqs))
    second = scheduler.schedule()

    assert len(second.decode_seqs) == 2
    assert second.prefill_seqs == [waiting]
    assert waiting.num_scheduled_tokens == 2


def test_fairness_reserve_does_not_waste_budget_on_short_prefill():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=9,
        block_size=4,
        starvation_token_budget=256,
    )
    scheduler.prefill_starvation_threshold = 1
    running = []
    for token in range(8):
        seq = Sequence([token + 1] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([99])
    scheduler.waiting.append(waiting)

    first = scheduler.schedule()
    assert first.decode_seqs == running
    assert first.prefill_seqs == []
    scheduler.postprocess_mixed(first, [100] * len(first.seqs))

    second = scheduler.schedule()

    assert len(second.decode_seqs) == 7
    assert second.prefill_seqs == [waiting]
    assert waiting.num_scheduled_tokens == 1
    assert second.num_decode_tokens + second.num_prefill_tokens == 8


def test_fairness_reserve_accounts_for_unbound_prefix_cache_hits():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=9,
        block_size=4,
        starvation_token_budget=256,
    )
    scheduler.prefill_starvation_threshold = 1
    original = Sequence(list(range(9)))
    scheduler.block_manager.allocate(original, 0)
    original.num_scheduled_tokens = len(original)
    scheduler.block_manager.hash_blocks(original)
    scheduler.block_manager.deallocate(original)

    running = []
    for token in range(8):
        seq = Sequence([token + 20] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence(list(range(9)))
    scheduler.waiting.append(waiting)

    first = scheduler.schedule()
    assert first.decode_seqs == running
    scheduler.postprocess_mixed(first, [100] * len(first.seqs))
    scheduler.block_manager.reset_cache_stats()

    second = scheduler.schedule()

    # Two cached blocks leave one actual prefill token. Reserving from the
    # stale zero num_cached_tokens value would schedule only four decodes.
    assert len(second.decode_seqs) == 7
    assert second.prefill_seqs == [waiting]
    assert waiting.num_cached_tokens == 8
    assert waiting.num_scheduled_tokens == 1
    assert second.num_decode_tokens + second.num_prefill_tokens == 8
    # The fairness forecast is deliberately excluded from cache hit metrics;
    # only the admission lookup is recorded.
    assert scheduler.block_manager.prefix_cache_queries == 1


def test_fairness_reserve_backfills_decode_when_prefill_cannot_allocate():
    scheduler = make_scheduler(
        max_tokens=8,
        max_seqs=9,
        block_size=4,
        num_blocks=8,
        starvation_token_budget=4,
    )
    scheduler.prefill_starvation_threshold = 1
    scheduler.current_prefill_starvation_steps = 1
    running = []
    for token in range(8):
        seq = Sequence([token + 1] * 4)
        scheduler.block_manager.allocate(seq, 0)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        seq.num_cached_tokens = len(seq)
        scheduler.running.append(seq)
        running.append(seq)
    waiting = Sequence([99] * 4)
    scheduler.waiting.append(waiting)

    result = scheduler.schedule()

    assert result.decode_seqs == running
    assert result.prefill_seqs == []
    assert result.num_decode_tokens == scheduler.max_num_batched_tokens
    assert list(scheduler.waiting) == [waiting]
    assert waiting.block_table == []
    assert scheduler.preemption_count == 0


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


def test_min_recompute_preemption_selects_less_advanced_request():
    scheduler = make_scheduler(preemption_policy="min_recompute")
    current = Sequence([1] * 20)
    expensive = Sequence([2] * 16)
    cheap = Sequence([3] * 4)
    current.num_cached_tokens = 20
    expensive.num_cached_tokens = 16
    cheap.num_cached_tokens = 4
    scheduler.running.extend((expensive, cheap))

    victim = scheduler.select_preemption_victim(current)

    assert victim is cheap
    assert list(scheduler.running) == [expensive]


def test_min_recompute_tie_preserves_older_request():
    scheduler = make_scheduler(preemption_policy="min_recompute")
    older = Sequence([1] * 4)
    newer = Sequence([2] * 4)
    older.num_cached_tokens = newer.num_cached_tokens = 4
    scheduler.running.append(newer)

    victim = scheduler.select_preemption_victim(older)

    assert victim is newer
    assert not scheduler.running


def test_min_recompute_reduces_progress_loss_for_heterogeneous_prompts():
    def run(policy):
        scheduler = make_scheduler(
            max_tokens=2048,
            max_seqs=4,
            block_size=256,
            num_blocks=5,
            preemption_policy=policy,
        )
        for token, length in ((1, 256), (2, 1024)):
            seq = Sequence([token] * length)
            seq.max_tokens = 16
            scheduler.add(seq)
        injected = False
        for _ in range(100):
            if scheduler.is_finished():
                break
            result = scheduler.schedule()
            scheduler.postprocess_mixed(result, [100] * len(result.seqs))
            if result.decode_seqs and not injected:
                for token in (3, 4):
                    seq = Sequence([token] * 512)
                    seq.max_tokens = 16
                    scheduler.add(seq)
                injected = True
        else:
            raise AssertionError("heterogeneous KV-pressure workload did not terminate")
        assert scheduler.is_finished()
        return scheduler

    fcfs = run("fcfs")
    min_recompute = run("min_recompute")

    assert fcfs.preemption_count == 2
    assert min_recompute.preemption_count == 1
    assert fcfs.preempted_token_progress == 1536
    assert min_recompute.preempted_token_progress == 256
    assert min_recompute.preempted_token_progress < fcfs.preempted_token_progress


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
