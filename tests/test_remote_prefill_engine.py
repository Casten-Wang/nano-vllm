from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import ScheduleResult, Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_engine(*, tensor_parallel_size=1):
    config = SimpleNamespace(
        max_model_len=32,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=8,
        enable_dynamic_chunked_prefill=True,
        preemption_policy="fcfs",
        model_spec=SimpleNamespace(is_hybrid=True),
    )
    Sequence.block_size = config.kvcache_block_size
    engine = object.__new__(LLMEngine)
    engine.config = config
    engine.scheduler = Scheduler(config)
    engine.model_runner = SimpleNamespace(call=Mock())
    return engine


def test_idle_remote_prefill_step_polls_without_running_model():
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(
        poll_remote_prefills=Mock(return_value=[]),
        schedule=Mock(return_value=ScheduleResult([], [])),
        num_waiting=1,
        num_running=0,
        block_manager=SimpleNamespace(num_used_blocks=2, num_total_blocks=8),
        prefill_starved_steps=0,
        max_prefill_starvation_steps=0,
        preemption_count=0,
        preempted_token_progress=0,
        max_preempted_token_progress=0,
        reclaimed_kv_blocks=0,
    )
    engine.metrics = SimpleNamespace(record_scheduler_state=Mock())
    engine.model_runner = SimpleNamespace(call=Mock())

    outputs, tokens, prefill_tokens, decode_tokens = engine.step()

    assert outputs == []
    assert (tokens, prefill_tokens, decode_tokens) == (0, 0, 0)
    engine.scheduler.poll_remote_prefills.assert_called_once()
    engine.model_runner.call.assert_not_called()


def test_engine_commits_remote_prefill_only_after_all_rank_receive():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )

    committed_id = engine.receive_remote_prefill(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )

    assert committed_id == seq_id
    seq = engine.scheduler.running[0]
    assert seq.status is SequenceStatus.RUNNING
    assert seq.completion_token_ids == [9]
    assert not engine.scheduler.remote_prefills
    engine.model_runner.call.assert_called_once()


def test_engine_receive_failure_releases_destination_and_requeues_prefill():
    engine = make_engine(tensor_parallel_size=1)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    engine.model_runner.call.side_effect = RuntimeError("checksum mismatch")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        engine.receive_remote_prefill(
            "request/attempt-1",
            9,
            [("127.0.0.1", 20001)],
        )

    seq = engine.scheduler.waiting[0]
    assert seq.seq_id == seq_id
    assert seq.status is SequenceStatus.WAITING
    assert seq.block_table == []
    assert seq.state_slot is None


def test_engine_releases_prefill_source_only_after_receiver_ack():
    engine = make_engine()
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
    engine.scheduler.add(seq)
    result = engine.scheduler.schedule()
    engine.scheduler.postprocess_mixed(result, [9])

    first_token_id = engine.send_remote_prefill(
        seq.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )

    assert first_token_id == 9
    assert seq.status is SequenceStatus.TRANSFERRED
    assert seq.state_slot is None
    assert not engine.scheduler.running
    assert engine.scheduler.block_manager.num_used_blocks == 0
    engine.model_runner.call.assert_called_once()


def test_engine_keeps_prefill_source_when_receiver_rejects_payload():
    engine = make_engine()
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
    engine.scheduler.add(seq)
    result = engine.scheduler.schedule()
    engine.scheduler.postprocess_mixed(result, [9])
    engine.model_runner.call.side_effect = RuntimeError("receiver rejected")

    with pytest.raises(RuntimeError, match="receiver rejected"):
        engine.send_remote_prefill(
            seq.seq_id,
            "request/attempt-1",
            [("127.0.0.1", 20001)],
        )

    assert seq.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [seq]
    assert engine.scheduler.block_manager.num_used_blocks == 1
    assert engine.scheduler.state_manager.num_used_slots == 1
