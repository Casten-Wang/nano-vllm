from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.metrics import EngineMetrics
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
    async_state = {"value": "receiving", "error": None}
    send_state = {"value": "sending", "error": None}

    def rank_results(method_name, *args):
        if method_name == "start_sequence_cache_receive":
            return [
                {"rank": rank, "started": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "poll_sequence_cache_receive":
            return [
                {
                    "rank": rank,
                    "state": async_state["value"],
                    **(
                        {"error": async_state["error"]}
                        if async_state["error"] is not None
                        else {}
                    ),
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "poll_sequence_cache_receives":
            transfer_ids = args[0]
            return [
                {
                    "rank": rank,
                    "receives": {
                        transfer_id: {
                            "state": async_state["value"],
                            **(
                                {"error": async_state["error"]}
                                if async_state["error"] is not None
                                else {}
                            ),
                        }
                        for transfer_id in transfer_ids
                    },
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "abort_sequence_cache_receive":
            return [
                {"rank": rank, "aborted": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "start_sequence_cache_send":
            return [
                {"rank": rank, "started": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "poll_sequence_cache_send":
            return [
                {
                    "rank": rank,
                    "state": send_state["value"],
                    **(
                        {"error": send_state["error"]}
                        if send_state["error"] is not None
                        else {}
                    ),
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "poll_sequence_cache_sends":
            transfer_ids = args[0]
            return [
                {
                    "rank": rank,
                    "sends": {
                        transfer_id: {
                            "state": send_state["value"],
                            **(
                                {"error": send_state["error"]}
                                if send_state["error"] is not None
                                else {}
                            ),
                        }
                        for transfer_id in transfer_ids
                    },
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "finish_sequence_cache_send":
            return [
                {"rank": rank, "sent_bytes": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "abort_sequence_cache_send":
            return [
                {"rank": rank, "aborted": 1}
                for rank in range(tensor_parallel_size)
            ]
        seq = args[0]
        value_name = (
            "cached_tokens"
            if method_name in {
                "receive_sequence_cache_from_endpoint",
                "install_sequence_cache_receive",
            }
            else "sent_bytes"
        )
        value = seq.num_prompt_tokens if value_name == "cached_tokens" else 1
        return [
            {"rank": rank, value_name: value}
            for rank in range(tensor_parallel_size)
        ]

    engine.model_runner = SimpleNamespace(
        call=Mock(),
        call_rank_results=Mock(side_effect=rank_results),
    )
    engine._remote_prefill_receive_tokens = {}
    engine._remote_prefill_receive_started_at = {}
    engine._remote_prefill_receive_errors = {}
    engine._remote_prefill_send_started_at = {}
    engine._remote_prefill_send_errors = {}
    engine._test_async_state = async_state
    engine._test_send_state = send_state
    engine.metrics = EngineMetrics()
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


def test_step_polls_active_async_receive_then_continues_scheduling():
    engine = make_engine()
    engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )
    engine.start_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()
    assert engine.step() == ([], 0, 0, 0)
    engine.model_runner.call_rank_results.assert_called_once_with(
        "poll_sequence_cache_receives",
        ["request/attempt-1"],
    )


def test_step_batches_multiple_async_receive_polls_into_one_tp_command():
    engine = make_engine()
    for index in range(2):
        transfer_id = f"request/attempt-{index}"
        engine.add_remote_prefill_request(
            [index + 1, 2, 3, 4],
            SamplingParams(max_tokens=4),
            transfer_id=transfer_id,
        )
        engine.start_remote_prefill_receive(
            transfer_id,
            9,
            [("127.0.0.1", 20001 + index)],
        )
    engine.model_runner.call_rank_results.reset_mock()

    assert engine.step() == ([], 0, 0, 0)
    engine.model_runner.call_rank_results.assert_called_once_with(
        "poll_sequence_cache_receives",
        ["request/attempt-0", "request/attempt-1"],
    )
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_poll_calls"] == 1
    assert metrics["remote_prefill_requests_polled"] == 2


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
    engine.model_runner.call_rank_results.assert_called_once()


def test_engine_receive_failure_releases_destination_and_requeues_prefill():
    engine = make_engine(tensor_parallel_size=1)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    engine.model_runner.call_rank_results.side_effect = RuntimeError(
        "checksum mismatch"
    )

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


def test_engine_receive_timeout_during_ack_releases_destination():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    _, session = engine.scheduler.remote_prefills["request/attempt-1"]
    session.deadline = 0.0

    with pytest.raises(RuntimeError, match="terminal"):
        engine.receive_remote_prefill(
            "request/attempt-1",
            9,
            [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
        )

    seq = engine.scheduler.waiting[0]
    assert seq.seq_id == seq_id
    assert seq.status is SequenceStatus.WAITING
    assert seq.block_table == []
    assert seq.state_slot is None
    assert not engine.scheduler.remote_prefills
    assert session.failure_reason == "cache transfer timed out"


def test_engine_rejects_inconsistent_rank_receive_results():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )
    engine.model_runner.call_rank_results.side_effect = None
    engine.model_runner.call_rank_results.return_value = [
        {"rank": 0, "cached_tokens": 4},
        {"rank": 1, "cached_tokens": 3},
    ]

    with pytest.raises(RuntimeError, match="expected 4"):
        engine.receive_remote_prefill(
            "request/attempt-1",
            9,
            [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
        )

    seq = engine.scheduler.waiting[0]
    assert seq.seq_id == seq_id
    assert seq.block_table == []
    assert seq.state_slot is None
    assert not engine.scheduler.remote_prefills


def test_engine_async_receive_commits_only_after_all_ranks_are_ready():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )

    started_id = engine.start_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    assert started_id == seq_id
    assert engine.poll_remote_prefill_receive("request/attempt-1") is None
    assert not engine.scheduler.running

    engine._test_async_state["value"] = "ready"
    assert engine.poll_remote_prefill_receive("request/attempt-1") == seq_id
    assert engine.scheduler.running[0].completion_token_ids == [9]
    assert not engine.scheduler.remote_prefills
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_receive_started"] == 1
    assert metrics["remote_prefill_receive_committed"] == 1
    assert metrics["remote_prefill_poll_calls"] == 2


def test_engine_async_receive_failure_aborts_all_ranks_and_falls_back():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )
    engine.start_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    engine._test_async_state.update(value="failed", error="checksum mismatch")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        engine.poll_remote_prefill_receive("request/attempt-1")

    assert engine.scheduler.waiting[0].seq_id == seq_id
    assert not engine.scheduler.remote_prefills
    assert "request/attempt-1" not in engine._remote_prefill_receive_tokens
    assert engine.metrics.to_dict()["remote_prefill_receive_failed"] == 1


def test_engine_async_receive_can_be_cancelled_explicitly():
    engine = make_engine()
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )
    engine.start_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001)],
    )

    assert engine.abort_remote_prefill_receive(
        "request/attempt-1",
        reason="client disconnected",
    ) == seq_id
    assert engine.scheduler.waiting[0].seq_id == seq_id
    assert not engine.scheduler.remote_prefills
    assert engine.metrics.to_dict()["remote_prefill_receive_cancelled"] == 1


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
    engine.model_runner.call_rank_results.assert_called_once()


def test_engine_keeps_prefill_source_when_receiver_rejects_payload():
    engine = make_engine()
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
    engine.scheduler.add(seq)
    result = engine.scheduler.schedule()
    engine.scheduler.postprocess_mixed(result, [9])
    engine.model_runner.call_rank_results.side_effect = RuntimeError(
        "receiver rejected"
    )

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


def _prepare_remote_prefill_source(engine, prompt_token: int):
    seq = Sequence([prompt_token, 2, 3, 4], SamplingParams(max_tokens=4))
    engine.scheduler.block_manager.allocate(seq, 0)
    seq.state_slot = engine.scheduler.state_manager.acquire(seq.seq_id)
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.append_token(9)
    engine.scheduler.running.append(seq)
    return seq


def test_async_send_pauses_decode_until_every_rank_acknowledges():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)

    assert engine.start_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    ) == 9
    assert seq.status is SequenceStatus.TRANSFERRING
    assert seq not in engine.scheduler.running
    assert engine.poll_remote_prefill_send("request/attempt-1") is None
    assert engine.scheduler.block_manager.num_used_blocks == 1

    engine._test_send_state["value"] = "ready"
    assert engine.poll_remote_prefill_send("request/attempt-1") == seq.seq_id
    assert seq.status is SequenceStatus.TRANSFERRED
    assert seq.state_slot is None
    assert engine.scheduler.block_manager.num_used_blocks == 0
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_send_started"] == 1
    assert metrics["remote_prefill_send_committed"] == 1
    assert metrics["remote_prefill_send_poll_calls"] == 2


def test_async_send_failure_restores_source_at_original_running_position():
    engine = make_engine(tensor_parallel_size=2)
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    assert list(engine.scheduler.running) == [first, second]
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    engine._test_send_state.update(value="failed", error="receiver rejected")

    with pytest.raises(RuntimeError, match="receiver rejected"):
        engine.poll_remote_prefill_send("request/attempt-1")

    assert first.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [first, second]
    assert first.block_table
    assert first.state_slot is not None
    assert engine.metrics.to_dict()["remote_prefill_send_failed"] == 1


def test_async_send_can_be_cancelled_without_releasing_source_state():
    engine = make_engine()
    seq = _prepare_remote_prefill_source(engine, 1)
    engine.start_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )

    assert engine.abort_remote_prefill_send("request/attempt-1") == seq.seq_id
    assert seq.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [seq]
    assert seq.block_table
    assert seq.state_slot is not None
    assert engine.metrics.to_dict()["remote_prefill_send_cancelled"] == 1


def test_step_batches_multiple_async_send_polls_into_one_tp_command():
    engine = make_engine()
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.start_remote_prefill_send(
        second.seq_id,
        "request/attempt-2",
        [("127.0.0.1", 20002)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    assert engine.step() == ([], 0, 0, 0)
    engine.model_runner.call_rank_results.assert_called_once_with(
        "poll_sequence_cache_sends",
        ["request/attempt-1", "request/attempt-2"],
    )
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_send_poll_calls"] == 1
    assert metrics["remote_prefill_send_requests_polled"] == 2
