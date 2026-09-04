from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.metrics import EngineMetrics
from nanovllm.engine.scheduler import ScheduleResult, Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_engine(*, tensor_parallel_size=1):
    config = SimpleNamespace(
        max_model_len=32,
        model_config=SimpleNamespace(vocab_size=32),
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=8,
        enable_dynamic_chunked_prefill=True,
        preemption_policy="fcfs",
        model_spec=SimpleNamespace(is_hybrid=True),
        max_remote_prefill_transfers=2,
        max_remote_prefill_staging_bytes=None,
    )
    Sequence.block_size = config.kvcache_block_size
    engine = object.__new__(LLMEngine)
    engine.config = config
    engine.scheduler = Scheduler(config)
    async_state = {"value": "receiving", "error": None}
    send_state = {"value": "sending", "error": None, "staged_bytes": None}

    def rank_results(method_name, *args):
        if method_name in {
            "start_sequence_cache_receive",
            "start_heterogeneous_sequence_cache_receive",
        }:
            return [
                {"rank": rank, "started": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name in {
            "poll_sequence_cache_receive",
            "poll_heterogeneous_sequence_cache_receive",
        }:
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
        if method_name in {
            "abort_sequence_cache_receive",
            "abort_heterogeneous_sequence_cache_receive",
        }:
            return [
                {"rank": rank, "aborted": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "start_sequence_cache_send":
            return [
                {
                    "rank": rank,
                    "started": 1,
                    "staged_bytes": 100 + rank,
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name in {
            "prepare_heterogeneous_sequence_cache_send",
            "start_heterogeneous_sequence_cache_send",
        }:
            return [
                {
                    "rank": rank,
                    "started": 1,
                    "staged_bytes": 100 + rank,
                }
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "estimate_sequence_cache_bytes":
            return [
                {"rank": rank, "staged_bytes": 100 + rank}
                for rank in range(tensor_parallel_size)
            ]
        if method_name == "estimate_cache_transfer_bytes_for_blocks":
            return [
                {"rank": rank, "staged_bytes": 100 + rank}
                for rank in range(tensor_parallel_size)
            ]
        if method_name in {
            "poll_sequence_cache_send",
            "poll_heterogeneous_sequence_cache_send",
        }:
            return [
                {
                    "rank": rank,
                    "state": send_state["value"],
                    "staged_bytes": (
                        send_state["staged_bytes"]
                        if send_state["staged_bytes"] is not None
                        else 100 + rank if send_state["value"] == "sending" else 0
                    ),
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
                            "staged_bytes": (
                                send_state["staged_bytes"]
                                if send_state["staged_bytes"] is not None
                                else 100 + rank
                                if send_state["value"] == "sending"
                                else 0
                            ),
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
        if method_name in {
            "finish_sequence_cache_send",
            "finish_heterogeneous_sequence_cache_send",
        }:
            return [
                {"rank": rank, "sent_bytes": 1}
                for rank in range(tensor_parallel_size)
            ]
        if method_name in {
            "abort_sequence_cache_send",
            "abort_heterogeneous_sequence_cache_send",
        }:
            return [
                {"rank": rank, "aborted": 1}
                for rank in range(tensor_parallel_size)
            ]
        seq = args[0]
        if method_name in {
            "receive_sequence_cache_from_endpoint",
            "install_sequence_cache_receive",
            "install_heterogeneous_sequence_cache_receive",
        }:
            return [
                {
                    "rank": rank,
                    "cached_tokens": seq.num_prompt_tokens,
                    "received_bytes": 100 + rank,
                }
                for rank in range(tensor_parallel_size)
            ]
        value_name = (
            "cached_tokens"
            if method_name in {
                "receive_sequence_cache_from_endpoint",
                "install_sequence_cache_receive",
                "install_heterogeneous_sequence_cache_receive",
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
        estimate_heterogeneous_cache_transfer_for_blocks=Mock(),
        build_heterogeneous_cache_transfer_plan_for_blocks=Mock(
            return_value=SimpleNamespace(
                profile=SimpleNamespace(
                    source_bytes=tuple(
                        200 + 2 * rank for rank in range(tensor_parallel_size)
                    ),
                    source_staging_bytes=tuple(
                        100 + rank for rank in range(tensor_parallel_size)
                    ),
                )
            )
        ),
        build_heterogeneous_cache_receive_plan_for_blocks=Mock(
            return_value=SimpleNamespace(
                profile=SimpleNamespace(
                    destination_bytes=tuple(
                        100 + rank for rank in range(tensor_parallel_size)
                    )
                )
            )
        ),
    )
    engine._remote_prefill_receive_tokens = {}
    engine._remote_prefill_receive_started_at = {}
    engine._remote_prefill_receive_reserved_staged_bytes = {}
    engine._remote_prefill_receive_staged_bytes = {}
    engine._remote_prefill_receive_expected_bytes = {}
    engine._remote_prefill_receive_errors = {}
    engine._heterogeneous_remote_prefill_source_tp_sizes = {}
    engine._remote_prefill_send_started_at = {}
    engine._remote_prefill_send_staged_bytes = {}
    engine._remote_prefill_send_errors = {}
    engine._heterogeneous_remote_prefill_send_ids = set()
    engine._test_async_state = async_state
    engine._test_send_state = send_state
    engine.metrics = EngineMetrics()
    return engine


def test_local_request_returns_id_and_can_be_cancelled():
    engine = make_engine()

    seq_id = engine.add_request([1, 2, 3], SamplingParams(max_tokens=2))

    assert engine.scheduler.waiting[0].seq_id == seq_id
    assert engine.abort_request(seq_id)
    assert engine.scheduler.is_finished()
    assert not engine.abort_request(seq_id)


@pytest.mark.parametrize(
    ("prompt", "error", "message"),
    [
        ([1, True], TypeError, "prompt token at index 1 must be an integer"),
        ([1, -1], ValueError, "outside the vocabulary range.*-1"),
        ([1, 32], ValueError, "outside the vocabulary range.*32"),
    ],
)
def test_local_request_rejects_invalid_token_ids_before_scheduling(
    prompt,
    error,
    message,
):
    engine = make_engine()

    with pytest.raises(error, match=message):
        engine.add_request(prompt, SamplingParams(max_tokens=2))

    assert engine.scheduler.is_finished()


@pytest.mark.parametrize(
    ("first_token_id", "error"),
    [
        (True, TypeError),
        (-1, ValueError),
        (32, ValueError),
    ],
)
def test_remote_prefill_rejects_invalid_first_token_before_receive(
    first_token_id,
    error,
):
    engine = make_engine()
    engine.add_remote_prefill_request(
        [1, 2, 3],
        SamplingParams(max_tokens=2),
        transfer_id="request/attempt-1",
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(error, match="first_token_id"):
        engine.start_remote_prefill_receive(
            "request/attempt-1",
            first_token_id,
            [("127.0.0.1", 8000)],
        )

    assert "request/attempt-1" in engine.scheduler.remote_prefills
    assert "request/attempt-1" not in engine._remote_prefill_receive_tokens
    engine.model_runner.call_rank_results.assert_not_called()


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
        waiting_prefill_preemptions=0,
        preempted_token_progress=0,
        max_preempted_token_progress=0,
        reclaimed_kv_blocks=0,
        aborted_requests=0,
        prefill_stopped_by_token_budget=0,
        prefill_stopped_by_sequence_capacity=0,
        prefill_stopped_by_kv_capacity=0,
        prefill_stopped_by_decode_kv_reservation=0,
        last_decode_kv_reserve_blocks=0,
    )
    engine.metrics = SimpleNamespace(record_scheduler_state=Mock())
    engine.model_runner = SimpleNamespace(call=Mock())

    outputs, tokens, prefill_tokens, decode_tokens = engine.step()

    assert outputs == []
    assert (tokens, prefill_tokens, decode_tokens) == (0, 0, 0)
    engine.scheduler.poll_remote_prefills.assert_called_once_with(
        now=pytest.approx(engine.scheduler.poll_remote_prefills.call_args.kwargs["now"]),
        exclude_transfer_ids=frozenset(),
    )
    engine.model_runner.call.assert_not_called()


def test_unstarted_receive_timeout_releases_engine_staging_reservation():
    engine = make_engine()
    engine.config.max_remote_prefill_staging_bytes = 150
    capacity_before = engine.remote_prefill_capacity_snapshot()
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    _, session = engine.scheduler.remote_prefills["request/attempt-1"]
    session.deadline = 0.0

    fallback = engine._poll_remote_prefill_reservations()

    assert [seq.seq_id for seq in fallback] == [seq_id]
    assert session.failure_reason == "cache transfer timed out"
    assert not engine.scheduler.remote_prefills
    assert not engine._remote_prefill_receive_reserved_staged_bytes
    assert not engine._remote_prefill_receive_expected_bytes
    capacity_after = engine.remote_prefill_capacity_snapshot()
    assert capacity_after["transfer_slots_used"] == 0
    assert capacity_after["transfer_slots_free"] == capacity_before[
        "transfer_slots_free"
    ]
    assert capacity_after["staging_bytes_reserved"] == 0
    assert capacity_after["staging_bytes_used"] == 0
    assert capacity_after["staging_bytes_free"] == capacity_before[
        "staging_bytes_free"
    ]
    assert capacity_after["waiting_requests"] == 1
    assert engine.metrics.to_dict()["remote_prefill_reservation_timed_out"] == 1


def test_reservation_poll_leaves_active_receive_owned_by_engine():
    engine = make_engine()
    engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    engine.start_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001)],
    )
    _, session = engine.scheduler.remote_prefills["request/attempt-1"]
    session.deadline = 0.0

    assert engine._poll_remote_prefill_reservations() == []

    assert "request/attempt-1" in engine.scheduler.remote_prefills
    assert "request/attempt-1" in engine._remote_prefill_receive_tokens
    assert engine._remote_prefill_receive_staged_bytes["request/attempt-1"] == 100


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
    assert [
        call.args[0]
        for call in engine.model_runner.call_rank_results.call_args_list
    ] == [
        "estimate_cache_transfer_bytes_for_blocks",
        "receive_sequence_cache_from_endpoint",
    ]


def test_engine_receive_failure_releases_destination_and_requeues_prefill():
    engine = make_engine(tensor_parallel_size=1)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        timeout_s=10.0,
    )
    original = engine.model_runner.call_rank_results.side_effect

    def fail_receive(method_name, *args):
        if method_name == "receive_sequence_cache_from_endpoint":
            raise RuntimeError("checksum mismatch")
        return original(method_name, *args)

    engine.model_runner.call_rank_results.side_effect = fail_receive

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
    assert not engine._remote_prefill_receive_expected_bytes
    assert session.failure_reason == "cache transfer timed out"


def test_engine_rejects_inconsistent_rank_receive_results():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )
    original = engine.model_runner.call_rank_results.side_effect

    def inconsistent_receive(method_name, *args):
        if method_name == "receive_sequence_cache_from_endpoint":
            return [
                {"rank": 0, "cached_tokens": 4, "received_bytes": 100},
                {"rank": 1, "cached_tokens": 3, "received_bytes": 101},
            ]
        return original(method_name, *args)

    engine.model_runner.call_rank_results.side_effect = inconsistent_receive

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
    assert not engine._remote_prefill_receive_expected_bytes


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
    assert metrics["remote_prefill_receive_staged_bytes"] == 201
    assert metrics["peak_remote_prefill_receive_staged_bytes"] == 201
    assert metrics["active_remote_prefill_receive_staged_bytes"] == 0
    assert not engine._remote_prefill_receive_staged_bytes
    assert not engine._remote_prefill_receive_expected_bytes


def test_heterogeneous_receive_commits_only_after_all_destination_ranks_install():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_heterogeneous_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        source_tp_size=4,
    )
    assert engine.remote_prefill_capacity_snapshot()["staging_bytes_reserved"] == 201

    assert engine.start_heterogeneous_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    ) == seq_id
    assert engine.poll_remote_prefill_receive("request/attempt-1") is None
    assert not engine.scheduler.running

    engine._test_async_state["value"] = "ready"
    assert engine.poll_remote_prefill_receive("request/attempt-1") == seq_id
    assert engine.scheduler.running[0].completion_token_ids == [9]
    assert not engine.scheduler.remote_prefills
    assert not engine._heterogeneous_remote_prefill_source_tp_sizes
    methods = [
        call.args[0]
        for call in engine.model_runner.call_rank_results.call_args_list
    ]
    assert methods == [
        "start_heterogeneous_sequence_cache_receive",
        "poll_heterogeneous_sequence_cache_receive",
        "poll_heterogeneous_sequence_cache_receive",
        "install_heterogeneous_sequence_cache_receive",
    ]


def test_step_polls_regular_and_heterogeneous_receives_through_separate_paths():
    engine = make_engine(tensor_parallel_size=2)
    engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="regular",
    )
    engine.start_remote_prefill_receive(
        "regular",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    engine.add_heterogeneous_remote_prefill_request(
        [5, 6, 7, 8],
        SamplingParams(max_tokens=4),
        transfer_id="heterogeneous",
        source_tp_size=4,
    )
    engine.start_heterogeneous_remote_prefill_receive(
        "heterogeneous",
        10,
        [("127.0.0.1", 20003), ("127.0.0.1", 20004)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    assert engine.step() == ([], 0, 0, 0)
    assert engine.model_runner.call_rank_results.call_args_list == [
        call("poll_sequence_cache_receives", ["regular"]),
        call("poll_heterogeneous_sequence_cache_receive", "heterogeneous"),
    ]


def test_heterogeneous_receive_failure_aborts_and_requeues_local_prefill():
    engine = make_engine(tensor_parallel_size=2)
    seq_id = engine.add_heterogeneous_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
        source_tp_size=4,
    )
    engine.start_heterogeneous_remote_prefill_receive(
        "request/attempt-1",
        9,
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    engine._test_async_state.update(value="failed", error="peer checksum mismatch")

    with pytest.raises(RuntimeError, match="peer checksum mismatch"):
        engine.poll_remote_prefill_receive("request/attempt-1")

    assert engine.scheduler.waiting[0].seq_id == seq_id
    assert not engine.scheduler.remote_prefills
    assert not engine._remote_prefill_receive_tokens
    assert not engine._heterogeneous_remote_prefill_source_tp_sizes
    assert any(
        call.args[0] == "abort_heterogeneous_sequence_cache_receive"
        for call in engine.model_runner.call_rank_results.call_args_list
    )


def test_heterogeneous_reservation_cancel_and_timeout_release_metadata():
    for expire in (False, True):
        engine = make_engine(tensor_parallel_size=2)
        engine.add_heterogeneous_remote_prefill_request(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=4),
            transfer_id="request/attempt-1",
            source_tp_size=4,
        )
        if expire:
            _, session = engine.scheduler.remote_prefills["request/attempt-1"]
            session.deadline = 0.0
            engine._poll_remote_prefill_reservations()
        else:
            engine.cancel_remote_prefill_reservation("request/attempt-1")
        assert not engine.scheduler.remote_prefills
        assert not engine._remote_prefill_receive_reserved_staged_bytes
        assert not engine._remote_prefill_receive_expected_bytes
        assert not engine._heterogeneous_remote_prefill_source_tp_sizes


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
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_receive_failed"] == 1
    assert metrics["active_remote_prefill_receive_staged_bytes"] == 0


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
    metrics = engine.metrics.to_dict()
    assert metrics["remote_prefill_receive_cancelled"] == 1
    assert metrics["active_remote_prefill_receive_staged_bytes"] == 0


def test_engine_async_receive_rejects_payload_size_mismatch_and_falls_back():
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
    original = engine.model_runner.call_rank_results.side_effect

    def inconsistent_install(method_name, *args):
        results = original(method_name, *args)
        if method_name == "install_sequence_cache_receive":
            results[1]["received_bytes"] += 1
        return results

    engine.model_runner.call_rank_results.side_effect = inconsistent_install
    engine._test_async_state["value"] = "ready"

    with pytest.raises(RuntimeError, match="differ from the preflight"):
        engine.poll_remote_prefill_receive("request/attempt-1")

    assert engine.scheduler.waiting[0].seq_id == seq_id
    assert not engine.scheduler.remote_prefills
    assert not engine._remote_prefill_receive_staged_bytes
    assert engine.metrics.to_dict()["active_remote_prefill_receive_staged_bytes"] == 0


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
    assert [
        call.args[0]
        for call in engine.model_runner.call_rank_results.call_args_list
    ] == ["estimate_sequence_cache_bytes", "send_sequence_cache_to_endpoint"]


def test_engine_keeps_prefill_source_when_receiver_rejects_payload():
    engine = make_engine()
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=4))
    engine.scheduler.add(seq)
    result = engine.scheduler.schedule()
    engine.scheduler.postprocess_mixed(result, [9])
    original = engine.model_runner.call_rank_results.side_effect

    def reject_send(method_name, *args):
        if method_name == "send_sequence_cache_to_endpoint":
            raise RuntimeError("receiver rejected")
        return original(method_name, *args)

    engine.model_runner.call_rank_results.side_effect = reject_send

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
    assert metrics["remote_prefill_send_staged_bytes"] == 201
    assert metrics["peak_remote_prefill_send_staged_bytes"] == 201
    assert metrics["active_remote_prefill_send_staged_bytes"] == 0
    assert metrics["remote_prefill_sent_bytes"] == 2


def test_heterogeneous_send_commits_only_after_every_source_rank_acknowledges():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)

    assert engine.start_heterogeneous_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        4,
        [("127.0.0.1", 20001 + rank) for rank in range(4)],
    ) == 9
    assert seq.status is SequenceStatus.TRANSFERRING
    assert engine.remote_prefill_capacity_snapshot()["staging_bytes_active"] == 201
    assert engine.poll_remote_prefill_send("request/attempt-1") is None
    assert seq.block_table

    engine._test_send_state["value"] = "ready"
    assert engine.poll_remote_prefill_send("request/attempt-1") == seq.seq_id
    assert seq.status is SequenceStatus.TRANSFERRED
    assert not seq.block_table
    assert seq.state_slot is None
    assert not engine._heterogeneous_remote_prefill_send_ids
    methods = [
        item.args[0]
        for item in engine.model_runner.call_rank_results.call_args_list
    ]
    assert methods == [
        "prepare_heterogeneous_sequence_cache_send",
        "start_heterogeneous_sequence_cache_send",
        "poll_heterogeneous_sequence_cache_send",
        "poll_heterogeneous_sequence_cache_send",
        "finish_heterogeneous_sequence_cache_send",
    ]


def test_heterogeneous_send_capacity_counts_shared_staging_not_fanout_bytes():
    engine = make_engine(tensor_parallel_size=2)
    engine.config.max_remote_prefill_staging_bytes = 300
    seq = _prepare_remote_prefill_source(engine, 1)

    engine.start_heterogeneous_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        4,
        [("127.0.0.1", 20001 + rank) for rank in range(4)],
    )

    profile = (
        engine.model_runner.build_heterogeneous_cache_transfer_plan_for_blocks
        .return_value.profile
    )
    assert sum(profile.source_staging_bytes) == 201
    assert sum(profile.source_bytes) == 402
    assert engine.remote_prefill_capacity_snapshot()["staging_bytes_active"] == 201


def test_heterogeneous_send_failure_restores_source_and_releases_capacity():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)
    engine.start_heterogeneous_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        4,
        [("127.0.0.1", 20001 + rank) for rank in range(4)],
    )
    engine._test_send_state.update(value="failed", error="destination rejected")

    with pytest.raises(RuntimeError, match="destination rejected"):
        engine.poll_remote_prefill_send("request/attempt-1")

    assert seq.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [seq]
    assert seq.block_table
    assert seq.state_slot is not None
    assert not engine._remote_prefill_send_staged_bytes
    assert not engine._heterogeneous_remote_prefill_send_ids
    assert any(
        item.args[0] == "abort_heterogeneous_sequence_cache_send"
        for item in engine.model_runner.call_rank_results.call_args_list
    )


def test_step_polls_regular_and_heterogeneous_sends_through_separate_paths():
    engine = make_engine()
    regular = _prepare_remote_prefill_source(engine, 1)
    heterogeneous = _prepare_remote_prefill_source(engine, 5)
    engine.start_remote_prefill_send(
        regular.seq_id,
        "regular",
        [("127.0.0.1", 20001)],
    )
    engine.start_heterogeneous_remote_prefill_send(
        heterogeneous.seq_id,
        "heterogeneous",
        2,
        [("127.0.0.1", 20002), ("127.0.0.1", 20003)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    assert engine.step() == ([], 0, 0, 0)
    assert engine.model_runner.call_rank_results.call_args_list == [
        call("poll_sequence_cache_sends", ["regular"]),
        call("poll_heterogeneous_sequence_cache_send", "heterogeneous"),
    ]


def test_heterogeneous_send_start_mismatch_aborts_before_metrics_commit():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)
    original = engine.model_runner.call_rank_results.side_effect

    def inconsistent_prepare(method_name, *args):
        results = original(method_name, *args)
        if method_name == "prepare_heterogeneous_sequence_cache_send":
            results[1]["staged_bytes"] += 1
        return results

    engine.model_runner.call_rank_results.side_effect = inconsistent_prepare

    with pytest.raises(RuntimeError, match="preflight estimate"):
        engine.start_heterogeneous_remote_prefill_send(
            seq.seq_id,
            "request/attempt-1",
            4,
            [("127.0.0.1", 20001 + rank) for rank in range(4)],
        )

    assert seq.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [seq]
    assert not engine._remote_prefill_send_started_at
    assert not engine._remote_prefill_send_staged_bytes
    assert not engine._heterogeneous_remote_prefill_send_ids
    assert engine.metrics.to_dict()["remote_prefill_send_started"] == 0


def test_async_send_releases_staging_budget_before_receiver_acknowledges():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)
    engine.start_remote_prefill_send(
        seq.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
    )
    engine._test_send_state["staged_bytes"] = 0

    assert engine.poll_remote_prefill_send("request/attempt-1") is None
    assert engine._remote_prefill_send_staged_bytes["request/attempt-1"] == 0
    assert engine.remote_prefill_capacity_snapshot()["staging_bytes_used"] == 0
    assert (
        engine.metrics.to_dict()["active_remote_prefill_send_staged_bytes"] == 0
    )
    assert seq.status is SequenceStatus.TRANSFERRING

    engine._test_send_state["value"] = "ready"
    assert engine.poll_remote_prefill_send("request/attempt-1") == seq.seq_id


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


def test_async_send_start_aborts_when_staged_size_differs_from_estimate():
    engine = make_engine(tensor_parallel_size=2)
    seq = _prepare_remote_prefill_source(engine, 1)
    original = engine.model_runner.call_rank_results.side_effect

    def inconsistent_start(method_name, *args):
        results = original(method_name, *args)
        if method_name == "start_sequence_cache_send":
            results[1]["staged_bytes"] += 1
        return results

    engine.model_runner.call_rank_results.side_effect = inconsistent_start

    with pytest.raises(RuntimeError, match="differ from the preflight"):
        engine.start_remote_prefill_send(
            seq.seq_id,
            "request/attempt-1",
            [("127.0.0.1", 20001), ("127.0.0.1", 20002)],
        )

    assert seq.status is SequenceStatus.RUNNING
    assert list(engine.scheduler.running) == [seq]
    assert not engine._remote_prefill_send_started_at
    assert not engine._remote_prefill_send_staged_bytes
    assert engine.metrics.to_dict()["remote_prefill_send_started"] == 0


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


def test_transfer_capacity_rejects_send_before_staging_or_source_mutation():
    engine = make_engine()
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    engine.config.max_remote_prefill_transfers = 1
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        engine.start_remote_prefill_send(
            second.seq_id,
            "request/attempt-2",
            [("127.0.0.1", 20002)],
        )

    assert second.status is SequenceStatus.RUNNING
    assert second in engine.scheduler.running
    assert "request/attempt-2" not in engine.scheduler.remote_prefill_sources
    engine.model_runner.call_rank_results.assert_not_called()
    assert engine.metrics.to_dict()["remote_prefill_send_backpressure"] == 1


def test_sync_send_respects_active_transfer_capacity_before_host_copy():
    engine = make_engine()
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    engine.config.max_remote_prefill_transfers = 1
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="1/1 reserved"):
        engine.send_remote_prefill(
            second.seq_id,
            "request/attempt-2",
            [("127.0.0.1", 20002)],
        )

    engine.model_runner.call_rank_results.assert_not_called()
    assert second.status is SequenceStatus.RUNNING
    assert engine.metrics.to_dict()["remote_prefill_send_backpressure"] == 1


def test_destination_reservation_respects_transfer_capacity_before_allocation():
    engine = make_engine()
    source = _prepare_remote_prefill_source(engine, 1)
    engine.config.max_remote_prefill_transfers = 1
    engine.start_remote_prefill_send(
        source.seq_id,
        "source/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()
    capacity_before = engine.remote_prefill_capacity_snapshot()

    with pytest.raises(RuntimeError, match="1/1 reserved"):
        engine.add_remote_prefill_request(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=4),
            transfer_id="destination/attempt-1",
        )

    engine.model_runner.call_rank_results.assert_not_called()
    assert "destination/attempt-1" not in engine.scheduler.remote_prefills
    assert engine.remote_prefill_capacity_snapshot() == capacity_before
    assert engine.metrics.to_dict()["remote_prefill_receive_backpressure"] == 1


def test_remote_prefill_capacity_snapshot_reports_live_routing_inputs():
    engine = make_engine()
    engine.config.max_remote_prefill_staging_bytes = 500
    source = _prepare_remote_prefill_source(engine, 1)
    engine.start_remote_prefill_send(
        source.seq_id,
        "source/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.add_remote_prefill_request(
        [5, 6, 7, 8],
        SamplingParams(max_tokens=4),
        transfer_id="destination/attempt-1",
    )
    engine.start_remote_prefill_receive(
        "destination/attempt-1",
        9,
        [("127.0.0.1", 20002)],
    )

    assert engine.remote_prefill_capacity_snapshot() == {
        "waiting_requests": 1,
        "running_requests": 1,
        "sequence_slots_total": 2,
        "sequence_slots_used": 2,
        "sequence_slots_free": 0,
        "sequence_slots_waiting_owned": 0,
        "sequence_slots_local_running": 0,
        "sequence_slots_remote_destination": 1,
        "sequence_slots_remote_source": 1,
        "state_slots_total": 2,
        "state_slots_used": 2,
        "state_slots_free": 0,
        "kv_blocks_total": 8,
        "kv_blocks_used": 2,
        "kv_blocks_free": 6,
        "kv_block_usage": 0.25,
        "last_prefill_stop_reason": None,
        "prefill_stopped_by_token_budget": 0,
        "prefill_stopped_by_sequence_capacity": 0,
        "prefill_stopped_by_kv_capacity": 0,
        "prefill_stopped_by_decode_kv_reservation": 0,
        "decode_kv_reserve_blocks": 0,
        "transfer_slots_total": 2,
        "transfer_slots_used": 2,
        "transfer_slots_free": 0,
        "staging_bytes_limit": 500,
        "staging_bytes_reserved": 0,
        "staging_bytes_active": 200,
        "staging_bytes_used": 200,
        "staging_bytes_free": 300,
    }


def test_remote_prefill_demand_matches_block_and_all_rank_staging_needs():
    engine = make_engine(tensor_parallel_size=2)
    capacity_before = engine.remote_prefill_capacity_snapshot()

    demand = engine.estimate_remote_prefill_demand(5)

    assert demand.kv_blocks == 2
    assert demand.staging_bytes == 201
    assert engine.remote_prefill_capacity_snapshot() == capacity_before
    assert engine.scheduler.is_finished()
    engine.model_runner.call_rank_results.assert_called_once_with(
        "estimate_cache_transfer_bytes_for_blocks",
        2,
    )


@pytest.mark.parametrize("num_prompt_tokens", [0, -1, True, 1.5])
def test_remote_prefill_demand_rejects_invalid_prompt_lengths(num_prompt_tokens):
    engine = make_engine()

    with pytest.raises(ValueError, match="positive integer"):
        engine.estimate_remote_prefill_demand(num_prompt_tokens)


def test_remote_prefill_demand_rejects_prompt_beyond_context_limit():
    engine = make_engine()

    with pytest.raises(ValueError, match="exceeds max_model_len"):
        engine.estimate_remote_prefill_demand(33)


def test_heterogeneous_remote_prefill_preflight_is_read_only():
    engine = make_engine(tensor_parallel_size=4)
    capacity_before = engine.remote_prefill_capacity_snapshot()
    report = {
        "source_tp_size": 4,
        "destination_tp_size": 8,
        "wire_bytes": 24,
        "source_egress_bytes": (6, 6, 6, 6),
        "source_staging_bytes": (3, 3, 3, 3),
        "destination_bytes": (3,) * 8,
        "source_peer_counts": (2,) * 4,
        "destination_peer_counts": (1,) * 8,
        "peer_bytes": (),
        "slice_count": 8,
    }
    engine.model_runner.estimate_heterogeneous_cache_transfer_for_blocks.return_value = (
        report
    )

    result = engine.estimate_heterogeneous_remote_prefill_transfer(5, 8)

    assert result == {**report, "kv_blocks": 2}
    assert result["wire_bytes"] == sum(result["source_egress_bytes"])
    assert result["wire_bytes"] == sum(result["destination_bytes"])
    assert engine.remote_prefill_capacity_snapshot() == capacity_before
    assert engine.scheduler.is_finished()
    engine.model_runner.estimate_heterogeneous_cache_transfer_for_blocks.assert_called_once_with(
        2,
        8,
    )


@pytest.mark.parametrize("num_prompt_tokens", [0, -1, True, 1.5])
def test_heterogeneous_remote_prefill_preflight_rejects_invalid_prompt_lengths(
    num_prompt_tokens,
):
    engine = make_engine()

    with pytest.raises(ValueError, match="num_prompt_tokens must be a positive integer"):
        engine.estimate_heterogeneous_remote_prefill_transfer(
            num_prompt_tokens,
            4,
        )


def test_heterogeneous_remote_prefill_preflight_rejects_prompt_beyond_context():
    engine = make_engine()

    with pytest.raises(ValueError, match="num_prompt_tokens exceeds max_model_len"):
        engine.estimate_heterogeneous_remote_prefill_transfer(33, 4)


@pytest.mark.parametrize("destination_tp_size", [0, -1, True, 1.5])
def test_heterogeneous_remote_prefill_preflight_rejects_invalid_destination_tp(
    destination_tp_size,
):
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="destination_tp_size must be a positive integer",
    ):
        engine.estimate_heterogeneous_remote_prefill_transfer(
            4,
            destination_tp_size,
        )


def test_cancel_remote_prefill_reservation_releases_all_capacity():
    engine = make_engine()
    engine.config.max_remote_prefill_staging_bytes = 250
    capacity_before = engine.remote_prefill_capacity_snapshot()
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=4),
        transfer_id="request/attempt-1",
    )

    reserved_capacity = engine.remote_prefill_capacity_snapshot()
    assert reserved_capacity != capacity_before
    assert reserved_capacity["transfer_slots_used"] == 1
    assert reserved_capacity["transfer_slots_free"] == 1
    assert reserved_capacity["staging_bytes_reserved"] == 100
    assert reserved_capacity["staging_bytes_active"] == 0
    assert reserved_capacity["staging_bytes_used"] == 100
    assert reserved_capacity["staging_bytes_free"] == 150
    assert engine.cancel_remote_prefill_reservation(
        "request/attempt-1",
    ) == seq_id

    assert engine.remote_prefill_capacity_snapshot() == capacity_before
    assert engine.scheduler.is_finished()


def test_unstarted_destination_reservation_prevents_slot_overcommit():
    engine = make_engine()
    engine.config.max_remote_prefill_transfers = 1
    engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="destination/attempt-1",
    )
    capacity_after_first = engine.remote_prefill_capacity_snapshot()

    with pytest.raises(RuntimeError, match="1/1 reserved"):
        engine.add_remote_prefill_request(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=4),
            transfer_id="destination/attempt-2",
        )

    assert tuple(engine.scheduler.remote_prefills) == ("destination/attempt-1",)
    assert engine.remote_prefill_capacity_snapshot() == capacity_after_first
    assert engine.metrics.to_dict()["remote_prefill_receive_backpressure"] == 1


def test_unstarted_destinations_cannot_overcommit_staging_budget():
    engine = make_engine()
    engine.config.max_remote_prefill_staging_bytes = 150
    engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="destination/attempt-1",
    )
    capacity_after_first = engine.remote_prefill_capacity_snapshot()

    with pytest.raises(RuntimeError, match="200/150 bytes requested"):
        engine.add_remote_prefill_request(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=4),
            transfer_id="destination/attempt-2",
        )

    assert tuple(engine.scheduler.remote_prefills) == ("destination/attempt-1",)
    assert engine.remote_prefill_capacity_snapshot() == capacity_after_first
    assert capacity_after_first["staging_bytes_reserved"] == 100
    assert capacity_after_first["staging_bytes_active"] == 0
    assert engine.metrics.to_dict()["remote_prefill_receive_backpressure"] == 1


def test_cancel_remote_prefill_reservation_rejects_active_receive():
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

    with pytest.raises(RuntimeError, match="receive is active"):
        engine.cancel_remote_prefill_reservation("request/attempt-1")


def test_transfer_capacity_is_shared_by_sends_and_receives():
    engine = make_engine()
    engine.config.max_remote_prefill_transfers = 1
    engine.add_remote_prefill_request(
        [5, 6, 7, 8],
        SamplingParams(max_tokens=4),
        transfer_id="destination/attempt-1",
    )
    source = _prepare_remote_prefill_source(engine, 1)
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="1/1 reserved"):
        engine.start_remote_prefill_send(
            source.seq_id,
            "source/attempt-1",
            [("127.0.0.1", 20001)],
        )

    assert source in engine.scheduler.running
    assert "source/attempt-1" not in engine.scheduler.remote_prefill_sources
    engine.model_runner.call_rank_results.assert_not_called()
    assert engine.metrics.to_dict()["remote_prefill_send_backpressure"] == 1


def test_reserved_destination_can_start_its_own_transfer_at_capacity_limit():
    engine = make_engine()
    engine.config.max_remote_prefill_transfers = 1
    seq_id = engine.add_remote_prefill_request(
        [1, 2, 3, 4],
        SamplingParams(max_tokens=4),
        transfer_id="destination/attempt-1",
    )

    assert engine.remote_prefill_capacity_snapshot()["transfer_slots_free"] == 0
    assert engine.start_remote_prefill_receive(
        "destination/attempt-1",
        9,
        [("127.0.0.1", 20001)],
    ) == seq_id


def test_staging_byte_capacity_rejects_before_host_copy():
    engine = make_engine()
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    engine.config.max_remote_prefill_staging_bytes = 150
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="200/150 bytes requested"):
        engine.start_remote_prefill_send(
            second.seq_id,
            "request/attempt-2",
            [("127.0.0.1", 20002)],
        )

    calls = engine.model_runner.call_rank_results.call_args_list
    assert [call.args[0] for call in calls] == ["estimate_sequence_cache_bytes"]
    assert second.status is SequenceStatus.RUNNING
    assert second in engine.scheduler.running
    assert engine.metrics.to_dict()["remote_prefill_send_backpressure"] == 1


def test_sync_send_respects_staging_capacity_before_host_copy():
    engine = make_engine()
    first = _prepare_remote_prefill_source(engine, 1)
    second = _prepare_remote_prefill_source(engine, 5)
    engine.config.max_remote_prefill_staging_bytes = 150
    engine.start_remote_prefill_send(
        first.seq_id,
        "request/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="200/150 bytes requested"):
        engine.send_remote_prefill(
            second.seq_id,
            "request/attempt-2",
            [("127.0.0.1", 20002)],
        )

    calls = engine.model_runner.call_rank_results.call_args_list
    assert [call.args[0] for call in calls] == ["estimate_sequence_cache_bytes"]
    assert second.status is SequenceStatus.RUNNING
    assert engine.metrics.to_dict()["remote_prefill_send_backpressure"] == 1


def test_receive_staging_capacity_rejects_before_destination_reservation():
    engine = make_engine(tensor_parallel_size=2)
    engine.config.max_remote_prefill_staging_bytes = 200

    with pytest.raises(RuntimeError, match="201/200 bytes requested"):
        engine.add_remote_prefill_request(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=4),
            transfer_id="request/attempt-1",
        )

    calls = engine.model_runner.call_rank_results.call_args_list
    assert [call.args[0] for call in calls] == [
        "estimate_cache_transfer_bytes_for_blocks"
    ]
    assert "request/attempt-1" not in engine.scheduler.remote_prefills
    assert not engine._remote_prefill_receive_reserved_staged_bytes
    assert not engine._remote_prefill_receive_staged_bytes
    assert engine.metrics.to_dict()["remote_prefill_receive_backpressure"] == 1


def test_staging_byte_capacity_is_shared_by_send_and_receive():
    engine = make_engine()
    engine.config.max_remote_prefill_staging_bytes = 150
    source = _prepare_remote_prefill_source(engine, 1)
    engine.start_remote_prefill_send(
        source.seq_id,
        "source/attempt-1",
        [("127.0.0.1", 20001)],
    )
    engine.model_runner.call_rank_results.reset_mock()

    with pytest.raises(RuntimeError, match="200/150 bytes requested"):
        engine.add_remote_prefill_request(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=4),
            transfer_id="destination/attempt-1",
        )

    calls = engine.model_runner.call_rank_results.call_args_list
    assert [call.args[0] for call in calls] == [
        "estimate_cache_transfer_bytes_for_blocks"
    ]
    assert "destination/attempt-1" not in engine.scheduler.remote_prefills
    assert engine.metrics.to_dict()["remote_prefill_receive_backpressure"] == 1
