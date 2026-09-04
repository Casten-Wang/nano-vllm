import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scheduler_workload_under_test",
    ROOT / "nanovllm" / "scheduler_workload.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def request(**overrides):
    item = {
        "request_id": "request-0",
        "arrival_step": 0,
        "input_len": 8,
        "output_len": 4,
        "workload_class": "short",
    }
    item.update(overrides)
    return item


def workload(requests=None, **overrides):
    payload = {
        "version": 1,
        "name": "test",
        "requests": [request()] if requests is None else requests,
    }
    payload.update(overrides)
    return payload


def test_normalizes_arrival_order_stably_and_hashes_canonical_payload():
    payload = workload(
        [
            request(request_id="late", arrival_step=2),
            request(request_id="first", arrival_step=0),
            request(request_id="second", arrival_step=0),
        ]
    )

    parsed = MODULE.SchedulerWorkload.from_dict(payload, max_model_len=12)
    reparsed = MODULE.SchedulerWorkload.from_dict(
        json.loads(json.dumps(parsed.to_dict(), indent=2)),
        max_model_len=12,
    )

    assert [item.request_id for item in parsed.requests] == [
        "first",
        "second",
        "late",
    ]
    assert parsed.digest == reparsed.digest
    assert parsed.summary() == {
        "name": "test",
        "digest": parsed.digest,
        "request_count": 3,
        "arrival_step_count": 2,
        "last_arrival_step": 2,
        "input_tokens": 24,
        "requested_output_tokens": 12,
        "requests_by_class": {"short": 3},
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "exactly version"),
        (workload(version=2), "unsupported workload version"),
        (workload(name=""), "name must not be empty"),
        (workload(requests=[]), "non-empty list"),
        (
            workload([request(), request()]),
            "request_id values must be unique",
        ),
        (
            workload([request(extra=True)]),
            "must contain exactly",
        ),
        (
            workload([request(request_id="")]),
            "request_id must not be empty",
        ),
        (
            workload([request(workload_class="")]),
            "workload_class must not be empty",
        ),
        (
            workload([request(arrival_step=True)]),
            "arrival_step must be a non-negative integer",
        ),
        (
            workload([request(input_len=0)]),
            "input_len must be a positive integer",
        ),
        (
            workload([request(output_len=-1)]),
            "output_len must be a non-negative integer",
        ),
    ],
)
def test_rejects_invalid_workload_contract(payload, message):
    with pytest.raises(ValueError, match=message):
        MODULE.SchedulerWorkload.from_dict(payload)


def test_rejects_request_that_exceeds_model_context():
    with pytest.raises(ValueError, match="13 tokens.*max_model_len=12"):
        MODULE.SchedulerWorkload.from_dict(
            workload([request(input_len=9, output_len=4)]),
            max_model_len=12,
        )


def test_built_in_profiles_cover_distinct_arrival_and_length_distributions():
    steady = MODULE.built_in_workload("steady")
    bursty = MODULE.built_in_workload("bursty")
    mixed = MODULE.built_in_workload("mixed")

    assert steady.summary()["request_count"] == 24
    assert steady.summary()["arrival_step_count"] == 24
    assert bursty.summary()["arrival_step_count"] == 3
    assert max(item.input_len for item in mixed.requests) == 4096
    assert max(item.output_len for item in mixed.requests) == 512
    assert mixed.summary()["requests_by_class"] == {
        "decode-heavy": 8,
        "prefill-heavy": 8,
        "short": 8,
    }
    assert len({profile.digest for profile in (steady, bursty, mixed)}) == 3


def test_built_in_profile_name_is_validated():
    with pytest.raises(ValueError, match="steady, bursty, mixed"):
        MODULE.built_in_workload("unknown")


class FakeMetrics:
    def __init__(self):
        self.request_samples = []
        self.steps = []

    def reset(self):
        self.request_samples.clear()
        self.steps.clear()

    def record_step(self, num_tokens, elapsed, **tokens):
        self.steps.append((num_tokens, elapsed, tokens))

    def to_dict(self):
        return {
            "num_finished_requests": len(self.request_samples),
            "recorded_steps": len(self.steps),
        }


class FakeScheduler:
    def __init__(self, engine):
        self.engine = engine

    def capacity_snapshot(self):
        return {
            "waiting_requests": len(self.engine.active),
            "kv_blocks_used": len(self.engine.active),
        }


class FakeEngine:
    def __init__(self, *, duplicate_seq_id=False, wrong_output=False):
        self.metrics = FakeMetrics()
        self.scheduler = FakeScheduler(self)
        self.active = []
        self.next_seq_id = 0
        self.duplicate_seq_id = duplicate_seq_id
        self.wrong_output = wrong_output

    def is_finished(self):
        return not self.active

    def add_request(self, prompt, sampling_params):
        seq_id = 0 if self.duplicate_seq_id else self.next_seq_id
        self.next_seq_id += 1
        self.active.append((seq_id, prompt, sampling_params))
        return seq_id

    def step(self):
        seq_id, prompt, output_len = self.active.pop(0)
        actual = output_len - int(self.wrong_output)
        self.metrics.request_samples.append(
            {
                "seq_id": seq_id,
                "prompt_tokens": len(prompt),
                "output_tokens": actual,
                "preemption_count": seq_id,
                "preempted_token_progress": seq_id * 8,
                "recomputed_tokens": seq_id * 4,
                "ttft_s": float(seq_id + 1),
                "tpot_s": 0.5,
                "latency_s": float(seq_id + 2),
            }
        )
        return [(seq_id, list(range(actual)))], -1, 0, 1


def two_request_workload():
    return MODULE.SchedulerWorkload.from_dict(
        workload(
            [
                request(request_id="first", arrival_step=0, workload_class="short"),
                request(
                    request_id="second",
                    arrival_step=3,
                    workload_class="decode-heavy",
                ),
            ]
        )
    )


def test_prompt_tokens_are_stable_per_request_and_seed():
    spec = two_request_workload().requests[0]

    first = MODULE.prompt_token_ids(spec, vocab_size=17, seed=4)
    second = MODULE.prompt_token_ids(spec, vocab_size=17, seed=4)

    assert first == second
    assert len(first) == spec.input_len
    assert all(0 <= token < 17 for token in first)
    assert first != MODULE.prompt_token_ids(spec, vocab_size=17, seed=5)


def test_replay_injects_at_logical_steps_and_preserves_raw_evidence():
    engine = FakeEngine()
    times = iter((0.0, 0.25, 1.0, 1.5))

    result = MODULE.replay_scheduler_workload(
        engine,
        two_request_workload(),
        prompt_factory=lambda item: [1] * item.input_len,
        sampling_params_factory=lambda item: item.output_len,
        clock=lambda: next(times),
    )

    assert result["engine_steps"] == 2
    assert result["idle_fast_forwards"] == 1
    assert [step["logical_step"] for step in result["step_samples"]] == [0, 3]
    assert result["step_samples"][0]["admitted_request_ids"] == ["first"]
    assert result["step_samples"][1]["admitted_request_ids"] == ["second"]
    assert result["request_samples"][1]["completion_step"] == 3
    assert result["request_samples"][1]["preemption_count"] == 1
    assert result["request_samples"][1]["preempted_token_progress"] == 8
    assert result["request_samples"][1]["recomputed_tokens"] == 4
    assert result["preemption"] == {
        "all": {
            "request_count": 2,
            "preempted_request_count": 1,
            "preempted_request_rate": 0.5,
            "total_preemption_count": 1,
            "max_preemptions_per_request": 1,
            "total_preempted_token_progress": 8,
            "p95_preempted_token_progress": pytest.approx(7.6),
            "max_preempted_token_progress": 8,
            "total_recomputed_tokens": 4,
            "p95_recomputed_tokens": pytest.approx(3.8),
            "max_recomputed_tokens": 4,
        },
        "by_class": {
            "decode-heavy": {
                "request_count": 1,
                "preempted_request_count": 1,
                "preempted_request_rate": 1.0,
                "total_preemption_count": 1,
                "max_preemptions_per_request": 1,
                "total_preempted_token_progress": 8,
                "p95_preempted_token_progress": 8,
                "max_preempted_token_progress": 8,
                "total_recomputed_tokens": 4,
                "p95_recomputed_tokens": 4,
                "max_recomputed_tokens": 4,
            },
            "short": {
                "request_count": 1,
                "preempted_request_count": 0,
                "preempted_request_rate": 0.0,
                "total_preemption_count": 0,
                "max_preemptions_per_request": 0,
                "total_preempted_token_progress": 0,
                "p95_preempted_token_progress": 0,
                "max_preempted_token_progress": 0,
                "total_recomputed_tokens": 0,
                "p95_recomputed_tokens": 0,
                "max_recomputed_tokens": 0,
            },
        },
    }
    assert result["output_token_ids"]["request_count"] == 2
    assert result["output_token_ids"]["token_count"] == 8
    assert len(result["output_token_ids"]["digest"]) == 64
    assert result["latency"]["by_class"]["decode-heavy"]["p99_ttft_s"] == 2.0
    assert result["engine_metrics"] == {
        "num_finished_requests": 2,
        "recorded_steps": 2,
    }


def test_replay_rejects_duplicate_engine_sequence_ids():
    with pytest.raises(RuntimeError, match="duplicate seq_id"):
        MODULE.replay_scheduler_workload(
            FakeEngine(duplicate_seq_id=True),
            MODULE.SchedulerWorkload.from_dict(
                workload(
                    [
                        request(request_id="first"),
                        request(request_id="second"),
                    ]
                )
            ),
            prompt_factory=lambda item: [1] * item.input_len,
            sampling_params_factory=lambda item: item.output_len,
        )


def test_replay_rejects_incomplete_generation():
    with pytest.raises(RuntimeError, match="produced 3/4 tokens"):
        MODULE.replay_scheduler_workload(
            FakeEngine(wrong_output=True),
            MODULE.SchedulerWorkload.from_dict(workload()),
            prompt_factory=lambda item: [1] * item.input_len,
            sampling_params_factory=lambda item: item.output_len,
        )


def test_replay_has_a_bounded_step_guard():
    class StuckEngine(FakeEngine):
        def step(self):
            return [], 0, 0, 0

    with pytest.raises(RuntimeError, match="max_engine_steps=2"):
        MODULE.replay_scheduler_workload(
            StuckEngine(),
            MODULE.SchedulerWorkload.from_dict(workload()),
            prompt_factory=lambda item: [1] * item.input_len,
            sampling_params_factory=lambda item: item.output_len,
            max_engine_steps=2,
        )
