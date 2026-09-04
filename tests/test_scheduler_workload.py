import importlib.util
import json
from pathlib import Path
import sys

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
    payload = workload([
        request(request_id="late", arrival_step=2),
        request(request_id="first", arrival_step=0),
        request(request_id="second", arrival_step=0),
    ])

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
