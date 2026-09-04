import json
from argparse import Namespace

import pytest

from scripts import benchmark_scheduler_workload as MODULE


def args(tmp_path, **overrides):
    values = {
        "workload": None,
        "profile": "mixed",
        "max_model_len": 8192,
        "vocab_size": 10000,
        "max_engine_steps": 100000,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 64,
        "prefill_starvation_token_budget": 256,
        "prefill_starvation_threshold": 0,
        "temperature": 0.6,
    }
    values.update(overrides)
    return Namespace(**values)


def test_loads_and_context_validates_built_in_profile(tmp_path):
    workload = MODULE.load_workload(args(tmp_path))

    assert workload.name == "mixed"
    assert workload.summary()["request_count"] == 24


def test_loads_external_workload_and_normalizes_order(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "custom",
                "requests": [
                    {
                        "request_id": "late",
                        "arrival_step": 2,
                        "input_len": 4,
                        "output_len": 2,
                        "workload_class": "short",
                    },
                    {
                        "request_id": "early",
                        "arrival_step": 0,
                        "input_len": 4,
                        "output_len": 2,
                        "workload_class": "short",
                    },
                ],
            }
        )
    )

    workload = MODULE.load_workload(args(tmp_path, workload=path, profile=None))

    assert [item.request_id for item in workload.requests] == ["early", "late"]


def test_external_workload_must_fit_selected_context(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "too-long",
                "requests": [
                    {
                        "request_id": "request",
                        "arrival_step": 0,
                        "input_len": 8,
                        "output_len": 5,
                        "workload_class": "long",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="13 tokens.*max_model_len=12"):
        MODULE.load_workload(
            args(tmp_path, workload=path, profile=None, max_model_len=12)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_engine_steps": 0}, "max_engine_steps"),
        ({"vocab_size": True}, "vocab_size"),
        ({"prefill_starvation_threshold": -1}, "non-negative"),
        ({"temperature": -0.1}, "temperature"),
    ],
)
def test_validates_cli_contract(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        MODULE.validate_args(args(tmp_path, **overrides))


def test_greedy_temperature_is_supported(tmp_path):
    MODULE.validate_args(args(tmp_path, temperature=0.0))
