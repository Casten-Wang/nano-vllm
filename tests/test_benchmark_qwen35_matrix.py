from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "benchmark_qwen35_matrix",
    ROOT / "scripts" / "benchmark_qwen35_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "model": "/models/qwen35",
        "num_seqs": 16,
        "input_len": 128,
        "output_len": 32,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 64,
        "seed": 7,
        "result_dir": "benchmark_results/matrix",
        "warmup": True,
        "repeats": 3,
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_matrix_covers_tp_state_and_kv_variants():
    cases = MODULE.build_cases((4, 8))

    assert len(cases) == 8
    assert {case.tensor_parallel_size for case in cases} == {4, 8}
    assert {case.recurrent_state_dtype for case in cases} == {"float32", "model"}
    assert {case.kv_cache_dtype for case in cases} == {"auto", "int8"}
    assert len({case.name for case in cases}) == len(cases)


def test_case_command_is_eager_and_fully_identified():
    case = MODULE.BenchmarkCase(8, "model", "int8")
    command = MODULE.command_for_case(args(), case, repeat=2)

    assert command[1].endswith("scripts/benchmark_baseline.py")
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--recurrent-state-dtype") + 1] == "model"
    assert command[command.index("--kv-cache-dtype") + 1] == "int8"
    assert command[command.index("--name") + 1] == "qwen35_tp8_state-model_kv-int8_r2"
    assert "--enforce-eager" in command


def test_no_warmup_is_forwarded():
    command = MODULE.command_for_case(
        args(warmup=False),
        MODULE.BenchmarkCase(4, "float32", "auto"),
    )

    assert "--no-warmup" in command


@pytest.mark.parametrize("value", ["", "0", "4,-1", "four"])
def test_invalid_tp_sizes_are_rejected(value):
    with pytest.raises((ValueError, MODULE.argparse.ArgumentTypeError)):
        MODULE.comma_separated_ints(value)
