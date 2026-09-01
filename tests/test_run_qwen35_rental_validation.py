from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "run_qwen35_rental_validation",
    ROOT / "scripts" / "run_qwen35_rental_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args():
    return Namespace(
        model="/models/qwen35",
        tp_sizes=(4, 8),
        num_seqs=64,
        input_len=512,
        output_len=128,
        max_num_seqs=64,
        repeats=3,
        run_id="rental-a",
        result_dir="benchmark_results/qwen35_rental",
        dry_run=True,
    )


def test_commands_are_fail_fast_and_cover_complete_validation_suite():
    stages = MODULE.commands(args())

    assert [name for name, _ in stages] == [
        "preflight",
        "kernels-tp4",
        "kernels-tp8",
        "performance-matrix",
        "quality-matrix",
    ]
    assert "--preflight-only" in stages[0][1]
    assert stages[1][1][stages[1][1].index("--tp-size") + 1] == "4"
    assert stages[2][1][stages[2][1].index("--tp-size") + 1] == "8"
    assert "--no-checkpoint-audit" in stages[-2][1]
    assert "--no-memory-preflight" in stages[-2][1]
    assert "--no-checkpoint-audit" in stages[-1][1]


@pytest.mark.parametrize("value", ["", "../run", "run id", "a/b"])
def test_run_id_rejects_unsafe_paths(value):
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.validate_run_id(value)


def test_tp_sizes_reject_unsupported_parallelism():
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.parse_tp_sizes("3,4")
