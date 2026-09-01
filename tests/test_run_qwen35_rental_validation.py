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
        resume=False,
    )


def test_commands_are_fail_fast_and_cover_complete_validation_suite():
    stages = MODULE.commands(args())

    assert [name for name, _ in stages] == [
        "preflight",
        "kernels-tp4",
        "cudagraph-tp4",
        "kernels-tp8",
        "cudagraph-tp8",
        "performance-matrix",
        "quality-matrix",
        "final-summary",
    ]
    assert "--preflight-only" in stages[0][1]
    assert stages[1][1][stages[1][1].index("--tp-size") + 1] == "4"
    assert stages[2][1][stages[2][1].index("--tensor-parallel-size") + 1] == "4"
    assert stages[3][1][stages[3][1].index("--tp-size") + 1] == "8"
    assert stages[4][1][stages[4][1].index("--tensor-parallel-size") + 1] == "8"
    assert "--no-checkpoint-audit" in stages[-3][1]
    assert "--no-memory-preflight" in stages[-3][1]
    assert "--include-moe-candidate" in stages[-3][1]
    assert "--no-checkpoint-audit" in stages[-2][1]
    assert stages[-1][1][-1].endswith("summary.json")


@pytest.mark.parametrize("value", ["", "../run", "run id", "a/b"])
def test_run_id_rejects_unsafe_paths(value):
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.validate_run_id(value)


def test_tp_sizes_reject_unsupported_parallelism():
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.parse_tp_sizes("3,4")


def test_manifest_resumes_only_identical_run(tmp_path):
    arguments = args()
    stages = MODULE.commands(arguments)
    plan = MODULE.manifest_plan(arguments, stages)
    path = tmp_path / "manifest.json"
    manifest = MODULE.prepare_manifest(path, plan, resume=False)
    artifact = tmp_path / "result.json"
    artifact.write_text('{"valid": true}\n')
    MODULE.mark_stage_completed(path, manifest, "preflight", [artifact])

    resumed = MODULE.prepare_manifest(path, plan, resume=True)

    assert resumed["completed_stages"] == ["preflight"]


def test_manifest_rejects_changed_completed_artifact(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    manifest = MODULE.prepare_manifest(path, plan, resume=False)
    artifact = tmp_path / "result.json"
    artifact.write_text('{"valid": true}\n')
    MODULE.mark_stage_completed(path, manifest, "preflight", [artifact])
    artifact.write_text('{"valid": false}\n')

    with pytest.raises(ValueError, match="missing or changed"):
        MODULE.prepare_manifest(path, plan, resume=True)


def test_manifest_rejects_missing_completed_artifact(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    manifest = MODULE.prepare_manifest(path, plan, resume=False)
    artifact = tmp_path / "result.json"
    artifact.write_text('{"valid": true}\n')
    MODULE.mark_stage_completed(path, manifest, "preflight", [artifact])
    artifact.unlink()

    with pytest.raises(ValueError, match="missing or changed"):
        MODULE.prepare_manifest(path, plan, resume=True)


def test_manifest_rejects_changed_resume_commands(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    MODULE.prepare_manifest(path, plan, resume=False)
    changed = {**plan, "model": "/different/model"}

    with pytest.raises(ValueError, match="does not match"):
        MODULE.prepare_manifest(path, changed, resume=True)


def test_manifest_refuses_accidental_overwrite(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    MODULE.prepare_manifest(path, plan, resume=False)

    with pytest.raises(ValueError, match="already exists"):
        MODULE.prepare_manifest(path, plan, resume=False)
