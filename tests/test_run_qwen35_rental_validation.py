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
        max_model_len=16_384,
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
        "official-checkpoint-audit",
        "preflight",
        "kernels-tp4",
        "mixed-tp4-r1",
        "mixed-tp4-r2",
        "mixed-tp4-r3",
        "kernels-long-prefill-tp4",
        "attention-short-tp4",
        "attention-long-tp4",
        "cudagraph-short-tp4",
        "cudagraph-long-tp4",
        "kernels-tp8",
        "mixed-tp8-r1",
        "mixed-tp8-r2",
        "mixed-tp8-r3",
        "kernels-long-prefill-tp8",
        "attention-short-tp8",
        "attention-long-tp8",
        "cudagraph-short-tp8",
        "cudagraph-long-tp8",
        "performance-matrix",
        "quality-matrix",
        "final-summary",
    ]
    official = stages[0][1]
    assert (
        official[official.index("--revision") + 1]
        == MODULE.OFFICIAL_CHECKPOINT_REVISION
    )
    assert official[official.index("--tp-sizes") + 1] == "4,8"
    assert "--preflight-only" in stages[1][1]
    assert "--verify-checkpoint-shards" in stages[1][1]
    assert stages[1][1][stages[1][1].index("--max-model-len") + 1] == "16384"
    assert stages[2][1][stages[2][1].index("--tp-size") + 1] == "4"
    commands = dict(stages)
    mixed = commands["mixed-tp4-r1"]
    assert mixed[mixed.index("--tensor-parallel-size") + 1] == "4"
    assert mixed[mixed.index("--qwen35-moe-decode-backend") + 1] == "batched"
    assert mixed[mixed.index("--temperature") + 1] == "0"
    assert "--enable-dynamic-chunked-prefill" in mixed
    assert mixed[mixed.index("--require-paths") + 1] == "mixed_eager"
    long_prefill = commands["kernels-long-prefill-tp4"]
    assert "--prefill-only" in long_prefill
    assert long_prefill[long_prefill.index("--prefill-tokens") + 1] == "8192"
    assert long_prefill[long_prefill.index("--prefill-batch") + 1] == "1"
    chunk_index = long_prefill.index("--delta-prefill-chunk-sizes")
    assert long_prefill[chunk_index + 1:chunk_index + 4] == ["32", "64", "128"]
    short_attention = commands["attention-short-tp4"]
    long_attention = commands["attention-long-tp4"]
    assert short_attention[short_attention.index("--context-len") + 1] == "4096"
    assert short_attention[short_attention.index("--num-heads") + 1] == "4"
    assert "--include-partitioned" not in short_attention
    assert long_attention[long_attention.index("--context-len") + 1] == "16384"
    assert "--include-partitioned" in long_attention
    assert long_attention[long_attention.index("--partition-sizes") + 1] == "256,512"
    short_graph = commands["cudagraph-short-tp4"]
    long_graph = commands["cudagraph-long-tp4"]
    assert short_graph[short_graph.index("--input-length-base") + 1] == "33"
    assert long_graph[long_graph.index("--input-length-base") + 1] == "8192"
    assert long_graph[long_graph.index("--batch-sizes") + 1] == "3"
    tp8_kernels = commands["kernels-tp8"]
    tp8_attention = commands["attention-short-tp8"]
    assert tp8_kernels[tp8_kernels.index("--tp-size") + 1] == "8"
    assert tp8_attention[tp8_attention.index("--num-heads") + 1] == "2"
    assert "--no-checkpoint-audit" in stages[-3][1]
    assert "--no-memory-preflight" in stages[-3][1]
    assert "--include-moe-candidate" in stages[-3][1]
    assert "--no-checkpoint-audit" in stages[-2][1]
    assert stages[-2][1][stages[-2][1].index("--prompt-lengths") + 1] == (
        "128,1024,3072,8192"
    )
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


def test_mixed_stage_collects_only_its_repeat(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    mixed_dir = tmp_path / arguments.run_id / "mixed" / "tp4"
    mixed_dir.mkdir(parents=True)
    first = mixed_dir / "r1.json"
    second = mixed_dir / "r2.json"
    first.write_text('{"repeat": 1}\n')
    second.write_text('{"repeat": 2}\n')

    artifacts = MODULE.collect_stage_artifacts(arguments, "mixed-tp4-r2")

    assert artifacts == [second]


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


def test_manifest_rejects_results_from_different_source_tree(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    MODULE.prepare_manifest(path, plan, resume=False)
    changed = {**plan, "source_tree_sha256": "0" * 64}

    with pytest.raises(ValueError, match="does not match"):
        MODULE.prepare_manifest(path, changed, resume=True)


def test_source_fingerprint_ignores_results_and_documentation(monkeypatch, tmp_path):
    source = tmp_path / "nanovllm"
    scripts = tmp_path / "scripts"
    source.mkdir()
    scripts.mkdir()
    (source / "model.py").write_text("VALUE = 1\n")
    (scripts / "run.py").write_text("print('run')\n")
    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\nname = 'test'\n")
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "SOURCE_ROOTS", (source, scripts))
    monkeypatch.setattr(MODULE, "SOURCE_FILES", (project,))

    baseline = MODULE.source_tree_sha256()
    (tmp_path / "benchmark_results.json").write_text("{}\n")
    (tmp_path / "notes.md").write_text("private notes\n")

    assert MODULE.source_tree_sha256() == baseline
    (source / "model.py").write_text("VALUE = 2\n")
    assert MODULE.source_tree_sha256() != baseline


def test_source_validation_detects_mid_run_code_change(monkeypatch):
    monkeypatch.setattr(MODULE, "source_tree_sha256", lambda: "current")

    with pytest.raises(ValueError, match="changed during the run"):
        MODULE.validate_source_tree({"source_tree_sha256": "original"})


def test_manifest_refuses_accidental_overwrite(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    MODULE.prepare_manifest(path, plan, resume=False)

    with pytest.raises(ValueError, match="already exists"):
        MODULE.prepare_manifest(path, plan, resume=False)
