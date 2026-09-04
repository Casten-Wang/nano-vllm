import json
import sys
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
SUMMARY_SPEC = spec_from_file_location(
    "summarize_qwen35_rental_contract",
    ROOT / "scripts" / "summarize_qwen35_rental.py",
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY_MODULE = module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)


def args():
    return Namespace(
        model="/models/qwen35",
        gptq_model=None,
        gptq_revision=MODULE.OFFICIAL_GPTQ_CHECKPOINT_REVISION,
        fp8_audit_model=None,
        fp8_model=None,
        fp8_revision=MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION,
        tp_sizes=(4, 8),
        num_seqs=64,
        input_len=512,
        output_len=128,
        max_model_len=16_384,
        max_num_seqs=64,
        sampling_chunk_size=32,
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
        "pd-export-auto-float32-tp4",
        "pd-export-bounded-auto-float32-tp4",
        "pd-transfer-auto-float32-tp4",
        "pd-export-int8-model-tp4",
        "pd-export-bounded-int8-model-tp4",
        "pd-transfer-int8-model-tp4",
        "kernels-tp4",
        "mixed-tp4-r1",
        "mixed-tp4-r2",
        "mixed-tp4-r3",
        "fairness-disabled-tp4-r1",
        "fairness-disabled-tp4-r2",
        "fairness-disabled-tp4-r3",
        "fairness-enabled-tp4-r1",
        "fairness-enabled-tp4-r2",
        "fairness-enabled-tp4-r3",
        "pressure-fcfs-tp4-r1",
        "pressure-fcfs-tp4-r2",
        "pressure-fcfs-tp4-r3",
        "pressure-min_recompute-tp4-r1",
        "pressure-min_recompute-tp4-r2",
        "pressure-min_recompute-tp4-r3",
        "pressure-min_recompute_reserved-tp4-r1",
        "pressure-min_recompute_reserved-tp4-r2",
        "pressure-min_recompute_reserved-tp4-r3",
        "scheduler-baseline-tp4-r1",
        "scheduler-baseline-tp4-r2",
        "scheduler-baseline-tp4-r3",
        "scheduler-optimized-tp4-r1",
        "scheduler-optimized-tp4-r2",
        "scheduler-optimized-tp4-r3",
        "kernels-long-prefill-tp4",
        "attention-short-tp4",
        "attention-long-tp4",
        "attention-max-tp4",
        "cudagraph-short-tp4",
        "cudagraph-conv-channel_accumulate-short-tp4",
        "cudagraph-long-tp4",
        "cudagraph-conv-channel_accumulate-long-tp4",
        "pd-export-auto-float32-tp8",
        "pd-export-bounded-auto-float32-tp8",
        "pd-transfer-auto-float32-tp8",
        "pd-export-int8-model-tp8",
        "pd-export-bounded-int8-model-tp8",
        "pd-transfer-int8-model-tp8",
        "kernels-tp8",
        "mixed-tp8-r1",
        "mixed-tp8-r2",
        "mixed-tp8-r3",
        "fairness-disabled-tp8-r1",
        "fairness-disabled-tp8-r2",
        "fairness-disabled-tp8-r3",
        "fairness-enabled-tp8-r1",
        "fairness-enabled-tp8-r2",
        "fairness-enabled-tp8-r3",
        "pressure-fcfs-tp8-r1",
        "pressure-fcfs-tp8-r2",
        "pressure-fcfs-tp8-r3",
        "pressure-min_recompute-tp8-r1",
        "pressure-min_recompute-tp8-r2",
        "pressure-min_recompute-tp8-r3",
        "pressure-min_recompute_reserved-tp8-r1",
        "pressure-min_recompute_reserved-tp8-r2",
        "pressure-min_recompute_reserved-tp8-r3",
        "scheduler-baseline-tp8-r1",
        "scheduler-baseline-tp8-r2",
        "scheduler-baseline-tp8-r3",
        "scheduler-optimized-tp8-r1",
        "scheduler-optimized-tp8-r2",
        "scheduler-optimized-tp8-r3",
        "kernels-long-prefill-tp8",
        "attention-short-tp8",
        "attention-long-tp8",
        "attention-max-tp8",
        "cudagraph-short-tp8",
        "cudagraph-conv-channel_accumulate-short-tp8",
        "cudagraph-long-tp8",
        "cudagraph-conv-channel_accumulate-long-tp8",
        "performance-matrix",
        "quality-matrix",
        "quality-conv-channel_accumulate",
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
    assert stages[2][1][stages[2][1].index("--kv-dtype") + 1] == "auto"
    assert stages[5][1][stages[5][1].index("--state-dtype") + 1] == "model"
    assert stages[6][1][stages[6][1].index("--tp-size") + 1] == "4"
    commands = dict(stages)
    mixed = commands["mixed-tp4-r1"]
    assert mixed[mixed.index("--tensor-parallel-size") + 1] == "4"
    assert mixed[mixed.index("--qwen35-moe-decode-backend") + 1] == "batched"
    assert mixed[mixed.index("--temperature") + 1] == "0"
    assert "--enable-dynamic-chunked-prefill" in mixed
    assert mixed[mixed.index("--require-paths") + 1] == "mixed_eager"
    fairness_disabled = commands["fairness-disabled-tp4-r1"]
    fairness_enabled = commands["fairness-enabled-tp4-r1"]
    assert fairness_disabled[
        fairness_disabled.index("--prefill-starvation-threshold") + 1
    ] == "0"
    assert fairness_disabled[
        fairness_disabled.index("--require-paths") + 1
    ] == "prefill_eager"
    assert fairness_enabled[
        fairness_enabled.index("--prefill-starvation-threshold") + 1
    ] == str(MODULE.FAIRNESS_THRESHOLD)
    assert fairness_enabled[
        fairness_enabled.index("--max-num-batched-tokens") + 1
    ] == str(MODULE.FAIRNESS_MAX_BATCHED_TOKENS)
    assert fairness_enabled[
        fairness_enabled.index("--initial-seqs") + 1
    ] == str(MODULE.FAIRNESS_INITIAL_SEQUENCES)
    assert fairness_enabled[
        fairness_enabled.index("--require-paths") + 1
    ] == "mixed_eager"
    pressure = commands["pressure-min_recompute-tp4-r1"]
    assert pressure[pressure.index("--num-kvcache-blocks-override") + 1] == "5"
    assert pressure[pressure.index("--initial-input-lens") + 1] == "256,1024"
    assert pressure[pressure.index("--injected-input-lens") + 1] == "512,512"
    assert pressure[pressure.index("--output-len") + 1] == "16"
    assert pressure[pressure.index("--preemption-policy") + 1] == "min_recompute"
    reserved_pressure = commands["pressure-min_recompute_reserved-tp4-r1"]
    assert "--enable-decode-kv-reservation" in reserved_pressure
    assert reserved_pressure[
        reserved_pressure.index("--preemption-policy") + 1
    ] == "min_recompute"
    scheduler_baseline = commands["scheduler-baseline-tp4-r1"]
    scheduler_optimized = commands["scheduler-optimized-tp4-r1"]
    assert scheduler_baseline[
        scheduler_baseline.index("--profile") + 1
    ] == "mixed"
    assert scheduler_baseline[
        scheduler_baseline.index("--temperature") + 1
    ] == "0"
    assert "--enable-dynamic-chunked-prefill" not in scheduler_baseline
    assert scheduler_baseline[
        scheduler_baseline.index("--require-paths") + 1
    ] == "prefill_eager"
    assert "--enable-dynamic-chunked-prefill" in scheduler_optimized
    assert "--enable-decode-kv-reservation" in scheduler_optimized
    assert scheduler_optimized[
        scheduler_optimized.index("--preemption-policy") + 1
    ] == "min_recompute"
    assert scheduler_optimized[
        scheduler_optimized.index("--require-paths") + 1
    ] == "mixed_eager"
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
    assert "--include-partitioned" in long_attention
    assert long_attention[long_attention.index("--context-len") + 1] == "16385"
    assert long_attention[long_attention.index("--partition-sizes") + 1] == "256,512"
    short_graph = commands["cudagraph-short-tp4"]
    long_graph = commands["cudagraph-long-tp4"]
    candidate_graph = commands[
        "cudagraph-conv-channel_accumulate-short-tp4"
    ]
    assert short_graph[short_graph.index("--input-length-base") + 1] == "33"
    assert long_graph[long_graph.index("--input-length-base") + 1] == "8192"
    assert long_graph[long_graph.index("--batch-sizes") + 1] == "3"
    assert candidate_graph[
        candidate_graph.index("--qwen35-decode-conv-backend") + 1
    ] == "channel_accumulate"
    tp8_kernels = commands["kernels-tp8"]
    tp8_attention = commands["attention-short-tp8"]
    assert tp8_kernels[tp8_kernels.index("--tp-size") + 1] == "8"
    assert tp8_attention[tp8_attention.index("--num-heads") + 1] == "2"
    performance = commands["performance-matrix"]
    quality = commands["quality-matrix"]
    candidate_quality = commands["quality-conv-channel_accumulate"]
    assert "--no-checkpoint-audit" in performance
    assert "--no-memory-preflight" in performance
    assert "--include-moe-candidate" in performance
    assert "--include-decode-conv-candidate" in performance
    assert "--no-checkpoint-audit" in quality
    assert quality[quality.index("--prompt-lengths") + 1] == (
        "128,1024,3072,8192"
    )
    assert quality[quality.index("--continuation-len") + 1] == "16"
    assert candidate_quality[
        candidate_quality.index("--qwen35-decode-conv-backend") + 1
    ] == "channel_accumulate"
    assert stages[-1][1][-1].endswith("summary.json")


def test_optional_gptq_checkpoint_adds_audited_eager_tp_matrix():
    arguments = args()
    arguments.gptq_model = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    stages = MODULE.commands(arguments)
    names = [name for name, _ in stages]

    assert names[-6:] == [
        "official-gptq-checkpoint-audit",
        "gptq-preflight",
        "gptq-performance-matrix",
        "gptq-quality-matrix",
        "gptq-vs-bf16-quality",
        "final-summary",
    ]
    commands = dict(stages)
    audit = commands["official-gptq-checkpoint-audit"]
    assert audit[audit.index("--repo") + 1] == MODULE.OFFICIAL_GPTQ_CHECKPOINT_REPO
    assert audit[audit.index("--revision") + 1] == MODULE.OFFICIAL_GPTQ_CHECKPOINT_REVISION
    preflight = commands["gptq-preflight"]
    assert "--preflight-only" in preflight
    assert "--verify-checkpoint-shards" in preflight
    assert preflight[preflight.index("--weight-quant-backend") + 1] == "auto"
    performance = commands["gptq-performance-matrix"]
    assert performance[performance.index("--weight-quant-backend") + 1] == "auto"
    assert "--no-checkpoint-audit" in performance
    assert "--no-memory-preflight" in performance
    quality = commands["gptq-quality-matrix"]
    assert quality[quality.index("--weight-quant-backend") + 1] == "auto"
    assert quality[quality.index("--qwen35-moe-decode-backend") + 1] == "sorted"
    assert "--no-checkpoint-audit" in quality
    comparison = commands["gptq-vs-bf16-quality"]
    assert comparison[comparison.index("--baseline-run-id") + 1] == arguments.run_id
    assert comparison[comparison.index("--candidate-run-id") + 1] == (
        f"{arguments.run_id}-gptq"
    )


def test_gptq_identity_is_part_of_resume_manifest():
    arguments = args()
    arguments.gptq_model = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))

    assert plan["gptq_model"] == arguments.gptq_model
    assert plan["gptq_revision"] == MODULE.OFFICIAL_GPTQ_CHECKPOINT_REVISION


def test_optional_fp8_checkpoint_adds_header_audit_only():
    arguments = args()
    arguments.fp8_audit_model = MODULE.OFFICIAL_FP8_CHECKPOINT_REPO

    stages = MODULE.commands(arguments)
    names = [name for name, _ in stages]

    assert names[-2:] == ["official-fp8-checkpoint-audit", "final-summary"]
    command = dict(stages)["official-fp8-checkpoint-audit"]
    assert (
        command[command.index("--repo") + 1]
        == MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
    )
    assert (
        command[command.index("--revision") + 1]
        == MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION
    )
    assert command[command.index("--tp-sizes") + 1] == "4,8"
    assert command[-1].endswith("fp8/official_checkpoint_header_audit.json")
    assert not any(name.startswith("fp8-performance") for name in names)
    assert not any(name.startswith("fp8-quality") for name in names)


def test_fp8_audit_identity_is_part_of_resume_manifest():
    arguments = args()
    arguments.fp8_audit_model = MODULE.OFFICIAL_FP8_CHECKPOINT_REPO

    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))

    assert plan["fp8_audit_model"] == MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
    assert plan["fp8_revision"] == MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION


def test_optional_fp8_checkpoint_adds_reference_execution_matrix():
    arguments = args()
    arguments.fp8_model = "/models/qwen35-fp8"

    stages = MODULE.commands(arguments)
    names = [name for name, _ in stages]

    assert names[-6:] == [
        "official-fp8-checkpoint-audit",
        "fp8-preflight",
        "fp8-performance-matrix",
        "fp8-quality-matrix",
        "fp8-vs-bf16-quality",
        "final-summary",
    ]
    commands = dict(stages)
    audit = commands["official-fp8-checkpoint-audit"]
    assert audit[audit.index("--repo") + 1] == MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
    for stage in (
        "fp8-preflight",
        "fp8-performance-matrix",
        "fp8-quality-matrix",
    ):
        command = commands[stage]
        assert command[command.index("--weight-quant-backend") + 1] == "reference"
    assert "--verify-checkpoint-shards" in commands["fp8-preflight"]
    assert "--no-checkpoint-audit" in commands["fp8-performance-matrix"]
    assert "--no-checkpoint-audit" in commands["fp8-quality-matrix"]
    comparison = commands["fp8-vs-bf16-quality"]
    assert comparison[comparison.index("--candidate-run-id") + 1] == (
        f"{arguments.run_id}-fp8"
    )


def test_optional_fp8_checkpoint_can_validate_resident_storage_backend():
    arguments = args()
    arguments.fp8_model = "/models/qwen35-fp8"
    arguments.fp8_runtime_backend = "resident"

    commands = dict(MODULE.commands(arguments))

    audit = commands["official-fp8-checkpoint-audit"]
    assert audit[audit.index("--fp8-runtime-backend") + 1] == "resident"
    for name in (
        "fp8-preflight",
        "fp8-performance-matrix",
        "fp8-quality-matrix",
    ):
        command = commands[name]
        assert command[command.index("--weight-quant-backend") + 1] == "resident"
    assert MODULE.manifest_plan(arguments, list(commands.items()))[
        "fp8_runtime_backend"
    ] == "resident"


def test_fp8_execution_identity_is_part_of_resume_manifest(tmp_path):
    arguments = args()
    arguments.fp8_model = str(tmp_path / "qwen35-fp8")

    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))

    assert plan["fp8_model"] == str((tmp_path / "qwen35-fp8").resolve())
    assert plan["fp8_audit_model"] == MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
    assert plan["fp8_revision"] == MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION


def test_fp8_audit_artifact_collection_is_isolated(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    fp8_dir = tmp_path / arguments.run_id / "fp8"
    fp8_dir.mkdir(parents=True)
    expected = fp8_dir / "official_checkpoint_header_audit.json"
    expected.write_text('{"valid": true}\n')
    (tmp_path / arguments.run_id / "unrelated.json").write_text("{}\n")

    artifacts = MODULE.collect_stage_artifacts(
        arguments, "official-fp8-checkpoint-audit"
    )

    assert artifacts == [expected]


def test_fp8_execution_artifacts_are_isolated(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    preflight = tmp_path / arguments.run_id / "fp8" / "preflight"
    preflight.mkdir(parents=True)
    mapping = preflight / "checkpoint_mapping_audit.json"
    memory = preflight / "memory_preflight.json"
    mapping.write_text('{"valid": true}\n')
    memory.write_text('{"valid": true}\n')

    assert MODULE.collect_stage_artifacts(arguments, "fp8-preflight") == [
        mapping,
        memory,
    ]


def test_gptq_artifact_collection_is_isolated(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    arguments.gptq_model = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    quality_dir = tmp_path / arguments.run_id / "gptq" / "quality"
    quality_dir.mkdir(parents=True)
    expected = quality_dir / f"{arguments.run_id}-gptq_summary.json"
    expected.write_text('{"valid": true}\n')
    (quality_dir / "unrelated.json").write_text('{}\n')

    artifacts = MODULE.collect_stage_artifacts(arguments, "gptq-quality-matrix")

    assert expected in artifacts
    assert all(path.is_relative_to(quality_dir) for path in artifacts)


def test_gptq_preflight_requires_mapping_and_memory_artifacts(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    preflight = tmp_path / arguments.run_id / "gptq" / "preflight"
    preflight.mkdir(parents=True)
    mapping = preflight / "checkpoint_mapping_audit.json"
    memory = preflight / "memory_preflight.json"
    mapping.write_text('{"valid": true}\n')
    memory.write_text('{"valid": true}\n')

    assert MODULE.collect_stage_artifacts(arguments, "gptq-preflight") == [
        mapping,
        memory,
    ]


def test_attention_commands_match_summary_contract():
    stages = dict(MODULE.commands(args()))
    for tp_size in (4, 8):
        for name, expected_context in (
            ("short", SUMMARY_MODULE.ATTENTION_SHORT_CONTEXT),
            ("long", SUMMARY_MODULE.ATTENTION_LONG_CONTEXT),
            ("max", SUMMARY_MODULE.ATTENTION_MAX_CONTEXT),
        ):
            command = stages[f"attention-{name}-tp{tp_size}"]
            def value(flag):
                return int(command[command.index(flag) + 1])

            assert value("--context-len") == expected_context
            assert value("--num-heads") == (
                SUMMARY_MODULE.QWEN35_TOTAL_QUERY_HEADS // tp_size
            )
            assert value("--num-kv-heads") == (
                SUMMARY_MODULE.QWEN35_KV_HEADS_PER_RANK
            )
            assert value("--head-dim") == SUMMARY_MODULE.QWEN35_HEAD_DIM

    max_attention = stages["attention-max-tp4"]
    assert max_attention[max_attention.index("--batch-size") + 1] == "1"
    assert max_attention[max_attention.index("--variants") + 1] == "v3"
    assert max_attention[max_attention.index("--iters") + 1] == "5"


@pytest.mark.parametrize("value", ["", "../run", "run id", "a/b"])
def test_run_id_rejects_unsafe_paths(value):
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.validate_run_id(value)


@pytest.mark.parametrize("value", ["3,4", "4,4", "8,4,8"])
def test_tp_sizes_reject_unsupported_or_duplicate_parallelism(value):
    with pytest.raises(MODULE.argparse.ArgumentTypeError):
        MODULE.parse_tp_sizes(value)


def test_visible_gpu_count_uses_cuda_runtime_instead_of_env_text(monkeypatch):
    class Cuda:
        @staticmethod
        def device_count():
            return 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"cuda": Cuda})())

    assert MODULE.visible_gpu_count() == 1


def test_commands_reject_duplicate_stage_names_from_programmatic_args():
    arguments = args()
    arguments.tp_sizes = (4, 4)

    with pytest.raises(ValueError, match="stage names must be unique"):
        MODULE.commands(arguments)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        (
            "--max-model-len",
            str(
                MODULE.QUALITY_MAX_PROMPT_LENGTH
                + MODULE.QUALITY_CONTINUATION_LENGTH
                - 1
            ),
            "fixed long-context quality gate",
        ),
        (
            "--max-num-seqs",
            str(MODULE.MIXED_CONCURRENT_SEQUENCES - 1),
            "mixed-workload gate",
        ),
    ],
)
def test_parse_args_rejects_settings_below_fixed_suite_requirements(
    monkeypatch,
    capsys,
    option,
    value,
    message,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_rental_validation.py",
            "--model",
            "/models/qwen35",
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        MODULE.parse_args()

    assert message in capsys.readouterr().err


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


def test_manifest_preserves_hugging_face_model_id():
    arguments = args()
    arguments.model = "Qwen/Qwen3.6-35B-A3B"

    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))

    assert plan["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert len(plan["source_commit"]) == 40


def checkpoint_reports(tmp_path, *, local_digest="shard-digest", fp8=False):
    local = {
        "checkpoint_manifest": {
            "config_sha256": "config-digest",
            "index_sha256": "index-digest",
            "shard_count": 1,
            "present_shard_count": 1,
            "missing_shards": [],
            "files": [
                {
                    "name": "model-00001-of-00001.safetensors",
                    "present": True,
                    "size_bytes": 123,
                    "content_sha256": local_digest,
                }
            ],
        }
    }
    remote = {
        "valid": True,
        "repo": (
            MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
            if fp8
            else MODULE.OFFICIAL_CHECKPOINT_REPO
        ),
        "resolved_revision": (
            MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION
            if fp8
            else MODULE.OFFICIAL_CHECKPOINT_REVISION
        ),
        "config_sha256": "config-digest",
        "index_sha256": "index-digest",
        "shard_count": 1,
        "checkpoint_shards": [
            {
                "name": "model-00001-of-00001.safetensors",
                "size_bytes": 123,
                "sha256": "shard-digest",
            }
        ],
    }
    preflight = tmp_path / "run" / ("fp8/preflight" if fp8 else "preflight")
    preflight.mkdir(parents=True)
    (preflight / "checkpoint_mapping_audit.json").write_text(json.dumps(local))
    remote_root = tmp_path / "run" / ("fp8" if fp8 else "preflight")
    (remote_root / "official_checkpoint_header_audit.json").write_text(
        json.dumps(remote)
    )


def test_preflight_accepts_checkpoint_matching_pinned_revision(tmp_path):
    checkpoint_reports(tmp_path)
    arguments = args()
    arguments.result_dir = str(tmp_path)
    arguments.run_id = "run"

    identity_name, attestation = MODULE.validate_preflight_checkpoint_identity(
        arguments, "preflight"
    )

    assert identity_name == "bf16"
    assert attestation == {
        "local_path": "/models/qwen35",
        "repository": MODULE.OFFICIAL_CHECKPOINT_REPO,
        "resolved_revision": MODULE.OFFICIAL_CHECKPOINT_REVISION,
        "config_sha256": "config-digest",
        "index_sha256": "index-digest",
        "shard_count": 1,
    }


def test_fp8_preflight_accepts_checkpoint_matching_pinned_revision(tmp_path):
    checkpoint_reports(tmp_path, fp8=True)
    arguments = args()
    arguments.result_dir = str(tmp_path)
    arguments.run_id = "run"
    arguments.fp8_model = "/models/qwen36-fp8"

    identity_name, attestation = MODULE.validate_preflight_checkpoint_identity(
        arguments, "fp8-preflight"
    )

    assert identity_name == "fp8"
    assert attestation["repository"] == MODULE.OFFICIAL_FP8_CHECKPOINT_REPO
    assert (
        attestation["resolved_revision"]
        == MODULE.OFFICIAL_FP8_CHECKPOINT_REVISION
    )


def test_preflight_rejects_checkpoint_not_matching_pinned_revision(tmp_path):
    checkpoint_reports(tmp_path, local_digest="different")
    arguments = args()
    arguments.result_dir = str(tmp_path)
    arguments.run_id = "run"

    with pytest.raises(RuntimeError, match="does not match.*official revision"):
        MODULE.validate_preflight_checkpoint_identity(arguments, "preflight")


def test_preflight_rejects_unpinned_remote_revision(tmp_path):
    checkpoint_reports(tmp_path)
    remote_path = (
        tmp_path / "run" / "preflight" / "official_checkpoint_header_audit.json"
    )
    remote = json.loads(remote_path.read_text())
    remote["resolved_revision"] = "0" * 40
    remote_path.write_text(json.dumps(remote))
    arguments = args()
    arguments.result_dir = str(tmp_path)
    arguments.run_id = "run"

    with pytest.raises(RuntimeError, match="does not match.*official revision"):
        MODULE.validate_preflight_checkpoint_identity(arguments, "preflight")


def test_manifest_rejects_resume_from_different_source_commit(tmp_path):
    arguments = args()
    plan = MODULE.manifest_plan(arguments, MODULE.commands(arguments))
    path = tmp_path / "manifest.json"
    MODULE.prepare_manifest(path, plan, resume=False)
    changed = {**plan, "source_commit": "0" * 40}

    with pytest.raises(ValueError, match="does not match"):
        MODULE.prepare_manifest(path, changed, resume=True)


def test_main_rejects_insufficient_gpus_before_building_stages(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_rental_validation.py",
            "--model",
            "Qwen/Qwen3.6-35B-A3B",
            "--tp-sizes",
            "4,8",
        ],
    )
    monkeypatch.setattr(MODULE, "validate_clean_worktree", lambda: None)
    monkeypatch.setattr(MODULE, "visible_gpu_count", lambda: 4)

    def unexpected_commands(_):
        raise AssertionError("stages were built before GPU validation")

    monkeypatch.setattr(MODULE, "commands", unexpected_commands)

    with pytest.raises(SystemExit, match="requires 8 visible GPUs.*no checkpoint"):
        MODULE.main()


def test_main_rejects_dirty_worktree_before_gpu_probe(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_rental_validation.py",
            "--model",
            "/models/qwen35",
        ],
    )

    def reject_dirty():
        raise ValueError("rental validation requires a clean Git worktree")

    monkeypatch.setattr(MODULE, "validate_clean_worktree", reject_dirty)
    monkeypatch.setattr(
        MODULE,
        "visible_gpu_count",
        lambda: pytest.fail("GPU probing started for a dirty worktree"),
    )

    with pytest.raises(SystemExit, match="clean Git worktree"):
        MODULE.main()


def test_validate_clean_worktree_reports_uncommitted_paths(monkeypatch):
    result = MODULE.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=" M nanovllm/engine/scheduler.py\n?? scratch.py\n",
        stderr="",
    )
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *_args, **_kwargs: result)

    with pytest.raises(ValueError, match=r"found 2 uncommitted path\(s\)"):
        MODULE.validate_clean_worktree()


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


def test_pressure_stage_collects_only_its_policy_repeat(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    pressure_dir = tmp_path / arguments.run_id / "pressure" / "tp4" / "fcfs"
    pressure_dir.mkdir(parents=True)
    artifact = pressure_dir / "r2.json"
    artifact.write_text('{"valid": true}\n')
    (pressure_dir / "r1.json").write_text('{"valid": true}\n')

    artifacts = MODULE.collect_stage_artifacts(
        arguments,
        "pressure-fcfs-tp4-r2",
    )

    assert artifacts == [artifact]


def test_fairness_stage_collects_only_its_mode_repeat(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    fairness_dir = (
        tmp_path / arguments.run_id / "fairness" / "enabled" / "tp4"
    )
    fairness_dir.mkdir(parents=True)
    artifact = fairness_dir / "r2.json"
    artifact.write_text('{"valid": true}\n')
    (fairness_dir / "r1.json").write_text('{"valid": true}\n')

    artifacts = MODULE.collect_stage_artifacts(
        arguments,
        "fairness-enabled-tp4-r2",
    )

    assert artifacts == [artifact]


def test_scheduler_stage_collects_only_its_mode_repeat(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    scheduler_dir = (
        tmp_path / arguments.run_id / "scheduler" / "optimized" / "tp4"
    )
    scheduler_dir.mkdir(parents=True)
    artifact = scheduler_dir / "r2.json"
    artifact.write_text('{"valid": true}\n')
    (scheduler_dir / "r1.json").write_text('{"valid": true}\n')

    artifacts = MODULE.collect_stage_artifacts(
        arguments,
        "scheduler-optimized-tp4-r2",
    )

    assert artifacts == [artifact]


def test_pd_transfer_stage_collects_its_exact_profile(tmp_path):
    arguments = args()
    arguments.result_dir = str(tmp_path)
    transfer_dir = tmp_path / arguments.run_id / "pd_transfer" / "tp4"
    transfer_dir.mkdir(parents=True)
    artifact = transfer_dir / "int8-model.json"
    artifact.write_text('{"valid": true}\n')
    (transfer_dir / "auto-float32.json").write_text('{"valid": true}\n')

    artifacts = MODULE.collect_stage_artifacts(
        arguments,
        "pd-transfer-int8-model-tp4",
    )

    assert artifacts == [artifact]


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


def test_source_fingerprint_ignores_repository_excluded_files(monkeypatch, tmp_path):
    source = tmp_path / "nanovllm"
    scripts = tmp_path / "scripts"
    source.mkdir()
    scripts.mkdir()
    (source / "model.py").write_text("VALUE = 1\n")
    (scripts / "run.py").write_text("print('run')\n")
    private_script = scripts / "private_interview.py"
    private_script.write_text("PRIVATE = 1\n")
    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\nname = 'test'\n")
    (tmp_path / ".gitignore").write_text("scripts/private_interview.py\n")
    MODULE.subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "SOURCE_ROOTS", (source, scripts))
    monkeypatch.setattr(MODULE, "SOURCE_FILES", (project,))

    baseline = MODULE.source_tree_sha256()
    (tmp_path / "benchmark_results.json").write_text("{}\n")
    (tmp_path / "notes.md").write_text("private notes\n")
    private_script.write_text("PRIVATE = 2\n")

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
