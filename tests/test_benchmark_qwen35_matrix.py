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
        "tp_sizes": (4, 8),
        "checkpoint_audit": True,
        "run_id": "test-run",
        "memory_preflight": True,
        "memory_headroom_gib": 2.0,
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


def test_default_sequence_capacity_tracks_workload(monkeypatch):
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "benchmark_qwen35_matrix.py",
            "--model",
            "/models/qwen35",
            "--num-seqs",
            "96",
        ],
    )

    parsed = MODULE.normalize_args(MODULE.parse_args())

    assert parsed.num_seqs == 96
    assert parsed.max_num_seqs == parsed.num_seqs


def test_explicit_sequence_capacity_is_preserved():
    parsed = args(max_num_seqs=32)

    assert MODULE.normalize_args(parsed).max_num_seqs == 32


def test_non_positive_repeat_count_is_rejected():
    with pytest.raises(ValueError, match="repeats must be positive"):
        MODULE.normalize_args(args(repeats=0))


def test_unsafe_run_id_is_rejected():
    with pytest.raises(ValueError, match="run-id"):
        MODULE.normalize_args(args(run_id="../outside"))


def test_negative_memory_headroom_is_rejected():
    with pytest.raises(ValueError, match="memory-headroom"):
        MODULE.normalize_args(args(memory_headroom_gib=-1))


def test_memory_preflight_covers_each_tp_rank():
    gib = 1024**3
    report = {
        "results": {
            "tp4": {"local_parameter_bytes": 16 * gib},
            "tp8": {"local_parameter_bytes": 8 * gib},
        }
    }

    result = MODULE.validate_memory_capacity(
        report,
        (4, 8),
        [20 * gib] * 8,
        2 * gib,
    )

    assert result["valid"]
    assert result["results"]["tp4"]["required_free_bytes_per_rank"] == 18 * gib
    assert len(result["results"]["tp8"]["free_bytes_by_rank"]) == 8


def test_memory_preflight_rejects_insufficient_rank():
    gib = 1024**3
    report = {"results": {"tp4": {"local_parameter_bytes": 16 * gib}}}

    with pytest.raises(ValueError, match="ranks 2"):
        MODULE.validate_memory_capacity(
            report,
            (4,),
            [20 * gib, 20 * gib, 17 * gib, 20 * gib],
            2 * gib,
        )


def test_case_command_is_eager_and_fully_identified():
    case = MODULE.BenchmarkCase(8, "model", "int8")
    command = MODULE.command_for_case(args(), case, repeat=2)

    assert command[1].endswith("scripts/benchmark_baseline.py")
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--recurrent-state-dtype") + 1] == "model"
    assert command[command.index("--kv-cache-dtype") + 1] == "int8"
    assert command[command.index("--name") + 1] == "qwen35_tp8_state-model_kv-int8_r2"
    assert command[command.index("--require-paths") + 1] == (
        "prefill_eager,decode_eager,int8_prefill,int8_fused_decode"
    )
    assert "--enforce-eager" in command


def test_float_kv_case_requires_float_attention_paths():
    paths = MODULE.required_paths(
        args(),
        MODULE.BenchmarkCase(4, "float32", "auto"),
    )

    assert paths == (
        "prefill_eager",
        "decode_eager",
        "float_flash_prefill",
        "float_flash_decode",
    )


def test_long_context_int8_case_requires_partitioned_decode():
    paths = MODULE.required_paths(
        args(input_len=8192),
        MODULE.BenchmarkCase(8, "model", "int8"),
    )

    assert paths[-1] == "int8_partitioned_decode"


def test_checkpoint_audit_covers_selected_tp_sizes():
    command = MODULE.checkpoint_audit_command(args())

    assert command[1].endswith("scripts/audit_checkpoint_mapping.py")
    assert command[command.index("--model") + 1] == "/models/qwen35"
    assert command[command.index("--tp-sizes") + 1] == "4,8"
    assert command[command.index("--output") + 1].endswith(
        "benchmark_results/matrix/checkpoint_mapping_audit.json"
    )


def test_no_warmup_is_forwarded():
    command = MODULE.command_for_case(
        args(warmup=False),
        MODULE.BenchmarkCase(4, "float32", "auto"),
    )

    assert "--no-warmup" in command


def test_matrix_uses_deterministic_result_stem():
    case = MODULE.BenchmarkCase(4, "model", "auto")
    command = MODULE.command_for_case(args(), case, repeat=2, run_id="rental-a")

    assert command[command.index("--output-stem") + 1] == (
        "rental-a_qwen35_tp4_state-model_kv-bf16_r2"
    )


def test_summary_uses_every_repeat_and_requires_output_parity():
    case = MODULE.BenchmarkCase(8, "float32", "int8")
    command = MODULE.summary_command(args(), case, "rental-a")

    result_paths = [item for item in command if item.endswith(".json")]
    assert len(result_paths) == 4
    assert result_paths[0].endswith("rental-a_qwen35_tp8_state-float32_kv-int8_r1.json")
    assert result_paths[2].endswith("rental-a_qwen35_tp8_state-float32_kv-int8_r3.json")
    assert result_paths[3].endswith("rental-a_qwen35_tp8_state-float32_kv-int8_summary.json")
    assert "--summarize-repeats" in command
    assert "--require-output-parity" in command


def test_matrix_summary_uses_all_configuration_summaries():
    cases = MODULE.build_cases((4, 8))
    command = MODULE.matrix_summary_command(args(), cases, "rental-a")

    summary_paths = [item for item in command if item.endswith("_summary.json")]
    assert len(summary_paths) == len(cases) + 1
    assert "--compare-repeat-summaries" in command
    assert command[-1].endswith("rental-a_matrix_summary.json")


@pytest.mark.parametrize("value", ["", "0", "4,-1", "four"])
def test_invalid_tp_sizes_are_rejected(value):
    with pytest.raises((ValueError, MODULE.argparse.ArgumentTypeError)):
        MODULE.comma_separated_ints(value)
