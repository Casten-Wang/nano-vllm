from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE = load_script("compare_kv_quality")
MMLU = load_script("evaluate_mmlu_kv")
TEACHER_FORCING = load_script("measure_kv_quality_teacher_forcing")


class HookableAttention:
    def register_forward_hook(self, hook):
        return hook


class Layer:
    pass


def test_mmlu_prompt_format_is_preserved():
    prompt = MMLU.format_prompt(
        {
            "question": "2 + 2?",
            "choices": ["1", "2", "3", "4"],
        }
    )

    assert prompt.endswith("D. 4\nAnswer:")


def test_hybrid_hook_selection_yields_only_full_attention_layers():
    gdn = Layer()
    gdn.linear_attn = object()
    full = Layer()
    full.self_attn = Layer()
    full.self_attn.attn = HookableAttention()
    unrelated = Layer()

    selected = list(COMPARE.iter_full_attention_modules([gdn, full, unrelated]))

    assert selected == [(1, full.self_attn.attn)]
    assert list(TEACHER_FORCING.iter_full_attention_modules([gdn, full, unrelated])) == [
        (1, full.self_attn.attn)
    ]


def test_quality_worker_command_forwards_tensor_parallel_size(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        Path(command[command.index("--worker-output-file") + 1]).touch()

    monkeypatch.setattr(COMPARE.subprocess, "run", fake_run)
    args = Namespace(
        model="/models/qwen35",
        tensor_parallel_size=8,
        output_len=8,
        max_model_len=128,
        max_num_batched_tokens=256,
        vocab_size=100,
        seed=1,
        capture_decode_steps=2,
        save_raw=False,
    )

    COMPARE.run_in_worker_process(
        args,
        mode="auto",
        prompts_file=tmp_path / "prompts.json",
        result_dir=tmp_path,
        batch_name="batch0",
    )

    command = captured["command"]
    assert command[command.index("--tensor-parallel-size") + 1] == "8"


def test_mmlu_worker_command_forwards_tensor_parallel_size(tmp_path):
    args = Namespace(
        model="/models/qwen35",
        tensor_parallel_size=4,
        data_dir="/data/mmlu",
        subjects="abstract_algebra",
        limit_per_subject=10,
        max_model_len=2048,
        batch_size=8,
        trace_max_events=128,
    )

    command = MMLU.build_worker_command(
        args,
        mode="int8",
        prompts_file=tmp_path / "prompts.json",
        output_file=tmp_path / "int8.pt",
        result_dir=tmp_path,
    )

    assert command[command.index("--tensor-parallel-size") + 1] == "4"


def test_mmlu_scored_result_preserves_worker_tensor_parallel_size():
    worker = {
        "mode": "auto",
        "tensor_parallel_size": 8,
        "logits": __import__("torch").tensor([[0.0, 1.0, 2.0, 3.0]]),
        "option_ids": [0, 1, 2, 3],
        "prompt_lengths": [10],
        "execution_stats": {},
        "shape_trace": {},
    }
    questions = [{"subject": "demo", "answer": "D"}]

    result = MMLU.score_worker(questions, worker)

    assert result["tensor_parallel_size"] == 8


def test_teacher_forcing_worker_command_forwards_tensor_parallel_size(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command

    monkeypatch.setattr(TEACHER_FORCING.subprocess, "run", fake_run)
    args = Namespace(
        model="/models/qwen35",
        tensor_parallel_size=8,
        max_model_len=4096,
        max_num_batched_tokens=8192,
    )

    TEACHER_FORCING.run_worker_process(
        args,
        mode="int8",
        cases_file=tmp_path / "cases.json",
        result_dir=tmp_path,
        batch_name="batch0",
    )

    command = captured["command"]
    assert command[command.index("--tensor-parallel-size") + 1] == "8"


def execution_worker(mode, *, include_decode=True):
    attention_paths = (
        {"float_flash_prefill": 1, "float_flash_decode": 1}
        if mode == "auto"
        else {"int8_prefill": 1, "int8_fused_decode": 1}
    )
    if not include_decode:
        attention_paths = {
            key: value
            for key, value in attention_paths.items()
            if "decode" not in key
        }
    return {
        "mode": mode,
        "forced_steps": 2 if include_decode else 1,
        "stage_records": (
            [{"stage": "prefill"}, {"stage": "decode"}]
            if include_decode
            else [{"stage": "prefill"}]
        ),
        "execution_stats": {
            "model_path_counts": (
                {"prefill_eager": 1, "decode_eager": 1}
                if include_decode
                else {"prefill_eager": 1}
            ),
            "attention_path_counts": attention_paths,
            "dropped_execution_signature_steps": 0,
        },
    }


@pytest.mark.parametrize("mode", ["auto", "int8"])
def test_teacher_forcing_execution_validation_requires_real_decode(mode):
    valid = TEACHER_FORCING.validate_worker_execution(
        execution_worker(mode)
    )
    invalid = TEACHER_FORCING.validate_worker_execution(
        execution_worker(mode, include_decode=False)
    )

    assert valid["valid"]
    assert not invalid["valid"]
    assert "decode_eager" in invalid["missing_paths"]


def test_decode_only_quality_aggregate_excludes_prefill_control():
    rows = [
        {
            "step": 0,
            "stage": "prefill",
            "row_count": 1,
            "logits_max_abs": 100.0,
            "logits_mean_abs": 100.0,
            "target_nll_bf16": 10.0,
            "target_nll_int8": 20.0,
        },
        {
            "step": 1,
            "stage": "decode",
            "row_count": 2,
            "logits_max_abs": 2.0,
            "logits_mean_abs": 1.0,
            "target_nll_bf16": 1.0,
            "target_nll_int8": 1.1,
        },
    ]
    decode_rows = [row for row in rows if row["stage"] == "decode"]

    aggregate = TEACHER_FORCING.aggregate_metric_rows(decode_rows)
    ppl = TEACHER_FORCING.perplexity_summary(decode_rows)

    assert aggregate["logits_max_abs"] == 2.0
    assert aggregate["logits_mean_abs"] == 1.0
    assert ppl["token_count"] == 2
    assert ppl["relative_change"] == pytest.approx(__import__("math").exp(0.1) - 1)


def test_worker_comparison_reports_decode_only_quality_scope():
    torch = __import__("torch")
    auto = execution_worker("auto")
    int8 = execution_worker("int8")
    for worker in (auto, int8):
        worker.update(
            {
                "tensor_parallel_size": 4,
                "target_matrix": [[1, 2]],
                "logits": [
                    torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]),
                    torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
                ],
                "kv_metrics": {},
                "shape_trace": {},
            }
        )
    int8["logits"][0] = int8["logits"][0] + 100.0

    result = TEACHER_FORCING.compare_workers(auto, int8)

    assert result["steps_compared"] == 2
    assert result["kv_sensitive_steps_compared"] == 1
    assert result["aggregate"]["logits_max_abs"] == 100.0
    assert result["decode_aggregate"]["logits_max_abs"] == 0.0
    assert result["decode_ppl"]["token_count"] == 1
