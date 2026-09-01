from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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
