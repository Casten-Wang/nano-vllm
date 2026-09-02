from types import SimpleNamespace
from unittest.mock import Mock

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import ScheduleResult


def test_idle_remote_prefill_step_polls_without_running_model():
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(
        poll_remote_prefills=Mock(return_value=[]),
        schedule=Mock(return_value=ScheduleResult([], [])),
        num_waiting=1,
        num_running=0,
        block_manager=SimpleNamespace(num_used_blocks=2, num_total_blocks=8),
        prefill_starved_steps=0,
        max_prefill_starvation_steps=0,
        preemption_count=0,
        preempted_token_progress=0,
        max_preempted_token_progress=0,
        reclaimed_kv_blocks=0,
    )
    engine.metrics = SimpleNamespace(record_scheduler_state=Mock())
    engine.model_runner = SimpleNamespace(call=Mock())

    outputs, tokens, prefill_tokens, decode_tokens = engine.step()

    assert outputs == []
    assert (tokens, prefill_tokens, decode_tokens) == (0, 0, 0)
    engine.scheduler.poll_remote_prefills.assert_called_once()
    engine.model_runner.call.assert_not_called()
