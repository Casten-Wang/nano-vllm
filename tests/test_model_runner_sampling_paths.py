from types import SimpleNamespace
from unittest.mock import Mock

import torch

from nanovllm.engine.model_runner import ModelRunner


def make_runner(*, rank=0):
    runner = object.__new__(ModelRunner)
    runner.rank = rank
    runner.config = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=8),
    )
    runner.prepare_prefill = Mock()
    runner.prepare_decode = Mock(
        return_value=(torch.tensor([1, 2]), torch.tensor([0, 1]))
    )
    runner.prepare_sample = Mock(
        return_value=(
            torch.ones(2),
            torch.full((2,), -1),
            torch.ones(2),
            "metadata",
        )
    )
    runner.run_model = Mock()
    runner.sampler = Mock()
    return runner


def test_all_greedy_batch_skips_full_logits_sampling_path():
    runner = make_runner()
    runner.run_model.return_value = torch.tensor([7, 9])
    seqs = [
        SimpleNamespace(temperature=0.0, top_k=-1),
        SimpleNamespace(temperature=1e-12, top_k=-1),
    ]

    assert runner.run(seqs, is_prefill=False) == [7, 9]

    runner.run_model.assert_called_once_with(
        runner.prepare_decode.return_value[0],
        runner.prepare_decode.return_value[1],
        False,
        greedy=True,
        top_k=None,
    )
    runner.prepare_sample.assert_not_called()
    runner.sampler.assert_not_called()


def test_mixed_sampling_batch_keeps_full_logits_sampler_path():
    runner = make_runner()
    logits = torch.randn(2, 8)
    runner.run_model.return_value = logits
    runner.sampler.return_value = torch.tensor([3, 5])
    seqs = [
        SimpleNamespace(temperature=0.0, top_k=-1),
        SimpleNamespace(temperature=0.7, top_k=2),
    ]

    assert runner.run(seqs, is_prefill=False) == [3, 5]

    runner.run_model.assert_called_once_with(
        runner.prepare_decode.return_value[0],
        runner.prepare_decode.return_value[1],
        False,
        greedy=False,
        top_k=None,
    )
    runner.prepare_sample.assert_called_once_with(seqs)
    runner.sampler.assert_called_once_with(
        logits,
        *runner.prepare_sample.return_value,
    )


def test_all_top_k_batch_uses_compact_candidate_sampling_path():
    runner = make_runner()
    selected_logits = torch.tensor([[9.0, 8.0], [7.0, 6.0]])
    selected_indices = torch.tensor([[1, 3], [4, 2]])
    runner.run_model.return_value = selected_logits, selected_indices
    runner.sampler.sample_top_k_candidates.return_value = torch.tensor([1, 4])
    runner.prepare_sample.return_value = (
        torch.tensor([0.7, 0.8]),
        torch.tensor([1, 2]),
        torch.tensor([1.0, 0.9]),
        "metadata",
    )
    seqs = [
        SimpleNamespace(temperature=0.7, top_k=1),
        SimpleNamespace(temperature=0.8, top_k=2),
    ]

    assert runner.run(seqs, is_prefill=False) == [1, 4]

    runner.run_model.assert_called_once_with(
        runner.prepare_decode.return_value[0],
        runner.prepare_decode.return_value[1],
        False,
        greedy=False,
        top_k=2,
    )
    runner.sampler.sample_top_k_candidates.assert_called_once_with(
        selected_logits,
        selected_indices,
        *runner.prepare_sample.return_value,
    )
    runner.sampler.assert_not_called()


def test_nonzero_rank_uses_same_greedy_collective_without_sampling():
    runner = make_runner(rank=1)
    runner.run_model.return_value = None
    seqs = [SimpleNamespace(temperature=0.0, top_k=-1)]

    assert runner.run(seqs, is_prefill=False) is None

    assert runner.run_model.call_args.kwargs == {"greedy": True, "top_k": None}
    runner.prepare_sample.assert_not_called()
    runner.sampler.assert_not_called()
