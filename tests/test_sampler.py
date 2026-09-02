import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"missing dependency: {exc}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sampler_module = load_module(
    "nanovllm_sampler_under_test",
    ROOT / "nanovllm" / "layers" / "sampler.py",
)
sampling_params_module = load_module(
    "nanovllm_sampling_params_under_test",
    ROOT / "nanovllm" / "sampling_params.py",
)
apply_top_k_top_p = sampler_module.apply_top_k_top_p
Sampler = sampler_module.Sampler
SamplingParams = sampling_params_module.SamplingParams


class SamplerTest(unittest.TestCase):
    def test_sampling_params_accept_greedy_temperature(self):
        params = SamplingParams(temperature=0.0)
        self.assertEqual(params.temperature, 0.0)

    def test_top_k_masks_tokens_outside_largest_k(self):
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        top_ks = torch.tensor([2], dtype=torch.int32)
        top_ps = torch.tensor([1.0], dtype=torch.float32)

        with unittest.mock.patch.object(
            torch,
            "sort",
            side_effect=AssertionError("full sort should not run"),
        ):
            filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        self.assertTrue(torch.isfinite(filtered[0, 1]))
        self.assertTrue(torch.isfinite(filtered[0, 2]))
        self.assertTrue(torch.isneginf(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 3]))

    def test_top_k_disabled_keeps_all_tokens_when_top_p_is_one(self):
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        top_ks = torch.tensor([-1], dtype=torch.int32)
        top_ps = torch.tensor([1.0], dtype=torch.float32)

        filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        self.assertIs(filtered, logits)

    def test_empty_batch_returns_input(self):
        logits = torch.empty((0, 4))
        filtered = apply_top_k_top_p(
            logits,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0),
        )

        self.assertIs(filtered, logits)

    def test_top_k_only_path_does_not_sort_full_vocabulary(self):
        logits = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]
        )
        top_ks = torch.tensor([2, -1], dtype=torch.int32)
        top_ps = torch.ones(2)

        with unittest.mock.patch.object(
            torch,
            "sort",
            side_effect=AssertionError("full sort should not run"),
        ):
            filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        self.assertTrue(torch.isfinite(filtered[0, :2]).all())
        self.assertTrue(torch.isneginf(filtered[0, 2:]).all())
        self.assertTrue(torch.equal(filtered[1], logits[1]))

    def test_top_p_keeps_minimum_probability_prefix(self):
        logits = torch.log(torch.tensor([[0.50, 0.25, 0.15, 0.10]]))
        top_ks = torch.tensor([-1], dtype=torch.int32)
        top_ps = torch.tensor([0.80], dtype=torch.float32)

        filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        # Sorted probabilities are 0.50, 0.25, 0.15, 0.10. With top_p=0.80,
        # nucleus sampling keeps the first token that crosses the threshold,
        # so the kept set is 0.50 + 0.25 + 0.15 = 0.90.
        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isfinite(filtered[0, 1]))
        self.assertTrue(torch.isfinite(filtered[0, 2]))
        self.assertTrue(torch.isneginf(filtered[0, 3]))

    def test_top_p_is_computed_after_top_k(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        top_ks = torch.tensor([2], dtype=torch.int32)
        top_ps = torch.tensor([0.70], dtype=torch.float32)

        filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        # After top-k, the first two probabilities are approximately
        # [0.731, 0.269], so top-p=0.70 keeps only the best token. Computing
        # top-p over the original full vocabulary would incorrectly keep two.
        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isneginf(filtered[0, 1]))
        self.assertTrue(torch.isneginf(filtered[0, 2]))
        self.assertTrue(torch.isneginf(filtered[0, 3]))

    def test_top_k_top_p_fast_path_supports_per_row_k(self):
        logits = torch.log(
            torch.tensor(
                [
                    [0.40, 0.30, 0.20, 0.10],
                    [0.40, 0.30, 0.20, 0.10],
                ]
            )
        )
        top_ks = torch.tensor([2, 3], dtype=torch.int32)
        top_ps = torch.tensor([0.50, 0.70], dtype=torch.float32)

        with unittest.mock.patch.object(
            torch,
            "sort",
            side_effect=AssertionError("full sort should not run"),
        ):
            filtered = apply_top_k_top_p(logits, top_ks, top_ps)

        self.assertTrue(torch.isfinite(filtered[0, :1]).all())
        self.assertTrue(torch.isneginf(filtered[0, 1:]).all())
        self.assertTrue(torch.isfinite(filtered[1, :2]).all())
        self.assertTrue(torch.isneginf(filtered[1, 2:]).all())

    def test_all_greedy_rows_use_argmax_without_sampling(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 9.0, 2.0], [8.0, 3.0, 4.0]])
        temperatures = torch.zeros(2)
        top_ks = torch.tensor([2, 2], dtype=torch.int32)
        top_ps = torch.tensor([0.5, 0.5])

        actual = sampler(logits, temperatures, top_ks, top_ps)

        self.assertTrue(torch.equal(actual, torch.tensor([1, 0])))


if __name__ == "__main__":
    unittest.main()
