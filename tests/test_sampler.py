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
build_sampling_metadata = sampler_module.build_sampling_metadata
Sampler = sampler_module.Sampler
SamplingParams = sampling_params_module.SamplingParams


class SamplerTest(unittest.TestCase):
    def test_host_metadata_describes_only_sampling_rows(self):
        metadata = build_sampling_metadata(
            [0.0, 0.7, 1.0],
            [-1, 3, 5],
            [1.0, 0.9, 1.0],
            vocab_size=4,
        )

        self.assertEqual(metadata.batch_size, 3)
        self.assertEqual(metadata.sample_count, 2)
        self.assertTrue(metadata.any_top_k_enabled)
        self.assertFalse(metadata.all_top_k_enabled)
        self.assertTrue(metadata.any_top_p_enabled)
        self.assertEqual(metadata.max_top_k, 3)

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

    def test_top_p_only_skips_top_k_workspace(self):
        logits = torch.log(torch.tensor([[0.50, 0.25, 0.15, 0.10]]))
        top_ks = torch.tensor([-1], dtype=torch.int32)
        top_ps = torch.tensor([0.80], dtype=torch.float32)
        metadata = build_sampling_metadata(
            [1.0],
            [-1],
            [0.80],
            vocab_size=4,
        )

        with (
            unittest.mock.patch.object(
                torch,
                "arange",
                side_effect=AssertionError("pure top-p must not build ranks"),
            ),
            unittest.mock.patch.object(
                torch,
                "full_like",
                side_effect=AssertionError("pure top-p must not build top-k bounds"),
            ),
        ):
            filtered = apply_top_k_top_p(
                logits,
                top_ks,
                top_ps,
                metadata,
            )

        self.assertTrue(torch.isfinite(filtered[0, :3]).all())
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

    def test_mixed_batch_promotes_only_sampling_rows(self):
        sampler = Sampler()
        logits = torch.tensor(
            [[1.0, 9.0, 2.0], [8.0, 3.0, 4.0]],
            dtype=torch.bfloat16,
        )
        temperatures = torch.tensor([0.0, 1.0])
        top_ks = torch.tensor([-1, -1], dtype=torch.int32)
        top_ps = torch.ones(2)
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )
        observed = []

        def record_dtype(
            sample_logits,
            sample_top_ks,
            sample_top_ps,
            sample_metadata,
            rank_buffer,
            *,
            inplace,
        ):
            observed.append((sample_logits.shape, sample_logits.dtype))
            self.assertIs(sample_metadata, metadata)
            self.assertIsNone(rank_buffer)
            self.assertTrue(inplace)
            return sample_logits

        with unittest.mock.patch.object(
            sampler_module,
            "apply_top_k_top_p",
            side_effect=record_dtype,
        ):
            actual = sampler(logits, temperatures, top_ks, top_ps, metadata)

        self.assertEqual(observed, [(torch.Size([1, 3]), torch.float32)])
        self.assertEqual(actual[0].item(), 1)

    def test_inplace_filter_reuses_logits_for_every_full_vocab_path(self):
        cases = (
            ([2, 2], [1.0, 1.0]),
            ([2, -1], [1.0, 1.0]),
            ([-1, -1], [0.8, 0.9]),
            ([2, -1], [0.8, 0.9]),
        )
        source = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]
        )
        for top_k_values, top_p_values in cases:
            with self.subTest(top_ks=top_k_values, top_ps=top_p_values):
                top_ks = torch.tensor(top_k_values, dtype=torch.int32)
                top_ps = torch.tensor(top_p_values)
                expected = apply_top_k_top_p(
                    source.clone(),
                    top_ks,
                    top_ps,
                )
                reusable = source.clone()
                actual = apply_top_k_top_p(
                    reusable,
                    top_ks,
                    top_ps,
                    inplace=True,
                )

                self.assertIs(actual, reusable)
                self.assertTrue(torch.equal(actual, expected))

    def test_inplace_filter_rejects_autograd_tensor(self):
        logits = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0]],
            requires_grad=True,
        )

        with self.assertRaisesRegex(ValueError, "inference-only"):
            apply_top_k_top_p(
                logits,
                torch.tensor([-1], dtype=torch.int32),
                torch.tensor([0.8]),
                inplace=True,
            )

    def test_sampler_can_reuse_fp32_inference_logits(self):
        sampler = Sampler()
        with torch.inference_mode():
            logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        metadata = build_sampling_metadata(
            [0.8],
            [-1],
            [0.9],
            vocab_size=4,
        )

        result = sampler(
            logits,
            torch.tensor([0.8]),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([0.9]),
            metadata,
        )

        self.assertEqual(result.shape, torch.Size([1]))

    def test_host_metadata_avoids_device_scalar_reads(self):
        sampler = Sampler()
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        temperatures = torch.ones(1)
        top_ks = torch.tensor([2], dtype=torch.int32)
        top_ps = torch.tensor([0.9])
        metadata = build_sampling_metadata(
            [1.0],
            [2],
            [0.9],
            vocab_size=4,
        )
        original_argmax = torch.Tensor.argmax
        argmax_inputs = []

        def record_argmax(tensor, *args, **kwargs):
            argmax_inputs.append(tensor)
            return original_argmax(tensor, *args, **kwargs)

        with (
            unittest.mock.patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("sampling must not read a device scalar"),
            ),
            unittest.mock.patch.object(
                torch.Tensor,
                "argmax",
                new=record_argmax,
            ),
        ):
            output = sampler(logits, temperatures, top_ks, top_ps, metadata)

        self.assertEqual(output.shape, torch.Size([1]))
        self.assertTrue(argmax_inputs)
        self.assertTrue(
            all(tensor.data_ptr() != logits.data_ptr() for tensor in argmax_inputs)
        )

    def test_host_metadata_matches_tensor_decision_paths(self):
        sampler = Sampler()
        logits = torch.tensor(
            [[5.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 5.0]]
        )
        cases = (
            ([0.0, 0.0], [-1, -1], [1.0, 1.0]),
            ([1.0, 0.8], [2, 3], [0.9, 0.8]),
            ([0.0, 1.0], [-1, 2], [1.0, 0.9]),
        )
        for temperatures_list, top_ks_list, top_ps_list in cases:
            temperatures = torch.tensor(temperatures_list)
            top_ks = torch.tensor(top_ks_list, dtype=torch.int32)
            top_ps = torch.tensor(top_ps_list)
            metadata = build_sampling_metadata(
                temperatures_list,
                top_ks_list,
                top_ps_list,
                vocab_size=logits.size(1),
            )
            torch.manual_seed(71)
            expected = sampler(logits, temperatures, top_ks, top_ps)
            torch.manual_seed(71)
            actual = sampler(logits, temperatures, top_ks, top_ps, metadata)

            self.assertTrue(torch.equal(actual, expected))

    def test_sampling_accepts_vocab_gather_transpose_view(self):
        sampler = Sampler()
        contiguous = torch.tensor(
            [[5.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 5.0]]
        )
        noncontiguous = contiguous.t().contiguous().t()
        self.assertFalse(noncontiguous.is_contiguous())
        temperatures = torch.tensor([0.0, 0.8])
        top_ks = torch.tensor([-1, 3], dtype=torch.int32)
        top_ps = torch.tensor([1.0, 0.9])
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=contiguous.size(1),
        )

        torch.manual_seed(79)
        expected = sampler(
            contiguous,
            temperatures,
            top_ks,
            top_ps,
            metadata,
        )
        torch.manual_seed(79)
        actual = sampler(
            noncontiguous,
            temperatures,
            top_ks,
            top_ps,
            metadata,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_all_top_k_sampling_avoids_full_vocab_filter(self):
        sampler = Sampler()
        logits = torch.tensor(
            [[1.0, 9.0, 2.0, 8.0, 3.0], [7.0, 1.0, 6.0, 2.0, 5.0]],
            dtype=torch.bfloat16,
        )
        temperatures = torch.tensor([1.0, 0.7])
        top_ks = torch.tensor([2, 3], dtype=torch.int32)
        top_ps = torch.tensor([1.0, 0.8])
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )
        def unit_exponential(tensor, *args, **kwargs):
            return tensor.fill_(1)

        with (
            unittest.mock.patch.object(
                sampler_module,
                "apply_top_k_top_p",
                side_effect=AssertionError("full-vocabulary filter must not run"),
            ),
            unittest.mock.patch.object(
                torch.Tensor,
                "exponential_",
                new=unit_exponential,
            ),
        ):
            actual = sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                metadata,
            )

        self.assertTrue(torch.equal(actual, torch.tensor([1, 0])))

    def test_mixed_greedy_and_top_k_sampling_uses_compact_candidates(self):
        sampler = Sampler()
        logits = torch.tensor(
            [[1.0, 9.0, 2.0, 8.0], [7.0, 1.0, 6.0, 2.0]],
            dtype=torch.bfloat16,
        )
        temperatures = torch.tensor([0.0, 0.7])
        top_ks = torch.tensor([-1, 2], dtype=torch.int32)
        top_ps = torch.tensor([1.0, 0.9])
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )

        with unittest.mock.patch.object(
            torch.Tensor,
            "exponential_",
            new=lambda tensor, *args, **kwargs: tensor.fill_(1),
        ):
            actual = sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                metadata,
            )

        self.assertTrue(torch.equal(actual, torch.tensor([1, 0])))

    def test_sampler_reuses_rank_buffer_across_top_k_steps(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        temperatures = torch.ones(1)
        top_ks = torch.tensor([3], dtype=torch.int32)
        top_ps = torch.ones(1)
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )
        with unittest.mock.patch.object(
            torch.Tensor,
            "exponential_",
            new=lambda tensor, *args, **kwargs: tensor.fill_(1),
        ):
            sampler(logits, temperatures, top_ks, top_ps, metadata)
            storage = sampler._rank_buffer.data_ptr()
            with unittest.mock.patch.object(
                sampler_module.torch,
                "arange",
                side_effect=AssertionError("rank buffer must be reused"),
            ):
                sampler(logits, temperatures, top_ks, top_ps, metadata)

        self.assertEqual(sampler._rank_buffer.data_ptr(), storage)

    def test_preselected_global_top_k_matches_full_logits_sampling(self):
        sampler = Sampler()
        logits = torch.tensor(
            [[1.0, 9.0, 4.0, 8.0, 3.0], [7.0, 1.0, 6.0, 2.0, 5.0]],
            dtype=torch.bfloat16,
        )
        temperatures = torch.tensor([0.7, 1.2])
        top_ks = torch.tensor([2, 3], dtype=torch.int32)
        top_ps = torch.tensor([0.8, 0.95])
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )
        selected_logits, selected_indices = torch.topk(
            logits,
            metadata.max_top_k,
            dim=-1,
        )

        with unittest.mock.patch.object(
            torch.Tensor,
            "exponential_",
            new=lambda tensor, *args, **kwargs: tensor.fill_(1),
        ):
            expected = sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                metadata,
            )
            actual = sampler.sample_top_k_candidates(
                selected_logits,
                selected_indices,
                temperatures,
                top_ks,
                top_ps,
                metadata,
            )

        self.assertTrue(torch.equal(actual, expected))

    def test_sampler_reuses_random_noise_buffer_across_steps(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        temperatures = torch.ones(1)
        top_ks = torch.tensor([3], dtype=torch.int32)
        top_ps = torch.ones(1)
        metadata = build_sampling_metadata(
            temperatures.tolist(),
            top_ks.tolist(),
            top_ps.tolist(),
            vocab_size=logits.size(1),
        )

        sampler(logits, temperatures, top_ks, top_ps, metadata)
        storage = sampler._noise_buffer.data_ptr()
        sampler(logits, temperatures, top_ks, top_ps, metadata)

        self.assertEqual(sampler._noise_buffer.data_ptr(), storage)
        self.assertEqual(
            sampler.storage_stats()["noise_buffer_bytes"],
            3 * torch.empty((), dtype=torch.float32).element_size(),
        )


if __name__ == "__main__":
    unittest.main()
