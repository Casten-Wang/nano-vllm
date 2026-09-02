import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_cudagraph_parity.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_cudagraph_parity_helpers",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class CudaGraphParityHelperTest(unittest.TestCase):
    def test_extract_decode_steps_excludes_prefill_shape_differences(self):
        artifact = {
            "logits_steps": [
                {"is_prefill": True, "shape": [1, 10], "logits": "prefill"},
                {"is_prefill": False, "shape": [3, 10], "logits": "decode-1"},
                {"is_prefill": False, "shape": [3, 10], "logits": "decode-2"},
            ]
        }

        self.assertEqual(
            module.extract_decode_steps(artifact),
            [
                {
                    "is_prefill": False,
                    "shape": [3, 10],
                    "logits": "decode-1",
                },
                {
                    "is_prefill": False,
                    "shape": [3, 10],
                    "logits": "decode-2",
                },
            ],
        )

    def test_scenario_lengths_cross_block_boundary_in_second_case(self):
        self.assertEqual(module.scenario_lengths(3, 0), [33, 65, 97])
        second = module.scenario_lengths(9, 1)
        self.assertIn(250, second)
        self.assertGreater(max(second), 256)

    def test_scenario_lengths_support_long_context_base(self):
        self.assertEqual(
            module.scenario_lengths(3, 0, 8192),
            [8192, 8224, 8256],
        )

    def test_primer_uses_a_different_padded_graph_bucket(self):
        self.assertEqual(module.primer_batch_size(3, 64), 5)
        self.assertEqual(module.primer_batch_size(9, 64), 7)
        with self.assertRaisesRegex(ValueError, "alternate padded bucket"):
            module.primer_batch_size(1, 1)

    def test_comparison_requires_attention_path_on_both_modes(self):
        import torch

        hidden_step = {
            "is_prefill": False,
            "shape": [1, 1],
            "hidden_states": torch.zeros(1, 1),
        }
        logits_step = {
            "is_prefill": False,
            "shape": [1, 1],
            "logits": torch.zeros(1, 1),
        }
        eager = {
            "output_tokens": [[1]],
            "hidden_steps": [hidden_step],
            "logits_steps": [logits_step],
            "execution_stats": {
                "model_path_counts": {"prefill_eager": 1, "decode_eager": 1},
                "attention_path_counts": {"int8_partitioned_decode": 1},
                "execution_signatures": [],
            },
        }
        graph = {
            **eager,
            "execution_stats": {
                "model_path_counts": {"prefill_eager": 1, "decode_cuda_graph": 1},
                "attention_path_counts": {"int8_fused_decode": 1},
                "execution_signatures": [
                    {"model_path": "decode_cuda_graph", "graph_bucket": 1}
                ],
            },
            "primer": {
                "execution_stats": {
                    "execution_signatures": [
                        {"model_path": "decode_cuda_graph", "graph_bucket": 2}
                    ]
                }
            },
        }

        result = module.compare_artifacts(
            eager,
            graph,
            atol=0,
            rtol=0,
            expected_graph_bucket=1,
            expected_primer_bucket=2,
            expected_attention_path="int8_partitioned_decode",
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["expected_eager_attention_path"])
        self.assertFalse(result["expected_graph_attention_path"])
        self.assertTrue(result["scratch_primed_across_bucket"])

    def test_comparison_rejects_unprimed_graph_scratch(self):
        import torch

        step = {
            "is_prefill": False,
            "shape": [1, 1],
            "logits": torch.zeros(1, 1),
        }
        hidden = {
            "is_prefill": False,
            "shape": [1, 1],
            "hidden_states": torch.zeros(1, 1),
        }
        base_stats = {
            "attention_path_counts": {"int8_fused_decode": 1},
        }
        eager = {
            "output_tokens": [[1]],
            "hidden_steps": [hidden],
            "logits_steps": [step],
            "execution_stats": {
                **base_stats,
                "model_path_counts": {"prefill_eager": 1, "decode_eager": 1},
                "execution_signatures": [],
            },
        }
        graph = {
            **eager,
            "execution_stats": {
                **base_stats,
                "model_path_counts": {
                    "prefill_eager": 1,
                    "decode_cuda_graph": 1,
                },
                "execution_signatures": [
                    {"model_path": "decode_cuda_graph", "graph_bucket": 4}
                ],
            },
            "primer": {"execution_stats": {"execution_signatures": []}},
        }

        result = module.compare_artifacts(
            eager,
            graph,
            atol=0,
            rtol=0,
            expected_graph_bucket=4,
            expected_primer_bucket=8,
            expected_attention_path="int8_fused_decode",
        )

        self.assertFalse(result["scratch_primed_across_bucket"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
