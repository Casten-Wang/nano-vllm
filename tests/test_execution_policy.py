import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "engine" / "execution.py"


def load_execution_module():
    spec = importlib.util.spec_from_file_location("nanovllm_execution", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExecutionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution = load_execution_module()

    def test_int8_decode_path_selection(self):
        select = self.execution.select_int8_decode_attention_path

        self.assertEqual(
            select(
                kv_dequant_backend="fused",
                max_context_len=4096,
                partition_threshold=8192,
                sliding_window_size=None,
            ),
            "int8_fused_decode",
        )
        self.assertEqual(
            select(
                kv_dequant_backend="fused",
                max_context_len=8192,
                partition_threshold=8192,
                sliding_window_size=None,
            ),
            "int8_partitioned_decode",
        )
        self.assertEqual(
            select(
                kv_dequant_backend="fused",
                max_context_len=16384,
                partition_threshold=8192,
                sliding_window_size=1024,
            ),
            "int8_fused_decode",
        )
        for backend in ("torch", "triton"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    select(
                        kv_dequant_backend=backend,
                        max_context_len=16384,
                        partition_threshold=8192,
                        sliding_window_size=None,
                    ),
                    "int8_dequant_flash",
                )

    def test_cuda_graph_buckets_cover_partial_maximum(self):
        buckets = self.execution.cuda_graph_buckets

        self.assertEqual(buckets(3), (1, 2, 3))
        self.assertEqual(buckets(10), (1, 2, 4, 8, 10))
        self.assertEqual(buckets(32), (1, 2, 4, 8, 16, 32))
        with self.assertRaisesRegex(ValueError, "positive"):
            buckets(0)

    def test_attention_paths_cover_prefill_decode_and_mixed(self):
        select = self.execution.select_attention_paths

        self.assertEqual(
            select(
                step_kind="prefill",
                kv_cache_dtype="auto",
                kv_dequant_backend="fused",
                max_context_len=0,
                partition_threshold=8192,
                sliding_window_size=None,
            ),
            ("float_flash_prefill",),
        )
        self.assertEqual(
            select(
                step_kind="decode",
                kv_cache_dtype="auto",
                kv_dequant_backend="fused",
                max_context_len=2048,
                partition_threshold=8192,
                sliding_window_size=None,
            ),
            ("float_flash_decode",),
        )
        self.assertEqual(
            select(
                step_kind="mixed",
                kv_cache_dtype="int8",
                kv_dequant_backend="fused",
                max_context_len=4096,
                partition_threshold=8192,
                sliding_window_size=None,
            ),
            ("int8_fused_decode", "int8_prefill"),
        )

    def test_model_path_selection_rejects_graph_for_non_decode(self):
        select = self.execution.select_model_path

        self.assertEqual(select("prefill", use_cuda_graph=False), "prefill_eager")
        self.assertEqual(select("mixed", use_cuda_graph=False), "mixed_eager")
        self.assertEqual(select("decode", use_cuda_graph=False), "decode_eager")
        self.assertEqual(select("decode", use_cuda_graph=True), "decode_cuda_graph")
        with self.assertRaisesRegex(ValueError, "CUDA Graph"):
            select("mixed", use_cuda_graph=True)

    def test_hybrid_cuda_graph_requires_graph_safe_moe_backend(self):
        supports = self.execution.supports_cudagraph_policy
        common = dict(
            enforce_eager=False,
            sliding_window_size=None,
            is_hybrid=True,
            kv_cache_dtype="auto",
            kv_dequant_backend="fused",
        )

        self.assertFalse(
            supports(qwen35_moe_decode_backend="sorted", **common)
        )
        self.assertTrue(
            supports(qwen35_moe_decode_backend="batched", **common)
        )
        self.assertFalse(
            supports(
                qwen35_moe_decode_backend="batched",
                **{**common, "kv_cache_dtype": "int8", "kv_dequant_backend": "torch"},
            )
        )

    def test_partition_count_uses_real_visible_context(self):
        count = self.execution.partition_count

        self.assertEqual(
            count(max_context_len=8193, partition_size=512),
            17,
        )
        self.assertEqual(
            count(
                max_context_len=16384,
                partition_size=512,
                sliding_window_size=1024,
            ),
            2,
        )
        self.assertEqual(count(max_context_len=0, partition_size=512), 1)

    def test_execution_stats_aggregate_per_step_not_per_layer(self):
        stats = self.execution.ExecutionStats()
        kwargs = dict(
            model_path="mixed_eager",
            attention_paths=("int8_fused_decode", "int8_prefill"),
            actual_batch_size=16,
            actual_input_rows=32,
            graph_bucket=None,
            max_context_len=1024,
            partition_threshold=8192,
            sliding_window_size=None,
            state_access_path="decode_contiguous_view",
        )

        stats.record(**kwargs)
        stats.record(**kwargs)
        result = stats.to_dict()

        self.assertEqual(result["model_path_counts"], {"mixed_eager": 2})
        self.assertEqual(
            result["attention_path_counts"],
            {"int8_fused_decode": 2, "int8_prefill": 2},
        )
        self.assertEqual(
            result["state_access_path_counts"],
            {"decode_contiguous_view": 2},
        )
        self.assertEqual(len(result["execution_signatures"]), 1)
        self.assertEqual(result["execution_signatures"][0]["count"], 2)
        self.assertEqual(
            result["execution_signatures"][0]["actual_input_rows"],
            32,
        )
        self.assertEqual(
            result["execution_signatures"][0]["attention_paths"],
            ["int8_fused_decode", "int8_prefill"],
        )
        self.assertEqual(
            result["execution_signatures"][0]["state_access_path"],
            "decode_contiguous_view",
        )

    def test_execution_stats_bound_unique_signatures(self):
        stats = self.execution.ExecutionStats(max_signatures=1)
        common = dict(
            model_path="decode_eager",
            attention_paths=("int8_fused_decode",),
            actual_batch_size=1,
            actual_input_rows=1,
            graph_bucket=None,
            partition_threshold=8192,
            sliding_window_size=None,
        )

        stats.record(max_context_len=1, **common)
        stats.record(max_context_len=2, **common)
        result = stats.to_dict()

        self.assertEqual(len(result["execution_signatures"]), 1)
        self.assertEqual(result["dropped_execution_signature_steps"], 1)


if __name__ == "__main__":
    unittest.main()
