import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "engine" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("nanovllm_metrics", MODULE_PATH)
metrics_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(metrics_module)
EngineMetrics = metrics_module.EngineMetrics


class EngineMetricsTest(unittest.TestCase):
    def test_pure_prefill_and_decode_have_independent_time(self):
        metrics = EngineMetrics()

        metrics.record_step(100, 2.0, prefill_tokens=100, decode_tokens=0)
        metrics.record_step(-20, 0.5, prefill_tokens=0, decode_tokens=20)
        result = metrics.to_dict()

        self.assertEqual(result["total_prefill_tokens"], 100)
        self.assertEqual(result["total_decode_tokens"], 20)
        self.assertEqual(result["pure_prefill_tokens"], 100)
        self.assertEqual(result["pure_decode_tokens"], 20)
        self.assertEqual(result["pure_prefill_time_s"], 2.0)
        self.assertEqual(result["pure_decode_time_s"], 0.5)
        self.assertEqual(result["pure_prefill_throughput_tok_s"], 50.0)
        self.assertEqual(result["pure_decode_throughput_tok_s"], 40.0)
        self.assertEqual(result["mixed_steps"], 0)

    def test_mixed_step_does_not_duplicate_wall_time(self):
        metrics = EngineMetrics()

        metrics.record_step(70, 0.25, prefill_tokens=80, decode_tokens=10)
        result = metrics.to_dict()

        self.assertEqual(result["total_prefill_tokens"], 80)
        self.assertEqual(result["total_decode_tokens"], 10)
        self.assertEqual(result["mixed_prefill_tokens"], 80)
        self.assertEqual(result["mixed_decode_tokens"], 10)
        self.assertEqual(result["mixed_step_time_s"], 0.25)
        self.assertEqual(result["mixed_steps"], 1)
        self.assertEqual(result["pure_prefill_time_s"], 0.0)
        self.assertEqual(result["pure_decode_time_s"], 0.0)
        self.assertEqual(result["pure_prefill_throughput_tok_s"], 0.0)
        self.assertEqual(result["pure_decode_throughput_tok_s"], 0.0)
        self.assertNotIn("prefill_throughput_tok_s", result)
        self.assertNotIn("decode_throughput_tok_s", result)

    def test_reset_clears_pure_and_mixed_counters(self):
        metrics = EngineMetrics()
        metrics.record_step(10, 1.0, prefill_tokens=10, decode_tokens=0)
        metrics.record_step(5, 2.0, prefill_tokens=6, decode_tokens=1)

        metrics.reset()
        result = metrics.to_dict()

        for name in (
            "total_prefill_tokens",
            "total_decode_tokens",
            "pure_prefill_tokens",
            "pure_decode_tokens",
            "mixed_prefill_tokens",
            "mixed_decode_tokens",
            "pure_prefill_steps",
            "pure_decode_steps",
            "mixed_steps",
        ):
            self.assertEqual(result[name], 0)
        for name in (
            "pure_prefill_time_s",
            "pure_decode_time_s",
            "mixed_step_time_s",
        ):
            self.assertEqual(result[name], 0.0)


if __name__ == "__main__":
    unittest.main()
