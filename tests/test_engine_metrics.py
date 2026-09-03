import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_scheduler_state_records_deterministic_prefill_starvation(self):
        metrics = EngineMetrics()

        metrics.record_scheduler_state(
            waiting_queue_len=3,
            running_queue_len=8,
            used_kvcache_blocks=10,
            total_kvcache_blocks=20,
            prefill_starved_steps=7,
            max_prefill_starvation_steps=4,
            preemption_count=2,
            preempted_token_progress=96,
            max_preempted_token_progress=64,
            reclaimed_kv_blocks=5,
        )
        result = metrics.to_dict()

        self.assertEqual(result["prefill_starved_steps"], 7)
        self.assertEqual(result["max_prefill_starvation_steps"], 4)
        self.assertEqual(result["preemption_count"], 2)
        self.assertEqual(result["preempted_token_progress"], 96)
        self.assertEqual(result["max_preempted_token_progress"], 64)
        self.assertEqual(result["reclaimed_kv_blocks"], 5)

        metrics.reset()
        result = metrics.to_dict()
        self.assertEqual(result["prefill_starved_steps"], 0)
        self.assertEqual(result["max_prefill_starvation_steps"], 0)
        self.assertEqual(result["preemption_count"], 0)
        self.assertEqual(result["preempted_token_progress"], 0)
        self.assertEqual(result["max_preempted_token_progress"], 0)
        self.assertEqual(result["reclaimed_kv_blocks"], 0)

    def test_request_latency_percentiles_capture_tail_distribution(self):
        metrics = EngineMetrics()
        sequences = []
        for latency in (1.0, 2.0, 3.0, 10.0):
            sequences.append(
                SimpleNamespace(
                    arrival_time=0.0,
                    first_token_time=latency / 2,
                    finish_time=latency,
                    num_completion_tokens=2,
                )
            )

        metrics.record_finished_sequences(sequences)
        result = metrics.to_dict()

        self.assertEqual(result["num_finished_requests"], 4)
        self.assertEqual(result["p50_ttft_s"], 1.25)
        self.assertAlmostEqual(result["p95_ttft_s"], 4.475)
        self.assertAlmostEqual(result["p99_ttft_s"], 4.895)
        self.assertEqual(result["p50_tpot_s"], 1.25)
        self.assertAlmostEqual(result["p95_tpot_s"], 4.475)
        self.assertAlmostEqual(result["p99_tpot_s"], 4.895)
        self.assertEqual(result["p50_request_latency_s"], 2.5)
        self.assertAlmostEqual(result["p95_request_latency_s"], 8.95)
        self.assertAlmostEqual(result["p99_request_latency_s"], 9.79)

    def test_empty_request_latency_percentiles_are_zero(self):
        result = EngineMetrics().to_dict()

        for metric in ("ttft", "tpot", "request_latency"):
            for percentile in ("p50", "p95", "p99"):
                self.assertEqual(result[f"{percentile}_{metric}_s"], 0.0)

    def test_remote_prefill_receive_metrics_track_batching_and_outcomes(self):
        metrics = EngineMetrics()
        metrics.record_remote_prefill_receive_started()
        metrics.record_remote_prefill_receive_started()
        metrics.record_remote_prefill_poll(2)
        metrics.record_remote_prefill_receive_finished(0.4, outcome="committed")
        metrics.record_remote_prefill_receive_finished(0.6, outcome="timed_out")

        result = metrics.to_dict()
        self.assertEqual(result["remote_prefill_receive_started"], 2)
        self.assertEqual(result["remote_prefill_receive_committed"], 1)
        self.assertEqual(result["remote_prefill_receive_timed_out"], 1)
        self.assertEqual(result["remote_prefill_receive_failed"], 0)
        self.assertEqual(result["remote_prefill_poll_calls"], 1)
        self.assertEqual(result["remote_prefill_requests_polled"], 2)
        self.assertEqual(result["remote_prefill_receive_time_s"], 1.0)
        self.assertEqual(result["avg_remote_prefill_receive_time_s"], 0.5)
        self.assertEqual(result["max_remote_prefill_receive_time_s"], 0.6)

        metrics.reset()
        self.assertEqual(metrics.to_dict()["remote_prefill_receive_started"], 0)
        self.assertEqual(metrics.to_dict()["remote_prefill_poll_calls"], 0)

    def test_remote_prefill_backpressure_metrics_track_each_direction(self):
        metrics = EngineMetrics()

        metrics.record_remote_prefill_backpressure(direction="send")
        metrics.record_remote_prefill_backpressure(direction="receive")

        result = metrics.to_dict()
        self.assertEqual(result["remote_prefill_send_backpressure"], 1)
        self.assertEqual(result["remote_prefill_receive_backpressure"], 1)
        with self.assertRaisesRegex(ValueError, "direction"):
            metrics.record_remote_prefill_backpressure(direction="unknown")


if __name__ == "__main__":
    unittest.main()
