import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "benchmark_metadata.py"
SPEC = importlib.util.spec_from_file_location("benchmark_metadata", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class BenchmarkMetadataTest(unittest.TestCase):
    def test_model_config_metadata_converts_non_json_values(self):
        class FakeConfig:
            def to_dict(self):
                return {
                    "dtype": object(),
                    "nested": {"values": (1, object())},
                }

        result = module.model_config_metadata(FakeConfig())

        self.assertIsInstance(result["dtype"], str)
        self.assertEqual(result["nested"]["values"][0], 1)
        self.assertIsInstance(result["nested"]["values"][1], str)

    def test_kv_cache_storage_metadata_counts_data_and_scales(self):
        class FakeTensor:
            def __init__(self, count, item_size, dtype):
                self._count = count
                self._item_size = item_size
                self.dtype = dtype

            def numel(self):
                return self._count

            def element_size(self):
                return self._item_size

        class FakeRunner:
            kv_cache = FakeTensor(100, 1, "int8")
            kv_scale = FakeTensor(20, 2, "float16")
            world_size = 2

        result = module.kv_cache_storage_metadata(FakeRunner())

        self.assertEqual(result["data_bytes"], 100)
        self.assertEqual(result["scale_bytes"], 40)
        self.assertEqual(result["total_bytes"], 140)
        self.assertEqual(result["estimated_all_ranks_bytes"], 280)
        self.assertEqual(result["data_dtype"], "int8")
        self.assertEqual(result["scale_dtype"], "float16")

    def test_metadata_has_reproducibility_fields_without_cuda(self):
        class FakeVersion:
            cuda = None

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            __version__ = "test"
            version = FakeVersion()
            cuda = FakeCuda()

        result = module.collect_benchmark_metadata(FakeTorch)

        for field in (
            "commit",
            "branch",
            "git_dirty",
            "command",
            "working_directory",
            "benchmark_timestamp",
            "python_version",
            "torch_version",
            "cuda_version",
            "transformers_version",
            "triton_version",
            "flash_attn_version",
            "nvidia_smi_gpus",
        ):
            self.assertIn(field, result)
        self.assertFalse(result["cuda_available"])
        self.assertEqual(result["cuda_device_count"], 0)
        self.assertIsNone(result["device_capability"])

    def test_execution_validation_requires_observed_paths(self):
        result = module.validate_execution_stats(
            {
                "model_path_counts": {"decode_cuda_graph": 3},
                "attention_path_counts": {"int8_fused_decode": 3},
            },
            ["decode_cuda_graph", "int8_fused_decode"],
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_paths"], [])

    def test_execution_validation_reports_missing_path(self):
        result = module.validate_execution_stats(
            {
                "model_path_counts": {"decode_eager": 3},
                "attention_path_counts": {"int8_fused_decode": 3},
            },
            ["decode_cuda_graph"],
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_paths"], ["decode_cuda_graph"])
        self.assertIn("missing required paths", result["reason"])

    def test_empty_execution_stats_are_invalid(self):
        result = module.validate_execution_stats(
            {
                "model_path_counts": {},
                "attention_path_counts": {},
            }
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "no execution path was recorded")

    def test_dropped_execution_signatures_invalidate_result(self):
        result = module.validate_execution_stats(
            {
                "model_path_counts": {"decode_eager": 3},
                "attention_path_counts": {"int8_fused_decode": 3},
                "dropped_execution_signature_steps": 2,
            },
            ["decode_eager", "int8_fused_decode"],
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["dropped_execution_signature_steps"], 2)
        self.assertIn("capacity was exceeded", result["reason"])


if __name__ == "__main__":
    unittest.main()
