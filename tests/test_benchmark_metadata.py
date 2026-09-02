import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "benchmark_metadata.py"
SPEC = importlib.util.spec_from_file_location("benchmark_metadata", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class BenchmarkMetadataTest(unittest.TestCase):
    def test_checkpoint_manifest_detects_shard_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"test"}')
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"first")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": shard.name}})
            )
            first = module.checkpoint_manifest_metadata(root)
            shard.write_bytes(b"other")
            stat = shard.stat()
            os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            second = module.checkpoint_manifest_metadata(root)

        self.assertEqual(first["shard_count"], 1)
        self.assertIsNotNone(first["config_sha256"])
        self.assertEqual(first["strength"], "metadata-only")
        self.assertNotEqual(first["digest"], second["digest"])

    def test_checkpoint_manifest_uses_content_addressed_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / ("a" * 64)
            blob.write_bytes(b"weights")
            shard = root / "model.safetensors"
            shard.symlink_to(blob)

            result = module.checkpoint_manifest_metadata(root)
            stat = blob.stat()
            os.utime(blob, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            after_mtime_change = module.checkpoint_manifest_metadata(root)

        self.assertEqual(result["strength"], "content-addressed")
        self.assertEqual(result["files"][0]["content_id"], "a" * 64)
        self.assertEqual(result["digest"], after_mtime_change["digest"])

    def test_checkpoint_manifest_detects_config_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            config = root / "config.json"
            config.write_text('{"hidden_size":4}')
            first = module.checkpoint_manifest_metadata(root)
            config.write_text('{"hidden_size":8}')
            second = module.checkpoint_manifest_metadata(root)

        self.assertNotEqual(first["config_sha256"], second["config_sha256"])
        self.assertNotEqual(first["digest"], second["digest"])

    def test_checkpoint_manifest_can_hash_regular_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model.safetensors"
            shard.write_bytes(b"weights")

            result = module.checkpoint_manifest_metadata(
                root,
                hash_shards=True,
            )

        self.assertEqual(result["strength"], "sha256")
        self.assertEqual(
            result["files"][0]["content_sha256"],
            module.hashlib.sha256(b"weights").hexdigest(),
        )

    def test_checkpoint_manifest_can_describe_index_without_local_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"test"}')
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
            )

            result = module.checkpoint_manifest_metadata(
                root,
                require_shards=False,
            )

        self.assertEqual(result["strength"], "index-only")
        self.assertEqual(result["present_shard_count"], 0)
        self.assertEqual(result["missing_shards"], ["model-00001.safetensors"])
        self.assertEqual(result["total_size_bytes"], 0)

    def test_token_digest_is_deterministic_and_length_aware(self):
        outputs = [
            {"token_ids": [1, 2, 3]},
            {"token_ids": [9]},
        ]

        first = module.token_ids_digest(outputs)
        second = module.token_ids_digest(outputs)
        changed = module.token_ids_digest(
            [{"token_ids": [1, 2, 4]}, {"token_ids": [9]}]
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first["digest"], changed["digest"])
        self.assertEqual(first["algorithm"], "sha256")
        self.assertEqual(first["sequence_lengths"], [3, 1])

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

    def test_generation_completion_accepts_exact_finished_workload(self):
        result = module.validate_generation_completion(
            [32, 32, 32],
            expected_num_seqs=3,
            expected_output_len=32,
            waiting_queue_len=0,
            running_queue_len=0,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["actual_output_tokens"], 96)

    def test_generation_completion_reports_incomplete_workload(self):
        result = module.validate_generation_completion(
            [32, 7],
            expected_num_seqs=3,
            expected_output_len=32,
            waiting_queue_len=1,
            running_queue_len=2,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(len(result["errors"]), 3)
        self.assertEqual(result["output_length_min"], 7)
        self.assertEqual(result["output_length_max"], 32)


if __name__ == "__main__":
    unittest.main()
