import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            "cuda_devices",
            "nccl_version",
            "transformers_version",
            "triton_version",
            "flash_attn_version",
            "nvidia_smi_gpus",
            "nvidia_smi_topology",
            "cuda_visible_devices",
            "cuda_device_order",
            "nccl_environment",
        ):
            self.assertIn(field, result)
        self.assertFalse(result["cuda_available"])
        self.assertEqual(result["cuda_device_count"], 0)
        self.assertEqual(result["cuda_devices"], [])
        self.assertIsNone(result["device_capability"])
        self.assertIsNone(result["nccl_version"])

    def test_metadata_records_every_gpu_topology_and_collective_runtime(self):
        class FakeVersion:
            cuda = "12.8"

        class FakeNccl:
            @staticmethod
            def version():
                return (2, 27, 3)

        class FakeCuda:
            nccl = FakeNccl()

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

            @staticmethod
            def get_device_name(index=0):
                return f"GPU-{index}"

            @staticmethod
            def get_device_capability(index=0):
                return (9, index)

            @staticmethod
            def get_device_properties(index=0):
                return type(
                    "Properties",
                    (),
                    {
                        "name": f"GPU-{index}",
                        "major": 9,
                        "minor": index,
                        "multi_processor_count": 100 + index,
                        "total_memory": (80 + index) * 1024**3,
                    },
                )()

        class FakeTorch:
            __version__ = "test"
            version = FakeVersion()
            cuda = FakeCuda()

        with (
            patch.object(module, "_nvidia_smi_query", return_value=["gpu-row"]),
            patch.object(module, "_nvidia_smi_topology", return_value="topology"),
            patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "3,1",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "NCCL_ALGO": "Ring",
                },
                clear=False,
            ),
        ):
            result = module.collect_benchmark_metadata(FakeTorch)

        self.assertEqual(
            result["cuda_devices"],
            [
                {
                    "index": 0,
                    "name": "GPU-0",
                    "capability": [9, 0],
                    "multiprocessor_count": 100,
                    "total_memory": 80 * 1024**3,
                },
                {
                    "index": 1,
                    "name": "GPU-1",
                    "capability": [9, 1],
                    "multiprocessor_count": 101,
                    "total_memory": 81 * 1024**3,
                },
            ],
        )
        self.assertEqual(result["nccl_version"], [2, 27, 3])
        self.assertEqual(result["nvidia_smi_topology"], "topology")
        self.assertEqual(result["cuda_visible_devices"], "3,1")
        self.assertEqual(result["cuda_device_order"], "PCI_BUS_ID")
        self.assertEqual(result["nccl_environment"]["NCCL_ALGO"], "Ring")

    def test_runtime_environment_identity_excludes_volatile_fields(self):
        metadata = {
            field: f"value-{field}"
            for field in module.RUNTIME_ENVIRONMENT_FIELDS
        }
        metadata.update(
            {
                "benchmark_timestamp": "later",
                "command": ["different"],
                "working_directory": "/different",
            }
        )

        identity = module.runtime_environment_identity(metadata)

        self.assertEqual(set(identity), set(module.RUNTIME_ENVIRONMENT_FIELDS))
        self.assertNotIn("benchmark_timestamp", identity)

    def test_execution_validation_requires_observed_paths(self):
        result = module.validate_execution_stats(
            {
                "model_path_counts": {"decode_cuda_graph": 3},
                "attention_path_counts": {"int8_fused_decode": 3},
                "state_access_path_counts": {"decode_graph_indexed": 3},
            },
            [
                "decode_cuda_graph",
                "int8_fused_decode",
                "decode_graph_indexed",
            ],
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

    def test_ranked_execution_validation_requires_every_rank(self):
        stats = {
            "model_path_counts": {"decode_eager": 3},
            "attention_path_counts": {"float_flash_decode": 3},
        }
        result = module.validate_execution_stats_by_rank(
            [{"rank": 1, **stats}, {"rank": 0, **stats}],
            expected_world_size=2,
            required_paths=["decode_eager"],
        )

        self.assertTrue(result["valid"])
        self.assertEqual([item["rank"] for item in result["by_rank"]], [0, 1])
        self.assertEqual(result["invalid_ranks"], [])

        with self.assertRaisesRegex(ValueError, "missing ranks: \\[1\\]"):
            module.validate_execution_stats_by_rank(
                [{"rank": 0, **stats}],
                expected_world_size=2,
            )

    def test_ranked_execution_validation_surfaces_one_invalid_rank(self):
        result = module.validate_execution_stats_by_rank(
            [
                {
                    "rank": 0,
                    "model_path_counts": {"decode_eager": 1},
                    "attention_path_counts": {"float_flash_decode": 1},
                },
                {
                    "rank": 1,
                    "model_path_counts": {},
                    "attention_path_counts": {},
                },
            ],
            expected_world_size=2,
            required_paths=["decode_eager"],
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["invalid_ranks"], [1])
        self.assertEqual(result["by_rank"][1]["reason"], "no execution path was recorded")

    def test_ranked_execution_validation_rejects_duplicate_rank(self):
        with self.assertRaisesRegex(ValueError, "invalid or duplicate rank"):
            module.validate_execution_stats_by_rank(
                [{"rank": 0}, {"rank": 0}],
                expected_world_size=2,
            )

    def test_ranked_records_are_complete_and_ordered(self):
        records = module.validate_ranked_records(
            [{"rank": 1, "bytes": 20}, {"rank": 0, "bytes": 10}],
            expected_world_size=2,
            record_name="memory stats",
        )

        self.assertEqual([item["rank"] for item in records], [0, 1])
        self.assertEqual([item["bytes"] for item in records], [10, 20])

    def test_ranked_records_reject_incomplete_or_duplicate_evidence(self):
        with self.assertRaisesRegex(ValueError, "missing ranks: \\[1\\]"):
            module.validate_ranked_records(
                [{"rank": 0}],
                expected_world_size=2,
                record_name="memory stats",
            )
        with self.assertRaisesRegex(ValueError, "invalid or duplicate rank"):
            module.validate_ranked_records(
                [{"rank": 0}, {"rank": 0}],
                expected_world_size=2,
                record_name="memory stats",
            )

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
