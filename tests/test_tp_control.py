import importlib.util
import pickle
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "engine" / "model_runner.py"
CONTEXT_MODULE_PATH = ROOT / "nanovllm" / "utils" / "context.py"


class FakeTensor:
    def __init__(self, values=None):
        self.values = list(values or [])

    def cuda(self, non_blocking=False):
        return self

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 2

    @property
    def dtype(self):
        return "fake16"


class FakeEvent:
    def __init__(self, ready=False):
        self.ready = ready

    def wait(self, timeout=None):
        return self.ready

    def clear(self):
        self.ready = False

    def set(self):
        self.ready = True


def load_model_runner_module():
    managed_names = (
        "torch",
        "torch.distributed",
        "nanovllm",
        "nanovllm.config",
        "nanovllm.engine",
        "nanovllm.engine.execution",
        "nanovllm.engine.decode_input_batch",
        "nanovllm.engine.sampling_input_batch",
        "nanovllm.engine.sequence",
        "nanovllm.engine.kv_cache_packing",
        "nanovllm.models",
        "nanovllm.models.registry",
        "nanovllm.models.cache_plan",
        "nanovllm.models.qwen3",
        "nanovllm.layers",
        "nanovllm.layers.sampler",
        "nanovllm.utils",
        "nanovllm.utils.context",
        "nanovllm.utils.loader",
    )
    saved = {name: sys.modules.get(name) for name in managed_names}
    try:
        torch_module = types.ModuleType("torch")
        torch_module.Tensor = FakeTensor
        torch_module.dtype = object
        torch_module.inference_mode = lambda: (lambda fn: fn)
        torch_module.int32 = object()
        torch_module.int64 = object()
        torch_module.float32 = object()
        torch_module.bool = object()
        torch_module.tensor = lambda values, **kwargs: FakeTensor(values)
        torch_module.cat = lambda tensors: FakeTensor(
            value for tensor in tensors for value in tensor.values
        )

        distributed_module = types.ModuleType("torch.distributed")
        torch_module.distributed = distributed_module

        nanovllm_module = types.ModuleType("nanovllm")
        config_module = types.ModuleType("nanovllm.config")
        engine_module = types.ModuleType("nanovllm.engine")
        execution_module = types.ModuleType("nanovllm.engine.execution")
        decode_input_module = types.ModuleType(
            "nanovllm.engine.decode_input_batch"
        )
        sampling_input_module = types.ModuleType(
            "nanovllm.engine.sampling_input_batch"
        )
        sequence_module = types.ModuleType("nanovllm.engine.sequence")
        packing_module = types.ModuleType("nanovllm.engine.kv_cache_packing")
        models_module = types.ModuleType("nanovllm.models")
        registry_module = types.ModuleType("nanovllm.models.registry")
        cache_plan_module = types.ModuleType("nanovllm.models.cache_plan")
        qwen_module = types.ModuleType("nanovllm.models.qwen3")
        layers_module = types.ModuleType("nanovllm.layers")
        sampler_module = types.ModuleType("nanovllm.layers.sampler")
        utils_module = types.ModuleType("nanovllm.utils")
        context_module = types.ModuleType("nanovllm.utils.context")
        loader_module = types.ModuleType("nanovllm.utils.loader")

        config_module.Config = object
        execution_module.ExecutionStats = object
        execution_module.cuda_graph_buckets = lambda value: (value,)
        execution_module.select_attention_paths = lambda **kwargs: ()
        execution_module.select_model_path = lambda *args, **kwargs: ""
        execution_module.supports_cudagraph_policy = lambda **kwargs: False
        decode_input_module.DecodeInputBatch = object
        decode_input_module.TokenInputBatch = object
        sampling_input_module.SamplingInputBatch = object
        sequence_module.Sequence = object
        packing_module.PackedBlockMetadata = object
        packing_module.build_packed_block_metadata = lambda *args, **kwargs: None
        qwen_module.Qwen3ForCausalLM = object
        registry_module.create_model = lambda *args, **kwargs: object()
        cache_plan_module.plan_cache_memory = lambda *args, **kwargs: None
        sampler_module.Sampler = object
        sampler_module.build_sampling_metadata = lambda *args, **kwargs: None
        context_module.set_context = lambda *args, **kwargs: None
        context_module.get_context = lambda: None
        context_module.reset_context = lambda: None
        loader_module.load_model = lambda *args, **kwargs: None

        sys.modules.update(
            {
                "torch": torch_module,
                "torch.distributed": distributed_module,
                "nanovllm": nanovllm_module,
                "nanovllm.config": config_module,
                "nanovllm.engine": engine_module,
                "nanovllm.engine.execution": execution_module,
                "nanovllm.engine.decode_input_batch": decode_input_module,
                "nanovllm.engine.sampling_input_batch": sampling_input_module,
                "nanovllm.engine.sequence": sequence_module,
                "nanovllm.engine.kv_cache_packing": packing_module,
                "nanovllm.models": models_module,
                "nanovllm.models.registry": registry_module,
                "nanovllm.models.cache_plan": cache_plan_module,
                "nanovllm.models.qwen3": qwen_module,
                "nanovllm.layers": layers_module,
                "nanovllm.layers.sampler": sampler_module,
                "nanovllm.utils": utils_module,
                "nanovllm.utils.context": context_module,
                "nanovllm.utils.loader": loader_module,
            }
        )

        spec = importlib.util.spec_from_file_location(
            "nanovllm_model_runner_under_test",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


model_runner_module = load_model_runner_module()
ModelRunner = model_runner_module.ModelRunner
CONTROL_STATUS_SIZE = model_runner_module.CONTROL_STATUS_SIZE
validate_initial_cache_capacity = model_runner_module.validate_initial_cache_capacity


def make_runner(rank: int, events):
    runner = object.__new__(ModelRunner)
    runner.rank = rank
    runner.world_size = 2
    runner.event = events
    return runner


class TPControlTest(unittest.TestCase):
    def test_initial_cache_capacity_reserves_state_and_one_kv_block(self):
        remaining = validate_initial_cache_capacity(
            free_bytes=900,
            total_bytes=1000,
            gpu_memory_utilization=0.8,
            state_bytes=500,
            minimum_kv_bytes=100,
        )

        self.assertEqual(remaining, 100)

    def test_initial_cache_capacity_fails_before_state_allocation(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"required 701 bytes .* available 700 bytes; at most 6 recurrent "
            r"state slots fit; reduce max_num_seqs",
        ):
            validate_initial_cache_capacity(
                free_bytes=900,
                total_bytes=1000,
                gpu_memory_utilization=0.8,
                state_bytes=601,
                minimum_kv_bytes=100,
                state_bytes_per_slot=100,
            )

    def test_collective_rpc_is_published_before_waiting_for_workers(self):
        runner = make_runner(
            0,
            [(FakeEvent(), FakeEvent(ready=True), bytearray(CONTROL_STATUS_SIZE))],
        )
        runner.worker_control_failed = False
        calls = []
        runner.write_shm = lambda method_name, *args: calls.append(
            ("publish", method_name)
        )
        runner.get_kv_cache_stats_by_rank = lambda: (
            calls.append(("collective", "get_kv_cache_stats_by_rank")) or [
                {"rank": 0},
                {"rank": 1},
            ]
        )
        runner.wait_for_workers = lambda: calls.append(("wait", None))

        stats = runner.call("get_kv_cache_stats_by_rank")

        self.assertEqual(stats, [{"rank": 0}, {"rank": 1}])
        self.assertEqual(
            calls,
            [
                ("publish", "get_kv_cache_stats_by_rank"),
                ("collective", "get_kv_cache_stats_by_rank"),
                ("wait", None),
            ],
        )

    def test_single_rank_cuda_memory_stats(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.world_size = 1
        cuda = SimpleNamespace(
            max_memory_allocated=lambda: 12_345,
            max_memory_reserved=lambda: 67_890,
        )
        original_cuda = getattr(model_runner_module.torch, "cuda", None)
        model_runner_module.torch.cuda = cuda
        try:
            stats = runner.get_cuda_memory_stats()
        finally:
            if original_cuda is None:
                del model_runner_module.torch.cuda
            else:
                model_runner_module.torch.cuda = original_cuda

        self.assertEqual(
            stats,
            [{
                "rank": 0,
                "peak_allocated_bytes": 12_345,
                "peak_reserved_bytes": 67_890,
            }],
        )

    def test_multi_rank_cuda_memory_stats_are_gathered(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.world_size = 2
        cuda = SimpleNamespace(
            max_memory_allocated=lambda: 100,
            max_memory_reserved=lambda: 200,
        )
        original_cuda = getattr(model_runner_module.torch, "cuda", None)
        original_gather = getattr(model_runner_module.dist, "all_gather_object", None)
        model_runner_module.torch.cuda = cuda

        def gather(output, local):
            output[:] = [local, {
                "rank": 1,
                "peak_allocated_bytes": 150,
                "peak_reserved_bytes": 250,
            }]

        model_runner_module.dist.all_gather_object = gather
        try:
            stats = runner.get_cuda_memory_stats()
        finally:
            if original_cuda is None:
                del model_runner_module.torch.cuda
            else:
                model_runner_module.torch.cuda = original_cuda
            if original_gather is None:
                del model_runner_module.dist.all_gather_object
            else:
                model_runner_module.dist.all_gather_object = original_gather

        self.assertEqual([item["rank"] for item in stats], [0, 1])
        self.assertEqual(
            max(item["peak_allocated_bytes"] for item in stats),
            150,
        )

    def test_multi_rank_recurrent_state_stats_are_gathered(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.world_size = 2
        runner.get_recurrent_state_stats = lambda: {
            "total_bytes_local_rank": 100,
        }
        original_gather = getattr(model_runner_module.dist, "all_gather_object", None)

        def gather(output, local):
            output[:] = [local, {"rank": 1, "total_bytes_local_rank": 120}]

        model_runner_module.dist.all_gather_object = gather
        try:
            stats = runner.get_recurrent_state_stats_by_rank()
        finally:
            if original_gather is None:
                del model_runner_module.dist.all_gather_object
            else:
                model_runner_module.dist.all_gather_object = original_gather

        self.assertEqual(stats[0], {"rank": 0, "total_bytes_local_rank": 100})
        self.assertEqual(stats[1], {"rank": 1, "total_bytes_local_rank": 120})

    def test_worker_success_status_round_trip(self):
        status_buffer = bytearray(CONTROL_STATUS_SIZE)
        runner = make_runner(1, (FakeEvent(), FakeEvent(), status_buffer))

        runner.write_worker_status(None)

        self.assertIsNone(runner.read_worker_status(1, status_buffer))

    def test_worker_error_status_is_reported_to_rank_zero(self):
        status_buffer = bytearray(CONTROL_STATUS_SIZE)
        worker = make_runner(1, (FakeEvent(), FakeEvent(), status_buffer))
        try:
            raise ValueError("worker boom")
        except ValueError as error:
            worker.write_worker_status(error)

        rank_zero = make_runner(
            0,
            [(FakeEvent(), FakeEvent(ready=True), status_buffer)],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "rank 1: ValueError: worker boom",
        ):
            rank_zero.wait_for_workers()

    def test_invalid_worker_status_size_is_rejected(self):
        status_buffer = bytearray(CONTROL_STATUS_SIZE)
        status_buffer[0:4] = CONTROL_STATUS_SIZE.to_bytes(4, "little")
        runner = make_runner(0, [])

        with self.assertRaisesRegex(RuntimeError, "invalid status payload size"):
            runner.read_worker_status(1, status_buffer)

    def test_worker_timeout_is_reported(self):
        status_buffer = bytearray(CONTROL_STATUS_SIZE)
        runner = make_runner(
            0,
            [(FakeEvent(), FakeEvent(ready=False), status_buffer)],
        )

        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            runner.wait_for_workers()

    def test_status_payload_must_be_a_dictionary(self):
        status_buffer = bytearray(CONTROL_STATUS_SIZE)
        payload = pickle.dumps(["not", "a", "dict"])
        status_buffer[0:4] = len(payload).to_bytes(4, "little")
        status_buffer[4 : len(payload) + 4] = payload
        runner = make_runner(0, [])

        with self.assertRaisesRegex(RuntimeError, "invalid status payload"):
            runner.read_worker_status(1, status_buffer)


class HybridStateContextTest(unittest.TestCase):
    def test_context_keeps_existing_positional_arguments_compatible(self):
        torch_module = types.ModuleType("torch")
        torch_module.Tensor = FakeTensor
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = torch_module
        try:
            spec = importlib.util.spec_from_file_location(
                "nanovllm_context_under_test",
                CONTEXT_MODULE_PATH,
            )
            context_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[spec.name] = context_module
            spec.loader.exec_module(context_module)
            dequant_ids = object()
            dequant_tables = object()
            context_module.set_context(
                True,
                None,
                None,
                0,
                0,
                None,
                None,
                None,
                dequant_ids,
                dequant_tables,
            )
        finally:
            sys.modules.pop("nanovllm_context_under_test", None)
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        context = context_module.get_context()
        self.assertIs(context.dequant_block_ids, dequant_ids)
        self.assertIs(context.dequant_block_tables, dequant_tables)
        self.assertIsNone(context.state_slots)

    def make_hybrid_runner(self):
        runner = object.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            model_spec=SimpleNamespace(is_hybrid=True),
            sliding_window_size=None,
        )
        return runner

    def test_state_slots_preserve_sequence_order(self):
        runner = self.make_hybrid_runner()
        seqs = [
            SimpleNamespace(state_slot=7, block_table=[1]),
            SimpleNamespace(state_slot=2, block_table=[2]),
        ]
        captured = {}
        original = model_runner_module.torch.tensor
        model_runner_module.torch.tensor = lambda values, **kwargs: (
            captured.update(kwargs) or FakeTensor(values)
        )

        try:
            slots = runner.prepare_state_slots(seqs)
        finally:
            model_runner_module.torch.tensor = original

        self.assertEqual(slots.values, [7, 2])
        self.assertIs(captured["dtype"], model_runner_module.torch.int64)

    def test_scheduled_hybrid_sequence_requires_state_slot(self):
        runner = self.make_hybrid_runner()
        seq = SimpleNamespace(state_slot=None, block_table=[1])

        with self.assertRaisesRegex(RuntimeError, "no recurrent state slot"):
            runner.prepare_state_slots([seq])

    def test_state_reset_slots_upload_only_new_sequences(self):
        runner = self.make_hybrid_runner()
        seqs = [
            SimpleNamespace(
                state_slot=7,
                block_table=[1],
                num_cached_tokens=3,
            ),
            SimpleNamespace(
                state_slot=2,
                block_table=[2],
                num_cached_tokens=0,
            ),
        ]
        captured = {}
        original = model_runner_module.torch.tensor
        model_runner_module.torch.tensor = lambda values, **kwargs: (
            captured.update(kwargs) or FakeTensor(values)
        )

        try:
            reset_slots = runner.prepare_state_reset_slots(seqs)
        finally:
            model_runner_module.torch.tensor = original

        self.assertEqual(reset_slots.values, [2])
        self.assertIs(captured["dtype"], model_runner_module.torch.int64)

    def test_state_reset_slots_skip_upload_when_no_reset_is_needed(self):
        runner = self.make_hybrid_runner()
        seq = SimpleNamespace(
            state_slot=7,
            block_table=[1],
            num_cached_tokens=3,
        )
        original = model_runner_module.torch.tensor
        model_runner_module.torch.tensor = lambda *args, **kwargs: self.fail(
            "no reset tensor should be allocated"
        )

        try:
            reset_slots = runner.prepare_state_reset_slots([seq])
        finally:
            model_runner_module.torch.tensor = original

        self.assertIsNone(reset_slots)

    def test_mixed_context_orders_decode_before_prefill_slots(self):
        runner = self.make_hybrid_runner()
        runner.block_size = 256
        metadata_options = []

        def build_decode_inputs(seqs, **kwargs):
            metadata_options.append(kwargs)
            return {
                "input_ids": [1, 2],
                "positions": [4, 5],
                "slot_mapping": [10, 11],
                "context_lens": [5, 6],
                "block_tables": FakeTensor(),
                "dequant_block_ids": None,
                "dequant_block_tables": None,
                "state_token_ranges": (),
            }

        def build_prefill_inputs(seqs, **kwargs):
            metadata_options.append(kwargs)
            return {
                "input_ids": [3, 4],
                "positions": [0, 1],
                "slot_mapping": [12, 13],
                "cu_seqlens_q": [0, 2],
                "cu_seqlens_k": [0, 2],
                "max_seqlen_q": 2,
                "max_seqlen_k": 2,
                "block_tables": None,
                "dequant_block_ids": None,
                "dequant_block_tables": None,
                "state_token_ranges": ((0, 2),),
            }

        runner.build_decode_inputs = build_decode_inputs
        runner.build_prefill_inputs = build_prefill_inputs
        runner.token_inputs = SimpleNamespace(
            update_tokens=lambda input_ids, positions, slots: (
                FakeTensor(input_ids),
                FakeTensor(positions),
                FakeTensor(slots),
            ),
            update_decode_context_lens=lambda values: FakeTensor(values),
            update_logits_indices=lambda values: FakeTensor(values),
            update_cu_seqlens=lambda query, key: (
                FakeTensor(query),
                FakeTensor(key),
            ),
        )
        decode_seqs = [object(), object()]
        prefill_seqs = [object()]
        state_metadata_calls = []
        runner.prepare_state_slots = lambda seqs, **kwargs: (
            state_metadata_calls.append(("slots", seqs))
            or FakeTensor([8, 9, 3])
        )
        runner.prepare_state_reset_slots = lambda seqs, **kwargs: (
            state_metadata_calls.append(("reset", seqs))
            or FakeTensor([3])
        )
        captured = {}
        original = model_runner_module.set_context
        model_runner_module.set_context = lambda *args, **kwargs: captured.update(kwargs)
        try:
            runner.prepare_mixed(prefill_seqs, decode_seqs)
        finally:
            model_runner_module.set_context = original

        self.assertEqual(captured["state_slots"].values, [8, 9, 3])
        self.assertEqual(
            captured["state_reset_slots"].values,
            [3],
        )
        self.assertEqual(captured["state_token_ranges"], ((2, 4),))
        self.assertEqual(
            metadata_options,
            [
                {"prepare_state_metadata": False},
                {"prepare_state_metadata": False},
            ],
        )
        self.assertEqual(
            state_metadata_calls,
            [
                ("slots", decode_seqs + prefill_seqs),
                ("reset", decode_seqs + prefill_seqs),
            ],
        )

    def test_hybrid_model_disables_cuda_graph(self):
        runner = self.make_hybrid_runner()
        runner.enforce_eager = False
        runner.config.kv_cache_dtype = "auto"
        runner.config.kv_dequant_backend = "fused"
        runner.config.qwen35_moe_decode_backend = "sorted"

        self.assertFalse(runner.supports_cudagraph())

    def test_contiguous_state_span_is_derived_without_device_reads(self):
        runner = self.make_hybrid_runner()
        contiguous = [
            SimpleNamespace(state_slot=3),
            SimpleNamespace(state_slot=4),
            SimpleNamespace(state_slot=5),
        ]
        interleaved = [
            SimpleNamespace(state_slot=3),
            SimpleNamespace(state_slot=5),
        ]

        self.assertEqual(runner.contiguous_state_span(contiguous), (3, 3))
        self.assertIsNone(runner.contiguous_state_span(interleaved))
        self.assertIsNone(runner.contiguous_state_span([]))

    def test_compressed_eager_state_span_is_reported_as_contiguous_view(self):
        runner = self.make_hybrid_runner()
        runner.config.recurrent_state_dtype = "model"
        context = SimpleNamespace(decode_state_span=(3, 2))

        self.assertEqual(
            runner._state_access_path(
                context,
                step_kind="decode",
                use_graph=False,
            ),
            "decode_contiguous_view",
        )

    def test_hybrid_model_enables_graph_safe_decode(self):
        runner = self.make_hybrid_runner()
        runner.enforce_eager = False
        runner.config.kv_cache_dtype = "auto"
        runner.config.kv_dequant_backend = "fused"
        runner.config.qwen35_moe_decode_backend = "batched"
        original = model_runner_module.supports_cudagraph_policy
        model_runner_module.supports_cudagraph_policy = (
            lambda **kwargs: kwargs["qwen35_moe_decode_backend"] == "batched"
        )
        try:
            self.assertTrue(runner.supports_cudagraph())
        finally:
            model_runner_module.supports_cudagraph_policy = original

    def test_recurrent_state_stats_report_local_storage(self):
        runner = object.__new__(ModelRunner)
        first = SimpleNamespace(
            state_pool=SimpleNamespace(
                recurrent=FakeTensor(range(6)),
                convolution=FakeTensor(range(4)),
            )
        )
        second = SimpleNamespace(state_pool=None)
        rotary = SimpleNamespace(
            state_pool=None,
            cos_sin_cache=FakeTensor(range(10)),
        )
        runner.model = SimpleNamespace(modules=lambda: [first, second, rotary])

        stats = runner.get_recurrent_state_stats()

        self.assertEqual(stats["layer_count"], 1)
        self.assertEqual(stats["recurrent_bytes_local_rank"], 12)
        self.assertEqual(stats["convolution_bytes_local_rank"], 8)
        self.assertEqual(stats["total_bytes_local_rank"], 20)
        self.assertEqual(stats["rotary_cache_bytes_local_rank"], 20)
        self.assertEqual(stats["total_model_state_bytes_local_rank"], 40)

    def test_single_rank_kv_cache_stats_report_data_and_scales(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.world_size = 1
        runner.kv_cache = FakeTensor(range(10))
        runner.kv_scale = FakeTensor(range(4))

        stats = runner.get_kv_cache_stats_by_rank()

        self.assertEqual(
            stats,
            [{"rank": 0, "data_bytes": 20, "scale_bytes": 8, "total_bytes": 28}],
        )

    def test_multi_rank_kv_cache_stats_are_gathered(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.world_size = 2
        runner.kv_cache = FakeTensor(range(10))
        runner.kv_scale = FakeTensor(range(4))
        original_gather = getattr(model_runner_module.dist, "all_gather_object", None)

        def gather(output, local):
            output[:] = [local, {
                "rank": 1,
                "data_bytes": 24,
                "scale_bytes": 10,
                "total_bytes": 34,
            }]

        model_runner_module.dist.all_gather_object = gather
        try:
            stats = runner.get_kv_cache_stats_by_rank()
        finally:
            if original_gather is None:
                del model_runner_module.dist.all_gather_object
            else:
                model_runner_module.dist.all_gather_object = original_gather

        self.assertEqual(
            stats,
            [
                {"rank": 0, "data_bytes": 20, "scale_bytes": 8, "total_bytes": 28},
                {"rank": 1, "data_bytes": 24, "scale_bytes": 10, "total_bytes": 34},
            ],
        )


if __name__ == "__main__":
    unittest.main()
