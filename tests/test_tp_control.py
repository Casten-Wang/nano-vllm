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
        sequence_module.Sequence = object
        packing_module.PackedBlockMetadata = object
        packing_module.build_packed_block_metadata = lambda *args, **kwargs: None
        qwen_module.Qwen3ForCausalLM = object
        registry_module.create_model = lambda *args, **kwargs: object()
        cache_plan_module.plan_cache_memory = lambda *args, **kwargs: None
        sampler_module.Sampler = object
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


def make_runner(rank: int, events):
    runner = object.__new__(ModelRunner)
    runner.rank = rank
    runner.world_size = 2
    runner.event = events
    return runner


class TPControlTest(unittest.TestCase):
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

        slots = runner.prepare_state_slots(seqs)

        self.assertEqual(slots.values, [7, 2])

    def test_scheduled_hybrid_sequence_requires_state_slot(self):
        runner = self.make_hybrid_runner()
        seq = SimpleNamespace(state_slot=None, block_table=[1])

        with self.assertRaisesRegex(RuntimeError, "no recurrent state slot"):
            runner.prepare_state_slots([seq])

    def test_mixed_context_orders_decode_before_prefill_slots(self):
        runner = self.make_hybrid_runner()
        runner.block_size = 256
        runner.build_decode_inputs = lambda seqs: {
            "input_ids": [1, 2],
            "positions": [4, 5],
            "slot_mapping": [10, 11],
            "context_lens": [5, 6],
            "block_tables": FakeTensor(),
            "dequant_block_ids": None,
            "dequant_block_tables": None,
            "state_slots": FakeTensor([8, 9]),
            "state_reset_mask": FakeTensor([False, False]),
            "state_token_ranges": (),
        }
        runner.build_prefill_inputs = lambda seqs: {
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
            "state_slots": FakeTensor([3]),
            "state_reset_mask": FakeTensor([True]),
            "state_token_ranges": ((0, 2),),
        }
        captured = {}
        original = model_runner_module.set_context
        model_runner_module.set_context = lambda *args, **kwargs: captured.update(kwargs)
        try:
            runner.prepare_mixed([object()], [object(), object()])
        finally:
            model_runner_module.set_context = original

        self.assertEqual(captured["state_slots"].values, [8, 9, 3])
        self.assertEqual(
            captured["state_reset_mask"].values,
            [False, False, True],
        )
        self.assertEqual(captured["state_token_ranges"], ((2, 4),))

    def test_hybrid_model_disables_cuda_graph(self):
        runner = self.make_hybrid_runner()
        runner.enforce_eager = False
        runner.config.kv_cache_dtype = "auto"

        self.assertFalse(runner.supports_cudagraph())

    def test_recurrent_state_stats_report_local_storage(self):
        runner = object.__new__(ModelRunner)
        first = SimpleNamespace(
            state_pool=SimpleNamespace(
                recurrent=FakeTensor(range(6)),
                convolution=FakeTensor(range(4)),
            )
        )
        second = SimpleNamespace(state_pool=None)
        runner.model = SimpleNamespace(modules=lambda: [first, second])

        stats = runner.get_recurrent_state_stats()

        self.assertEqual(stats["layer_count"], 1)
        self.assertEqual(stats["recurrent_bytes_local_rank"], 12)
        self.assertEqual(stats["convolution_bytes_local_rank"], 8)
        self.assertEqual(stats["total_bytes_local_rank"], 20)


if __name__ == "__main__":
    unittest.main()
