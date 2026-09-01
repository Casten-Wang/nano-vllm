import importlib.util
import pickle
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "engine" / "model_runner.py"


class FakeTensor:
    pass


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

        distributed_module = types.ModuleType("torch.distributed")
        torch_module.distributed = distributed_module

        nanovllm_module = types.ModuleType("nanovllm")
        config_module = types.ModuleType("nanovllm.config")
        engine_module = types.ModuleType("nanovllm.engine")
        execution_module = types.ModuleType("nanovllm.engine.execution")
        sequence_module = types.ModuleType("nanovllm.engine.sequence")
        packing_module = types.ModuleType("nanovllm.engine.kv_cache_packing")
        models_module = types.ModuleType("nanovllm.models")
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


if __name__ == "__main__":
    unittest.main()
