import pickle
import traceback
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.execution import (
    ExecutionStats,
    cuda_graph_buckets,
    select_attention_paths,
    select_model_path,
    supports_cudagraph_policy,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.kv_cache_packing import PackedBlockMetadata, build_packed_block_metadata
from nanovllm.models.registry import create_model
from nanovllm.models.cache_plan import plan_cache_memory
from nanovllm.layers.sampler import Sampler, build_sampling_metadata
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
try:
    from nanovllm.shape_trace import ShapeTrace, activate, restore
except ModuleNotFoundError:
    # A few CPU-only protocol tests load this file directly with a synthetic
    # ``nanovllm`` module instead of importing the package. Keep that test
    # harness compatible without weakening the normal package import.
    class ShapeTrace:
        def __init__(self, *args, **kwargs):
            self.enabled = False

        def reset(self):
            return None

        def record_model_step(self, *args, **kwargs):
            return None

        def record_attention(self, *args, **kwargs):
            return None

        def to_dict(self):
            return {
                "enabled": False,
                "max_events": 0,
                "dropped_events": 0,
                "events": [],
            }

    def activate(trace):
        return None

    def restore(previous):
        return None


CONTROL_SHM_SIZE = 2**20
CONTROL_ACK_TIMEOUT_S = 300.0
CONTROL_STATUS_SIZE = 64 * 1024


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype, device="cpu").element_size()


def validate_initial_cache_capacity(
    *,
    free_bytes: int,
    total_bytes: int,
    gpu_memory_utilization: float,
    state_bytes: int,
    minimum_kv_bytes: int,
) -> int:
    """Return the remaining device budget or fail before large state allocation."""

    used_bytes = total_bytes - free_bytes
    available_bytes = max(
        int(total_bytes * gpu_memory_utilization) - used_bytes,
        0,
    )
    required_bytes = state_bytes + minimum_kv_bytes
    if required_bytes > available_bytes:
        raise RuntimeError(
            "recurrent state cache leaves no room for KV cache within "
            f"gpu_memory_utilization: required {required_bytes} bytes "
            f"({state_bytes} state + {minimum_kv_bytes} minimum KV), "
            f"available {available_bytes} bytes; reduce max_num_seqs, use "
            "recurrent_state_dtype='model', increase tensor_parallel_size, "
            "or increase gpu_memory_utilization"
        )
    return available_bytes - required_bytes


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: (
            tuple[Event, Event, object]
            | list[tuple[Event, Event, object]]
        ),
    ):
        self.config = config
        hf_config = config.model_config
        model_spec = config.model_spec
        if hf_config is None or model_spec is None:
            raise RuntimeError("model configuration was not initialized")
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.worker_control_failed = False
        self._resources_released = False
        self.execution_stats = ExecutionStats()
        self.execution_stats_enabled = False
        self.shape_trace = ShapeTrace()
        self.cudagraph_capture_stats: dict = {
            "supported": False,
            "buckets": [],
        }

        port = config.distributed_port
        if port is None:
            raise ValueError("distributed_port must be assigned before ModelRunner starts")
        dist.init_process_group(
            "nccl",
            f"tcp://127.0.0.1:{port}",
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = create_model(model_spec.architecture, hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_recurrent_state_cache()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.supports_cudagraph():
            self.capture_cudagraph()
        # Model warmup and CUDA Graph capture are initialization work, not
        # benchmark execution. Enable counters only after both are complete.
        self.execution_stats_enabled = True
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                if not config.shared_memory_name:
                    raise ValueError(
                        "shared_memory_name must be assigned for tensor parallelism"
                    )
                self.shm = SharedMemory(
                    name=config.shared_memory_name,
                    create=True,
                    size=CONTROL_SHM_SIZE,
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shared_memory_name)
                self.loop()

    def exit(self):
        self._release_resources(synchronize_cuda=True)

    def abort(self):
        """Release local resources without waiting for failed peer ranks."""

        self.worker_control_failed = True
        self._release_resources(synchronize_cuda=False)

    def _release_resources(self, *, synchronize_cuda: bool):
        if self._resources_released:
            return
        self._resources_released = True
        if self.world_size > 1 and hasattr(self, "shm"):
            self.shm.close()
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        if synchronize_cuda:
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def supports_cudagraph(self) -> bool:
        model_spec = self.config.model_spec
        return supports_cudagraph_policy(
            enforce_eager=self.enforce_eager,
            sliding_window_size=self.config.sliding_window_size,
            is_hybrid=model_spec is not None and model_spec.is_hybrid,
            qwen35_moe_decode_backend=self.config.qwen35_moe_decode_backend,
            kv_cache_dtype=self.config.kv_cache_dtype,
            kv_dequant_backend=self.config.kv_dequant_backend,
        )

    def should_use_cudagraph(self, input_ids: torch.Tensor, is_prefill: bool) -> bool:
        if not hasattr(self, "graphs"):
            return False
        if is_prefill or input_ids.size(0) > 512:
            return False
        if input_ids.size(0) > self.graph_bs[-1]:
            return False
        if self.config.kv_cache_dtype == "int8":
            context = get_context()
            if context.max_context_len >= self.config.int8_partitioned_decode_threshold:
                return False
        return True

    def reset_execution_stats(self):
        self.execution_stats.reset()
        self.execution_stats_enabled = True

    def get_execution_stats(self):
        return self.execution_stats.to_dict()

    def reset_shape_trace(self):
        self.shape_trace.reset()

    def get_shape_trace(self):
        return self.shape_trace.to_dict()

    def get_cudagraph_capture_stats(self):
        return self.cudagraph_capture_stats

    def reset_cuda_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats()

    def get_cuda_memory_stats(self):
        local = {
            "rank": self.rank,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if self.world_size == 1:
            return [local]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        return gathered

    def get_recurrent_state_stats(self):
        recurrent_bytes = 0
        convolution_bytes = 0
        rotary_cache_bytes = 0
        recurrent_dtypes = set()
        convolution_dtypes = set()
        layer_count = 0
        for module in self.model.modules():
            rotary_cache = getattr(module, "cos_sin_cache", None)
            if rotary_cache is not None:
                rotary_cache_bytes += (
                    rotary_cache.numel() * rotary_cache.element_size()
                )
            pool = getattr(module, "state_pool", None)
            if pool is None:
                continue
            layer_count += 1
            recurrent_bytes += pool.recurrent.numel() * pool.recurrent.element_size()
            convolution_bytes += (
                pool.convolution.numel() * pool.convolution.element_size()
            )
            recurrent_dtypes.add(str(pool.recurrent.dtype))
            convolution_dtypes.add(str(pool.convolution.dtype))
        return {
            "layer_count": layer_count,
            "recurrent_bytes_local_rank": recurrent_bytes,
            "convolution_bytes_local_rank": convolution_bytes,
            "total_bytes_local_rank": recurrent_bytes + convolution_bytes,
            "rotary_cache_bytes_local_rank": rotary_cache_bytes,
            "total_model_state_bytes_local_rank": (
                recurrent_bytes + convolution_bytes + rotary_cache_bytes
            ),
            "recurrent_dtypes": sorted(recurrent_dtypes),
            "convolution_dtypes": sorted(convolution_dtypes),
            "graph_padding_slots": int(
                getattr(self, "recurrent_graph_padding_slot", None) is not None
            ),
        }

    def get_recurrent_state_stats_by_rank(self):
        local = {"rank": self.rank, **self.get_recurrent_state_stats()}
        if self.world_size == 1:
            return [local]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        return gathered

    def get_kv_cache_stats_by_rank(self):
        scale = getattr(self, "kv_scale", None)
        data_bytes = self.kv_cache.numel() * self.kv_cache.element_size()
        scale_bytes = scale.numel() * scale.element_size() if scale is not None else 0
        local = {
            "rank": self.rank,
            "data_bytes": data_bytes,
            "scale_bytes": scale_bytes,
            "total_bytes": data_bytes + scale_bytes,
        }
        if self.world_size == 1:
            return [local]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local)
        return gathered

    def _record_execution(
        self,
        *,
        model_path: str,
        attention_paths: tuple[str, ...],
        actual_batch_size: int,
        actual_input_rows: int,
        graph_bucket: int | None,
        max_context_len: int,
        state_access_path: str | None,
    ):
        # Worker ranks execute the same model step, but benchmark results are
        # reported from rank 0. Recording only there avoids multiplying counts
        # by tensor-parallel world size.
        if self.rank != 0 or not self.execution_stats_enabled:
            return
        self.execution_stats.record(
            model_path=model_path,
            attention_paths=attention_paths,
            actual_batch_size=actual_batch_size,
            actual_input_rows=actual_input_rows,
            graph_bucket=graph_bucket,
            max_context_len=max_context_len,
            partition_threshold=self.config.int8_partitioned_decode_threshold,
            sliding_window_size=self.config.sliding_window_size,
            state_access_path=state_access_path,
        )

    def _state_access_path(
        self,
        context,
        *,
        step_kind: str,
        use_graph: bool,
    ) -> str | None:
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return None
        if step_kind == "prefill":
            return "prefill_indexed"
        if use_graph:
            return "decode_graph_indexed"
        if getattr(context, "decode_state_span", None) is not None:
            return "decode_contiguous_view"
        return "decode_indexed_copy"

    def loop(self):
        while True:
            method_name = None
            error = None
            try:
                method_name, args = self.read_shm()
                self.call(method_name, *args)
            except BaseException as exc:
                error = exc
                self.write_worker_status(exc)
            else:
                self.write_worker_status(None)
            finally:
                # The rank-0 process must not publish another command until
                # every worker has consumed and completed (or failed) this one.
                self.event[1].set()
            if error is not None:
                raise error
            if method_name == "exit":
                break

    def write_worker_status(self, error: BaseException | None):
        if not (self.world_size > 1 and self.rank > 0):
            raise RuntimeError("only tensor-parallel worker ranks write status")
        status_buffer = self.event[2]
        if error is None:
            status_buffer[0:4] = (0).to_bytes(4, "little")
            return
        payload = pickle.dumps(
            {
                "rank": self.rank,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if len(payload) > CONTROL_STATUS_SIZE - 4:
            payload = pickle.dumps(
                {
                    "rank": self.rank,
                    "error_type": type(error).__name__,
                    "message": str(error)[:4096],
                    "traceback": "worker traceback exceeded status buffer",
                }
            )
        status_buffer[0:4] = len(payload).to_bytes(4, "little")
        status_buffer[4 : len(payload) + 4] = payload

    def read_worker_status(self, worker_rank: int, status_buffer) -> dict | None:
        n = int.from_bytes(bytes(status_buffer[0:4]), "little")
        if n == 0:
            return None
        if n > CONTROL_STATUS_SIZE - 4:
            raise RuntimeError(
                f"tensor-parallel worker rank {worker_rank} returned an "
                f"invalid status payload size: {n}"
            )
        payload = bytes(status_buffer[4 : n + 4])
        status_buffer[0:4] = (0).to_bytes(4, "little")
        decoded = pickle.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"tensor-parallel worker rank {worker_rank} returned an "
                "invalid status payload"
            )
        return decoded

    def read_shm(self):
        if not (self.world_size > 1 and self.rank > 0):
            raise RuntimeError("only tensor-parallel worker ranks read commands")
        command_event, _ack_event, _status_buffer = self.event
        command_event.wait()
        try:
            n = int.from_bytes(self.shm.buf[0:4], "little")
            if n <= 0 or n > CONTROL_SHM_SIZE - 4:
                raise RuntimeError(f"invalid shared-memory payload size: {n}")
            payload = bytes(self.shm.buf[4 : n + 4])
            decoded = pickle.loads(payload)
            if not isinstance(decoded, list) or not decoded:
                raise RuntimeError("invalid shared-memory command payload")
            method_name, *args = decoded
            if not isinstance(method_name, str):
                raise RuntimeError("shared-memory method name must be a string")
            return method_name, args
        finally:
            command_event.clear()

    def write_shm(self, method_name, *args):
        if not (self.world_size > 1 and self.rank == 0):
            raise RuntimeError("only tensor-parallel rank 0 writes commands")
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n > CONTROL_SHM_SIZE - 4:
            raise ValueError(
                f"shared-memory command is too large: {n} bytes "
                f"(limit {CONTROL_SHM_SIZE - 4})"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for command_event, ack_event, status_buffer in self.event:
            ack_event.clear()
            status_buffer[0:4] = (0).to_bytes(4, "little")
            command_event.set()

    def wait_for_workers(self):
        if self.world_size <= 1 or self.rank != 0:
            return
        worker_errors = []
        for worker_rank, (
            _command_event,
            ack_event,
            status_buffer,
        ) in enumerate(self.event, start=1):
            if not ack_event.wait(timeout=CONTROL_ACK_TIMEOUT_S):
                self.worker_control_failed = True
                raise RuntimeError(
                    f"tensor-parallel worker rank {worker_rank} did not "
                    f"acknowledge the command within {CONTROL_ACK_TIMEOUT_S:.0f}s"
                )
            ack_event.clear()
            try:
                worker_status = self.read_worker_status(
                    worker_rank,
                    status_buffer,
                )
            except BaseException:
                self.worker_control_failed = True
                raise
            if worker_status is not None:
                worker_errors.append(worker_status)
        if worker_errors:
            self.worker_control_failed = True
            details = "\n".join(
                "rank {rank}: {error_type}: {message}\n{traceback}".format(
                    **status
                )
                for status in worker_errors
            )
            raise RuntimeError(
                "tensor-parallel worker command failed:\n" + details
            )

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"unknown ModelRunner method: {method_name}")
        if self.world_size > 1 and self.rank == 0:
            if self.worker_control_failed:
                raise RuntimeError(
                    "tensor-parallel worker control channel has failed"
                )
            self.write_shm(method_name, *args)
            try:
                return method(*args)
            finally:
                self.wait_for_workers()
        return method(*args)

    def _synchronize_kv_block_count(self, local_num_blocks: int) -> int:
        counts = torch.tensor(
            [local_num_blocks],
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if self.world_size > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.MIN)
        shared_num_blocks = int(counts.item())
        if shared_num_blocks <= 0:
            raise RuntimeError(
                "no KV cache blocks are available on all tensor-parallel ranks "
                f"(local rank {self.rank} computed {local_num_blocks})"
            )
        return shared_num_blocks

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        if self.config.kv_cache_dtype == "int8":
            self.allocate_int8_kv_cache()
        else:
            self.allocate_float_kv_cache()

    def allocate_recurrent_state_cache(self):
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return
        model_config = self.config.model_config
        if model_config is None:
            raise RuntimeError("model configuration was not initialized")
        recurrent_dtype = (
            torch.float32
            if self.config.recurrent_state_dtype == "float32"
            else model_config.dtype
        )
        allocated_layers = 0
        num_state_slots = self.config.max_num_seqs
        if self.supports_cudagraph():
            self.recurrent_graph_padding_slot = num_state_slots
            num_state_slots += 1
        else:
            self.recurrent_graph_padding_slot = None
        kv_dtype = (
            torch.int8
            if self.config.kv_cache_dtype == "int8"
            else model_config.dtype
        )
        cache_plan = plan_cache_memory(
            model_spec,
            self.world_size,
            kv_dtype_bytes=dtype_nbytes(kv_dtype),
            recurrent_dtype_bytes=dtype_nbytes(recurrent_dtype),
            convolution_dtype_bytes=dtype_nbytes(model_config.dtype),
        )
        state_bytes = num_state_slots * (
            cache_plan.recurrent_bytes_per_sequence
            + cache_plan.convolution_bytes_per_sequence
        )
        minimum_kv_bytes = cache_plan.kv_bytes_per_token * self.block_size
        if self.config.kv_cache_dtype == "int8":
            minimum_kv_bytes += (
                cache_plan.int8_scale_bytes_per_token * self.block_size
            )
        # Model loading may leave reusable blocks in PyTorch's allocator.
        # Release those blocks so device-free memory is not underestimated.
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        validate_initial_cache_capacity(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            state_bytes=state_bytes,
            minimum_kv_bytes=minimum_kv_bytes,
        )
        for module in self.model.modules():
            allocate = getattr(module, "allocate_state_cache", None)
            if allocate is not None and callable(allocate):
                allocate(
                    num_state_slots,
                    torch.cuda.current_device(),
                    recurrent_dtype=recurrent_dtype,
                )
                allocated_layers += 1
        if allocated_layers != len(model_spec.linear_attention_layers):
            raise RuntimeError(
                f"allocated {allocated_layers} recurrent state layers, expected "
                f"{len(model_spec.linear_attention_layers)}"
            )

    def allocate_float_kv_cache(self):
        config = self.config
        hf_config = config.model_config
        model_spec = config.model_spec
        if hf_config is None or model_spec is None:
            raise RuntimeError("model configuration was not initialized")
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        cache_plan = plan_cache_memory(
            model_spec,
            self.world_size,
            kv_dtype_bytes=dtype_nbytes(hf_config.dtype),
        )
        num_kv_heads = cache_plan.local_kv_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_kv_layers = model_spec.num_kv_cache_layers
        block_bytes = cache_plan.kv_bytes_per_token * self.block_size
        local_num_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        config.num_kvcache_blocks = self._synchronize_kv_block_count(local_num_blocks)
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.kv_scale = None
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.layer_id = layer_id
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.kv_cache_dtype = "auto"
                module.kv_dequant_backend = config.kv_dequant_backend
                module.int8_partitioned_decode_threshold = config.int8_partitioned_decode_threshold
                module.int8_partitioned_decode_partition_size = config.int8_partitioned_decode_partition_size
                layer_id += 1
        if layer_id != num_kv_layers:
            raise RuntimeError(
                f"attached {layer_id} KV cache layers, expected "
                f"{num_kv_layers}"
            )

    def allocate_int8_kv_cache(self):
        config = self.config
        hf_config = config.model_config
        model_spec = config.model_spec
        if hf_config is None or model_spec is None:
            raise RuntimeError("model configuration was not initialized")
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_layers = model_spec.num_kv_cache_layers
        cache_plan = plan_cache_memory(
            model_spec,
            self.world_size,
            kv_dtype_bytes=dtype_nbytes(torch.int8),
        )
        num_kv_heads = cache_plan.local_kv_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_data_bytes = cache_plan.kv_bytes_per_token * self.block_size
        scale_bytes = cache_plan.int8_scale_bytes_per_token * self.block_size
        block_bytes = kv_data_bytes + scale_bytes
        local_num_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        config.num_kvcache_blocks = self._synchronize_kv_block_count(local_num_blocks)
        self.kv_cache = torch.empty(
            2,
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.int8,
        )
        self.kv_scale = torch.empty(
            2,
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            dtype=torch.float16,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.layer_id = layer_id
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.k_scale = self.kv_scale[0, layer_id]
                module.v_scale = self.kv_scale[1, layer_id]
                module.kv_cache_dtype = "int8"
                module.kv_dequant_backend = config.kv_dequant_backend
                module.int8_partitioned_decode_threshold = config.int8_partitioned_decode_threshold
                module.int8_partitioned_decode_partition_size = config.int8_partitioned_decode_partition_size
                layer_id += 1
        if layer_id != num_layers:
            raise RuntimeError(
                f"attached {layer_id} KV cache layers, expected {num_layers}"
            )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_packed_block_metadata(self, metadata: PackedBlockMetadata):
        selected_block_ids = torch.tensor(metadata.selected_block_ids, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_block_tables = torch.tensor(metadata.packed_block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return selected_block_ids, packed_block_tables

    def prepare_state_slots(self, seqs: list[Sequence]):
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return None
        state_slots = []
        for warmup_slot, seq in enumerate(seqs):
            if seq.state_slot is None:
                if seq.block_table:
                    raise RuntimeError(
                        "scheduled hybrid sequence has no recurrent state slot"
                    )
                state_slots.append(warmup_slot)
            else:
                state_slots.append(seq.state_slot)
        return torch.tensor(
            state_slots,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)

    def contiguous_state_span(
        self,
        seqs: list[Sequence],
    ) -> tuple[int, int] | None:
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid or not seqs:
            return None
        slots = [
            warmup_slot if seq.state_slot is None else seq.state_slot
            for warmup_slot, seq in enumerate(seqs)
        ]
        start = slots[0]
        if slots == list(range(start, start + len(slots))):
            return start, len(slots)
        return None

    def prepare_state_reset_mask(self, seqs: list[Sequence]):
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return None
        return torch.tensor(
            [seq.num_cached_tokens == 0 for seq in seqs],
            dtype=torch.bool,
            pin_memory=True,
        ).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        dequant_block_ids = None
        dequant_block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
            if self.config.kv_cache_dtype == "int8":
                prefix_lens = [seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs]
                query_start_lens = [seq.num_cached_tokens for seq in seqs]
                metadata = build_packed_block_metadata(
                    seqs,
                    self.block_size,
                    sliding_window_size=self.config.sliding_window_size,
                    seq_lens=prefix_lens,
                    query_start_lens=query_start_lens,
                )
                dequant_block_ids, dequant_block_tables = self.prepare_packed_block_metadata(metadata)
        state_token_ranges = tuple(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:]))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            dequant_block_ids,
            dequant_block_tables,
            self.config.sliding_window_size,
            state_slots=self.prepare_state_slots(seqs),
            state_reset_mask=self.prepare_state_reset_mask(seqs),
            state_token_ranges=state_token_ranges,
        )
        return input_ids, positions

    def build_prefill_inputs(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        dequant_block_ids = None
        dequant_block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
            if self.config.kv_cache_dtype == "int8":
                prefix_lens = [seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs]
                query_start_lens = [seq.num_cached_tokens for seq in seqs]
                metadata = build_packed_block_metadata(
                    seqs,
                    self.block_size,
                    sliding_window_size=self.config.sliding_window_size,
                    seq_lens=prefix_lens,
                    query_start_lens=query_start_lens,
                )
                dequant_block_ids, dequant_block_tables = self.prepare_packed_block_metadata(metadata)

        return dict(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            dequant_block_ids=dequant_block_ids,
            dequant_block_tables=dequant_block_tables,
            state_slots=self.prepare_state_slots(seqs),
            state_reset_mask=self.prepare_state_reset_mask(seqs),
            state_token_ranges=tuple(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])),
        )

    def build_decode_inputs(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        block_tables = self.prepare_block_tables(seqs)
        dequant_block_ids = None
        dequant_block_tables = None
        if self.config.kv_cache_dtype == "int8" and self.config.kv_dequant_backend != "fused":
            metadata = build_packed_block_metadata(
                seqs,
                self.block_size,
                sliding_window_size=self.config.sliding_window_size,
            )
            dequant_block_ids, dequant_block_tables = self.prepare_packed_block_metadata(metadata)
        return dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            dequant_block_ids=dequant_block_ids,
            dequant_block_tables=dequant_block_tables,
            max_context_len=max(context_lens) if context_lens else 0,
            state_slots=self.prepare_state_slots(seqs),
            state_reset_mask=self.prepare_state_reset_mask(seqs),
            state_token_ranges=(),
            decode_state_span=self.contiguous_state_span(seqs),
        )

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        max_context_len = max(context_lens) if context_lens else 0
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        dequant_block_ids = None
        dequant_block_tables = None
        if self.config.kv_cache_dtype == "int8" and self.config.kv_dequant_backend != "fused":
            metadata = build_packed_block_metadata(
                seqs,
                self.block_size,
                sliding_window_size=self.config.sliding_window_size,
            )
            dequant_block_ids, dequant_block_tables = self.prepare_packed_block_metadata(metadata)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            dequant_block_ids=dequant_block_ids,
            dequant_block_tables=dequant_block_tables,
            sliding_window_size=self.config.sliding_window_size,
            max_context_len=max_context_len,
            state_slots=self.prepare_state_slots(seqs),
            state_reset_mask=self.prepare_state_reset_mask(seqs),
            state_token_ranges=(),
            decode_state_span=self.contiguous_state_span(seqs),
        )
        return input_ids, positions

    def prepare_mixed(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        decode = self.build_decode_inputs(decode_seqs)
        prefill = self.build_prefill_inputs(prefill_seqs)
        input_ids = decode["input_ids"] + prefill["input_ids"]
        positions = decode["positions"] + prefill["positions"]
        slot_mapping = decode["slot_mapping"] + prefill["slot_mapping"]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        decode_context_lens = torch.tensor(decode["context_lens"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefill_cu_seqlens_q = torch.tensor(prefill["cu_seqlens_q"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefill_cu_seqlens_k = torch.tensor(prefill["cu_seqlens_k"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            False,
            slot_mapping=slot_mapping,
            sliding_window_size=self.config.sliding_window_size,
            is_mixed=True,
            decode_token_count=len(decode_seqs),
            prefill_token_count=len(prefill["input_ids"]),
            decode_context_lens=decode_context_lens,
            decode_max_context_len=max(decode["context_lens"]) if decode["context_lens"] else 0,
            decode_block_tables=decode["block_tables"],
            decode_dequant_block_ids=decode["dequant_block_ids"],
            decode_dequant_block_tables=decode["dequant_block_tables"],
            prefill_cu_seqlens_q=prefill_cu_seqlens_q,
            prefill_cu_seqlens_k=prefill_cu_seqlens_k,
            prefill_max_seqlen_q=prefill["max_seqlen_q"],
            prefill_max_seqlen_k=prefill["max_seqlen_k"],
            prefill_block_tables=prefill["block_tables"],
            prefill_dequant_block_ids=prefill["dequant_block_ids"],
            prefill_dequant_block_tables=prefill["dequant_block_tables"],
            state_slots=torch.cat(
                (decode["state_slots"], prefill["state_slots"]),
            )
            if decode["state_slots"] is not None
            else None,
            state_reset_mask=torch.cat(
                (decode["state_reset_mask"], prefill["state_reset_mask"]),
            )
            if decode["state_reset_mask"] is not None
            else None,
            state_token_ranges=tuple(
                (
                    len(decode_seqs) + start,
                    len(decode_seqs) + end,
                )
                for start, end in prefill["state_token_ranges"]
            ),
            decode_state_span=decode.get("decode_state_span"),
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        top_ks = [seq.top_k for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        metadata = build_sampling_metadata(
            temperatures,
            top_ks,
            top_ps,
            int(self.config.model_config.vocab_size),
        )
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ks, top_ps, metadata

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        use_graph = self.should_use_cudagraph(input_ids, is_prefill)
        graph_bucket = None
        if use_graph:
            graph_bucket = next(x for x in self.graph_bs if x >= input_ids.size(0))
        step_kind = "prefill" if is_prefill else "decode"
        model_path = select_model_path(step_kind, use_cuda_graph=use_graph)
        attention_paths = select_attention_paths(
            step_kind=step_kind,
            kv_cache_dtype=self.config.kv_cache_dtype,
            kv_dequant_backend=self.config.kv_dequant_backend,
            max_context_len=context.max_context_len,
            partition_threshold=self.config.int8_partitioned_decode_threshold,
            sliding_window_size=self.config.sliding_window_size,
        )
        state_access_path = self._state_access_path(
            context,
            step_kind=step_kind,
            use_graph=use_graph,
        )
        self._record_execution(
            model_path=model_path,
            attention_paths=attention_paths,
            actual_batch_size=(
                context.cu_seqlens_q.numel() - 1
                if is_prefill and context.cu_seqlens_q is not None
                else input_ids.size(0)
            ),
            actual_input_rows=input_ids.size(0),
            graph_bucket=graph_bucket,
            max_context_len=context.max_context_len,
            state_access_path=state_access_path,
        )
        self.shape_trace.record_model_step(
            input_ids=input_ids,
            positions=positions,
            context=context,
            model_path=model_path,
            attention_paths=attention_paths,
            graph_bucket=graph_bucket,
            state_access_path=state_access_path,
        )
        if not use_graph:
            previous_trace = activate(self.shape_trace)
            try:
                return self.model.compute_logits(self.model(input_ids, positions))
            finally:
                restore(previous_trace)

        bs = input_ids.size(0)
        graph = self.graphs[graph_bucket]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs].fill_(-1)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        if graph_vars["state_slots"] is not None:
            graph_vars["state_slots"].fill_(self.recurrent_graph_padding_slot)
            graph_vars["state_slots"][:bs] = context.state_slots
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_mixed(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> list[int]:
        input_ids, positions = self.prepare_mixed(prefill_seqs, decode_seqs)
        context = get_context()
        attention_paths = select_attention_paths(
            step_kind="mixed",
            kv_cache_dtype=self.config.kv_cache_dtype,
            kv_dequant_backend=self.config.kv_dequant_backend,
            max_context_len=context.decode_max_context_len,
            partition_threshold=self.config.int8_partitioned_decode_threshold,
            sliding_window_size=self.config.sliding_window_size,
        )
        state_access_path = self._state_access_path(
            context,
            step_kind="mixed",
            use_graph=False,
        )
        self._record_execution(
            model_path=select_model_path("mixed", use_cuda_graph=False),
            attention_paths=attention_paths,
            actual_batch_size=len(prefill_seqs) + len(decode_seqs),
            actual_input_rows=input_ids.size(0),
            graph_bucket=None,
            max_context_len=context.decode_max_context_len,
            state_access_path=state_access_path,
        )
        seqs = decode_seqs + prefill_seqs
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        self.shape_trace.record_model_step(
            input_ids=input_ids,
            positions=positions,
            context=context,
            model_path=select_model_path("mixed", use_cuda_graph=False),
            attention_paths=attention_paths,
            graph_bucket=None,
            state_access_path=state_access_path,
        )
        previous_trace = activate(self.shape_trace)
        try:
            logits = self.model.compute_logits(self.model(input_ids, positions))
        finally:
            restore(previous_trace)
        token_ids = self.sampler(logits, *sample_args).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.model_config
        if hf_config is None:
            raise RuntimeError("model configuration was not initialized")
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        state_slots = None
        if config.model_spec is not None and config.model_spec.is_hybrid:
            if self.recurrent_graph_padding_slot is None:
                raise RuntimeError("hybrid CUDA Graph has no padding state slot")
            state_slots = torch.full(
                (max_bs,),
                self.recurrent_graph_padding_slot,
                dtype=torch.int64,
            )
        self.graph_bs = cuda_graph_buckets(max_bs)
        self.graphs = {}
        self.graph_pool = None
        self.cudagraph_capture_stats = {
            "supported": True,
            "max_batch_size": max_bs,
            "max_num_blocks": max_num_blocks,
            "hybrid_recurrent_state": (
                config.model_spec is not None and config.model_spec.is_hybrid
            ),
            "recurrent_padding_slot": self.recurrent_graph_padding_slot,
            "buckets": [],
        }

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                max_context_len=0,
                state_slots=state_slots[:bs] if state_slots is not None else None,
            )
            warmup_start = perf_counter()
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            torch.cuda.synchronize()
            warmup_time_s = perf_counter() - warmup_start
            capture_start = perf_counter()
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            torch.cuda.synchronize()
            capture_time_s = perf_counter() - capture_start
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            self.cudagraph_capture_stats["buckets"].append(
                {
                    "batch_size": bs,
                    "warmup_time_s": warmup_time_s,
                    "capture_time_s": capture_time_s,
                    "graph_pool": str(self.graph_pool),
                }
            )
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            state_slots=state_slots,
        )
