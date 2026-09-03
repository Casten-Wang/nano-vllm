import pickle
import traceback
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.decode_input_batch import DecodeInputBatch, TokenInputBatch
from nanovllm.engine.execution import (
    ExecutionStats,
    cuda_graph_buckets,
    select_attention_paths,
    select_model_path,
    supports_cudagraph_policy,
)
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.sampling_input_batch import SamplingInputBatch
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
CONTROL_RESULT_COMMAND = "__nanovllm_call_with_result__"
_NO_CONTROL_RESULT = object()


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype, device="cpu").element_size()


def validate_initial_cache_capacity(
    *,
    free_bytes: int,
    total_bytes: int,
    gpu_memory_utilization: float,
    state_bytes: int,
    minimum_kv_bytes: int,
    state_bytes_per_slot: int | None = None,
) -> int:
    """Return the remaining device budget or fail before large state allocation."""

    used_bytes = total_bytes - free_bytes
    available_bytes = max(
        int(total_bytes * gpu_memory_utilization) - used_bytes,
        0,
    )
    required_bytes = state_bytes + minimum_kv_bytes
    if required_bytes > available_bytes:
        slot_hint = ""
        if state_bytes_per_slot is not None:
            if state_bytes_per_slot <= 0:
                raise ValueError("state_bytes_per_slot must be positive")
            max_slots = max(
                (available_bytes - minimum_kv_bytes) // state_bytes_per_slot,
                0,
            )
            slot_hint = f"; at most {max_slots} recurrent state slots fit"
        raise RuntimeError(
            "recurrent state cache leaves no room for KV cache within "
            f"gpu_memory_utilization: required {required_bytes} bytes "
            f"({state_bytes} state + {minimum_kv_bytes} minimum KV), "
            f"available {available_bytes} bytes{slot_hint}; reduce max_num_seqs, use "
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
        self._pending_cache_receives = {}
        self._pending_cache_sends = {}
        self._cache_send_staging_pool = None
        self._cache_receive_staging_pool = None
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
        self.sampler = Sampler(
            config.sampling_chunk_size,
            config.tp_top_k_reduction_max_width,
        )
        self.sampler.reserve_runtime_buffers(int(config.model_config.vocab_size))
        self.reserve_runtime_buffers()
        self.sampling_inputs = (
            SamplingInputBatch(config.max_num_seqs)
            if self.rank == 0
            else None
        )
        self.decode_inputs = DecodeInputBatch(
            config.max_num_seqs,
            (config.max_model_len + self.block_size - 1) // self.block_size,
        )
        self.token_inputs = TokenInputBatch(
            config.max_num_batched_tokens,
            config.max_num_seqs,
            (config.max_model_len + self.block_size - 1) // self.block_size,
        )
        self.allocate_recurrent_state_cache()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.supports_cudagraph():
            self.capture_cudagraph()
        # Model warmup and CUDA Graph capture are initialization work, not
        # runtime evidence. Clear both path and MoE-dispatch counters after
        # initialization; benchmark warmups call this method again.
        self.reset_execution_stats()
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
        pending_receives = getattr(self, "_pending_cache_receives", {})
        for receive in pending_receives.values():
            receive.finish(accepted=False)
        pending_receives.clear()
        pending_sends = getattr(self, "_pending_cache_sends", {})
        for send in pending_sends.values():
            send.finish()
        pending_sends.clear()
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
            weight_quant_backend=getattr(
                self.config,
                "weight_quant_backend",
                "auto",
            ),
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
        self.sampler.reset_stats()
        for module in self.model.modules():
            reset_dispatch_stats = getattr(module, "reset_dispatch_stats", None)
            if reset_dispatch_stats is not None:
                reset_dispatch_stats()
        self.execution_stats_enabled = True

    def reserve_runtime_buffers(self) -> int:
        """Allocate predictable persistent scratch before KV-cache sizing."""

        reserved_modules = 0
        for module in self.model.modules():
            reserve = getattr(module, "reserve_runtime_buffers", None)
            if reserve is not None and callable(reserve):
                reserve(self.config.max_num_seqs)
                reserved_modules += 1
        return reserved_modules

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

    def get_model_parameter_stats(self):
        """Report physical parameter storage, deduplicating tied weights."""

        by_dtype: dict[str, dict[str, int]] = {}
        seen_storage = set()
        logical_parameter_count = 0
        logical_numel = 0
        for parameter in self.model.parameters():
            logical_parameter_count += 1
            logical_numel += parameter.numel()
            storage = parameter.untyped_storage()
            storage_key = (
                str(parameter.device),
                storage.data_ptr(),
                storage.nbytes(),
            )
            if storage_key in seen_storage:
                continue
            seen_storage.add(storage_key)
            dtype = str(parameter.dtype)
            item = by_dtype.setdefault(
                dtype,
                {"storage_count": 0, "bytes": 0},
            )
            item["storage_count"] += 1
            item["bytes"] += storage.nbytes()
        total_bytes = sum(item["bytes"] for item in by_dtype.values())
        return {
            "logical_parameter_count": logical_parameter_count,
            "logical_numel": logical_numel,
            "unique_storage_count": len(seen_storage),
            "by_dtype": by_dtype,
            "total_bytes_local_rank": total_bytes,
        }

    def get_model_parameter_stats_by_rank(self):
        local = {"rank": self.rank, **self.get_model_parameter_stats()}
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

    def get_runtime_buffer_stats(self):
        pools = {}
        dequant_pools = {}
        moe_weight_pools = {}
        resident_fp8_weight_pools = {}
        gptq_workspace_pools = {}
        resident_fp8_storage_stats = []
        qwen35_key_pools = {}
        tp_logits_stats = []
        moe_dispatch_stats = []
        for module in self.model.modules():
            pool = getattr(module, "int8_partitioned_decode_pool", None)
            if pool is not None:
                pools[id(pool)] = pool
            dequant_pool = getattr(module, "int8_dequant_pool", None)
            if dequant_pool is not None:
                dequant_pools[id(dequant_pool)] = dequant_pool
            moe_weight_pool = getattr(
                module,
                "decode_weight_buffer_pool",
                None,
            )
            if moe_weight_pool is not None:
                moe_weight_pools[id(moe_weight_pool)] = moe_weight_pool
            resident_fp8_pool = getattr(module, "resident_weight_buffer_pool", None)
            if resident_fp8_pool is not None:
                resident_fp8_weight_pools[id(resident_fp8_pool)] = resident_fp8_pool
            gptq_workspace_pool = getattr(
                module, "gptq_workspace_pool", None
            )
            if gptq_workspace_pool is not None:
                gptq_workspace_pools[id(gptq_workspace_pool)] = (
                    gptq_workspace_pool
                )
            resident_storage = getattr(module, "resident_fp8_storage_stats", None)
            if resident_storage is not None:
                item = resident_storage()
                if item["total_bytes"]:
                    resident_fp8_storage_stats.append(item)
            qwen35_key_pool = getattr(module, "key_buffer_pool", None)
            if qwen35_key_pool is not None:
                qwen35_key_pools[id(qwen35_key_pool)] = qwen35_key_pool
            logits_stats = getattr(module, "tp_logits_storage_stats", None)
            if logits_stats is not None:
                tp_logits_stats.append(logits_stats())
            dispatch_stats = getattr(module, "dispatch_stats", None)
            if dispatch_stats is not None:
                moe_dispatch_stats.append(dispatch_stats())
        stats = [pool.storage_stats() for pool in pools.values()]
        dequant_stats = [
            pool.storage_stats() for pool in dequant_pools.values()
        ]
        moe_weight_stats = [
            pool.storage_stats() for pool in moe_weight_pools.values()
        ]
        resident_fp8_weight_stats = [
            pool.storage_stats() for pool in resident_fp8_weight_pools.values()
        ]
        gptq_workspace_stats = [
            pool.storage_stats() for pool in gptq_workspace_pools.values()
        ]
        qwen35_key_stats = [
            pool.storage_stats() for pool in qwen35_key_pools.values()
        ]
        partitioned_total = sum(item["total_bytes"] for item in stats)
        dequant_total = sum(item["total_bytes"] for item in dequant_stats)
        moe_weight_total = sum(
            item["storage_bytes"] for item in moe_weight_stats
        )
        moe_workspace_total = sum(
            item.get("workspace_bytes", 0) for item in moe_weight_stats
        )
        resident_fp8_workspace_total = sum(
            item["storage_bytes"] for item in resident_fp8_weight_stats
        )
        gptq_workspace_total = sum(
            item["storage_bytes"] for item in gptq_workspace_stats
        )
        qwen35_key_total = sum(
            item["storage_bytes"] for item in qwen35_key_stats
        )
        sampler_stats = self.sampler.storage_stats()
        sampler_runtime_stats = self.sampler.runtime_stats()
        sampler_total = sum(sampler_stats.values())
        tp_logits_total = sum(item["total_bytes"] for item in tp_logits_stats)
        host_staging_pool = getattr(self, "_cache_send_staging_pool", None)
        host_staging_stats = (
            host_staging_pool.storage_stats()
            if host_staging_pool is not None
            else {
                "storage_bytes": 0,
                "allocation_count": 0,
                "reuse_count": 0,
                "transient_allocation_count": 0,
                "leased": 0,
            }
        )
        receive_staging_pool = getattr(
            self,
            "_cache_receive_staging_pool",
            None,
        )
        receive_staging_stats = (
            receive_staging_pool.storage_stats()
            if receive_staging_pool is not None
            else {
                "storage_bytes": 0,
                "allocation_count": 0,
                "reuse_count": 0,
                "transient_allocation_count": 0,
                "leased": 0,
            }
        )
        return {
            "int8_partitioned_decode_pool_count": len(stats),
            "int8_partitioned_workspace_bytes": sum(
                item["workspace_bytes"] for item in stats
            ),
            "int8_partitioned_output_bytes": sum(
                item["output_bytes"] for item in stats
            ),
            "int8_dequant_pool_count": len(dequant_stats),
            "int8_dequant_buffer_bytes": dequant_total,
            "moe_decode_weight_pool_count": len(moe_weight_stats),
            "moe_decode_weight_buffer_bytes": moe_weight_total,
            "moe_decode_workspace_bytes": moe_workspace_total,
            "moe_sorted_dispatch_count": sum(
                item["sorted_dispatch_count"] for item in moe_dispatch_stats
            ),
            "moe_sorted_decode_dispatch_count": sum(
                item["sorted_decode_dispatch_count"]
                for item in moe_dispatch_stats
            ),
            "moe_sorted_prefill_dispatch_count": sum(
                item["sorted_prefill_dispatch_count"]
                for item in moe_dispatch_stats
            ),
            "moe_batched_dispatch_count": sum(
                item["batched_dispatch_count"] for item in moe_dispatch_stats
            ),
            "moe_host_route_sync_count": sum(
                item["host_route_sync_count"] for item in moe_dispatch_stats
            ),
            "moe_host_route_sync_items": sum(
                item["host_route_sync_items"] for item in moe_dispatch_stats
            ),
            "moe_decode_host_route_sync_count": sum(
                item["decode_host_route_sync_count"]
                for item in moe_dispatch_stats
            ),
            "moe_prefill_host_route_sync_count": sum(
                item["prefill_host_route_sync_count"]
                for item in moe_dispatch_stats
            ),
            "resident_fp8_weight_pool_count": len(resident_fp8_weight_stats),
            "resident_fp8_dequant_workspace_bytes": resident_fp8_workspace_total,
            "resident_fp8_dequant_workspace_allocation_count": sum(
                item["allocation_count"] for item in resident_fp8_weight_stats
            ),
            "resident_fp8_dequant_workspace_reuse_count": sum(
                item["reuse_count"] for item in resident_fp8_weight_stats
            ),
            "gptq_expert_workspace_pool_count": len(gptq_workspace_stats),
            "gptq_expert_workspace_bytes": sum(
                item["storage_bytes"] for item in gptq_workspace_stats
            ),
            "gptq_expert_workspace_allocation_count": sum(
                item["allocation_count"] for item in gptq_workspace_stats
            ),
            "gptq_expert_workspace_reuse_count": sum(
                item["reuse_count"] for item in gptq_workspace_stats
            ),
            "resident_fp8_expert_layer_count": len(resident_fp8_storage_stats),
            "resident_fp8_expert_weight_bytes": sum(
                item["weight_bytes"] for item in resident_fp8_storage_stats
            ),
            "resident_fp8_expert_scale_bytes": sum(
                item["scale_bytes"] for item in resident_fp8_storage_stats
            ),
            "qwen35_key_buffer_pool_count": len(qwen35_key_stats),
            "qwen35_key_buffer_bytes": qwen35_key_total,
            "qwen35_key_buffer_allocation_count": sum(
                item["allocation_count"] for item in qwen35_key_stats
            ),
            "qwen35_key_buffer_reuse_count": sum(
                item["reuse_count"] for item in qwen35_key_stats
            ),
            "sampling_rank_buffer_bytes": sampler_stats["rank_buffer_bytes"],
            "sampling_noise_buffer_bytes": sampler_stats["noise_buffer_bytes"],
            "full_sampling_call_count": sampler_runtime_stats[
                "full_sampling_call_count"
            ],
            "full_sampling_row_count": sampler_runtime_stats[
                "full_sampling_row_count"
            ],
            "full_sampling_chunk_count": sampler_runtime_stats[
                "full_sampling_chunk_count"
            ],
            "max_full_sampling_chunk_rows": sampler_runtime_stats[
                "max_full_sampling_chunk_rows"
            ],
            "configured_sampling_chunk_rows": sampler_runtime_stats[
                "configured_sampling_chunk_rows"
            ],
            "tp_logits_local_buffer_bytes": sum(
                item["local_bytes"] for item in tp_logits_stats
            ),
            "tp_logits_gathered_buffer_bytes": sum(
                item["gathered_bytes"] for item in tp_logits_stats
            ),
            "tp_logits_buffer_allocation_count": sum(
                item["allocation_count"] for item in tp_logits_stats
            ),
            "tp_logits_buffer_reuse_count": sum(
                item["reuse_count"] for item in tp_logits_stats
            ),
            "tp_greedy_reduction_count": sum(
                item["greedy_reduction_count"] for item in tp_logits_stats
            ),
            "tp_greedy_candidate_bytes": sum(
                item["greedy_candidate_bytes"] for item in tp_logits_stats
            ),
            "tp_greedy_full_gather_avoided_bytes": sum(
                item["greedy_full_gather_avoided_bytes"]
                for item in tp_logits_stats
            ),
            "tp_top_k_reduction_count": sum(
                item["top_k_reduction_count"] for item in tp_logits_stats
            ),
            "tp_top_k_candidate_bytes": sum(
                item["top_k_candidate_bytes"] for item in tp_logits_stats
            ),
            "tp_top_k_full_gather_avoided_bytes": sum(
                item["top_k_full_gather_avoided_bytes"]
                for item in tp_logits_stats
            ),
            "pd_send_host_staging_bytes": host_staging_stats["storage_bytes"],
            "pd_send_host_staging_allocation_count": host_staging_stats[
                "allocation_count"
            ],
            "pd_send_host_staging_reuse_count": host_staging_stats[
                "reuse_count"
            ],
            "pd_send_host_staging_transient_allocation_count": (
                host_staging_stats["transient_allocation_count"]
            ),
            "pd_send_host_staging_leased": host_staging_stats["leased"],
            "pd_receive_host_staging_bytes": receive_staging_stats[
                "storage_bytes"
            ],
            "pd_receive_host_staging_allocation_count": receive_staging_stats[
                "allocation_count"
            ],
            "pd_receive_host_staging_reuse_count": receive_staging_stats[
                "reuse_count"
            ],
            "pd_receive_host_staging_transient_allocation_count": (
                receive_staging_stats["transient_allocation_count"]
            ),
            "pd_receive_host_staging_leased": receive_staging_stats["leased"],
            "total_bytes_local_rank": (
                partitioned_total
                + dequant_total
                + moe_weight_total
                + moe_workspace_total
                + resident_fp8_workspace_total
                + gptq_workspace_total
                + sum(item["scale_bytes"] for item in resident_fp8_storage_stats)
                + qwen35_key_total
                + sampler_total
                + tp_logits_total
            ),
        }

    def get_runtime_buffer_stats_by_rank(self):
        local = {"rank": self.rank, **self.get_runtime_buffer_stats()}
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
            groups = getattr(context, "state_prefill_groups", ())
            contiguous = [group[3] is not None for group in groups]
            if contiguous and all(contiguous):
                return "prefill_contiguous_view"
            if any(contiguous):
                return "prefill_mixed_state_access"
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
                return_result = method_name == CONTROL_RESULT_COMMAND
                if return_result:
                    if not args or not isinstance(args[0], str):
                        raise RuntimeError("invalid worker result command")
                    method_name, *args = args
                result = self.call(method_name, *args)
                self.write_worker_status(
                    None,
                    result=result if return_result else _NO_CONTROL_RESULT,
                )
            except BaseException as exc:
                error = exc
                self.write_worker_status(exc)
            finally:
                # The rank-0 process must not publish another command until
                # every worker has consumed and completed (or failed) this one.
                self.event[1].set()
            if error is not None:
                raise error
            if method_name == "exit":
                break

    def write_worker_status(
        self,
        error: BaseException | None,
        *,
        result=_NO_CONTROL_RESULT,
    ):
        if not (self.world_size > 1 and self.rank > 0):
            raise RuntimeError("only tensor-parallel worker ranks write status")
        status_buffer = self.event[2]
        if error is None and result is _NO_CONTROL_RESULT:
            status_buffer[0:4] = (0).to_bytes(4, "little")
            return
        if error is None:
            payload = pickle.dumps(
                {"kind": "result", "rank": self.rank, "value": result}
            )
            if len(payload) > CONTROL_STATUS_SIZE - 4:
                raise ValueError(
                    "worker result exceeds control status buffer: "
                    f"{len(payload)} bytes"
                )
        else:
            payload = pickle.dumps(
                {
                    "kind": "error",
                    "rank": self.rank,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        if len(payload) > CONTROL_STATUS_SIZE - 4:
            payload = pickle.dumps(
                {
                    "kind": "error",
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

    def wait_for_workers(self, *, expect_results: bool = False):
        if self.world_size <= 1 or self.rank != 0:
            return [] if expect_results else None
        worker_errors = []
        worker_results = []
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
                if worker_status.get("rank") != worker_rank:
                    self.worker_control_failed = True
                    raise RuntimeError(
                        f"tensor-parallel worker rank {worker_rank} returned "
                        "a mismatched status identity"
                    )
                kind = worker_status.get("kind", "error")
                if kind == "result" and expect_results:
                    if "value" not in worker_status:
                        self.worker_control_failed = True
                        raise RuntimeError(
                            f"tensor-parallel worker rank {worker_rank} "
                            "returned a result without a value"
                        )
                    worker_results.append(worker_status["value"])
                elif kind == "error":
                    worker_errors.append(worker_status)
                else:
                    self.worker_control_failed = True
                    raise RuntimeError(
                        f"tensor-parallel worker rank {worker_rank} returned "
                        "an unexpected control result"
                    )
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
        return worker_results if expect_results else None

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

    def call_rank_results(self, method_name, *args) -> list[object]:
        """Run a command on every rank and collect small control-plane results."""

        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"unknown ModelRunner method: {method_name}")
        if self.world_size <= 1:
            return [method(*args)]
        if self.rank != 0:
            raise RuntimeError("only tensor-parallel rank 0 collects rank results")
        if self.worker_control_failed:
            raise RuntimeError("tensor-parallel worker control channel has failed")
        self.write_shm(CONTROL_RESULT_COMMAND, method_name, *args)
        try:
            local_result = method(*args)
            local_payload = pickle.dumps(local_result)
            if len(local_payload) > CONTROL_STATUS_SIZE - 4:
                raise ValueError(
                    "rank-0 result exceeds control status buffer: "
                    f"{len(local_payload)} bytes"
                )
        finally:
            worker_results = self.wait_for_workers(expect_results=True)
        return [local_result, *worker_results]

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
        override = self.config.num_kvcache_blocks_override
        if override is not None:
            shared_num_blocks = min(shared_num_blocks, override)
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
            state_bytes_per_slot=(
                cache_plan.recurrent_bytes_per_sequence
                + cache_plan.convolution_bytes_per_sequence
            ),
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
        from nanovllm.layers.int8_fused_attention import (
            PartitionedDecodeBufferPool,
        )
        from nanovllm.layers.kv_cache_quant import Int8DequantBufferPool

        config = self.config
        hf_config = config.model_config
        model_spec = config.model_spec
        if hf_config is None or model_spec is None:
            raise RuntimeError("model configuration was not initialized")
        num_layers = model_spec.num_kv_cache_layers
        cache_plan = plan_cache_memory(
            model_spec,
            self.world_size,
            kv_dtype_bytes=dtype_nbytes(torch.int8),
        )
        num_kv_heads = cache_plan.local_kv_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        partitioned_decode_pool = PartitionedDecodeBufferPool()
        if (
            config.kv_dequant_backend == "fused"
            and config.sliding_window_size is None
            and config.max_model_len >= config.int8_partitioned_decode_threshold
        ):
            partitioned_decode_pool.reserve(
                num_seqs=min(config.max_num_seqs, config.max_num_batched_tokens),
                num_heads=hf_config.num_attention_heads // self.world_size,
                num_partitions=(
                    config.max_model_len
                    + config.int8_partitioned_decode_partition_size
                    - 1
                )
                // config.int8_partitioned_decode_partition_size,
                head_dim=head_dim,
                dtype=hf_config.dtype,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
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
        dequant_pool = Int8DequantBufferPool()
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
                module.int8_partitioned_decode_pool = partitioned_decode_pool
                module.int8_dequant_pool = dequant_pool
                layer_id += 1
        if layer_id != num_layers:
            raise RuntimeError(
                f"attached {layer_id} KV cache layers, expected {num_layers}"
            )

    def prepare_block_tables(
        self,
        seqs: list[Sequence],
        *,
        reuse_decode_buffer: bool = False,
    ):
        if reuse_decode_buffer:
            return self.decode_inputs.update_block_tables(
                [seq.block_table for seq in seqs]
            )
        return self.token_inputs.update_block_tables(
            [seq.block_table for seq in seqs]
        )

    def prepare_packed_block_metadata(
        self,
        metadata: PackedBlockMetadata,
        *,
        slot: int = 0,
    ):
        return self.token_inputs.update_packed_block_metadata(
            metadata.selected_block_ids,
            metadata.packed_block_tables,
            slot=slot,
        )

    def _state_slot_views(
        self,
        state_slot: int | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return (), ()
        if state_slot is None:
            raise RuntimeError(
                "hybrid cache transfer requires a recurrent state slot"
            )
        layers = [
            (
                module.layer_idx,
                module.state_pool,
            )
            for module in self.model.modules()
            if getattr(module, "state_pool", None) is not None
        ]
        layers.sort(key=lambda item: item[0])
        if len(layers) != len(model_spec.linear_attention_layers):
            raise RuntimeError(
                "hybrid cache transfer found an unexpected state layer count"
            )
        layer_ids = tuple(layer_idx for layer_idx, _ in layers)
        if layer_ids != tuple(sorted(model_spec.linear_attention_layers)):
            raise RuntimeError(
                "hybrid cache transfer layer ids do not match model spec"
            )
        recurrent = tuple(
            pool.recurrent[0, state_slot] for _, pool in layers
        )
        convolution = tuple(
            pool.convolution[0, state_slot] for _, pool in layers
        )
        return recurrent, convolution

    def _sequence_state_views(
        self,
        seq: Sequence,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return self._state_slot_views(seq.state_slot)

    def export_sequence_cache(
        self,
        seq: Sequence,
        *,
        transfer_id: str,
        to_host: bool = False,
        host_staging_pool=None,
    ):
        """Create the rank-local tensor payload needed by a decode worker."""

        from nanovllm.engine.cache_transfer import export_rank_cache

        recurrent, convolution = self._sequence_state_views(seq)
        return export_rank_cache(
            self.kv_cache,
            self.kv_scale,
            seq.block_table,
            transfer_id=transfer_id,
            tensor_parallel_rank=self.rank,
            tensor_parallel_size=self.world_size,
            block_size=self.block_size,
            cached_tokens=seq.num_cached_tokens,
            recurrent_states=recurrent,
            convolution_states=convolution,
            to_host=to_host,
            host_staging_pool=host_staging_pool,
        )

    def _host_staging_pool(self):
        from nanovllm.engine.cache_transfer import HostStagingBufferPool

        if getattr(self, "_cache_send_staging_pool", None) is None:
            self._cache_send_staging_pool = HostStagingBufferPool()
        return self._cache_send_staging_pool

    def _host_receive_staging_pool(self):
        from nanovllm.engine.cache_transfer import HostStagingBufferPool

        if getattr(self, "_cache_receive_staging_pool", None) is None:
            self._cache_receive_staging_pool = HostStagingBufferPool()
        return self._cache_receive_staging_pool

    def estimate_sequence_cache_bytes(self, seq: Sequence) -> dict:
        """Estimate rank-local host staging bytes without copying tensors."""

        from nanovllm.engine.cache_transfer import (
            estimate_rank_cache_transfer_bytes,
        )

        recurrent, convolution = self._sequence_state_views(seq)
        staged_bytes = estimate_rank_cache_transfer_bytes(
            self.kv_cache,
            self.kv_scale,
            seq.block_table,
            recurrent_states=recurrent,
            convolution_states=convolution,
        )
        return {"rank": self.rank, "staged_bytes": staged_bytes}

    def estimate_cache_transfer_bytes_for_blocks(self, num_blocks: int) -> dict:
        """Estimate rank-local staging before scheduler state is reserved."""

        from nanovllm.engine.cache_transfer import (
            estimate_rank_cache_transfer_bytes_for_blocks,
        )

        model_spec = self.config.model_spec
        state_slot = 0 if model_spec is not None and model_spec.is_hybrid else None
        recurrent, convolution = self._state_slot_views(state_slot)
        staged_bytes = estimate_rank_cache_transfer_bytes_for_blocks(
            self.kv_cache,
            self.kv_scale,
            num_blocks,
            recurrent_states=recurrent,
            convolution_states=convolution,
        )
        return {"rank": self.rank, "staged_bytes": staged_bytes}

    def import_sequence_cache(
        self,
        seq: Sequence,
        payload,
        *,
        transfer_id: str,
    ) -> None:
        """Install a validated rank-local payload into preallocated slots."""

        from nanovllm.engine.cache_transfer import import_rank_cache

        expected_cached_tokens = (
            seq.num_prompt_tokens
            if getattr(seq, "status", None) is SequenceStatus.TRANSFERRING
            else seq.num_cached_tokens
        )
        if payload.cached_tokens != expected_cached_tokens:
            raise ValueError(
                "cache transfer token count does not match destination sequence"
            )
        recurrent, convolution = self._sequence_state_views(seq)
        import_rank_cache(
            payload,
            self.kv_cache,
            self.kv_scale,
            seq.block_table,
            transfer_id=transfer_id,
            tensor_parallel_rank=self.rank,
            tensor_parallel_size=self.world_size,
            block_size=self.block_size,
            recurrent_states=recurrent,
            convolution_states=convolution,
        )

    def _rank_cache_endpoint(
        self,
        endpoints: list[tuple[str, int]],
    ) -> tuple[str, int]:
        if len(endpoints) != self.world_size:
            raise ValueError(
                "cache transfer requires one endpoint per tensor-parallel rank"
            )
        endpoint = endpoints[self.rank]
        if (
            not isinstance(endpoint, (list, tuple))
            or len(endpoint) != 2
            or not isinstance(endpoint[0], str)
            or not endpoint[0]
            or not isinstance(endpoint[1], int)
            or isinstance(endpoint[1], bool)
            or not 1 <= endpoint[1] <= 65535
        ):
            raise ValueError("cache transfer endpoint is invalid")
        return endpoint[0], endpoint[1]

    def send_sequence_cache_to_endpoint(
        self,
        seq: Sequence,
        transfer_id: str,
        endpoints: list[tuple[str, int]],
        timeout_s: float = 30.0,
    ) -> dict:
        """Send this TP rank directly to its matching decode rank."""

        from nanovllm.engine.cache_transfer_wire import (
            send_rank_cache_to_endpoint,
        )

        if timeout_s <= 0:
            raise ValueError("cache transfer endpoint timeout must be positive")
        host, port = self._rank_cache_endpoint(endpoints)
        payload = self.export_sequence_cache(
            seq,
            transfer_id=transfer_id,
            to_host=True,
            host_staging_pool=self._host_staging_pool(),
        )
        sent_bytes = send_rank_cache_to_endpoint(
            host,
            port,
            payload,
            timeout_s=timeout_s,
        )
        return {"rank": self.rank, "sent_bytes": sent_bytes}

    def start_sequence_cache_send(
        self,
        seq: Sequence,
        transfer_id: str,
        endpoints: list[tuple[str, int]],
        timeout_s: float = 30.0,
    ) -> dict:
        """Stage rank-local cache on host, then send it in the background."""

        from nanovllm.engine.cache_transfer_wire import PendingRankCacheSend

        if transfer_id in self._pending_cache_sends:
            raise ValueError("cache send id is already active")
        if timeout_s <= 0:
            raise ValueError("cache transfer endpoint timeout must be positive")
        host, port = self._rank_cache_endpoint(endpoints)
        payload = self.export_sequence_cache(
            seq,
            transfer_id=transfer_id,
            to_host=True,
            host_staging_pool=self._host_staging_pool(),
        )
        try:
            send = PendingRankCacheSend(
                host,
                port,
                payload,
                timeout_s=timeout_s,
            )
        except BaseException:
            payload.release_host_staging()
            raise
        self._pending_cache_sends[transfer_id] = send
        try:
            send.start()
        except BaseException:
            self._pending_cache_sends.pop(transfer_id, None)
            send.finish()
            raise
        return {
            "rank": self.rank,
            "started": 1,
            "staged_bytes": payload.nbytes,
        }

    def poll_sequence_cache_send(self, transfer_id: str) -> dict:
        send = self._pending_cache_sends.get(transfer_id)
        if send is None:
            raise ValueError("cache send id is not active")
        state, error = send.poll()
        result = {
            "rank": self.rank,
            "state": state,
            "staged_bytes": send.staged_bytes,
        }
        if error is not None:
            result["error"] = error
        return result

    def poll_sequence_cache_sends(self, transfer_ids: list[str]) -> dict:
        """Poll multiple host-side sends in one TP control command."""

        if (
            not isinstance(transfer_ids, list)
            or not transfer_ids
            or any(not isinstance(item, str) or not item for item in transfer_ids)
            or len(set(transfer_ids)) != len(transfer_ids)
        ):
            raise ValueError("cache send ids must be unique non-empty strings")
        sends = {}
        for transfer_id in transfer_ids:
            result = self.poll_sequence_cache_send(transfer_id)
            sends[transfer_id] = {
                key: value for key, value in result.items() if key != "rank"
            }
        return {"rank": self.rank, "sends": sends}

    def finish_sequence_cache_send(self, transfer_id: str) -> dict:
        send = self._pending_cache_sends.get(transfer_id)
        if send is None:
            raise ValueError("cache send id is not active")
        try:
            sent_bytes = send.result()
        finally:
            send.finish()
            self._pending_cache_sends.pop(transfer_id, None)
        return {"rank": self.rank, "sent_bytes": sent_bytes}

    def abort_sequence_cache_send(self, transfer_id: str) -> dict:
        send = self._pending_cache_sends.pop(transfer_id, None)
        if send is not None:
            send.finish()
        return {"rank": self.rank, "aborted": 1}

    def receive_sequence_cache_from_endpoint(
        self,
        seq: Sequence,
        transfer_id: str,
        bind_endpoints: list[tuple[str, int]],
        timeout_s: float = 30.0,
        max_payload_bytes: int = 16 * 1024**3,
        expected_payload_bytes: list[int] | None = None,
    ) -> dict:
        """Receive, verify, and install this TP rank's remote prefill state."""

        from nanovllm.engine.cache_transfer_wire import RankCacheReceiver

        expected_bytes = None
        if expected_payload_bytes is not None:
            if (
                len(expected_payload_bytes) != self.world_size
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in expected_payload_bytes
                )
            ):
                raise ValueError("expected payload bytes must cover every TP rank")
            expected_bytes = expected_payload_bytes[self.rank]
            max_payload_bytes = min(max_payload_bytes, expected_bytes)

        def install_verified(payload) -> None:
            if expected_bytes is not None and payload.nbytes != expected_bytes:
                raise ValueError(
                    "cache receive payload bytes differ from the preflight estimate"
                )
            self.import_sequence_cache(seq, payload, transfer_id=transfer_id)

        host, port = self._rank_cache_endpoint(bind_endpoints)
        with RankCacheReceiver(
            host,
            port,
            timeout_s=timeout_s,
            max_payload_bytes=max_payload_bytes,
            host_staging_pool=self._host_receive_staging_pool(),
        ) as receiver:
            payload = receiver.receive(
                timeout_s=timeout_s,
                on_verified=install_verified,
            )
        cached_tokens = payload.cached_tokens
        received_bytes = payload.nbytes
        try:
            return {
                "rank": self.rank,
                "cached_tokens": cached_tokens,
                "received_bytes": received_bytes,
            }
        finally:
            payload.release_host_staging()

    def start_sequence_cache_receive(
        self,
        transfer_id: str,
        bind_endpoints: list[tuple[str, int]],
        timeout_s: float = 30.0,
        max_payload_bytes: int = 16 * 1024**3,
        expected_payload_bytes: list[int] | None = None,
    ) -> dict:
        """Start rank-local TCP receive without touching CUDA state."""

        from nanovllm.engine.cache_transfer_wire import PendingRankCacheReceive

        if transfer_id in self._pending_cache_receives:
            raise ValueError("cache receive id is already active")
        if expected_payload_bytes is not None:
            if (
                len(expected_payload_bytes) != self.world_size
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in expected_payload_bytes
                )
            ):
                raise ValueError("expected payload bytes must cover every TP rank")
            max_payload_bytes = min(
                max_payload_bytes,
                expected_payload_bytes[self.rank],
            )
        host, port = self._rank_cache_endpoint(bind_endpoints)
        receive = PendingRankCacheReceive(
            host,
            port,
            timeout_s=timeout_s,
            max_payload_bytes=max_payload_bytes,
            host_staging_pool=self._host_receive_staging_pool(),
        )
        self._pending_cache_receives[transfer_id] = receive
        try:
            receive.start()
        except BaseException:
            self._pending_cache_receives.pop(transfer_id, None)
            receive.finish(accepted=False)
            raise
        return {"rank": self.rank, "started": 1}

    def poll_sequence_cache_receive(self, transfer_id: str) -> dict:
        receive = self._pending_cache_receives.get(transfer_id)
        if receive is None:
            raise ValueError("cache receive id is not active")
        state, error = receive.poll()
        result = {"rank": self.rank, "state": state}
        if error is not None:
            result["error"] = error
        return result

    def poll_sequence_cache_receives(self, transfer_ids: list[str]) -> dict:
        """Poll multiple CPU receive tasks in one TP control command."""

        if (
            not isinstance(transfer_ids, list)
            or not transfer_ids
            or any(not isinstance(item, str) or not item for item in transfer_ids)
            or len(set(transfer_ids)) != len(transfer_ids)
        ):
            raise ValueError("cache receive ids must be unique non-empty strings")
        receives = {}
        for transfer_id in transfer_ids:
            result = self.poll_sequence_cache_receive(transfer_id)
            receives[transfer_id] = {
                key: value for key, value in result.items() if key != "rank"
            }
        return {"rank": self.rank, "receives": receives}

    def install_sequence_cache_receive(
        self,
        seq: Sequence,
        transfer_id: str,
        expected_payload_bytes: list[int] | None = None,
    ) -> dict:
        """Install a ready CPU payload, then ACK the producer."""

        receive = self._pending_cache_receives.get(transfer_id)
        if receive is None:
            raise ValueError("cache receive id is not active")
        try:
            payload = receive.payload()
            if expected_payload_bytes is not None:
                if (
                    len(expected_payload_bytes) != self.world_size
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value <= 0
                        for value in expected_payload_bytes
                    )
                ):
                    raise ValueError(
                        "expected payload bytes must cover every TP rank"
                    )
                if payload.nbytes != expected_payload_bytes[self.rank]:
                    raise ValueError(
                        "cache receive payload bytes differ from the preflight estimate"
                    )
            self.import_sequence_cache(seq, payload, transfer_id=transfer_id)
            cached_tokens = payload.cached_tokens
            received_bytes = payload.nbytes
        except BaseException:
            receive.finish(accepted=False)
            self._pending_cache_receives.pop(transfer_id, None)
            raise
        receive.finish(accepted=True)
        self._pending_cache_receives.pop(transfer_id, None)
        return {
            "rank": self.rank,
            "cached_tokens": cached_tokens,
            "received_bytes": received_bytes,
        }

    def abort_sequence_cache_receive(self, transfer_id: str) -> dict:
        receive = self._pending_cache_receives.pop(transfer_id, None)
        if receive is not None:
            receive.finish(accepted=False)
        return {"rank": self.rank, "aborted": 1}

    def prepare_state_slots(
        self,
        seqs: list[Sequence],
        *,
        reuse_decode_buffer: bool = False,
    ):
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
        if reuse_decode_buffer:
            return self.decode_inputs.update_state_slots(state_slots)
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

    def contiguous_prefill_state_spans(
        self,
        seqs: list[Sequence],
        *,
        slot_offset: int = 0,
    ) -> tuple[tuple[int, int] | None, ...]:
        """Describe equal-length prefill slot groups without device reads."""

        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid or not seqs:
            return ()
        grouped_slots: dict[int, list[int]] = {}
        for sequence_index, seq in enumerate(seqs):
            slot = (
                slot_offset + sequence_index
                if seq.state_slot is None
                else seq.state_slot
            )
            grouped_slots.setdefault(seq.num_scheduled_tokens, []).append(slot)
        spans = []
        for slots in grouped_slots.values():
            start = slots[0]
            spans.append(
                (start, len(slots))
                if slots == list(range(start, start + len(slots)))
                else None
            )
        return tuple(spans)

    def prepare_state_reset_slots(
        self,
        seqs: list[Sequence],
        *,
        reuse_decode_buffer: bool = False,
    ):
        model_spec = self.config.model_spec
        if model_spec is None or not model_spec.is_hybrid:
            return None
        reset_slots = []
        for warmup_slot, seq in enumerate(seqs):
            if seq.num_cached_tokens != 0:
                continue
            if seq.state_slot is None:
                if seq.block_table:
                    raise RuntimeError(
                        "scheduled hybrid sequence has no recurrent state slot"
                    )
                reset_slots.append(warmup_slot)
            else:
                reset_slots.append(seq.state_slot)
        if not reset_slots:
            return None
        if reuse_decode_buffer:
            return self.decode_inputs.update_reset_slots(reset_slots)
        return torch.tensor(
            reset_slots,
            dtype=torch.int64,
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
                dequant_block_ids, dequant_block_tables = (
                    self.prepare_packed_block_metadata(metadata)
                )
        state_token_ranges = tuple(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:]))
        logits_indices = self.token_inputs.update_logits_indices(
            [end - 1 for end in cu_seqlens_q[1:]]
        )
        input_ids, positions, slot_mapping = self.token_inputs.update_tokens(
            input_ids,
            positions,
            slot_mapping,
        )
        cu_seqlens_q, cu_seqlens_k = self.token_inputs.update_cu_seqlens(
            cu_seqlens_q,
            cu_seqlens_k,
        )
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
            state_slots=self.prepare_state_slots(
                seqs,
                reuse_decode_buffer=True,
            ),
            state_reset_slots=self.prepare_state_reset_slots(
                seqs,
                reuse_decode_buffer=True,
            ),
            state_token_ranges=state_token_ranges,
            state_prefill_spans=self.contiguous_prefill_state_spans(seqs),
            logits_indices=logits_indices,
        )
        return input_ids, positions

    def build_prefill_inputs(
        self,
        seqs: list[Sequence],
        *,
        prepare_state_metadata: bool = True,
    ):
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
                dequant_block_ids, dequant_block_tables = (
                    self.prepare_packed_block_metadata(metadata, slot=1)
                )

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
            state_slots=(
                self.prepare_state_slots(seqs)
                if prepare_state_metadata
                else None
            ),
            state_reset_slots=(
                self.prepare_state_reset_slots(seqs)
                if prepare_state_metadata
                else None
            ),
            state_token_ranges=tuple(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])),
        )

    def build_decode_inputs(
        self,
        seqs: list[Sequence],
        *,
        prepare_state_metadata: bool = True,
    ):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        block_tables = self.prepare_block_tables(
            seqs,
            reuse_decode_buffer=True,
        )
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
            state_slots=(
                self.prepare_state_slots(seqs)
                if prepare_state_metadata
                else None
            ),
            state_reset_slots=(
                self.prepare_state_reset_slots(seqs)
                if prepare_state_metadata
                else None
            ),
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
        input_ids, positions, slot_mapping, context_lens = self.decode_inputs.update(
            input_ids,
            positions,
            slot_mapping,
            context_lens,
        )
        block_tables = self.prepare_block_tables(
            seqs,
            reuse_decode_buffer=True,
        )
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
            state_slots=self.prepare_state_slots(
                seqs,
                reuse_decode_buffer=True,
            ),
            state_reset_slots=self.prepare_state_reset_slots(
                seqs,
                reuse_decode_buffer=True,
            ),
            state_token_ranges=(),
            decode_state_span=self.contiguous_state_span(seqs),
        )
        return input_ids, positions

    def prepare_mixed(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        decode = self.build_decode_inputs(
            decode_seqs,
            prepare_state_metadata=False,
        )
        prefill = self.build_prefill_inputs(
            prefill_seqs,
            prepare_state_metadata=False,
        )
        mixed_seqs = [*decode_seqs, *prefill_seqs]
        input_ids = decode["input_ids"] + prefill["input_ids"]
        positions = decode["positions"] + prefill["positions"]
        slot_mapping = decode["slot_mapping"] + prefill["slot_mapping"]

        input_ids, positions, slot_mapping = self.token_inputs.update_tokens(
            input_ids,
            positions,
            slot_mapping,
        )
        decode_context_lens = self.token_inputs.update_decode_context_lens(
            decode["context_lens"]
        )
        prefill_cu_seqlens_q, prefill_cu_seqlens_k = (
            self.token_inputs.update_cu_seqlens(
                prefill["cu_seqlens_q"],
                prefill["cu_seqlens_k"],
            )
        )
        logits_indices = self.token_inputs.update_logits_indices(
            [*range(len(decode_seqs))]
            + [
                len(decode_seqs) + end - 1
                for end in prefill["cu_seqlens_q"][1:]
            ]
        )

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
            state_slots=self.prepare_state_slots(
                mixed_seqs,
                reuse_decode_buffer=True,
            ),
            state_reset_slots=self.prepare_state_reset_slots(
                mixed_seqs,
                reuse_decode_buffer=True,
            ),
            state_token_ranges=tuple(
                (
                    len(decode_seqs) + start,
                    len(decode_seqs) + end,
                )
                for start, end in prefill["state_token_ranges"]
            ),
            state_prefill_spans=self.contiguous_prefill_state_spans(
                prefill_seqs,
                slot_offset=len(decode_seqs),
            ),
            decode_state_span=decode.get("decode_state_span"),
            logits_indices=logits_indices,
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
        if self.sampling_inputs is None:
            raise RuntimeError("sampling inputs are only available on rank 0")
        temperatures, top_ks, top_ps = self.sampling_inputs.update(
            temperatures,
            top_ks,
            top_ps,
        )
        return temperatures, top_ks, top_ps, metadata

    def _prepare_sampling_path(self, seqs: list[Sequence]):
        if all(seq.temperature <= 1e-10 for seq in seqs):
            path = "greedy"
            top_k = None
        else:
            vocab_size = int(self.config.model_config.vocab_size)
            top_k_reduction_limit = int(
                getattr(self.config, "tp_top_k_reduction_max_width", 256)
            )
            top_k_enabled = all(
                seq.temperature > 1e-10
                and 0 < seq.top_k < vocab_size
                and seq.top_k <= top_k_reduction_limit
                for seq in seqs
            )
            path = "top_k" if top_k_enabled else "full"
            top_k = max(seq.top_k for seq in seqs) if top_k_enabled else None
        sample_args = (
            self.prepare_sample(seqs)
            if self.rank == 0 and path != "greedy"
            else None
        )
        return path, top_k, sample_args

    def _finish_sampling(self, model_output, *, path: str, sample_args):
        if self.rank != 0:
            return None
        if path == "greedy":
            return model_output.tolist()
        if path == "top_k":
            selected_logits, selected_indices = model_output
            return self.sampler.sample_top_k_candidates(
                selected_logits,
                selected_indices,
                *sample_args,
            ).tolist()
        if path != "full":
            raise ValueError("sampling path is invalid")
        return self.sampler(model_output, *sample_args).tolist()

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        *,
        greedy: bool = False,
        top_k: int | None = None,
    ):
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
                return self.model.compute_logits(
                    self.model(input_ids, positions),
                    greedy=greedy,
                    top_k=top_k,
                )
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
        return self.model.compute_logits(
            graph_vars["outputs"][:bs],
            greedy=greedy,
            top_k=top_k,
        )

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sampling_path, top_k, sample_args = self._prepare_sampling_path(seqs)
        model_output = self.run_model(
            input_ids,
            positions,
            is_prefill,
            greedy=sampling_path == "greedy",
            top_k=top_k,
        )
        token_ids = self._finish_sampling(
            model_output,
            path=sampling_path,
            sample_args=sample_args,
        )
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
        sampling_path, top_k, sample_args = self._prepare_sampling_path(seqs)
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
            model_output = self.model.compute_logits(
                self.model(input_ids, positions),
                greedy=sampling_path == "greedy",
                top_k=top_k,
            )
        finally:
            restore(previous_trace)
        token_ids = self._finish_sampling(
            model_output,
            path=sampling_path,
            sample_args=sample_args,
        )
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
