import json
import os
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import AutoConfig

from nanovllm.models.cache_plan import validate_cache_parallelism
from nanovllm.models.model_spec import (
    ModelSpec,
    QWEN35_MOE_ARCHITECTURES,
    resolve_model_spec,
    validate_weight_parallelism,
)


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def resolve_eos_token_ids(
    model: str,
    fallback: int | list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve every generation EOS token from a local model directory."""

    path = Path(model) / "generation_config.json"
    value = fallback
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            configured = json.load(handle).get("eos_token_id")
        if configured is not None:
            value = configured
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ValueError("eos_token_id must contain integer token ids")
    if isinstance(value, int):
        values = (value,)
    elif isinstance(value, (list, tuple)) and value:
        values = tuple(value)
    else:
        raise ValueError("eos_token_id must be an integer or non-empty list")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in values
    ):
        raise ValueError("eos_token_id must contain integer token ids")
    if any(token_id < 0 for token_id in values):
        raise ValueError("eos_token_id must contain non-negative token ids")
    return tuple(dict.fromkeys(values))


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    model_config: object | None = None
    model_spec: ModelSpec | None = None
    eos: int | tuple[int, ...] = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_kvcache_blocks_override: int | None = None
    kv_cache_dtype: str = "auto"
    kv_dequant_backend: str = "fused"
    sliding_window_size: int | None = None
    enable_dynamic_chunked_prefill: bool = False
    prefill_starvation_threshold: int = 0
    prefill_starvation_token_budget: int = 256
    preemption_policy: str = "fcfs"
    enable_decode_kv_reservation: bool = False
    int8_partitioned_decode_threshold: int = 8192
    int8_partitioned_decode_partition_size: int = 512
    recurrent_state_dtype: str = "float32"
    qwen35_decode_conv_backend: str = "weighted"
    qwen35_moe_decode_backend: str = "sorted"
    qwen35_moe_decode_chunk_size: int = 8
    tp_top_k_reduction_max_width: int = 256
    sampling_chunk_size: int = 32
    weight_quant_backend: str = "auto"
    max_remote_prefill_transfers: int = 2
    max_remote_prefill_staging_bytes: int | None = None
    distributed_port: int | None = None
    shared_memory_name: str | None = None

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model directory does not exist: {self.model}")
        _require_positive_int(
            self.max_num_batched_tokens,
            "max_num_batched_tokens",
        )
        _require_positive_int(self.max_num_seqs, "max_num_seqs")
        _require_positive_int(self.max_model_len, "max_model_len")
        if (
            isinstance(self.gpu_memory_utilization, bool)
            or not isinstance(self.gpu_memory_utilization, (int, float))
            or not 0.0 < self.gpu_memory_utilization <= 1.0
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        _require_bool(self.enforce_eager, "enforce_eager")
        _require_positive_int(self.kvcache_block_size, "kvcache_block_size")
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if self.num_kvcache_blocks_override is not None:
            _require_positive_int(
                self.num_kvcache_blocks_override,
                "num_kvcache_blocks_override",
            )
        if (
            not isinstance(self.tensor_parallel_size, int)
            or isinstance(self.tensor_parallel_size, bool)
            or not 1 <= self.tensor_parallel_size <= 8
        ):
            raise ValueError("tensor_parallel_size must be an integer in [1, 8]")
        if self.kv_cache_dtype not in ("auto", "int8"):
            raise ValueError("kv_cache_dtype must be 'auto' or 'int8'")
        if self.kv_dequant_backend not in ("triton", "torch", "fused"):
            raise ValueError(
                "kv_dequant_backend must be one of: 'triton', 'torch', 'fused'"
            )
        if self.sliding_window_size is not None:
            _require_positive_int(self.sliding_window_size, "sliding_window_size")
        _require_bool(
            self.enable_dynamic_chunked_prefill,
            "enable_dynamic_chunked_prefill",
        )
        _require_non_negative_int(
            self.prefill_starvation_threshold,
            "prefill_starvation_threshold",
        )
        _require_positive_int(
            self.prefill_starvation_token_budget,
            "prefill_starvation_token_budget",
        )
        if self.preemption_policy not in ("fcfs", "min_recompute"):
            raise ValueError("preemption_policy must be 'fcfs' or 'min_recompute'")
        if not isinstance(self.enable_decode_kv_reservation, bool):
            raise ValueError("enable_decode_kv_reservation must be a boolean")
        _require_positive_int(
            self.int8_partitioned_decode_threshold,
            "int8_partitioned_decode_threshold",
        )
        _require_positive_int(
            self.int8_partitioned_decode_partition_size,
            "int8_partitioned_decode_partition_size",
        )
        if self.recurrent_state_dtype not in ("float32", "model"):
            raise ValueError("recurrent_state_dtype must be 'float32' or 'model'")
        if self.qwen35_decode_conv_backend not in ("weighted", "channel_accumulate"):
            raise ValueError(
                "qwen35_decode_conv_backend must be 'weighted' or "
                "'channel_accumulate'"
            )
        if self.qwen35_moe_decode_backend not in ("sorted", "batched"):
            raise ValueError(
                "qwen35_moe_decode_backend must be 'sorted' or 'batched'"
            )
        _require_positive_int(
            self.qwen35_moe_decode_chunk_size,
            "qwen35_moe_decode_chunk_size",
        )
        if (
            not isinstance(self.tp_top_k_reduction_max_width, int)
            or isinstance(self.tp_top_k_reduction_max_width, bool)
            or self.tp_top_k_reduction_max_width <= 0
        ):
            raise ValueError("tp_top_k_reduction_max_width must be positive")
        if (
            not isinstance(self.sampling_chunk_size, int)
            or isinstance(self.sampling_chunk_size, bool)
            or self.sampling_chunk_size <= 0
        ):
            raise ValueError("sampling_chunk_size must be positive")
        if self.weight_quant_backend not in ("auto", "reference", "resident", "triton"):
            raise ValueError(
                "weight_quant_backend must be 'auto', 'reference', 'resident', "
                "or 'triton'"
            )
        if (
            not isinstance(self.max_remote_prefill_transfers, int)
            or isinstance(self.max_remote_prefill_transfers, bool)
            or self.max_remote_prefill_transfers <= 0
        ):
            raise ValueError("max_remote_prefill_transfers must be positive")
        if (
            self.max_remote_prefill_staging_bytes is not None
            and (
                not isinstance(self.max_remote_prefill_staging_bytes, int)
                or isinstance(self.max_remote_prefill_staging_bytes, bool)
                or self.max_remote_prefill_staging_bytes <= 0
            )
        ):
            raise ValueError("max_remote_prefill_staging_bytes must be positive")
        if self.distributed_port is not None and (
            not isinstance(self.distributed_port, int)
            or isinstance(self.distributed_port, bool)
            or not 1 <= self.distributed_port <= 65535
        ):
            raise ValueError("distributed_port must be an integer in [1, 65535]")
        if self.shared_memory_name is not None and (
            not isinstance(self.shared_memory_name, str)
            or not self.shared_memory_name
        ):
            raise ValueError("shared_memory_name must be a non-empty string")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.model_spec = resolve_model_spec(self.hf_config)
        self.model_config = self.model_spec.text_config
        validate_cache_parallelism(self.model_spec, self.tensor_parallel_size)
        validate_weight_parallelism(self.model_spec, self.tensor_parallel_size)
        quantization = self.model_spec.quantization
        if quantization.format == "gptq_int4":
            if self.weight_quant_backend == "auto":
                # ModelRunner is CUDA-only. Prefer the fused executor by
                # default while retaining the explicit reference backend for
                # correctness checks and diagnostics.
                self.weight_quant_backend = "triton"
            if self.qwen35_moe_decode_backend != "sorted":
                raise ValueError(
                    "the current GPTQ expert backend requires "
                    "qwen35_moe_decode_backend='sorted'"
                )
            self.model_config.nanovllm_quantization_spec = quantization
            self.model_config.nanovllm_weight_quant_backend = self.weight_quant_backend
        elif quantization.format == "fp8_block":
            if self.model_spec.architecture not in QWEN35_MOE_ARCHITECTURES:
                raise NotImplementedError(
                    "block-FP8 loading is implemented only for Qwen3.6-compatible MoE"
                )
            if self.weight_quant_backend == "auto":
                self.weight_quant_backend = "reference"
            if self.weight_quant_backend not in ("reference", "resident"):
                raise ValueError(
                    "block-FP8 checkpoints require weight_quant_backend="
                    "'reference' or 'resident'"
                )
            if (
                self.weight_quant_backend == "resident"
                and self.qwen35_moe_decode_backend != "sorted"
            ):
                raise ValueError(
                    "resident FP8 experts currently require "
                    "qwen35_moe_decode_backend='sorted'"
                )
            self.model_config.nanovllm_quantization_spec = quantization
            self.model_config.nanovllm_weight_quant_backend = self.weight_quant_backend
        elif not quantization.is_quantized and self.weight_quant_backend != "auto":
            raise ValueError(
                f"weight_quant_backend={self.weight_quant_backend!r} requires "
                "a GPTQ-Int4 checkpoint"
            )
        else:
            quantization.require_runtime_support()
        self.model_config.qwen35_moe_decode_backend = (
            self.qwen35_moe_decode_backend
        )
        self.model_config.qwen35_decode_conv_backend = (
            self.qwen35_decode_conv_backend
        )
        self.model_config.qwen35_moe_decode_chunk_size = (
            self.qwen35_moe_decode_chunk_size
        )
        self.model_config.nanovllm_tp_top_k_reduction_max_width = (
            self.tp_top_k_reduction_max_width
        )
        if not hasattr(self.model_config, "dtype"):
            torch_dtype = getattr(self.model_config, "torch_dtype", None)
            self.model_config.dtype = (
                torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16
            )
        max_position_embeddings = getattr(
            self.model_config,
            "max_position_embeddings",
            self.max_model_len,
        )
        self.max_model_len = min(self.max_model_len, max_position_embeddings)
        # Model modules only need positional state for lengths the engine can
        # actually schedule. Keep the checkpoint's architectural limit intact
        # while exposing the smaller runtime allocation bound.
        self.model_config.nanovllm_max_model_len = self.max_model_len
        self.model_config.nanovllm_max_num_batched_tokens = (
            self.max_num_batched_tokens
        )
