import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig

from nanovllm.models.model_spec import ModelSpec, resolve_model_spec


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
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "auto"
    kv_dequant_backend: str = "fused"
    sliding_window_size: int | None = None
    enable_dynamic_chunked_prefill: bool = False
    int8_partitioned_decode_threshold: int = 8192
    int8_partitioned_decode_partition_size: int = 512
    distributed_port: int | None = None
    shared_memory_name: str | None = None

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"model directory does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be in [1, 8]")
        if self.kv_cache_dtype not in ("auto", "int8"):
            raise ValueError("kv_cache_dtype must be 'auto' or 'int8'")
        if self.kv_dequant_backend not in ("triton", "torch", "fused"):
            raise ValueError(
                "kv_dequant_backend must be one of: 'triton', 'torch', 'fused'"
            )
        if self.sliding_window_size is not None and self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive when provided")
        if self.int8_partitioned_decode_threshold <= 0:
            raise ValueError("int8_partitioned_decode_threshold must be positive")
        if self.int8_partitioned_decode_partition_size <= 0:
            raise ValueError("int8_partitioned_decode_partition_size must be positive")
        if self.distributed_port is not None and not 1 <= self.distributed_port <= 65535:
            raise ValueError("distributed_port must be in [1, 65535]")
        if self.shared_memory_name is not None and not self.shared_memory_name:
            raise ValueError("shared_memory_name must not be empty")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.model_spec = resolve_model_spec(self.hf_config)
        self.model_config = self.model_spec.text_config
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
