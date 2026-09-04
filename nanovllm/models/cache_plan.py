"""Per-rank cache sizing for dense and hybrid attention models."""

from __future__ import annotations

from dataclasses import dataclass

from nanovllm.models.model_spec import ModelSpec


@dataclass(frozen=True, slots=True)
class CacheMemoryPlan:
    tensor_parallel_size: int
    local_kv_heads: int
    kv_head_replication: int
    kv_bytes_per_token: int
    int8_scale_bytes_per_token: int
    recurrent_bytes_per_sequence: int
    convolution_bytes_per_sequence: int

    def bytes_per_sequence(self, context_length: int) -> int:
        if context_length < 0:
            raise ValueError("context_length must be non-negative")
        return (
            self.kv_bytes_per_token * context_length
            + self.recurrent_bytes_per_sequence
            + self.convolution_bytes_per_sequence
        )


def _local_heads(total_heads: int, tensor_parallel_size: int) -> tuple[int, int]:
    if total_heads <= 0:
        raise ValueError("head count must be positive")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if total_heads >= tensor_parallel_size:
        if total_heads % tensor_parallel_size:
            raise ValueError(
                f"{total_heads} heads cannot be sharded across "
                f"TP={tensor_parallel_size}"
            )
        return total_heads // tensor_parallel_size, 1
    if tensor_parallel_size % total_heads:
        raise ValueError(
            f"{total_heads} heads cannot be replicated across "
            f"TP={tensor_parallel_size}"
        )
    return 1, tensor_parallel_size // total_heads


def validate_cache_parallelism(
    model_spec: ModelSpec,
    tensor_parallel_size: int,
) -> None:
    """Validate cache-bearing head layouts before distributed workers start."""

    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size <= 0
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    config = model_spec.text_config
    num_attention_heads = int(config.num_attention_heads)
    if num_attention_heads % tensor_parallel_size:
        raise ValueError(
            f"{num_attention_heads} attention heads cannot be sharded across "
            f"TP={tensor_parallel_size}"
        )
    _local_heads(int(config.num_key_value_heads), tensor_parallel_size)
    if getattr(model_spec, "linear_attention_layers", ()):
        _local_heads(int(config.linear_num_key_heads), tensor_parallel_size)
        _local_heads(int(config.linear_num_value_heads), tensor_parallel_size)


def plan_cache_memory(
    model_spec: ModelSpec,
    tensor_parallel_size: int,
    *,
    kv_dtype_bytes: int = 2,
    recurrent_dtype_bytes: int = 4,
    convolution_dtype_bytes: int = 2,
) -> CacheMemoryPlan:
    """Calculate persistent inference-state bytes allocated on each rank."""

    for name, value in (
        ("kv_dtype_bytes", kv_dtype_bytes),
        ("recurrent_dtype_bytes", recurrent_dtype_bytes),
        ("convolution_dtype_bytes", convolution_dtype_bytes),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    validate_cache_parallelism(model_spec, tensor_parallel_size)
    config = model_spec.text_config
    num_attention_heads = int(config.num_attention_heads)
    local_kv_heads, kv_head_replication = _local_heads(
        int(config.num_key_value_heads),
        tensor_parallel_size,
    )
    head_dim = int(
        getattr(
            config,
            "head_dim",
            int(config.hidden_size) // num_attention_heads,
        )
    )
    kv_bytes_per_token = (
        2
        * model_spec.num_kv_cache_layers
        * local_kv_heads
        * head_dim
        * kv_dtype_bytes
    )
    int8_scale_bytes_per_token = (
        2
        * model_spec.num_kv_cache_layers
        * local_kv_heads
        * 2  # Runtime stores one FP16 scale per K/V token and local KV head.
    )

    recurrent_bytes = 0
    convolution_bytes = 0
    if model_spec.linear_attention_layers:
        local_key_heads, _ = _local_heads(
            int(config.linear_num_key_heads),
            tensor_parallel_size,
        )
        local_value_heads, _ = _local_heads(
            int(config.linear_num_value_heads),
            tensor_parallel_size,
        )
        key_head_dim = int(config.linear_key_head_dim)
        value_head_dim = int(config.linear_value_head_dim)
        num_linear_layers = len(model_spec.linear_attention_layers)
        recurrent_bytes = (
            num_linear_layers
            * local_value_heads
            * key_head_dim
            * value_head_dim
            * recurrent_dtype_bytes
        )
        local_convolution_width = (
            2 * local_key_heads * key_head_dim
            + local_value_heads * value_head_dim
        )
        convolution_bytes = (
            num_linear_layers
            * local_convolution_width
            * int(config.linear_conv_kernel_dim)
            * convolution_dtype_bytes
        )

    return CacheMemoryPlan(
        tensor_parallel_size=tensor_parallel_size,
        local_kv_heads=local_kv_heads,
        kv_head_replication=kv_head_replication,
        kv_bytes_per_token=kv_bytes_per_token,
        int8_scale_bytes_per_token=int8_scale_bytes_per_token,
        recurrent_bytes_per_sequence=recurrent_bytes,
        convolution_bytes_per_sequence=convolution_bytes,
    )
