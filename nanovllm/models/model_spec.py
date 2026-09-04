"""Architecture metadata used to plan model execution and cache storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanovllm.models.quantization_spec import (
    QuantizationSpec,
    resolve_quantization_spec,
)


QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"
QWEN35_MOE_ARCHITECTURES = frozenset(
    {
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Normalized text-model information independent of HF wrapper layout."""

    architecture: str
    text_config: Any
    full_attention_layers: tuple[int, ...]
    linear_attention_layers: tuple[int, ...]
    quantization: QuantizationSpec

    @property
    def num_hidden_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def num_kv_cache_layers(self) -> int:
        return len(self.full_attention_layers)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.linear_attention_layers)


def validate_weight_parallelism(
    model_spec: ModelSpec,
    tensor_parallel_size: int,
) -> None:
    """Validate dimensions sharded by the supported model implementations."""

    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size <= 0
    ):
        raise ValueError("tensor_parallel_size must be a positive integer")
    config = model_spec.text_config
    for field in (
        "vocab_size",
        "intermediate_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        value = getattr(config, field, None)
        if value is None:
            continue
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{field} must be a positive integer")
        if value % tensor_parallel_size:
            raise ValueError(
                f"{field}={value} cannot be sharded across "
                f"TP={tensor_parallel_size}"
            )


def _architecture(hf_config: Any) -> str:
    architectures = getattr(hf_config, "architectures", None)
    if not architectures or not isinstance(architectures, (list, tuple)):
        raise ValueError("model config must declare at least one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model architecture must be a non-empty string")
    return architecture


def _validate_qwen35_semantics(text_config: Any) -> None:
    """Reject variants whose configured math is not implemented locally."""

    if not bool(getattr(text_config, "attn_output_gate", True)):
        raise ValueError("Qwen3.6-compatible MoE requires attn_output_gate=True")
    if getattr(text_config, "hidden_act", "silu") != "silu":
        raise ValueError("Qwen3.6-compatible MoE supports only hidden_act='silu'")
    if tuple(getattr(text_config, "mlp_only_layers", ())) != ():
        raise ValueError("Qwen3.6-compatible MoE mlp_only_layers are not supported")
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    if rope_parameters.get("rope_type", "default") != "default":
        raise ValueError("Qwen3.6-compatible MoE supports only default RoPE")


def resolve_model_spec(hf_config: Any) -> ModelSpec:
    """Resolve dense and hybrid Qwen configs to a common text-only view."""

    architecture = _architecture(hf_config)
    quantization = resolve_quantization_spec(hf_config)
    if architecture == QWEN3_ARCHITECTURE:
        num_layers = int(hf_config.num_hidden_layers)
        return ModelSpec(
            architecture=architecture,
            text_config=hf_config,
            full_attention_layers=tuple(range(num_layers)),
            linear_attention_layers=(),
            quantization=quantization,
        )

    if architecture in QWEN35_MOE_ARCHITECTURES:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            if architecture == "Qwen3_5MoeForCausalLM":
                text_config = hf_config
            else:
                raise ValueError(
                    "Qwen3.6-compatible conditional-generation config is missing text_config"
                )
        num_layers = int(text_config.num_hidden_layers)
        layer_types = tuple(getattr(text_config, "layer_types", ()))
        if len(layer_types) != num_layers:
            raise ValueError(
                "Qwen3.6-compatible layer_types length must equal num_hidden_layers: "
                f"{len(layer_types)} != {num_layers}"
            )
        unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
        if unknown:
            raise ValueError(f"unsupported Qwen3.6-compatible layer types: {unknown}")
        full_attention_layers = tuple(
            index for index, kind in enumerate(layer_types) if kind == "full_attention"
        )
        linear_attention_layers = tuple(
            index for index, kind in enumerate(layer_types) if kind == "linear_attention"
        )
        if not full_attention_layers or not linear_attention_layers:
            raise ValueError(
                "Qwen3.6-compatible MoE must contain both full and linear attention layers"
            )
        _validate_qwen35_semantics(text_config)
        return ModelSpec(
            architecture=architecture,
            text_config=text_config,
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            quantization=quantization,
        )

    raise ValueError(f"unsupported model architecture: {architecture}")
