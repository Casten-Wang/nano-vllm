"""Architecture metadata used to plan model execution and cache storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    @property
    def num_hidden_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def num_kv_cache_layers(self) -> int:
        return len(self.full_attention_layers)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.linear_attention_layers)


def _architecture(hf_config: Any) -> str:
    architectures = getattr(hf_config, "architectures", None)
    if not architectures or not isinstance(architectures, (list, tuple)):
        raise ValueError("model config must declare at least one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model architecture must be a non-empty string")
    return architecture


def resolve_model_spec(hf_config: Any) -> ModelSpec:
    """Resolve dense and hybrid Qwen configs to a common text-only view."""

    architecture = _architecture(hf_config)
    if architecture == QWEN3_ARCHITECTURE:
        num_layers = int(hf_config.num_hidden_layers)
        return ModelSpec(
            architecture=architecture,
            text_config=hf_config,
            full_attention_layers=tuple(range(num_layers)),
            linear_attention_layers=(),
        )

    if architecture in QWEN35_MOE_ARCHITECTURES:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            if architecture == "Qwen3_5MoeForCausalLM":
                text_config = hf_config
            else:
                raise ValueError(
                    "Qwen3.5 conditional-generation config is missing text_config"
                )
        num_layers = int(text_config.num_hidden_layers)
        layer_types = tuple(getattr(text_config, "layer_types", ()))
        if len(layer_types) != num_layers:
            raise ValueError(
                "Qwen3.5 layer_types length must equal num_hidden_layers: "
                f"{len(layer_types)} != {num_layers}"
            )
        unknown = sorted(set(layer_types) - {"full_attention", "linear_attention"})
        if unknown:
            raise ValueError(f"unsupported Qwen3.5 layer types: {unknown}")
        full_attention_layers = tuple(
            index for index, kind in enumerate(layer_types) if kind == "full_attention"
        )
        linear_attention_layers = tuple(
            index for index, kind in enumerate(layer_types) if kind == "linear_attention"
        )
        if not full_attention_layers or not linear_attention_layers:
            raise ValueError("Qwen3.5 must contain both full and linear attention layers")
        return ModelSpec(
            architecture=architecture,
            text_config=text_config,
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
        )

    raise ValueError(f"unsupported model architecture: {architecture}")
