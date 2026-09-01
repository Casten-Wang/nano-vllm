"""Lazy model registry for architecture-specific runtime implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from nanovllm.models.model_spec import QWEN35_MOE_ARCHITECTURES


MODEL_FACTORIES = {
    "Qwen3ForCausalLM": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
}


def create_model(architecture: str, model_config: Any):
    """Instantiate a registered model without importing every backend."""

    target = MODEL_FACTORIES.get(architecture)
    if target is None:
        if architecture in QWEN35_MOE_ARCHITECTURES:
            raise NotImplementedError(
                "Qwen3.5 MoE config is recognized, but its text runtime is not "
                "implemented yet"
            )
        raise ValueError(f"unsupported model architecture: {architecture}")
    module_name, class_name = target
    model_class = getattr(import_module(module_name), class_name)
    return model_class(model_config)
