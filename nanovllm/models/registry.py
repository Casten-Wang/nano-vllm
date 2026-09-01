"""Lazy model registry for architecture-specific runtime implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

MODEL_FACTORIES = {
    "Qwen3ForCausalLM": ("nanovllm.models.qwen3", "Qwen3ForCausalLM"),
    "Qwen3_5MoeForCausalLM": (
        "nanovllm.models.qwen35",
        "Qwen3_5MoeForCausalLM",
    ),
    "Qwen3_5MoeForConditionalGeneration": (
        "nanovllm.models.qwen35",
        "Qwen3_5MoeForConditionalGeneration",
    ),
}


def create_model(architecture: str, model_config: Any):
    """Instantiate a registered model without importing every backend."""

    target = MODEL_FACTORIES.get(architecture)
    if target is None:
        raise ValueError(f"unsupported model architecture: {architecture}")
    module_name, class_name = target
    model_class = getattr(import_module(module_name), class_name)
    return model_class(model_config)
