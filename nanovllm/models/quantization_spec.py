"""Validated checkpoint quantization metadata, independent of execution kernels."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"quantization_config.{field} must be a positive integer")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"quantization_config.{field} must be a list of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"quantization_config.{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"quantization_config.{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Normalized on-disk format; this does not imply runtime support."""

    format: str
    weight_bits: int = 16
    activation_scheme: str | None = None
    weight_block_size: tuple[int, int] | None = None
    group_size: int | None = None
    symmetric: bool | None = None
    desc_act: bool | None = None
    ignored_modules: tuple[str, ...] = ()
    ignored_patterns: tuple[str, ...] = ()

    @property
    def is_quantized(self) -> bool:
        return self.format != "bf16"

    def ignores_module(self, module_name: str) -> bool:
        """Match complete module paths/subtrees, never incidental substrings."""

        if not isinstance(module_name, str) or not module_name:
            raise ValueError("module_name must be a non-empty string")
        if any(
            module_name == ignored or module_name.startswith(f"{ignored}.")
            for ignored in self.ignored_modules
        ):
            return True
        return any(
            re.fullmatch(pattern, module_name) for pattern in self.ignored_patterns
        )

    def require_runtime_support(self) -> None:
        """Reject recognized formats until their loaders and kernels exist."""

        if self.is_quantized:
            raise NotImplementedError(
                f"{self.format} checkpoints are recognized but not executable yet; "
                "use the BF16 checkpoint"
            )


BF16_QUANTIZATION_SPEC = QuantizationSpec(format="bf16")


def _parse_fp8(config: Mapping[str, object]) -> QuantizationSpec:
    if config.get("activation_scheme") != "dynamic":
        raise ValueError("FP8 activation_scheme must be 'dynamic'")
    if config.get("weight_per_tensor") is not False:
        raise ValueError("FP8 weight_per_tensor must be false for block quantization")
    if config.get("act_per_tensor") is not False:
        raise ValueError("FP8 act_per_tensor must be false for dynamic activations")
    raw_block = config.get("weight_block_size")
    if not isinstance(raw_block, (list, tuple)) or len(raw_block) != 2:
        raise ValueError("FP8 weight_block_size must contain two dimensions")
    block = tuple(
        _require_int(value, f"weight_block_size[{index}]")
        for index, value in enumerate(raw_block)
    )
    if block != (128, 128):
        raise ValueError(
            "only the official FP8 weight_block_size [128, 128] is supported"
        )
    return QuantizationSpec(
        format="fp8_block",
        weight_bits=8,
        activation_scheme="dynamic",
        weight_block_size=block,
        ignored_modules=_string_tuple(
            config.get("modules_to_not_convert"),
            "modules_to_not_convert",
        ),
    )


def _parse_gptq(config: Mapping[str, object]) -> QuantizationSpec:
    bits = _require_int(config.get("bits"), "bits")
    group_size = _require_int(config.get("group_size"), "group_size")
    if bits != 4:
        raise ValueError("only official GPTQ bits=4 checkpoints are supported")
    if group_size != 128:
        raise ValueError("only official GPTQ group_size=128 checkpoints are supported")
    if config.get("sym") is not True:
        raise ValueError("official GPTQ checkpoints require sym=true")
    if config.get("desc_act") is not False:
        raise ValueError("official GPTQ checkpoints require desc_act=false")

    dynamic = config.get("dynamic", {})
    if not isinstance(dynamic, Mapping):
        raise ValueError("quantization_config.dynamic must be a mapping")
    ignored_patterns = []
    for rule, override in dynamic.items():
        if not isinstance(rule, str) or not rule:
            raise ValueError(
                "quantization_config.dynamic keys must be non-empty strings"
            )
        if not isinstance(override, Mapping):
            raise ValueError("quantization_config.dynamic values must be mappings")
        if rule.startswith("-:"):
            pattern = rule[2:]
            if not pattern:
                raise ValueError("GPTQ exclusion patterns must not be empty")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid GPTQ exclusion pattern: {pattern!r}"
                ) from error
            ignored_patterns.append(pattern)

    return QuantizationSpec(
        format="gptq_int4",
        weight_bits=bits,
        group_size=group_size,
        symmetric=True,
        desc_act=False,
        ignored_modules=_string_tuple(
            config.get("modules_to_not_convert"),
            "modules_to_not_convert",
        ),
        ignored_patterns=tuple(ignored_patterns),
    )


def resolve_quantization_spec(hf_config: Any) -> QuantizationSpec:
    """Parse the checkpoint format declared by a Hugging Face config."""

    config = getattr(hf_config, "quantization_config", None)
    if config is None:
        return BF16_QUANTIZATION_SPEC
    if not isinstance(config, Mapping):
        raise ValueError("quantization_config must be a mapping")
    method = config.get("quant_method")
    if method == "fp8":
        return _parse_fp8(config)
    if method == "gptq":
        return _parse_gptq(config)
    raise ValueError(f"unsupported checkpoint quantization method: {method!r}")
