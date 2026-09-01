from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = -1
    top_p: float = 1.0

    def __post_init__(self):
        if not isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.top_k != -1 and self.top_k <= 0:
            raise ValueError("top_k must be -1 or positive")
        if not isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
