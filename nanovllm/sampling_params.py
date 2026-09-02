from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert (
            isinstance(self.max_tokens, int)
            and not isinstance(self.max_tokens, bool)
            and self.max_tokens > 0
        ), "max_tokens must be a positive integer"

    def validate_request_length(self, prompt_length: int, max_model_len: int):
        requested_tokens = prompt_length + self.max_tokens
        if requested_tokens > max_model_len:
            raise ValueError(
                f"request requires {requested_tokens} tokens, exceeding "
                f"max_model_len={max_model_len}"
            )
