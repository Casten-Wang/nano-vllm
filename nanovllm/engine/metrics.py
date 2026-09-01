from dataclasses import dataclass


@dataclass(slots=True)
class EngineMetrics:
    """Lightweight aggregate counters for one LLMEngine instance.

    These metrics are intentionally simple: they only summarize engine-level
    work that is already visible in the generate loop. Request-level TTFT,
    TPOT, and latency are recorded when Sequence timestamps are available.
    Mixed steps retain their wall time as one combined measurement instead of
    attributing the same elapsed time to both pure prefill and pure decode.
    """

    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    pure_prefill_tokens: int = 0
    pure_prefill_time: float = 0.0
    pure_prefill_steps: int = 0
    pure_decode_tokens: int = 0
    pure_decode_time: float = 0.0
    pure_decode_steps: int = 0
    mixed_prefill_tokens: int = 0
    mixed_decode_tokens: int = 0
    mixed_step_time: float = 0.0
    mixed_steps: int = 0
    max_waiting_queue_len: int = 0
    max_running_queue_len: int = 0
    peak_used_kvcache_blocks: int = 0
    peak_kv_block_usage: float = 0.0
    request_ttfts: list[float] | None = None
    request_tpots: list[float] | None = None
    request_latencies: list[float] | None = None

    def __post_init__(self):
        self.request_ttfts = []
        self.request_tpots = []
        self.request_latencies = []

    def reset(self):
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.pure_prefill_tokens = 0
        self.pure_prefill_time = 0.0
        self.pure_prefill_steps = 0
        self.pure_decode_tokens = 0
        self.pure_decode_time = 0.0
        self.pure_decode_steps = 0
        self.mixed_prefill_tokens = 0
        self.mixed_decode_tokens = 0
        self.mixed_step_time = 0.0
        self.mixed_steps = 0
        self.max_waiting_queue_len = 0
        self.max_running_queue_len = 0
        self.peak_used_kvcache_blocks = 0
        self.peak_kv_block_usage = 0.0
        self.request_ttfts.clear()
        self.request_tpots.clear()
        self.request_latencies.clear()

    def record_step(
        self,
        num_tokens: int,
        elapsed: float,
        prefill_tokens: int | None = None,
        decode_tokens: int | None = None,
    ):
        """Record one scheduler/model step.

        LLMEngine.step returns a positive token count for prefill work and a
        negative token count for decode work. The sign is part of nano-vLLM's
        existing convention, so this class follows it instead of introducing a
        new enum or mode flag.
        """
        if prefill_tokens is not None or decode_tokens is not None:
            prefill_tokens = prefill_tokens or 0
            decode_tokens = decode_tokens or 0
        elif num_tokens > 0:
            prefill_tokens = num_tokens
            decode_tokens = 0
        elif num_tokens < 0:
            prefill_tokens = 0
            decode_tokens = -num_tokens
        else:
            prefill_tokens = 0
            decode_tokens = 0

        self.total_prefill_tokens += prefill_tokens
        self.total_decode_tokens += decode_tokens
        if prefill_tokens and decode_tokens:
            self.mixed_prefill_tokens += prefill_tokens
            self.mixed_decode_tokens += decode_tokens
            self.mixed_step_time += elapsed
            self.mixed_steps += 1
        elif prefill_tokens:
            self.pure_prefill_tokens += prefill_tokens
            self.pure_prefill_time += elapsed
            self.pure_prefill_steps += 1
        elif decode_tokens:
            self.pure_decode_tokens += decode_tokens
            self.pure_decode_time += elapsed
            self.pure_decode_steps += 1

    def record_scheduler_state(
        self,
        waiting_queue_len: int,
        running_queue_len: int,
        used_kvcache_blocks: int,
        total_kvcache_blocks: int,
    ):
        """Record queue and KV block high-water marks.

        This should be called after scheduling, while the scheduled sequences
        still hold their KV blocks. If we only look after generation finishes,
        finished sequences may already have released their blocks and the usage
        number would misleadingly be zero.
        """
        self.max_waiting_queue_len = max(self.max_waiting_queue_len, waiting_queue_len)
        self.max_running_queue_len = max(self.max_running_queue_len, running_queue_len)
        self.peak_used_kvcache_blocks = max(self.peak_used_kvcache_blocks, used_kvcache_blocks)
        if total_kvcache_blocks > 0:
            self.peak_kv_block_usage = max(
                self.peak_kv_block_usage,
                used_kvcache_blocks / total_kvcache_blocks,
            )

    def record_finished_sequences(self, seqs):
        """Record request-level latency metrics for finished sequences.

        TTFT measures time from request arrival to the first generated token.
        TPOT measures the average time between generated tokens after the first
        token. A one-token completion has no post-first-token interval, so its
        TPOT is recorded as 0.
        """
        for seq in seqs:
            if seq.arrival_time is None or seq.first_token_time is None or seq.finish_time is None:
                continue
            ttft = seq.first_token_time - seq.arrival_time
            latency = seq.finish_time - seq.arrival_time
            num_output_tokens = seq.num_completion_tokens
            tpot = 0.0
            if num_output_tokens > 1:
                tpot = (seq.finish_time - seq.first_token_time) / (num_output_tokens - 1)
            self.request_ttfts.append(ttft)
            self.request_tpots.append(tpot)
            self.request_latencies.append(latency)

    @staticmethod
    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _max(values: list[float]) -> float:
        if not values:
            return 0.0
        return max(values)

    @property
    def pure_prefill_throughput(self) -> float:
        if self.pure_prefill_time == 0.0:
            return 0.0
        return self.pure_prefill_tokens / self.pure_prefill_time

    @property
    def pure_decode_throughput(self) -> float:
        if self.pure_decode_time == 0.0:
            return 0.0
        return self.pure_decode_tokens / self.pure_decode_time

    @property
    def prefill_throughput(self) -> float:
        """Compatibility alias for the pure-prefill progress display."""

        return self.pure_prefill_throughput

    @property
    def decode_throughput(self) -> float:
        """Compatibility alias for the pure-decode progress display."""

        return self.pure_decode_throughput

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "pure_prefill_tokens": self.pure_prefill_tokens,
            "pure_prefill_time_s": self.pure_prefill_time,
            "pure_prefill_steps": self.pure_prefill_steps,
            "pure_prefill_throughput_tok_s": self.pure_prefill_throughput,
            "pure_decode_tokens": self.pure_decode_tokens,
            "pure_decode_time_s": self.pure_decode_time,
            "pure_decode_steps": self.pure_decode_steps,
            "pure_decode_throughput_tok_s": self.pure_decode_throughput,
            "mixed_prefill_tokens": self.mixed_prefill_tokens,
            "mixed_decode_tokens": self.mixed_decode_tokens,
            "mixed_step_time_s": self.mixed_step_time,
            "mixed_steps": self.mixed_steps,
            "max_waiting_queue_len": self.max_waiting_queue_len,
            "max_running_queue_len": self.max_running_queue_len,
            "peak_used_kvcache_blocks": self.peak_used_kvcache_blocks,
            "peak_kv_block_usage": self.peak_kv_block_usage,
            "num_finished_requests": len(self.request_latencies),
            "avg_ttft_s": self._avg(self.request_ttfts),
            "max_ttft_s": self._max(self.request_ttfts),
            "avg_tpot_s": self._avg(self.request_tpots),
            "max_tpot_s": self._max(self.request_tpots),
            "avg_request_latency_s": self._avg(self.request_latencies),
            "max_request_latency_s": self._max(self.request_latencies),
        }
