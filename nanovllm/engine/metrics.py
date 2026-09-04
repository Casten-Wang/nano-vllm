from dataclasses import dataclass
import math


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
    prefill_starved_steps: int = 0
    max_prefill_starvation_steps: int = 0
    preemption_count: int = 0
    waiting_prefill_preemptions: int = 0
    preempted_token_progress: int = 0
    max_preempted_token_progress: int = 0
    reclaimed_kv_blocks: int = 0
    aborted_requests: int = 0
    prefill_stopped_by_token_budget: int = 0
    prefill_stopped_by_sequence_capacity: int = 0
    prefill_stopped_by_kv_capacity: int = 0
    prefill_stopped_by_decode_kv_reservation: int = 0
    peak_decode_kv_reserve_blocks: int = 0
    remote_prefill_receive_started: int = 0
    remote_prefill_receive_committed: int = 0
    remote_prefill_receive_failed: int = 0
    remote_prefill_receive_timed_out: int = 0
    remote_prefill_receive_cancelled: int = 0
    remote_prefill_reservation_timed_out: int = 0
    remote_prefill_receive_time: float = 0.0
    max_remote_prefill_receive_time: float = 0.0
    remote_prefill_poll_calls: int = 0
    remote_prefill_requests_polled: int = 0
    remote_prefill_send_started: int = 0
    remote_prefill_send_committed: int = 0
    remote_prefill_send_failed: int = 0
    remote_prefill_send_cancelled: int = 0
    remote_prefill_send_time: float = 0.0
    max_remote_prefill_send_time: float = 0.0
    remote_prefill_send_poll_calls: int = 0
    remote_prefill_send_requests_polled: int = 0
    remote_prefill_receive_backpressure: int = 0
    remote_prefill_send_backpressure: int = 0
    remote_prefill_receive_staged_bytes: int = 0
    active_remote_prefill_receive_staged_bytes: int = 0
    peak_remote_prefill_receive_staged_bytes: int = 0
    remote_prefill_send_staged_bytes: int = 0
    active_remote_prefill_send_staged_bytes: int = 0
    peak_remote_prefill_send_staged_bytes: int = 0
    remote_prefill_sent_bytes: int = 0
    request_ttfts: list[float] | None = None
    request_time_to_first_schedules: list[float] | None = None
    request_first_token_service_times: list[float] | None = None
    request_tpots: list[float] | None = None
    request_latencies: list[float] | None = None
    request_samples: list[dict[str, int | float]] | None = None

    def __post_init__(self):
        self.request_ttfts = []
        self.request_time_to_first_schedules = []
        self.request_first_token_service_times = []
        self.request_tpots = []
        self.request_latencies = []
        self.request_samples = []

    def reset(self):
        active_receive_staged_bytes = self.active_remote_prefill_receive_staged_bytes
        active_send_staged_bytes = self.active_remote_prefill_send_staged_bytes
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
        self.prefill_starved_steps = 0
        self.max_prefill_starvation_steps = 0
        self.preemption_count = 0
        self.waiting_prefill_preemptions = 0
        self.preempted_token_progress = 0
        self.max_preempted_token_progress = 0
        self.reclaimed_kv_blocks = 0
        self.aborted_requests = 0
        self.prefill_stopped_by_token_budget = 0
        self.prefill_stopped_by_sequence_capacity = 0
        self.prefill_stopped_by_kv_capacity = 0
        self.prefill_stopped_by_decode_kv_reservation = 0
        self.peak_decode_kv_reserve_blocks = 0
        self.remote_prefill_receive_started = 0
        self.remote_prefill_receive_committed = 0
        self.remote_prefill_receive_failed = 0
        self.remote_prefill_receive_timed_out = 0
        self.remote_prefill_receive_cancelled = 0
        self.remote_prefill_reservation_timed_out = 0
        self.remote_prefill_receive_time = 0.0
        self.max_remote_prefill_receive_time = 0.0
        self.remote_prefill_poll_calls = 0
        self.remote_prefill_requests_polled = 0
        self.remote_prefill_send_started = 0
        self.remote_prefill_send_committed = 0
        self.remote_prefill_send_failed = 0
        self.remote_prefill_send_cancelled = 0
        self.remote_prefill_send_time = 0.0
        self.max_remote_prefill_send_time = 0.0
        self.remote_prefill_send_poll_calls = 0
        self.remote_prefill_send_requests_polled = 0
        self.remote_prefill_receive_backpressure = 0
        self.remote_prefill_send_backpressure = 0
        self.remote_prefill_receive_staged_bytes = 0
        self.active_remote_prefill_receive_staged_bytes = active_receive_staged_bytes
        self.peak_remote_prefill_receive_staged_bytes = active_receive_staged_bytes
        self.remote_prefill_send_staged_bytes = 0
        self.active_remote_prefill_send_staged_bytes = active_send_staged_bytes
        self.peak_remote_prefill_send_staged_bytes = active_send_staged_bytes
        self.remote_prefill_sent_bytes = 0
        self.request_ttfts.clear()
        self.request_time_to_first_schedules.clear()
        self.request_first_token_service_times.clear()
        self.request_tpots.clear()
        self.request_latencies.clear()
        self.request_samples.clear()

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
        prefill_starved_steps: int = 0,
        max_prefill_starvation_steps: int = 0,
        preemption_count: int = 0,
        waiting_prefill_preemptions: int = 0,
        preempted_token_progress: int = 0,
        max_preempted_token_progress: int = 0,
        reclaimed_kv_blocks: int = 0,
        aborted_requests: int = 0,
        prefill_stopped_by_token_budget: int = 0,
        prefill_stopped_by_sequence_capacity: int = 0,
        prefill_stopped_by_kv_capacity: int = 0,
        prefill_stopped_by_decode_kv_reservation: int = 0,
        decode_kv_reserve_blocks: int = 0,
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
        self.prefill_starved_steps = prefill_starved_steps
        self.max_prefill_starvation_steps = max_prefill_starvation_steps
        self.preemption_count = preemption_count
        self.waiting_prefill_preemptions = waiting_prefill_preemptions
        self.preempted_token_progress = preempted_token_progress
        self.max_preempted_token_progress = max_preempted_token_progress
        self.reclaimed_kv_blocks = reclaimed_kv_blocks
        self.aborted_requests = aborted_requests
        self.prefill_stopped_by_token_budget = prefill_stopped_by_token_budget
        self.prefill_stopped_by_sequence_capacity = (
            prefill_stopped_by_sequence_capacity
        )
        self.prefill_stopped_by_kv_capacity = prefill_stopped_by_kv_capacity
        self.prefill_stopped_by_decode_kv_reservation = (
            prefill_stopped_by_decode_kv_reservation
        )
        self.peak_decode_kv_reserve_blocks = max(
            self.peak_decode_kv_reserve_blocks,
            decode_kv_reserve_blocks,
        )

    def record_finished_sequences(self, seqs):
        """Record request-level latency metrics for finished sequences.

        TTFT measures time from request arrival to the first generated token.
        TPOT measures the average time between generated tokens after the first
        token. A one-token completion has no post-first-token interval, so its
        TPOT is recorded as 0.
        """
        for seq in seqs:
            if (
                seq.arrival_time is None
                or seq.first_scheduled_time is None
                or seq.first_token_time is None
                or seq.finish_time is None
            ):
                continue
            time_to_first_schedule = seq.first_scheduled_time - seq.arrival_time
            first_token_service = seq.first_token_time - seq.first_scheduled_time
            ttft = seq.first_token_time - seq.arrival_time
            latency = seq.finish_time - seq.arrival_time
            num_output_tokens = seq.num_completion_tokens
            tpot = 0.0
            if num_output_tokens > 1:
                tpot = (seq.finish_time - seq.first_token_time) / (num_output_tokens - 1)
            self.request_ttfts.append(ttft)
            self.request_time_to_first_schedules.append(time_to_first_schedule)
            self.request_first_token_service_times.append(first_token_service)
            self.request_tpots.append(tpot)
            self.request_latencies.append(latency)
            self.request_samples.append({
                "seq_id": seq.seq_id,
                "prompt_tokens": seq.num_prompt_tokens,
                "output_tokens": num_output_tokens,
                "preemption_count": seq.num_preemptions,
                "preempted_token_progress": seq.preempted_token_progress,
                "recomputed_tokens": seq.recomputed_tokens,
                "time_to_first_schedule_s": time_to_first_schedule,
                "first_token_service_s": first_token_service,
                "ttft_s": ttft,
                "tpot_s": tpot,
                "latency_s": latency,
            })

    def record_remote_prefill_receive_started(self, staged_bytes: int = 0) -> None:
        if staged_bytes < 0:
            raise ValueError("remote prefill staged bytes must be non-negative")
        self.remote_prefill_receive_started += 1
        self.remote_prefill_receive_staged_bytes += staged_bytes
        self.active_remote_prefill_receive_staged_bytes += staged_bytes
        self.peak_remote_prefill_receive_staged_bytes = max(
            self.peak_remote_prefill_receive_staged_bytes,
            self.active_remote_prefill_receive_staged_bytes,
        )

    def record_remote_prefill_poll(self, request_count: int) -> None:
        if request_count <= 0:
            raise ValueError("remote prefill poll request count must be positive")
        self.remote_prefill_poll_calls += 1
        self.remote_prefill_requests_polled += request_count

    def record_remote_prefill_reservation_timeout(
        self,
        request_count: int = 1,
    ) -> None:
        if request_count <= 0:
            raise ValueError("expired reservation count must be positive")
        self.remote_prefill_reservation_timed_out += request_count

    def record_remote_prefill_receive_finished(
        self,
        elapsed: float,
        *,
        outcome: str,
        staged_bytes: int = 0,
    ) -> None:
        if elapsed < 0.0:
            raise ValueError("remote prefill receive elapsed time must be non-negative")
        counters = {
            "committed": "remote_prefill_receive_committed",
            "failed": "remote_prefill_receive_failed",
            "timed_out": "remote_prefill_receive_timed_out",
            "cancelled": "remote_prefill_receive_cancelled",
        }
        counter = counters.get(outcome)
        if counter is None:
            raise ValueError("remote prefill receive outcome is invalid")
        if (
            staged_bytes < 0
            or staged_bytes > self.active_remote_prefill_receive_staged_bytes
        ):
            raise ValueError("remote prefill finished staged bytes are invalid")
        setattr(self, counter, getattr(self, counter) + 1)
        self.active_remote_prefill_receive_staged_bytes -= staged_bytes
        self.remote_prefill_receive_time += elapsed
        self.max_remote_prefill_receive_time = max(
            self.max_remote_prefill_receive_time,
            elapsed,
        )

    def record_remote_prefill_send_started(self, staged_bytes: int = 0) -> None:
        if staged_bytes < 0:
            raise ValueError("remote prefill staged bytes must be non-negative")
        self.remote_prefill_send_started += 1
        self.remote_prefill_send_staged_bytes += staged_bytes
        self.active_remote_prefill_send_staged_bytes += staged_bytes
        self.peak_remote_prefill_send_staged_bytes = max(
            self.peak_remote_prefill_send_staged_bytes,
            self.active_remote_prefill_send_staged_bytes,
        )

    def record_remote_prefill_backpressure(self, *, direction: str) -> None:
        counters = {
            "receive": "remote_prefill_receive_backpressure",
            "send": "remote_prefill_send_backpressure",
        }
        counter = counters.get(direction)
        if counter is None:
            raise ValueError("remote prefill backpressure direction is invalid")
        setattr(self, counter, getattr(self, counter) + 1)

    def record_remote_prefill_send_poll(self, request_count: int) -> None:
        if request_count <= 0:
            raise ValueError("remote prefill send poll request count must be positive")
        self.remote_prefill_send_poll_calls += 1
        self.remote_prefill_send_requests_polled += request_count

    def record_remote_prefill_send_staging_released(self, staged_bytes: int) -> None:
        """Account for host staging released before the receiver ACK arrives."""

        if (
            staged_bytes < 0
            or staged_bytes > self.active_remote_prefill_send_staged_bytes
        ):
            raise ValueError("remote prefill released staged bytes are invalid")
        self.active_remote_prefill_send_staged_bytes -= staged_bytes

    def record_remote_prefill_send_finished(
        self,
        elapsed: float,
        *,
        outcome: str,
        staged_bytes: int = 0,
        sent_bytes: int = 0,
    ) -> None:
        if elapsed < 0.0:
            raise ValueError("remote prefill send elapsed time must be non-negative")
        counters = {
            "committed": "remote_prefill_send_committed",
            "failed": "remote_prefill_send_failed",
            "cancelled": "remote_prefill_send_cancelled",
        }
        counter = counters.get(outcome)
        if counter is None:
            raise ValueError("remote prefill send outcome is invalid")
        if staged_bytes < 0 or staged_bytes > self.active_remote_prefill_send_staged_bytes:
            raise ValueError("remote prefill finished staged bytes are invalid")
        if sent_bytes < 0 or (outcome != "committed" and sent_bytes):
            raise ValueError("remote prefill sent bytes are invalid")
        setattr(self, counter, getattr(self, counter) + 1)
        self.active_remote_prefill_send_staged_bytes -= staged_bytes
        self.remote_prefill_sent_bytes += sent_bytes
        self.remote_prefill_send_time += elapsed
        self.max_remote_prefill_send_time = max(
            self.max_remote_prefill_send_time,
            elapsed,
        )

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

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Return a linearly interpolated percentile over observed requests."""

        if not values:
            return 0.0
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be between 0 and 1")
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

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
    def avg_remote_prefill_receive_time(self) -> float:
        finished = (
            self.remote_prefill_receive_committed
            + self.remote_prefill_receive_failed
            + self.remote_prefill_receive_timed_out
            + self.remote_prefill_receive_cancelled
        )
        if finished == 0:
            return 0.0
        return self.remote_prefill_receive_time / finished

    @property
    def avg_remote_prefill_send_time(self) -> float:
        finished = (
            self.remote_prefill_send_committed
            + self.remote_prefill_send_failed
            + self.remote_prefill_send_cancelled
        )
        if finished == 0:
            return 0.0
        return self.remote_prefill_send_time / finished

    @property
    def prefill_throughput(self) -> float:
        """Compatibility alias for the pure-prefill progress display."""

        return self.pure_prefill_throughput

    @property
    def decode_throughput(self) -> float:
        """Compatibility alias for the pure-decode progress display."""

        return self.pure_decode_throughput

    def to_dict(self) -> dict[str, object]:
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
            "prefill_starved_steps": self.prefill_starved_steps,
            "max_prefill_starvation_steps": self.max_prefill_starvation_steps,
            "preemption_count": self.preemption_count,
            "waiting_prefill_preemptions": self.waiting_prefill_preemptions,
            "preempted_token_progress": self.preempted_token_progress,
            "max_preempted_token_progress": self.max_preempted_token_progress,
            "reclaimed_kv_blocks": self.reclaimed_kv_blocks,
            "aborted_requests": self.aborted_requests,
            "prefill_stopped_by_token_budget": self.prefill_stopped_by_token_budget,
            "prefill_stopped_by_sequence_capacity": (
                self.prefill_stopped_by_sequence_capacity
            ),
            "prefill_stopped_by_kv_capacity": self.prefill_stopped_by_kv_capacity,
            "prefill_stopped_by_decode_kv_reservation": (
                self.prefill_stopped_by_decode_kv_reservation
            ),
            "peak_decode_kv_reserve_blocks": self.peak_decode_kv_reserve_blocks,
            "remote_prefill_receive_started": self.remote_prefill_receive_started,
            "remote_prefill_receive_committed": self.remote_prefill_receive_committed,
            "remote_prefill_receive_failed": self.remote_prefill_receive_failed,
            "remote_prefill_receive_timed_out": self.remote_prefill_receive_timed_out,
            "remote_prefill_receive_cancelled": self.remote_prefill_receive_cancelled,
            "remote_prefill_reservation_timed_out": (
                self.remote_prefill_reservation_timed_out
            ),
            "remote_prefill_receive_time_s": self.remote_prefill_receive_time,
            "avg_remote_prefill_receive_time_s": self.avg_remote_prefill_receive_time,
            "max_remote_prefill_receive_time_s": self.max_remote_prefill_receive_time,
            "remote_prefill_poll_calls": self.remote_prefill_poll_calls,
            "remote_prefill_requests_polled": self.remote_prefill_requests_polled,
            "remote_prefill_send_started": self.remote_prefill_send_started,
            "remote_prefill_send_committed": self.remote_prefill_send_committed,
            "remote_prefill_send_failed": self.remote_prefill_send_failed,
            "remote_prefill_send_cancelled": self.remote_prefill_send_cancelled,
            "remote_prefill_send_time_s": self.remote_prefill_send_time,
            "avg_remote_prefill_send_time_s": self.avg_remote_prefill_send_time,
            "max_remote_prefill_send_time_s": self.max_remote_prefill_send_time,
            "remote_prefill_send_poll_calls": self.remote_prefill_send_poll_calls,
            "remote_prefill_send_requests_polled": self.remote_prefill_send_requests_polled,
            "remote_prefill_receive_backpressure": self.remote_prefill_receive_backpressure,
            "remote_prefill_send_backpressure": self.remote_prefill_send_backpressure,
            "remote_prefill_receive_staged_bytes": self.remote_prefill_receive_staged_bytes,
            "active_remote_prefill_receive_staged_bytes": self.active_remote_prefill_receive_staged_bytes,
            "peak_remote_prefill_receive_staged_bytes": self.peak_remote_prefill_receive_staged_bytes,
            "remote_prefill_send_staged_bytes": self.remote_prefill_send_staged_bytes,
            "active_remote_prefill_send_staged_bytes": self.active_remote_prefill_send_staged_bytes,
            "peak_remote_prefill_send_staged_bytes": self.peak_remote_prefill_send_staged_bytes,
            "remote_prefill_sent_bytes": self.remote_prefill_sent_bytes,
            "num_finished_requests": len(self.request_latencies),
            "request_samples": list(self.request_samples),
            "avg_ttft_s": self._avg(self.request_ttfts),
            "p50_ttft_s": self._percentile(self.request_ttfts, 0.50),
            "p95_ttft_s": self._percentile(self.request_ttfts, 0.95),
            "p99_ttft_s": self._percentile(self.request_ttfts, 0.99),
            "max_ttft_s": self._max(self.request_ttfts),
            "avg_time_to_first_schedule_s": self._avg(
                self.request_time_to_first_schedules
            ),
            "p50_time_to_first_schedule_s": self._percentile(
                self.request_time_to_first_schedules,
                0.50,
            ),
            "p95_time_to_first_schedule_s": self._percentile(
                self.request_time_to_first_schedules,
                0.95,
            ),
            "p99_time_to_first_schedule_s": self._percentile(
                self.request_time_to_first_schedules,
                0.99,
            ),
            "max_time_to_first_schedule_s": self._max(
                self.request_time_to_first_schedules
            ),
            "avg_first_token_service_s": self._avg(
                self.request_first_token_service_times
            ),
            "p50_first_token_service_s": self._percentile(
                self.request_first_token_service_times,
                0.50,
            ),
            "p95_first_token_service_s": self._percentile(
                self.request_first_token_service_times,
                0.95,
            ),
            "p99_first_token_service_s": self._percentile(
                self.request_first_token_service_times,
                0.99,
            ),
            "max_first_token_service_s": self._max(
                self.request_first_token_service_times
            ),
            "avg_tpot_s": self._avg(self.request_tpots),
            "p50_tpot_s": self._percentile(self.request_tpots, 0.50),
            "p95_tpot_s": self._percentile(self.request_tpots, 0.95),
            "p99_tpot_s": self._percentile(self.request_tpots, 0.99),
            "max_tpot_s": self._max(self.request_tpots),
            "avg_request_latency_s": self._avg(self.request_latencies),
            "p50_request_latency_s": self._percentile(
                self.request_latencies,
                0.50,
            ),
            "p95_request_latency_s": self._percentile(
                self.request_latencies,
                0.95,
            ),
            "p99_request_latency_s": self._percentile(
                self.request_latencies,
                0.99,
            ),
            "max_request_latency_s": self._max(self.request_latencies),
        }
