from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    TRANSFERRING = auto()
    TRANSFERRED = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ):
        if not token_ids:
            raise ValueError("token_ids must contain at least one token")
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.state_slot: int | None = None
        self.arrival_time: float | None = None
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        # Scheduler-owned attribution. These counters stay on the request so
        # tail-latency samples can be connected to the pressure that caused
        # them instead of relying only on engine-wide totals.
        self.num_preemptions = 0
        self.preempted_token_progress = 0
        self.computed_token_ranges: list[tuple[int, int]] = []
        self.recomputed_tokens = 0
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def record_computed_span(self, start: int, count: int) -> int:
        """Record an executed token span and return its prior overlap."""

        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, count)
        ):
            raise TypeError("computed token span must use integer bounds")
        if start < 0 or count < 0:
            raise ValueError("computed token span must be non-negative")
        end = start + count
        if count == 0:
            return 0
        overlap = sum(
            max(min(end, range_end) - max(start, range_start), 0)
            for range_start, range_end in self.computed_token_ranges
        )
        merged = []
        inserted = False
        for range_start, range_end in self.computed_token_ranges:
            if range_end < start:
                merged.append((range_start, range_end))
            elif end < range_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((range_start, range_end))
            else:
                start = min(start, range_start)
                end = max(end, range_end)
        if not inserted:
            merged.append((start, end))
        self.computed_token_ranges = merged
        return overlap

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
        ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
