from collections import deque
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.state_manager import StateSlotManager


@dataclass(slots=True)
class ScheduleResult:
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]

    @property
    def is_mixed(self) -> bool:
        return bool(self.prefill_seqs and self.decode_seqs)

    @property
    def is_prefill(self) -> bool:
        return bool(self.prefill_seqs and not self.decode_seqs)

    @property
    def is_decode(self) -> bool:
        return bool(self.decode_seqs and not self.prefill_seqs)

    @property
    def seqs(self) -> list[Sequence]:
        return self.decode_seqs + self.prefill_seqs

    @property
    def num_prefill_tokens(self) -> int:
        return sum(seq.num_scheduled_tokens for seq in self.prefill_seqs)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_seqs)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        eos = config.eos if isinstance(config.eos, tuple) else (config.eos,)
        self.eos_token_ids = frozenset(eos)
        self.block_size = config.kvcache_block_size
        self.enable_dynamic_chunked_prefill = config.enable_dynamic_chunked_prefill
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        model_spec = getattr(config, "model_spec", None)
        self.state_manager = (
            StateSlotManager(config.max_num_seqs)
            if model_spec is not None and model_spec.is_hybrid
            else None
        )
        # KV-only prefix entries cannot reconstruct linear-attention state.
        # Hybrid models must replay the prompt until joint state snapshots
        # are implemented.
        self.prefix_cache_enabled = self.state_manager is None
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.schedule_steps = 0
        self.prefill_starved_steps = 0
        self.current_prefill_starvation_steps = 0
        self.max_prefill_starvation_steps = 0

    def is_finished(self):
        return not self.waiting and not self.running

    @property
    def num_waiting(self):
        return len(self.waiting)

    @property
    def num_running(self):
        return len(self.running)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool] | ScheduleResult:
        waiting_before = len(self.waiting)
        if self.enable_dynamic_chunked_prefill:
            result = self.schedule_dynamic_chunked_prefill()
        else:
            result = self.schedule_legacy()
        if isinstance(result, ScheduleResult):
            prefill_progress = bool(result.prefill_seqs)
        else:
            prefill_progress = result[1] and bool(result[0])
        # A complete prefix-cache hit can leave the waiting queue without a
        # prefill kernel, so queue shrinkage also counts as admission progress.
        prefill_progress = prefill_progress or len(self.waiting) < waiting_before
        if waiting_before and not prefill_progress:
            self.prefill_starved_steps += 1
            self.current_prefill_starvation_steps += 1
            self.max_prefill_starvation_steps = max(
                self.max_prefill_starvation_steps,
                self.current_prefill_starvation_steps,
            )
        else:
            self.current_prefill_starvation_steps = 0
        self.schedule_steps += 1
        seqs = result.seqs if isinstance(result, ScheduleResult) else result[0]
        if self.state_manager is not None:
            slots = self.state_manager.acquire_many(seq.seq_id for seq in seqs)
            for seq, slot in zip(seqs, slots):
                if seq.state_slot is not None and seq.state_slot != slot:
                    raise RuntimeError("sequence recurrent state slot changed unexpectedly")
                seq.state_slot = slot
        return result

    def schedule_legacy(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while (
            self.waiting
            and len(self.running) + len(scheduled_seqs) < self.max_num_seqs
        ):
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = (
                    self.block_manager.get_num_cached_blocks(seq)
                    if self.prefix_cache_enabled
                    else 0
                )
                num_cached_blocks = self.block_manager.can_allocate(
                    seq,
                    num_cached_blocks=num_cached_blocks,
                )
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def schedule_dynamic_chunked_prefill(self) -> ScheduleResult:
        # Final serving path:
        # 1. Decode first to protect TPOT/ITL for already-running requests.
        # 2. Spend the remaining token and sequence budget on prefill chunks.
        # 3. Return both groups so ModelRunner can execute one mixed forward.
        decode_seqs = self.schedule_decode_first()
        prefill_budget = self.max_num_batched_tokens - len(decode_seqs)
        # ``self.running`` still contains decode requests that did not fit the
        # current token budget. They continue to own KV/state slots and must be
        # counted before admitting new prefill requests.
        active_decode_seqs = len(self.running) + len(decode_seqs)
        prefill_slots = max(self.max_num_seqs - active_decode_seqs, 0)
        prefill_seqs = self.schedule_prefill_with_budget(prefill_budget, prefill_slots)
        if decode_seqs:
            # Rotate decoded requests to the back. Using extendleft here
            # repeatedly selected the same queue head when the token budget
            # was smaller than the number of running requests.
            self.running.extend(decode_seqs)
        if not decode_seqs and not prefill_seqs:
            if self.waiting:
                seq = self.waiting[0]
                raise RuntimeError(
                    "unable to schedule waiting request: insufficient free KV "
                    f"blocks for prompt length {len(seq)}"
                )
            raise RuntimeError("scheduler has no runnable sequence")
        return ScheduleResult(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

    def schedule_decode_first(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs and len(scheduled_seqs) < self.max_num_batched_tokens:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        return scheduled_seqs

    def schedule_prefill_with_budget(self, token_budget: int, seq_budget: int) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < seq_budget:
            remaining = token_budget - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = (
                    self.block_manager.get_num_cached_blocks(seq)
                    if self.prefix_cache_enabled
                    else 0
                )
                seq.num_cached_tokens = num_cached_blocks * self.block_size
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                if num_tokens <= 0:
                    # A complete prefix-cache hit still needs to bind the
                    # cached blocks to this sequence. Otherwise the next
                    # decode step would access an empty block table.
                    if (
                        self.block_manager.can_allocate(
                            seq,
                            num_blocks=seq.num_blocks,
                            num_cached_blocks=num_cached_blocks,
                        )
                        == -1
                    ):
                        break
                    self.block_manager.allocate(
                        seq,
                        num_cached_blocks,
                        num_blocks=seq.num_blocks,
                    )
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                    continue
                scheduled_tokens = min(num_tokens, remaining)
                target_blocks = (
                    seq.num_cached_tokens + scheduled_tokens + self.block_size - 1
                ) // self.block_size
                if self.block_manager.can_allocate(
                    seq,
                    num_blocks=target_blocks,
                    num_cached_blocks=num_cached_blocks,
                ) == -1:
                    break
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                    num_blocks=target_blocks,
                )
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
                if num_tokens <= 0:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                    continue
                scheduled_tokens = min(num_tokens, remaining)
                target_blocks = (
                    seq.num_cached_tokens + scheduled_tokens + self.block_size - 1
                ) // self.block_size
                if not self.block_manager.can_grow(seq, target_blocks):
                    break
                self.block_manager.grow(seq, target_blocks)

            # Recompute after allocation because a prefix hit may have
            # advanced num_cached_tokens.
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            if num_tokens <= 0:
                # The prompt is fully covered by prefix cache. It does not need
                # a prefill chunk; it can enter RUNNING and decode next step.
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                continue
            # Dynamic chunked prefill allows every scheduled waiting request to
            # consume only the remaining token budget instead of forcing the
            # whole prompt into this step.
            seq.is_prefill = True
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            target_blocks = (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
                + self.block_size
                - 1
            ) // self.block_size
            if not self.block_manager.can_grow(seq, target_blocks):
                # The initial allocation above already covers this target;
                # this branch only protects unusual state mutations.
                raise RuntimeError("insufficient KV blocks for prefill chunk")
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
            seq.state_slot = None
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        self._validate_sample_count(seqs, token_ids)
        for seq, token_id in zip(seqs, token_ids):
            self.postprocess_one(seq, token_id, is_prefill)

    def postprocess_mixed(self, result: ScheduleResult, token_ids: list[int]):
        seqs = result.decode_seqs + result.prefill_seqs
        self._validate_sample_count(seqs, token_ids)
        for seq, token_id in zip(seqs, token_ids):
            self.postprocess_one(seq, token_id, seq.is_prefill)

    @staticmethod
    def _validate_sample_count(
        seqs: list[Sequence],
        token_ids: list[int],
    ) -> None:
        if len(token_ids) != len(seqs):
            raise RuntimeError(
                "sampler returned an unexpected number of tokens: "
                f"expected {len(seqs)}, got {len(token_ids)}"
            )

    def postprocess_one(self, seq: Sequence, token_id: int, is_prefill: bool):
        if self.prefix_cache_enabled:
            self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            return
        if seq.num_completion_tokens == 0:
            seq.first_token_time = perf_counter()
        seq.append_token(token_id)
        if (
            not seq.ignore_eos and token_id in self.eos_token_ids
        ) or seq.num_completion_tokens == seq.max_tokens:
            seq.finish_time = perf_counter()
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            if self.state_manager is not None:
                self.state_manager.release(seq.seq_id)
                seq.state_slot = None
            self.running.remove(seq)
