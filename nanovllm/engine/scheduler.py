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
        self.prefill_starvation_threshold = getattr(
            config,
            "prefill_starvation_threshold",
            0,
        )
        self.prefill_starvation_token_budget = getattr(
            config,
            "prefill_starvation_token_budget",
            1,
        )
        self.preemption_policy = getattr(config, "preemption_policy", "fcfs")
        self.enable_decode_kv_reservation = getattr(
            config,
            "enable_decode_kv_reservation",
            False,
        )
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
        self.remote_prefills: dict[str, tuple[Sequence, object]] = {}
        self.remote_prefill_sources: dict[str, tuple[Sequence, int]] = {}
        self.schedule_steps = 0
        self.prefill_starved_steps = 0
        self.current_prefill_starvation_steps = 0
        self.max_prefill_starvation_steps = 0
        self.preemption_count = 0
        self.waiting_prefill_preemptions = 0
        self.preempted_token_progress = 0
        self.max_preempted_token_progress = 0
        self.reclaimed_kv_blocks = 0
        self.aborted_requests = 0
        self.last_prefill_stop_reason: str | None = None
        self.prefill_stopped_by_token_budget = 0
        self.prefill_stopped_by_sequence_capacity = 0
        self.prefill_stopped_by_kv_capacity = 0
        self.prefill_stopped_by_decode_kv_reservation = 0
        self.last_decode_kv_reserve_blocks = 0

    def _mark_prefill_stop(self, reason: str) -> None:
        """Record the first admission boundary reached in this step."""

        if self.last_prefill_stop_reason is not None:
            return
        if reason == "token_budget":
            self.prefill_stopped_by_token_budget += 1
        elif reason == "sequence_capacity":
            self.prefill_stopped_by_sequence_capacity += 1
        elif reason == "kv_capacity":
            self.prefill_stopped_by_kv_capacity += 1
        elif reason == "decode_kv_reservation":
            self.prefill_stopped_by_decode_kv_reservation += 1
        else:
            raise ValueError(f"unknown prefill stop reason: {reason}")
        self.last_prefill_stop_reason = reason

    def _mark_kv_stop(
        self,
        reserve_free_blocks: int,
        fits_without_reserve: bool,
    ) -> None:
        self._mark_prefill_stop(
            "decode_kv_reservation"
            if reserve_free_blocks and fits_without_reserve
            else "kv_capacity"
        )

    def capacity_snapshot(self) -> dict[str, int | float | str | None]:
        """Return scheduler-owned capacity and admission-pressure state."""

        active_waiting = sum(
            bool(seq.block_table) or seq.state_slot is not None
            for seq in self.waiting
        )
        used_sequence_slots = (
            len(self.running)
            + len(self.remote_prefills)
            + len(self.remote_prefill_sources)
            + active_waiting
        )
        return {
            "sequence_slots_total": self.max_num_seqs,
            "sequence_slots_used": used_sequence_slots,
            "sequence_slots_free": max(self.max_num_seqs - used_sequence_slots, 0),
            "kv_blocks_total": self.block_manager.num_total_blocks,
            "kv_blocks_used": self.block_manager.num_used_blocks,
            "kv_blocks_free": self.block_manager.num_free_blocks,
            "kv_block_usage": self.block_manager.usage,
            "waiting_requests": self.num_waiting,
            "running_requests": self.num_running,
            "last_prefill_stop_reason": self.last_prefill_stop_reason,
            "prefill_stopped_by_token_budget": self.prefill_stopped_by_token_budget,
            "prefill_stopped_by_sequence_capacity": (
                self.prefill_stopped_by_sequence_capacity
            ),
            "prefill_stopped_by_kv_capacity": self.prefill_stopped_by_kv_capacity,
            "prefill_stopped_by_decode_kv_reservation": (
                self.prefill_stopped_by_decode_kv_reservation
            ),
            "decode_kv_reserve_blocks": self.last_decode_kv_reserve_blocks,
        }

    def _next_decode_block_demand(
        self,
        scheduled: list[Sequence],
        pending: deque[Sequence],
    ) -> int:
        """Count blocks required to keep active decode requests runnable."""

        if not self.enable_decode_kv_reservation:
            return 0
        next_step_demand = sum(
            len(seq) % self.block_size == 0
            and seq.num_completion_tokens + 1 < seq.max_tokens
            for seq in scheduled
        )
        pending_demand = sum(
            len(seq) % self.block_size == 1
            for seq in pending
        )
        return next_step_demand + pending_demand

    def is_finished(self):
        return (
            not self.waiting
            and not self.running
            and not self.remote_prefills
            and not self.remote_prefill_sources
        )

    @property
    def num_waiting(self):
        return len(self.waiting) + len(self.remote_prefills)

    @property
    def num_running(self):
        return len(self.running) + len(self.remote_prefill_sources)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        """Cancel one local request and release all scheduler-owned state.

        Remote-prefill requests have rank-wide transfer state and must use the
        corresponding transfer cancellation API instead. Unknown or already
        finished requests are intentionally idempotent for disconnect races.
        """

        if not isinstance(seq_id, int) or isinstance(seq_id, bool):
            raise TypeError("sequence id must be an integer")
        if any(
            seq.seq_id == seq_id
            for seq, _session in self.remote_prefills.values()
        ) or any(
            seq.seq_id == seq_id
            for seq, _position in self.remote_prefill_sources.values()
        ):
            raise RuntimeError(
                "remote-prefill requests must use their transfer cancellation API"
            )

        owner = None
        seq = None
        for queue in (self.waiting, self.running):
            seq = next(
                (candidate for candidate in queue if candidate.seq_id == seq_id),
                None,
            )
            if seq is not None:
                owner = queue
                break
        if seq is None:
            return False

        owner.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
        seq.state_slot = None
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.FINISHED
        seq.finish_time = perf_counter()
        self.aborted_requests += 1
        return True

    def reserve_remote_prefill(self, seq: Sequence, session) -> None:
        """Reserve destination cache/state without making ``seq`` runnable."""

        transfer_id = session.transfer_id
        if (
            transfer_id in self.remote_prefills
            or transfer_id in self.remote_prefill_sources
        ):
            raise ValueError("cache transfer id is already reserved")
        if seq not in self.waiting:
            raise ValueError("remote prefill sequence must be waiting")
        if seq.block_table or seq.state_slot is not None:
            raise ValueError("remote prefill sequence already owns cache state")
        if (
            len(self.running)
            + len(self.remote_prefills)
            + len(self.remote_prefill_sources)
            >= self.max_num_seqs
        ):
            raise RuntimeError("no sequence slot is available for remote prefill")
        target_blocks = (
            seq.num_prompt_tokens + self.block_size - 1
        ) // self.block_size
        if (
            self.block_manager.can_allocate(
                seq,
                num_blocks=target_blocks,
                num_cached_blocks=0,
            )
            == -1
        ):
            raise RuntimeError("insufficient KV blocks for remote prefill")

        state_slot = None
        if self.state_manager is not None:
            state_slot = self.state_manager.acquire(seq.seq_id)
        try:
            self.block_manager.allocate(
                seq,
                0,
                num_blocks=target_blocks,
            )
        except BaseException:
            if self.state_manager is not None:
                self.state_manager.release(seq.seq_id)
            raise
        seq.state_slot = state_slot
        seq.status = SequenceStatus.TRANSFERRING
        self.waiting.remove(seq)
        self.remote_prefills[transfer_id] = (seq, session)

    def _release_remote_prefill(
        self,
        transfer_id: str,
        *,
        fallback: bool,
    ) -> Sequence:
        seq, _ = self.remote_prefills.pop(transfer_id)
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
        seq.state_slot = None
        seq.status = (
            SequenceStatus.WAITING if fallback else SequenceStatus.FINISHED
        )
        seq.num_scheduled_tokens = 0
        if fallback:
            seq.is_prefill = True
            self.waiting.appendleft(seq)
        return seq

    def _rollback_remote_prefill(self, transfer_id: str) -> Sequence:
        return self._release_remote_prefill(transfer_id, fallback=True)

    def fail_remote_prefill(
        self,
        transfer_id: str,
        rank: int,
        reason: str,
        *,
        now: float,
    ) -> Sequence:
        _, session = self.remote_prefills[transfer_id]
        session.fail(rank, reason, now=now)
        return self._rollback_remote_prefill(transfer_id)

    def abort_remote_prefill(
        self,
        transfer_id: str,
        reason: str,
        *,
        now: float,
    ) -> Sequence:
        """Abort a coordinator operation, preserving an expired session reason."""

        _, session = self.remote_prefills[transfer_id]
        session.poll(now=now)
        if not session.fallback_required:
            session.fail(0, reason, now=now)
        return self._rollback_remote_prefill(transfer_id)

    def cancel_remote_prefill(
        self,
        transfer_id: str,
        reason: str,
        *,
        now: float,
    ) -> Sequence:
        """Release an unstarted destination reservation without local fallback."""

        _, session = self.remote_prefills[transfer_id]
        session.poll(now=now)
        if not session.fallback_required:
            session.fail(0, reason, now=now)
        return self._release_remote_prefill(transfer_id, fallback=False)

    def poll_remote_prefills(self, *, now: float) -> list[Sequence]:
        fallback = []
        for transfer_id, (_, session) in tuple(self.remote_prefills.items()):
            session.poll(now=now)
            if session.fallback_required:
                fallback.append(self._rollback_remote_prefill(transfer_id))
        return fallback

    def commit_remote_prefill(
        self,
        transfer_id: str,
        first_token_id: int,
        *,
        now: float,
    ) -> Sequence:
        seq, session = self.remote_prefills[transfer_id]
        session.commit(now=now)
        self.remote_prefills.pop(transfer_id)
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.num_scheduled_tokens = 0
        seq.is_prefill = False
        seq.first_token_time = perf_counter()
        seq.append_token(first_token_id)
        if (
            not seq.ignore_eos and first_token_id in self.eos_token_ids
        ) or seq.num_completion_tokens == seq.max_tokens:
            seq.finish_time = perf_counter()
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            if self.state_manager is not None:
                self.state_manager.release(seq.seq_id)
                seq.state_slot = None
        else:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        return seq

    def complete_remote_prefill_source(self, seq: Sequence) -> None:
        """Release producer resources only after the decode side ACKs."""

        if seq not in self.running:
            raise ValueError("remote prefill source sequence must be running")
        if (
            seq.num_cached_tokens != seq.num_prompt_tokens
            or seq.num_completion_tokens != 1
            or seq.num_scheduled_tokens != 0
        ):
            raise ValueError("remote prefill source is not ready for handoff")
        self.running.remove(seq)
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
            seq.state_slot = None
        seq.status = SequenceStatus.TRANSFERRED

    def reserve_remote_prefill_source(
        self,
        seq: Sequence,
        transfer_id: str,
    ) -> None:
        """Pause a handoff source so decode cannot mutate it while sending."""

        if transfer_id in self.remote_prefill_sources:
            raise ValueError("cache transfer source id is already active")
        if transfer_id in self.remote_prefills:
            raise ValueError("cache transfer id is already reserved as a destination")
        if seq not in self.running:
            raise ValueError("remote prefill source sequence is not running")
        if (
            seq.num_cached_tokens != seq.num_prompt_tokens
            or seq.num_completion_tokens != 1
            or seq.num_scheduled_tokens != 0
        ):
            raise ValueError("remote prefill source is not ready for handoff")
        position = self.running.index(seq)
        for _pending_seq, pending_position in sorted(
            self.remote_prefill_sources.values(),
            key=lambda item: item[1],
        ):
            if pending_position <= position:
                position += 1
        self.running.remove(seq)
        seq.status = SequenceStatus.TRANSFERRING
        self.remote_prefill_sources[transfer_id] = (seq, position)

    def abort_remote_prefill_source(self, transfer_id: str) -> Sequence:
        seq, position = self.remote_prefill_sources.pop(transfer_id)
        seq.status = SequenceStatus.RUNNING
        pending_before = sum(
            pending_position < position
            for _pending_seq, pending_position in self.remote_prefill_sources.values()
        )
        running_position = position - pending_before
        self.running.insert(min(running_position, len(self.running)), seq)
        return seq

    def commit_remote_prefill_source(self, transfer_id: str) -> Sequence:
        seq, _position = self.remote_prefill_sources.pop(transfer_id)
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
            seq.state_slot = None
        seq.status = SequenceStatus.TRANSFERRED
        return seq

    def schedule(self) -> tuple[list[Sequence], bool] | ScheduleResult:
        self.last_prefill_stop_reason = None
        self.last_decode_kv_reserve_blocks = 0
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
        num_running = (
            len(self.running)
            + len(self.remote_prefills)
            + len(self.remote_prefill_sources)
        )

        # prefill
        while (
            self.waiting
            and num_running + len(scheduled_seqs) < self.max_num_seqs
        ):
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                self._mark_prefill_stop("token_budget")
                break
            if not seq.block_table:
                num_cached_blocks = (
                    self.block_manager.get_num_cached_blocks(seq)
                    if self.prefix_cache_enabled
                    else 0
                )
                cached_tokens = num_cached_blocks * self.block_size
                num_tokens = seq.num_tokens - cached_tokens
            else:
                cached_tokens = seq.num_cached_tokens
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if num_tokens <= 0:
                if not seq.block_table:
                    if (
                        self.block_manager.can_allocate(
                            seq,
                            num_blocks=seq.num_blocks,
                            num_cached_blocks=num_cached_blocks,
                        )
                        == -1
                    ):
                        self._mark_prefill_stop("kv_capacity")
                        break
                    self.block_manager.allocate(
                        seq,
                        num_cached_blocks,
                        num_blocks=seq.num_blocks,
                    )
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                num_running += 1
                continue
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                self._mark_prefill_stop("token_budget")
                break
            scheduled_tokens = min(num_tokens, remaining)
            target_blocks = (
                cached_tokens
                + scheduled_tokens
                + self.block_size
                - 1
            ) // self.block_size
            if not seq.block_table:
                target_blocks = max(target_blocks, num_cached_blocks)
                if (
                    self.block_manager.can_allocate(
                        seq,
                        num_blocks=target_blocks,
                        num_cached_blocks=num_cached_blocks,
                    )
                    == -1
                ):
                    self._mark_prefill_stop("kv_capacity")
                    break
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                    num_blocks=target_blocks,
                )
            elif not self.block_manager.can_grow(seq, target_blocks):
                self._mark_prefill_stop("kv_capacity")
                break
            else:
                self.block_manager.grow(seq, target_blocks)
            seq.num_scheduled_tokens = scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if (
            self.waiting
            and num_running + len(scheduled_seqs) >= self.max_num_seqs
        ):
            self._mark_prefill_stop("sequence_capacity")

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while (
            self.running
            and len(scheduled_seqs) < self.max_num_seqs
            and len(scheduled_seqs) < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                victim = self.select_preemption_victim(seq)
                self.preempt(victim)
                if victim is seq:
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            if self.remote_prefills or self.remote_prefill_sources:
                return [], False
            raise RuntimeError("scheduler has no runnable sequence")
        # Requests that did not fit this step's token budget go first next
        # time; append completed decode work behind them to prevent starvation.
        self.running.extend(scheduled_seqs)
        return scheduled_seqs, False

    def schedule_dynamic_chunked_prefill(self) -> ScheduleResult:
        # Final serving path:
        # 1. Decode first to protect TPOT/ITL for already-running requests.
        # 2. Spend the remaining token and sequence budget on prefill chunks.
        # 3. Return both groups so ModelRunner can execute one mixed forward.
        decode_budget = self.max_num_batched_tokens
        active_waiting = sum(
            seq.state_slot is not None or bool(seq.block_table)
            for seq in self.waiting
        )
        waiting_head_is_active = bool(self.waiting) and (
            self.waiting[0].state_slot is not None
            or bool(self.waiting[0].block_table)
        )
        has_prefill_slot = (
            len(self.running)
            + len(self.remote_prefills)
            + len(self.remote_prefill_sources)
            + active_waiting
            < self.max_num_seqs
            or waiting_head_is_active
        )
        if (
            self.waiting
            and has_prefill_slot
            and self.prefill_starvation_threshold > 0
            and self.current_prefill_starvation_steps
            >= self.prefill_starvation_threshold
        ):
            # Reserve a useful prefill chunk instead of merely one token, but
            # retain at least half of a multi-token step for decode latency.
            reserve_cap = max(1, self.max_num_batched_tokens // 2)
            head = self.waiting[0]
            cached_tokens = head.num_cached_tokens
            if self.prefix_cache_enabled and not head.block_table:
                cached_tokens = (
                    self.block_manager.peek_num_cached_blocks(head)
                    * self.block_size
                )
            head_remaining_tokens = max(
                head.num_tokens - cached_tokens,
                0,
            )
            prefill_reserve = min(
                self.prefill_starvation_token_budget,
                reserve_cap,
                head_remaining_tokens,
            )
            decode_budget = max(decode_budget - prefill_reserve, 0)
        decode_seqs = self.schedule_decode_first(decode_budget)
        decode_kv_reserve_blocks = self._next_decode_block_demand(
            decode_seqs,
            self.running,
        )
        self.last_decode_kv_reserve_blocks = decode_kv_reserve_blocks
        prefill_budget = self.max_num_batched_tokens - len(decode_seqs)
        # ``self.running`` still contains decode requests that did not fit the
        # current token budget. They continue to own KV/state slots and must be
        # counted before admitting new prefill requests.
        active_decode_seqs = (
            len(self.running)
            + len(decode_seqs)
            + len(self.remote_prefills)
            + len(self.remote_prefill_sources)
        )
        active_waiting_seqs = sum(
            seq.state_slot is not None or bool(seq.block_table)
            for seq in self.waiting
        )
        prefill_batch_slots = max(
            self.max_num_seqs - len(decode_seqs),
            0,
        )
        prefill_admission_slots = max(
            self.max_num_seqs
            - active_decode_seqs
            - active_waiting_seqs,
            0,
        )
        running_before_prefill = len(self.running)
        prefill_seqs = self.schedule_prefill_with_budget(
            prefill_budget,
            prefill_batch_slots,
            admission_budget=prefill_admission_slots,
            reserve_free_blocks=decode_kv_reserve_blocks,
        )
        # Remove requests admitted by this prefill step before considering
        # decode backfill. They have not been postprocessed yet and therefore
        # must never be selected for decode in the same scheduler step.
        newly_admitted = []
        while len(self.running) > running_before_prefill:
            newly_admitted.append(self.running.pop())
        newly_admitted.reverse()
        unused_tokens = (
            self.max_num_batched_tokens
            - len(decode_seqs)
            - sum(seq.num_scheduled_tokens for seq in prefill_seqs)
        )
        if unused_tokens > 0 and self.running:
            # A fairness reserve can go unused when prefill admission fails
            # under KV pressure. Reclaim only that unused budget for older
            # runnable decode requests instead of returning a partial batch.
            # If prefill made no progress at all, preemption is required to
            # avoid a deadlock where boundary decode requests and the waiting
            # partial prefill collectively own every KV block.
            decode_seqs.extend(
                self.schedule_decode_first(
                    unused_tokens,
                    allow_preemption=not prefill_seqs,
                )
            )
        if decode_seqs:
            # Rotate decoded requests behind unscheduled running requests, but
            # keep them ahead of requests admitted by prefill in this step.
            # Otherwise a newly admitted request jumps the FCFS queue and may
            # cause an older request at the tail to be selected for preemption.
            self.running.extend(decode_seqs)
        self.running.extend(newly_admitted)
        if not decode_seqs and not prefill_seqs:
            if self.remote_prefills or self.remote_prefill_sources:
                return ScheduleResult(prefill_seqs=[], decode_seqs=[])
            if self.waiting:
                seq = self.waiting[0]
                raise RuntimeError(
                    "unable to schedule waiting request: insufficient free KV "
                    f"blocks for prompt length {len(seq)}"
                )
            raise RuntimeError("scheduler has no runnable sequence")
        return ScheduleResult(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

    def schedule_decode_first(
        self,
        token_budget: int | None = None,
        *,
        allow_preemption: bool = True,
    ) -> list[Sequence]:
        if token_budget is None:
            token_budget = self.max_num_batched_tokens
        if token_budget < 0:
            raise ValueError("decode token budget must be non-negative")
        scheduled_seqs = []
        while (
            self.running
            and len(scheduled_seqs) < self.max_num_seqs
            and len(scheduled_seqs) < token_budget
        ):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if not allow_preemption:
                    self.running.appendleft(seq)
                    return scheduled_seqs
                victim = self.select_preemption_victim(seq)
                self.preempt(victim)
                if victim is seq:
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        return scheduled_seqs

    def select_preemption_victim(self, current: Sequence) -> Sequence:
        """Remove and return the request to evict under the configured policy."""

        active_waiting = [seq for seq in self.waiting if seq.block_table]
        if self.preemption_policy == "fcfs":
            victim = max(
                (current, *self.running, *active_waiting),
                key=lambda seq: seq.seq_id,
            )
        elif self.preemption_policy == "min_recompute":
            victim = min(
                (current, *self.running, *active_waiting),
                key=lambda seq: (
                    seq.num_cached_tokens,
                    len(seq.block_table),
                    -seq.seq_id,
                ),
            )
        else:
            raise RuntimeError(
                f"unsupported preemption policy: {self.preemption_policy}"
            )
        if victim is current:
            return victim
        if victim in self.running:
            self.running.remove(victim)
        else:
            self.waiting.remove(victim)
        return victim

    def _reclaim_waiting_prefill(self, current: Sequence) -> bool:
        """Release a different partial prefill that blocks the queue head.

        Chunked prefills retain their KV blocks while waiting for another
        chunk. Under pressure, several such requests can consume every free
        block while no decode request remains available to trigger the normal
        preemption path. Preserve the queue head and reclaim one other owner so
        the oldest request can make forward progress.
        """

        candidates = [
            seq
            for seq in self.waiting
            if seq is not current and bool(seq.block_table)
        ]
        if not candidates:
            return False
        if self.preemption_policy == "fcfs":
            victim = max(candidates, key=lambda seq: seq.seq_id)
        elif self.preemption_policy == "min_recompute":
            victim = min(
                candidates,
                key=lambda seq: (
                    seq.num_cached_tokens,
                    len(seq.block_table),
                    -seq.seq_id,
                ),
            )
        else:
            raise RuntimeError(
                f"unsupported preemption policy: {self.preemption_policy}"
            )
        self.waiting.remove(victim)
        self.preempt(victim)
        self.waiting_prefill_preemptions += 1
        self.waiting.remove(current)
        self.waiting.appendleft(current)
        return True

    def schedule_prefill_with_budget(
        self,
        token_budget: int,
        seq_budget: int,
        *,
        admission_budget: int | None = None,
        reserve_free_blocks: int = 0,
    ) -> list[Sequence]:
        if admission_budget is None:
            admission_budget = seq_budget
        if min(token_budget, seq_budget, admission_budget) < 0:
            raise ValueError("prefill budgets must be non-negative")
        if reserve_free_blocks < 0:
            raise ValueError("reserve_free_blocks must be non-negative")
        scheduled_seqs = []
        num_batched_tokens = 0
        admitted_seqs = 0
        while self.waiting and len(scheduled_seqs) < seq_budget:
            remaining = token_budget - num_batched_tokens
            if remaining <= 0:
                self._mark_prefill_stop("token_budget")
                break
            seq = self.waiting[0]
            owns_slot = seq.state_slot is not None or bool(seq.block_table)
            if not owns_slot and admitted_seqs >= admission_budget:
                self._mark_prefill_stop("sequence_capacity")
                break
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
                            reserve_free_blocks=reserve_free_blocks,
                        )
                        == -1
                    ):
                        if self._reclaim_waiting_prefill(seq):
                            continue
                        self._mark_kv_stop(
                            reserve_free_blocks,
                            self.block_manager.can_allocate(
                                seq,
                                num_blocks=seq.num_blocks,
                                num_cached_blocks=num_cached_blocks,
                            )
                            != -1,
                        )
                        break
                    self.block_manager.allocate(
                        seq,
                        num_cached_blocks,
                        num_blocks=seq.num_blocks,
                    )
                    if not owns_slot:
                        admitted_seqs += 1
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
                    reserve_free_blocks=reserve_free_blocks,
                ) == -1:
                    if self._reclaim_waiting_prefill(seq):
                        continue
                    self._mark_kv_stop(
                        reserve_free_blocks,
                        self.block_manager.can_allocate(
                            seq,
                            num_blocks=target_blocks,
                            num_cached_blocks=num_cached_blocks,
                        )
                        != -1,
                    )
                    break
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                    num_blocks=target_blocks,
                )
                if not owns_slot:
                    admitted_seqs += 1
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
                if not self.block_manager.can_grow(
                    seq,
                    target_blocks,
                    reserve_free_blocks=reserve_free_blocks,
                ):
                    if self._reclaim_waiting_prefill(seq):
                        continue
                    self._mark_kv_stop(
                        reserve_free_blocks,
                        self.block_manager.can_grow(seq, target_blocks),
                    )
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
        if self.waiting and self.last_prefill_stop_reason is None:
            if token_budget - num_batched_tokens <= 0:
                self._mark_prefill_stop("token_budget")
            elif len(scheduled_seqs) >= seq_budget:
                self._mark_prefill_stop("sequence_capacity")
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        discarded_tokens = seq.num_cached_tokens
        used_blocks_before = self.block_manager.num_used_blocks
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        reclaimed_blocks = used_blocks_before - self.block_manager.num_used_blocks
        if self.state_manager is not None:
            self.state_manager.release(seq.seq_id)
            seq.state_slot = None
        self.waiting.appendleft(seq)
        self.preemption_count += 1
        self.preempted_token_progress += discarded_tokens
        self.max_preempted_token_progress = max(
            self.max_preempted_token_progress,
            discarded_tokens,
        )
        self.reclaimed_kv_blocks += reclaimed_blocks

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
