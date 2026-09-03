import atexit
from dataclasses import fields
import os
import socket
import uuid
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config, resolve_eos_token_ids
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import ScheduleResult, Scheduler
from nanovllm.engine.cache_transfer import CacheTransferPhase, CacheTransferSession
from nanovllm.engine.model_runner import CONTROL_STATUS_SIZE, ModelRunner
from nanovllm.engine.metrics import EngineMetrics
from nanovllm.engine.remote_prefill_router import RemotePrefillDemand


def _find_free_port() -> int:
    """Ask the OS for an unused local TCP port for the NCCL rendezvous."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_rank_results(
    results: list[object],
    tensor_parallel_size: int,
    value_name: str,
    *,
    expected_value: int | None = None,
) -> dict[int, int]:
    """Validate one small completion record from every tensor-parallel rank."""

    by_rank = {}
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("tensor-parallel rank result must be a dictionary")
        rank = result.get("rank")
        value = result.get(value_name)
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < tensor_parallel_size
        ):
            raise RuntimeError("tensor-parallel rank result has an invalid rank")
        if rank in by_rank:
            raise RuntimeError("tensor-parallel rank result is duplicated")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                f"tensor-parallel rank result has invalid {value_name}"
            )
        if expected_value is not None and value != expected_value:
            raise RuntimeError(
                f"tensor-parallel rank {rank} reported {value_name}={value}; "
                f"expected {expected_value}"
            )
        by_rank[rank] = value
    expected_ranks = set(range(tensor_parallel_size))
    if set(by_rank) != expected_ranks:
        raise RuntimeError("tensor-parallel rank results are incomplete")
    return by_rank


def _validate_rank_receive_polls(
    results: list[object],
    tensor_parallel_size: int,
    transfer_ids: tuple[str, ...],
) -> dict[str, list[dict]]:
    """Validate one batched receive-state snapshot from every TP rank."""

    expected_ids = set(transfer_ids)
    if len(expected_ids) != len(transfer_ids):
        raise RuntimeError("cache receive batch ids must be unique")
    by_transfer = {transfer_id: [] for transfer_id in transfer_ids}
    seen_ranks = set()
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("cache receive batch result must be a dictionary")
        rank = result.get("rank")
        receives = result.get("receives")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < tensor_parallel_size
            or rank in seen_ranks
        ):
            raise RuntimeError("cache receive batch result has an invalid rank")
        if not isinstance(receives, dict) or set(receives) != expected_ids:
            raise RuntimeError("cache receive batch result has incomplete ids")
        seen_ranks.add(rank)
        for transfer_id, status in receives.items():
            if (
                not isinstance(status, dict)
                or not set(status).issubset({"state", "error"})
            ):
                raise RuntimeError("cache receive batch status must be a dictionary")
            by_transfer[transfer_id].append({"rank": rank, **status})
    if seen_ranks != set(range(tensor_parallel_size)):
        raise RuntimeError("cache receive batch results are incomplete")
    return by_transfer


def _validate_rank_send_polls(
    results: list[object],
    tensor_parallel_size: int,
    transfer_ids: tuple[str, ...],
) -> dict[str, list[dict]]:
    """Validate one batched send-state snapshot from every TP rank."""

    expected_ids = set(transfer_ids)
    if len(expected_ids) != len(transfer_ids):
        raise RuntimeError("cache send batch ids must be unique")
    by_transfer = {transfer_id: [] for transfer_id in transfer_ids}
    seen_ranks = set()
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("cache send batch result must be a dictionary")
        rank = result.get("rank")
        sends = result.get("sends")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < tensor_parallel_size
            or rank in seen_ranks
        ):
            raise RuntimeError("cache send batch result has an invalid rank")
        if not isinstance(sends, dict) or set(sends) != expected_ids:
            raise RuntimeError("cache send batch result has incomplete ids")
        seen_ranks.add(rank)
        for transfer_id, status in sends.items():
            if (
                not isinstance(status, dict)
                or not set(status).issubset({"state", "error", "staged_bytes"})
            ):
                raise RuntimeError("cache send batch status must be a dictionary")
            by_transfer[transfer_id].append({"rank": rank, **status})
    if seen_ranks != set(range(tensor_parallel_size)):
        raise RuntimeError("cache send batch results are incomplete")
    return by_transfer


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        if config.distributed_port is None:
            config.distributed_port = _find_free_port()
        if config.tensor_parallel_size > 1:
            if config.shared_memory_name is None:
                config.shared_memory_name = (
                    f"nanovllm_{os.getpid()}_{uuid.uuid4().hex[:12]}"
                )
        self.config = config
        self._exited = False
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            command_event = ctx.Event()
            ack_event = ctx.Event()
            status_buffer = ctx.Array(
                "B",
                CONTROL_STATUS_SIZE,
                lock=False,
            )
            process = ctx.Process(
                target=ModelRunner,
                args=(
                    config,
                    i,
                    (command_event, ack_event, status_buffer),
                ),
            )
            process.start()
            self.ps.append(process)
            self.events.append((command_event, ack_event, status_buffer))
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = resolve_eos_token_ids(
            config.model,
            self.tokenizer.eos_token_id,
        )
        self.scheduler = Scheduler(config)
        self.metrics = EngineMetrics()
        self._remote_prefill_receive_tokens: dict[str, int] = {}
        self._remote_prefill_receive_started_at: dict[str, float] = {}
        self._remote_prefill_receive_staged_bytes: dict[str, int] = {}
        self._remote_prefill_receive_expected_bytes: dict[str, dict[int, int]] = {}
        self._remote_prefill_receive_errors: dict[str, str] = {}
        self._remote_prefill_send_started_at: dict[str, float] = {}
        self._remote_prefill_send_staged_bytes: dict[str, int] = {}
        self._remote_prefill_send_errors: dict[str, str] = {}
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        runner = getattr(self, "model_runner", None)
        shutdown_error = None
        if runner is not None:
            try:
                runner.call("exit")
            except BaseException as exc:
                shutdown_error = exc
                # A failed worker must not leave rank-0 waiting forever in
                # an atexit handler. Abort only performs local cleanup; the
                # worker processes are joined/terminated below.
                try:
                    runner.abort()
                except BaseException:
                    pass
            finally:
                del self.model_runner

        for process in self.ps:
            process.join(timeout=5.0)
        for process in self.ps:
            if process.is_alive():
                process.terminate()
        for process in self.ps:
            process.join(timeout=5.0)

        if shutdown_error is not None:
            raise shutdown_error

    def _create_sequence(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, (list, tuple)):
            raise TypeError("prompt must be a string or a token list")
        prompt = list(prompt)
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a SamplingParams instance")
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if len(prompt) > self.config.max_model_len:
            raise ValueError(
                f"prompt length {len(prompt)} exceeds max_model_len "
                f"{self.config.max_model_len}"
            )
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError(
                "prompt length plus max_tokens exceeds max_model_len: "
                f"{len(prompt)} + {sampling_params.max_tokens} > "
                f"{self.config.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        return seq

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        seq = self._create_sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def add_remote_prefill_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        transfer_id: str,
        timeout_s: float = 30.0,
    ) -> int:
        """Reserve decode-side cache/state for a remote prefill request."""

        seq = self._create_sequence(prompt, sampling_params)
        session = CacheTransferSession(
            transfer_id,
            self.config.tensor_parallel_size,
            started_at=perf_counter(),
            timeout_s=timeout_s,
        )
        self.scheduler.add(seq)
        try:
            self.scheduler.reserve_remote_prefill(seq, session)
        except BaseException:
            if seq in self.scheduler.waiting:
                self.scheduler.waiting.remove(seq)
            raise
        return seq.seq_id

    def receive_remote_prefill(
        self,
        transfer_id: str,
        first_token_id: int,
        bind_endpoints: list[tuple[str, int]],
        *,
        timeout_s: float = 30.0,
        max_payload_bytes: int = 16 * 1024**3,
    ) -> int:
        """Receive every TP rank, then atomically admit the request to decode."""

        if not isinstance(first_token_id, int) or isinstance(first_token_id, bool):
            raise TypeError("first_token_id must be an integer")
        if transfer_id in getattr(self, "_remote_prefill_receive_tokens", {}):
            raise ValueError("cache receive id is already active")
        seq, session = self.scheduler.remote_prefills[transfer_id]
        estimates = self.model_runner.call_rank_results(
            "estimate_sequence_cache_bytes",
            seq,
        )
        expected_by_rank = _validate_rank_results(
            estimates,
            self.config.tensor_parallel_size,
            "staged_bytes",
        )
        if any(expected > max_payload_bytes for expected in expected_by_rank.values()):
            raise ValueError(
                "expected cache receive payload exceeds max_payload_bytes"
            )
        staged_bytes = sum(expected_by_rank.values())
        self._ensure_remote_prefill_staging_capacity(
            staged_bytes,
            direction="receive",
        )
        receive_staged_bytes = getattr(
            self,
            "_remote_prefill_receive_staged_bytes",
            None,
        )
        if receive_staged_bytes is None:
            receive_staged_bytes = self._remote_prefill_receive_staged_bytes = {}
        receive_staged_bytes[transfer_id] = staged_bytes
        receive_started_at = perf_counter()
        self.metrics.record_remote_prefill_receive_started(staged_bytes)
        try:
            rank_results = self.model_runner.call_rank_results(
                "receive_sequence_cache_from_endpoint",
                seq,
                transfer_id,
                bind_endpoints,
                timeout_s,
                max_payload_bytes,
                [expected_by_rank[rank] for rank in range(self.config.tensor_parallel_size)],
            )
            received = _validate_rank_results(
                rank_results,
                self.config.tensor_parallel_size,
                "cached_tokens",
                expected_value=seq.num_prompt_tokens,
            )
            received_bytes = _validate_rank_results(
                rank_results,
                self.config.tensor_parallel_size,
                "received_bytes",
            )
            if received_bytes != expected_by_rank:
                raise RuntimeError(
                    "cache receive payload bytes differ from the preflight estimate"
                )
            now = perf_counter()
            for rank in received:
                session.acknowledge(rank, now=now)
            self.scheduler.commit_remote_prefill(
                transfer_id,
                first_token_id,
                now=now,
            )
        except BaseException as exc:
            if transfer_id in self.scheduler.remote_prefills:
                self.scheduler.abort_remote_prefill(
                    transfer_id,
                    f"rank-local cache receive failed: {exc}",
                    now=perf_counter(),
                )
            outcome = (
                "timed_out"
                if session.phase is CacheTransferPhase.TIMED_OUT
                else "failed"
            )
            self.metrics.record_remote_prefill_receive_finished(
                perf_counter() - receive_started_at,
                outcome=outcome,
                staged_bytes=receive_staged_bytes.pop(transfer_id),
            )
            raise
        self.metrics.record_remote_prefill_receive_finished(
            perf_counter() - receive_started_at,
            outcome="committed",
            staged_bytes=receive_staged_bytes.pop(transfer_id),
        )
        return seq.seq_id

    def start_remote_prefill_receive(
        self,
        transfer_id: str,
        first_token_id: int,
        bind_endpoints: list[tuple[str, int]],
        *,
        timeout_s: float = 30.0,
        max_payload_bytes: int = 16 * 1024**3,
    ) -> int:
        """Start CPU-side rank receives while decode scheduling continues."""

        if not isinstance(first_token_id, int) or isinstance(first_token_id, bool):
            raise TypeError("first_token_id must be an integer")
        if transfer_id not in self.scheduler.remote_prefills:
            raise ValueError("cache transfer id is not reserved")
        receive_tokens = getattr(self, "_remote_prefill_receive_tokens", None)
        if receive_tokens is None:
            receive_tokens = self._remote_prefill_receive_tokens = {}
        if transfer_id in receive_tokens:
            raise ValueError("cache receive id is already active")
        self._ensure_remote_prefill_transfer_capacity(direction="receive")
        seq, _session = self.scheduler.remote_prefills[transfer_id]
        estimates = self.model_runner.call_rank_results(
            "estimate_sequence_cache_bytes",
            seq,
        )
        expected_by_rank = _validate_rank_results(
            estimates,
            self.config.tensor_parallel_size,
            "staged_bytes",
        )
        if any(expected > max_payload_bytes for expected in expected_by_rank.values()):
            raise ValueError(
                "expected cache receive payload exceeds max_payload_bytes"
            )
        staged_bytes = sum(expected_by_rank.values())
        self._ensure_remote_prefill_staging_capacity(
            staged_bytes,
            direction="receive",
        )
        receive_started_at = perf_counter()
        try:
            rank_results = self.model_runner.call_rank_results(
                "start_sequence_cache_receive",
                transfer_id,
                bind_endpoints,
                timeout_s,
                max_payload_bytes,
                [expected_by_rank[rank] for rank in range(self.config.tensor_parallel_size)],
            )
            _validate_rank_results(
                rank_results,
                self.config.tensor_parallel_size,
                "started",
                expected_value=1,
            )
        except BaseException as exc:
            try:
                self.model_runner.call_rank_results(
                    "abort_sequence_cache_receive",
                    transfer_id,
                )
            except BaseException:
                pass
            if transfer_id in self.scheduler.remote_prefills:
                self.scheduler.abort_remote_prefill(
                    transfer_id,
                    f"cache receive start failed: {exc}",
                    now=perf_counter(),
                )
            raise
        receive_tokens[transfer_id] = first_token_id
        started_at = getattr(self, "_remote_prefill_receive_started_at", None)
        if started_at is None:
            started_at = self._remote_prefill_receive_started_at = {}
        started_at[transfer_id] = receive_started_at
        receive_staged_bytes = getattr(
            self,
            "_remote_prefill_receive_staged_bytes",
            None,
        )
        if receive_staged_bytes is None:
            receive_staged_bytes = self._remote_prefill_receive_staged_bytes = {}
        receive_staged_bytes[transfer_id] = staged_bytes
        expected_bytes = getattr(
            self,
            "_remote_prefill_receive_expected_bytes",
            None,
        )
        if expected_bytes is None:
            expected_bytes = self._remote_prefill_receive_expected_bytes = {}
        expected_bytes[transfer_id] = expected_by_rank
        self.metrics.record_remote_prefill_receive_started(staged_bytes)
        getattr(self, "_remote_prefill_receive_errors", {}).pop(transfer_id, None)
        return seq.seq_id

    def cancel_remote_prefill_reservation(
        self,
        transfer_id: str,
        *,
        reason: str = "remote prefill destination rejected",
    ) -> int:
        """Release an unstarted destination so a router can try another node."""

        if transfer_id not in self.scheduler.remote_prefills:
            raise ValueError("cache transfer id is not reserved")
        if transfer_id in self._remote_prefill_receive_tokens:
            raise RuntimeError(
                "cache receive is active; abort it before releasing the request"
            )
        if not reason:
            raise ValueError("cache reservation cancellation reason must not be empty")
        seq = self.scheduler.cancel_remote_prefill(
            transfer_id,
            reason,
            now=perf_counter(),
        )
        return seq.seq_id

    def _abort_remote_prefill_receive(
        self,
        transfer_id: str,
        reason: str,
        *,
        outcome: str = "failed",
    ) -> None:
        try:
            self.model_runner.call_rank_results(
                "abort_sequence_cache_receive",
                transfer_id,
            )
        except BaseException:
            pass
        self._remote_prefill_receive_tokens.pop(transfer_id, None)
        staged_bytes = getattr(
            self,
            "_remote_prefill_receive_staged_bytes",
            {},
        ).pop(transfer_id, 0)
        getattr(
            self,
            "_remote_prefill_receive_expected_bytes",
            {},
        ).pop(transfer_id, None)
        started_at = getattr(self, "_remote_prefill_receive_started_at", {}).pop(
            transfer_id,
            None,
        )
        if started_at is not None:
            self.metrics.record_remote_prefill_receive_finished(
                perf_counter() - started_at,
                outcome=outcome,
                staged_bytes=staged_bytes,
            )
        if transfer_id in self.scheduler.remote_prefills:
            self.scheduler.abort_remote_prefill(
                transfer_id,
                reason,
                now=perf_counter(),
            )

    def abort_remote_prefill_receive(
        self,
        transfer_id: str,
        *,
        reason: str = "cache receive cancelled",
    ) -> int:
        """Cancel an active receive and return its request to local prefill."""

        if transfer_id not in getattr(self, "_remote_prefill_receive_tokens", {}):
            raise ValueError("cache receive id is not active")
        if not reason:
            raise ValueError("cache receive cancellation reason must not be empty")
        seq, _session = self.scheduler.remote_prefills[transfer_id]
        self._abort_remote_prefill_receive(
            transfer_id,
            reason,
            outcome="cancelled",
        )
        return seq.seq_id

    def _advance_remote_prefill_receive(
        self,
        transfer_id: str,
        rank_results: list[object],
    ) -> int | None:
        receive_tokens = getattr(self, "_remote_prefill_receive_tokens", {})
        if transfer_id not in receive_tokens:
            raise ValueError("cache receive id is not active")
        seq, session = self.scheduler.remote_prefills[transfer_id]
        now = perf_counter()
        if session.poll(now=now) is CacheTransferPhase.TIMED_OUT:
            self._abort_remote_prefill_receive(
                transfer_id,
                "cache transfer timed out",
                outcome="timed_out",
            )
            raise TimeoutError("cache transfer timed out")
        try:
            states = {}
            failures = []
            for result in rank_results:
                if not isinstance(result, dict):
                    raise RuntimeError("cache receive poll result must be a dictionary")
                rank = result.get("rank")
                state = result.get("state")
                if (
                    not isinstance(rank, int)
                    or isinstance(rank, bool)
                    or not 0 <= rank < self.config.tensor_parallel_size
                    or rank in states
                ):
                    raise RuntimeError("cache receive poll result has an invalid rank")
                if state not in {"receiving", "ready", "failed"}:
                    raise RuntimeError("cache receive poll result has an invalid state")
                states[rank] = state
                if state == "failed":
                    failures.append(f"rank {rank}: {result.get('error', 'unknown error')}")
            if set(states) != set(range(self.config.tensor_parallel_size)):
                raise RuntimeError("cache receive poll results are incomplete")
            if failures:
                raise RuntimeError("; ".join(failures))
            if any(state == "receiving" for state in states.values()):
                return None
            installed = self.model_runner.call_rank_results(
                "install_sequence_cache_receive",
                seq,
                transfer_id,
                [
                    getattr(
                        self,
                        "_remote_prefill_receive_expected_bytes",
                        {},
                    )[transfer_id][rank]
                    for rank in range(self.config.tensor_parallel_size)
                ],
            )
            received = _validate_rank_results(
                installed,
                self.config.tensor_parallel_size,
                "cached_tokens",
                expected_value=seq.num_prompt_tokens,
            )
            received_bytes = _validate_rank_results(
                installed,
                self.config.tensor_parallel_size,
                "received_bytes",
            )
            expected_bytes = getattr(
                self,
                "_remote_prefill_receive_expected_bytes",
                {},
            ).get(transfer_id)
            if received_bytes != expected_bytes:
                raise RuntimeError(
                    "cache receive payload bytes differ from the preflight estimate"
                )
            now = perf_counter()
            for rank in received:
                session.acknowledge(rank, now=now)
            self.scheduler.commit_remote_prefill(
                transfer_id,
                receive_tokens[transfer_id],
                now=now,
            )
        except BaseException as exc:
            self._abort_remote_prefill_receive(
                transfer_id,
                f"rank-local async cache receive failed: {exc}",
            )
            raise
        receive_tokens.pop(transfer_id, None)
        staged_bytes = getattr(
            self,
            "_remote_prefill_receive_staged_bytes",
            {},
        ).pop(transfer_id, 0)
        getattr(
            self,
            "_remote_prefill_receive_expected_bytes",
            {},
        ).pop(transfer_id, None)
        started_at = getattr(self, "_remote_prefill_receive_started_at", {}).pop(
            transfer_id,
            None,
        )
        if started_at is not None:
            self.metrics.record_remote_prefill_receive_finished(
                perf_counter() - started_at,
                outcome="committed",
                staged_bytes=staged_bytes,
            )
        return seq.seq_id

    def poll_remote_prefill_receive(self, transfer_id: str) -> int | None:
        """Progress one receive; return its sequence id only after all-rank commit."""

        if transfer_id not in getattr(self, "_remote_prefill_receive_tokens", {}):
            raise ValueError("cache receive id is not active")
        self.metrics.record_remote_prefill_poll(1)
        try:
            rank_results = self.model_runner.call_rank_results(
                "poll_sequence_cache_receive",
                transfer_id,
            )
            return self._advance_remote_prefill_receive(transfer_id, rank_results)
        except BaseException as exc:
            if transfer_id in getattr(self, "_remote_prefill_receive_tokens", {}):
                self._abort_remote_prefill_receive(
                    transfer_id,
                    f"cache receive poll failed: {exc}",
                )
            raise

    def _poll_remote_prefill_receives(self) -> None:
        errors = getattr(self, "_remote_prefill_receive_errors", None)
        if errors is None:
            errors = self._remote_prefill_receive_errors = {}
        transfer_ids = tuple(getattr(self, "_remote_prefill_receive_tokens", {}))
        if not transfer_ids:
            return
        self.metrics.record_remote_prefill_poll(len(transfer_ids))
        try:
            rank_results = self.model_runner.call_rank_results(
                "poll_sequence_cache_receives",
                list(transfer_ids),
            )
            by_transfer = _validate_rank_receive_polls(
                rank_results,
                self.config.tensor_parallel_size,
                transfer_ids,
            )
        except Exception as exc:
            for transfer_id in transfer_ids:
                self._abort_remote_prefill_receive(
                    transfer_id,
                    f"batched cache receive poll failed: {exc}",
                )
                errors[transfer_id] = str(exc)
            return
        for transfer_id in transfer_ids:
            try:
                self._advance_remote_prefill_receive(
                    transfer_id,
                    by_transfer[transfer_id],
                )
            except Exception as exc:
                errors[transfer_id] = str(exc)

    def send_remote_prefill(
        self,
        seq_id: int,
        transfer_id: str,
        endpoints: list[tuple[str, int]],
        *,
        timeout_s: float = 30.0,
    ) -> int:
        """Send every TP rank and release producer state after all ACKs."""

        seq = next(
            (candidate for candidate in self.scheduler.running if candidate.seq_id == seq_id),
            None,
        )
        if seq is None:
            raise ValueError("remote prefill source sequence is not running")
        if (
            seq.num_cached_tokens != seq.num_prompt_tokens
            or seq.num_completion_tokens != 1
            or seq.num_scheduled_tokens != 0
        ):
            raise ValueError("remote prefill source is not ready for handoff")
        rank_results = self.model_runner.call_rank_results(
            "send_sequence_cache_to_endpoint",
            seq,
            transfer_id,
            endpoints,
            timeout_s,
        )
        _validate_rank_results(
            rank_results,
            self.config.tensor_parallel_size,
            "sent_bytes",
        )
        first_token_id = seq.completion_token_ids[0]
        self.scheduler.complete_remote_prefill_source(seq)
        return first_token_id

    def _ensure_remote_prefill_transfer_capacity(self, *, direction: str) -> None:
        active_receives = len(
            getattr(self, "_remote_prefill_receive_tokens", {})
        )
        active_sends = len(
            getattr(self, "_remote_prefill_send_started_at", {})
        )
        limit = self.config.max_remote_prefill_transfers
        if active_receives + active_sends < limit:
            return
        self.metrics.record_remote_prefill_backpressure(direction=direction)
        raise RuntimeError(
            "remote prefill transfer capacity is exhausted: "
            f"{active_receives + active_sends}/{limit} active"
        )

    def remote_prefill_capacity_snapshot(self) -> dict[str, int | float | None]:
        """Return live capacity inputs for an external PD request router."""

        active_receives = len(self._remote_prefill_receive_tokens)
        active_sends = len(self._remote_prefill_send_started_at)
        active_transfers = active_receives + active_sends
        active_staging_bytes = sum(
            self._remote_prefill_receive_staged_bytes.values()
        ) + sum(self._remote_prefill_send_staged_bytes.values())
        staging_limit = self.config.max_remote_prefill_staging_bytes
        active_waiting = sum(
            bool(seq.block_table) or seq.state_slot is not None
            for seq in self.scheduler.waiting
        )
        used_sequence_slots = (
            len(self.scheduler.running)
            + len(self.scheduler.remote_prefills)
            + len(self.scheduler.remote_prefill_sources)
            + active_waiting
        )
        block_manager = self.scheduler.block_manager
        return {
            "waiting_requests": self.scheduler.num_waiting,
            "running_requests": self.scheduler.num_running,
            "sequence_slots_total": self.config.max_num_seqs,
            "sequence_slots_used": used_sequence_slots,
            "sequence_slots_free": max(
                self.config.max_num_seqs - used_sequence_slots,
                0,
            ),
            "kv_blocks_total": block_manager.num_total_blocks,
            "kv_blocks_used": block_manager.num_used_blocks,
            "kv_blocks_free": block_manager.num_free_blocks,
            "kv_block_usage": block_manager.usage,
            "transfer_slots_total": self.config.max_remote_prefill_transfers,
            "transfer_slots_used": active_transfers,
            "transfer_slots_free": max(
                self.config.max_remote_prefill_transfers - active_transfers,
                0,
            ),
            "staging_bytes_limit": staging_limit,
            "staging_bytes_used": active_staging_bytes,
            "staging_bytes_free": (
                None
                if staging_limit is None
                else max(staging_limit - active_staging_bytes, 0)
            ),
        }

    def estimate_remote_prefill_demand(
        self,
        num_prompt_tokens: int,
    ) -> RemotePrefillDemand:
        """Estimate decode-side resources without reserving scheduler state."""

        if (
            not isinstance(num_prompt_tokens, int)
            or isinstance(num_prompt_tokens, bool)
            or num_prompt_tokens <= 0
        ):
            raise ValueError("num_prompt_tokens must be a positive integer")
        if num_prompt_tokens > self.config.max_model_len:
            raise ValueError("num_prompt_tokens exceeds max_model_len")
        num_blocks = (
            num_prompt_tokens + self.config.kvcache_block_size - 1
        ) // self.config.kvcache_block_size
        estimates = self.model_runner.call_rank_results(
            "estimate_cache_transfer_bytes_for_blocks",
            num_blocks,
        )
        staged_by_rank = _validate_rank_results(
            estimates,
            self.config.tensor_parallel_size,
            "staged_bytes",
        )
        return RemotePrefillDemand(
            kv_blocks=num_blocks,
            staging_bytes=sum(staged_by_rank.values()),
        )

    def _ensure_remote_prefill_staging_capacity(
        self,
        staged_bytes: int,
        *,
        direction: str = "send",
    ) -> None:
        limit = self.config.max_remote_prefill_staging_bytes
        if limit is None:
            return
        active_bytes = sum(
            getattr(self, "_remote_prefill_send_staged_bytes", {}).values()
        ) + sum(
            getattr(self, "_remote_prefill_receive_staged_bytes", {}).values()
        )
        if active_bytes + staged_bytes <= limit:
            return
        self.metrics.record_remote_prefill_backpressure(direction=direction)
        raise RuntimeError(
            "remote prefill staging capacity is exhausted: "
            f"{active_bytes + staged_bytes}/{limit} bytes requested"
        )

    def start_remote_prefill_send(
        self,
        seq_id: int,
        transfer_id: str,
        endpoints: list[tuple[str, int]],
        *,
        timeout_s: float = 30.0,
    ) -> int:
        """Stage a source on every rank and send it without blocking scheduling."""

        seq = next(
            (candidate for candidate in self.scheduler.running if candidate.seq_id == seq_id),
            None,
        )
        if seq is None:
            raise ValueError("remote prefill source sequence is not running")
        started_at = getattr(self, "_remote_prefill_send_started_at", None)
        if started_at is None:
            started_at = self._remote_prefill_send_started_at = {}
        if transfer_id in started_at:
            raise ValueError("cache send id is already active")
        self._ensure_remote_prefill_transfer_capacity(direction="send")
        estimates = self.model_runner.call_rank_results(
            "estimate_sequence_cache_bytes",
            seq,
        )
        estimated_by_rank = _validate_rank_results(
            estimates,
            self.config.tensor_parallel_size,
            "staged_bytes",
        )
        staged_bytes = sum(estimated_by_rank.values())
        self._ensure_remote_prefill_staging_capacity(staged_bytes)
        self.scheduler.reserve_remote_prefill_source(seq, transfer_id)
        try:
            rank_results = self.model_runner.call_rank_results(
                "start_sequence_cache_send",
                seq,
                transfer_id,
                endpoints,
                timeout_s,
            )
            _validate_rank_results(
                rank_results,
                self.config.tensor_parallel_size,
                "started",
                expected_value=1,
            )
            staged_by_rank = _validate_rank_results(
                rank_results,
                self.config.tensor_parallel_size,
                "staged_bytes",
            )
            if staged_by_rank != estimated_by_rank:
                raise RuntimeError(
                    "cache send staged bytes differ from the preflight estimate"
                )
        except BaseException:
            try:
                self.model_runner.call_rank_results(
                    "abort_sequence_cache_send",
                    transfer_id,
                )
            except BaseException:
                pass
            if transfer_id in self.scheduler.remote_prefill_sources:
                self.scheduler.abort_remote_prefill_source(transfer_id)
            raise
        started_at[transfer_id] = perf_counter()
        staged_by_transfer = getattr(
            self,
            "_remote_prefill_send_staged_bytes",
            None,
        )
        if staged_by_transfer is None:
            staged_by_transfer = self._remote_prefill_send_staged_bytes = {}
        staged_by_transfer[transfer_id] = staged_bytes
        errors = getattr(self, "_remote_prefill_send_errors", None)
        if errors is None:
            errors = self._remote_prefill_send_errors = {}
        errors.pop(transfer_id, None)
        self.metrics.record_remote_prefill_send_started(staged_bytes)
        return seq.completion_token_ids[0]

    def _abort_remote_prefill_send(
        self,
        transfer_id: str,
        *,
        outcome: str,
    ) -> Sequence | None:
        try:
            self.model_runner.call_rank_results(
                "abort_sequence_cache_send",
                transfer_id,
            )
        except BaseException:
            pass
        started_at = getattr(self, "_remote_prefill_send_started_at", {}).pop(
            transfer_id,
            None,
        )
        staged_bytes = getattr(self, "_remote_prefill_send_staged_bytes", {}).pop(
            transfer_id,
            0,
        )
        if started_at is not None:
            self.metrics.record_remote_prefill_send_finished(
                perf_counter() - started_at,
                outcome=outcome,
                staged_bytes=staged_bytes,
            )
        if transfer_id in self.scheduler.remote_prefill_sources:
            return self.scheduler.abort_remote_prefill_source(transfer_id)
        return None

    def abort_remote_prefill_send(self, transfer_id: str) -> int:
        """Cancel a source send while retaining its local KV and decode state."""

        if transfer_id not in getattr(self, "_remote_prefill_send_started_at", {}):
            raise ValueError("cache send id is not active")
        seq, _position = self.scheduler.remote_prefill_sources[transfer_id]
        self._abort_remote_prefill_send(transfer_id, outcome="cancelled")
        return seq.seq_id

    def _advance_remote_prefill_send(
        self,
        transfer_id: str,
        rank_results: list[object],
    ) -> int | None:
        if transfer_id not in getattr(self, "_remote_prefill_send_started_at", {}):
            raise ValueError("cache send id is not active")
        states = {}
        failures = []
        retained_staged_bytes = 0
        try:
            for result in rank_results:
                if not isinstance(result, dict):
                    raise RuntimeError("cache send poll result must be a dictionary")
                rank = result.get("rank")
                state = result.get("state")
                staged_bytes = result.get("staged_bytes")
                if (
                    not isinstance(rank, int)
                    or isinstance(rank, bool)
                    or not 0 <= rank < self.config.tensor_parallel_size
                    or rank in states
                ):
                    raise RuntimeError("cache send poll result has an invalid rank")
                if state not in {"sending", "ready", "failed"}:
                    raise RuntimeError("cache send poll result has an invalid state")
                if (
                    not isinstance(staged_bytes, int)
                    or isinstance(staged_bytes, bool)
                    or staged_bytes < 0
                    or (state in {"ready", "failed"} and staged_bytes != 0)
                ):
                    raise RuntimeError(
                        "cache send poll result has invalid staged bytes"
                    )
                states[rank] = state
                retained_staged_bytes += staged_bytes
                if state == "failed":
                    failures.append(f"rank {rank}: {result.get('error', 'unknown error')}")
            if set(states) != set(range(self.config.tensor_parallel_size)):
                raise RuntimeError("cache send poll results are incomplete")
            previous_staged_bytes = self._remote_prefill_send_staged_bytes[
                transfer_id
            ]
            if retained_staged_bytes > previous_staged_bytes:
                raise RuntimeError("cache send staged bytes increased while active")
            released_staged_bytes = previous_staged_bytes - retained_staged_bytes
            if released_staged_bytes:
                self.metrics.record_remote_prefill_send_staging_released(
                    released_staged_bytes
                )
                self._remote_prefill_send_staged_bytes[
                    transfer_id
                ] = retained_staged_bytes
            if failures:
                raise RuntimeError("; ".join(failures))
            if any(state == "sending" for state in states.values()):
                return None
            finished = self.model_runner.call_rank_results(
                "finish_sequence_cache_send",
                transfer_id,
            )
            sent_by_rank = _validate_rank_results(
                finished,
                self.config.tensor_parallel_size,
                "sent_bytes",
            )
        except BaseException:
            self._abort_remote_prefill_send(transfer_id, outcome="failed")
            raise
        seq = self.scheduler.commit_remote_prefill_source(transfer_id)
        started_at = self._remote_prefill_send_started_at.pop(transfer_id)
        staged_bytes = self._remote_prefill_send_staged_bytes.pop(transfer_id)
        self.metrics.record_remote_prefill_send_finished(
            perf_counter() - started_at,
            outcome="committed",
            staged_bytes=staged_bytes,
            sent_bytes=sum(sent_by_rank.values()),
        )
        return seq.seq_id

    def poll_remote_prefill_send(self, transfer_id: str) -> int | None:
        """Progress one source send and commit only after every rank ACKs."""

        if transfer_id not in getattr(self, "_remote_prefill_send_started_at", {}):
            raise ValueError("cache send id is not active")
        self.metrics.record_remote_prefill_send_poll(1)
        try:
            rank_results = self.model_runner.call_rank_results(
                "poll_sequence_cache_send",
                transfer_id,
            )
            return self._advance_remote_prefill_send(transfer_id, rank_results)
        except BaseException as exc:
            if transfer_id in getattr(self, "_remote_prefill_send_started_at", {}):
                self._abort_remote_prefill_send(transfer_id, outcome="failed")
            getattr(self, "_remote_prefill_send_errors", {})[transfer_id] = str(exc)
            raise

    def _poll_remote_prefill_sends(self) -> None:
        transfer_ids = tuple(getattr(self, "_remote_prefill_send_started_at", {}))
        if not transfer_ids:
            return
        self.metrics.record_remote_prefill_send_poll(len(transfer_ids))
        try:
            rank_results = self.model_runner.call_rank_results(
                "poll_sequence_cache_sends",
                list(transfer_ids),
            )
            by_transfer = _validate_rank_send_polls(
                rank_results,
                self.config.tensor_parallel_size,
                transfer_ids,
            )
        except Exception as exc:
            for transfer_id in transfer_ids:
                self._abort_remote_prefill_send(transfer_id, outcome="failed")
                self._remote_prefill_send_errors[transfer_id] = str(exc)
            return
        for transfer_id in transfer_ids:
            try:
                self._advance_remote_prefill_send(
                    transfer_id,
                    by_transfer[transfer_id],
                )
            except Exception as exc:
                self._remote_prefill_send_errors[transfer_id] = str(exc)

    def step(self):
        self._poll_remote_prefill_receives()
        self._poll_remote_prefill_sends()
        self.scheduler.poll_remote_prefills(now=perf_counter())
        schedule_result = self.scheduler.schedule()
        if isinstance(schedule_result, ScheduleResult):
            seqs = schedule_result.seqs
            is_prefill = schedule_result.is_prefill
            num_tokens = schedule_result.num_prefill_tokens - schedule_result.num_decode_tokens
            prefill_tokens = schedule_result.num_prefill_tokens
            decode_tokens = schedule_result.num_decode_tokens
        else:
            seqs, is_prefill = schedule_result
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            prefill_tokens = num_tokens if num_tokens > 0 else 0
            decode_tokens = -num_tokens if num_tokens < 0 else 0
        self.metrics.record_scheduler_state(
            self.scheduler.num_waiting,
            self.scheduler.num_running,
            self.scheduler.block_manager.num_used_blocks,
            self.scheduler.block_manager.num_total_blocks,
            self.scheduler.prefill_starved_steps,
            self.scheduler.max_prefill_starvation_steps,
            self.scheduler.preemption_count,
            self.scheduler.preempted_token_progress,
            self.scheduler.max_preempted_token_progress,
            self.scheduler.reclaimed_kv_blocks,
        )
        if not seqs:
            return [], 0, 0, 0
        if isinstance(schedule_result, ScheduleResult) and schedule_result.is_mixed:
            token_ids = self.model_runner.call("run_mixed", schedule_result.prefill_seqs, schedule_result.decode_seqs)
            self.scheduler.postprocess_mixed(schedule_result, token_ids)
        elif isinstance(schedule_result, ScheduleResult):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess_mixed(schedule_result, token_ids)
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        finished_seqs = [seq for seq in seqs if seq.is_finished]
        self.metrics.record_finished_sequences(finished_seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
        return outputs, num_tokens, prefill_tokens, decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if isinstance(sampling_params, (list, tuple)):
            if len(sampling_params) != len(prompts):
                raise ValueError(
                    "sampling_params must have the same length as prompts"
                )
            params = list(sampling_params)
        else:
            params = [sampling_params] * len(prompts)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        for prompt, sp in zip(prompts, params):
            self.add_request(prompt, sp)
        self.metrics.reset()
        outputs = {}
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, prefill_tokens, decode_tokens = self.step()
            self.metrics.record_step(
                num_tokens,
                perf_counter() - t,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
            )
            pbar.set_postfix({
                "Pure prefill": f"{int(self.metrics.pure_prefill_throughput)}tok/s",
                "Pure decode": f"{int(self.metrics.pure_decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
