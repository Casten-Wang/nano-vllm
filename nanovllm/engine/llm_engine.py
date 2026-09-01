import atexit
from dataclasses import fields
import os
import socket
import uuid
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import ScheduleResult, Scheduler
from nanovllm.engine.model_runner import CONTROL_STATUS_SIZE, ModelRunner
from nanovllm.engine.metrics import EngineMetrics


def _find_free_port() -> int:
    """Ask the OS for an unused local TCP port for the NCCL rendezvous."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.metrics = EngineMetrics()
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

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
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
        self.scheduler.add(seq)

    def step(self):
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
        )
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
