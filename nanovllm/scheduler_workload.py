"""Deterministic request-arrival traces for scheduler evaluation.

The offline engine cannot accept work while a synchronous model step is in
flight.  Scheduler experiments therefore use logical arrival steps rather than
pretending to measure a wall-clock request rate.  The same normalized trace can
be replayed against every policy and paired with an online serving benchmark
when a concurrent server path is available.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from time import perf_counter

TRACE_VERSION = 1
PROFILE_NAMES = ("steady", "bursty", "mixed")


def _plain_non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _plain_positive_int(value: object, name: str) -> int:
    value = _plain_non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class SchedulerRequest:
    request_id: str
    arrival_step: int
    input_len: int
    output_len: int
    workload_class: str

    @classmethod
    def from_dict(cls, item: object, index: int) -> SchedulerRequest:
        if not isinstance(item, dict):
            raise TypeError(f"requests[{index}] must be an object")
        required = {
            "request_id",
            "arrival_step",
            "input_len",
            "output_len",
            "workload_class",
        }
        if set(item) != required:
            raise ValueError(
                f"requests[{index}] must contain exactly {sorted(required)}"
            )
        request_id = item["request_id"]
        workload_class = item["workload_class"]
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"requests[{index}].request_id must not be empty")
        if not isinstance(workload_class, str) or not workload_class:
            raise ValueError(f"requests[{index}].workload_class must not be empty")
        return cls(
            request_id=request_id,
            arrival_step=_plain_non_negative_int(
                item["arrival_step"],
                f"requests[{index}].arrival_step",
            ),
            input_len=_plain_positive_int(
                item["input_len"],
                f"requests[{index}].input_len",
            ),
            output_len=_plain_positive_int(
                item["output_len"],
                f"requests[{index}].output_len",
            ),
            workload_class=workload_class,
        )


@dataclass(frozen=True, slots=True)
class SchedulerWorkload:
    name: str
    requests: tuple[SchedulerRequest, ...]

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        max_model_len: int | None = None,
    ) -> SchedulerWorkload:
        if not isinstance(payload, dict):
            raise TypeError("workload must be an object")
        if set(payload) != {"version", "name", "requests"}:
            raise ValueError(
                "workload must contain exactly version, name, and requests"
            )
        if payload["version"] != TRACE_VERSION:
            raise ValueError(f"unsupported workload version: {payload['version']!r}")
        name = payload["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("workload name must not be empty")
        raw_requests = payload["requests"]
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("workload requests must be a non-empty list")
        requests = tuple(
            SchedulerRequest.from_dict(item, index)
            for index, item in enumerate(raw_requests)
        )
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("workload request_id values must be unique")
        if max_model_len is not None:
            max_model_len = _plain_positive_int(max_model_len, "max_model_len")
            for request in requests:
                total = request.input_len + request.output_len
                if total > max_model_len:
                    raise ValueError(
                        f"request {request.request_id!r} requires {total} tokens, "
                        f"exceeding max_model_len={max_model_len}"
                    )
        return cls(
            name=name,
            requests=tuple(
                request
                for _, request in sorted(
                    enumerate(requests),
                    key=lambda pair: (pair[1].arrival_step, pair[0]),
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": TRACE_VERSION,
            "name": self.name,
            "requests": [asdict(request) for request in self.requests],
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return sha256(encoded).hexdigest()

    def summary(self) -> dict[str, object]:
        by_class: dict[str, int] = {}
        for request in self.requests:
            by_class[request.workload_class] = (
                by_class.get(request.workload_class, 0) + 1
            )
        return {
            "name": self.name,
            "digest": self.digest,
            "request_count": len(self.requests),
            "arrival_step_count": len(
                {request.arrival_step for request in self.requests}
            ),
            "last_arrival_step": max(request.arrival_step for request in self.requests),
            "input_tokens": sum(request.input_len for request in self.requests),
            "requested_output_tokens": sum(
                request.output_len for request in self.requests
            ),
            "requests_by_class": dict(sorted(by_class.items())),
        }


def _request(
    profile: str,
    index: int,
    arrival_step: int,
    input_len: int,
    output_len: int,
    workload_class: str,
) -> dict[str, object]:
    return {
        "request_id": f"{profile}-{index:03d}",
        "arrival_step": arrival_step,
        "input_len": input_len,
        "output_len": output_len,
        "workload_class": workload_class,
    }


def built_in_workload(name: str) -> SchedulerWorkload:
    """Return fixed scheduler traces suitable for TP4/TP8 comparisons."""

    if name not in PROFILE_NAMES:
        raise ValueError(f"workload profile must be one of: {', '.join(PROFILE_NAMES)}")
    requests = []
    if name == "steady":
        for index in range(24):
            requests.append(
                _request(
                    name,
                    index,
                    index * 2,
                    256 if index % 2 == 0 else 1024,
                    128,
                    "steady",
                )
            )
    elif name == "bursty":
        for index in range(24):
            requests.append(
                _request(
                    name,
                    index,
                    (index // 8) * 8,
                    512,
                    128,
                    "burst",
                )
            )
    else:
        shapes = (
            (256, 512, "decode-heavy"),
            (4096, 64, "prefill-heavy"),
            (128, 32, "short"),
        )
        for index in range(24):
            input_len, output_len, workload_class = shapes[index % len(shapes)]
            requests.append(
                _request(
                    name,
                    index,
                    (index // 3) * 2,
                    input_len,
                    output_len,
                    workload_class,
                )
            )
    return SchedulerWorkload.from_dict(
        {
            "version": TRACE_VERSION,
            "name": name,
            "requests": requests,
        }
    )


def prompt_token_ids(
    request: SchedulerRequest,
    *,
    vocab_size: int,
    seed: int,
) -> list[int]:
    """Build stable synthetic tokens without depending on request order."""

    vocab_size = _plain_positive_int(vocab_size, "vocab_size")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    material = f"{seed}\0{request.request_id}".encode()
    request_seed = int.from_bytes(sha256(material).digest()[:8], "big")
    rng = random.Random(request_seed)
    return [rng.randrange(vocab_size) for _ in range(request.input_len)]


def _percentile(values: list[float], rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * rank
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _request_latency_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    by_class: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        by_class.setdefault(str(sample["workload_class"]), []).append(sample)

    def summarize(items: list[dict[str, object]]) -> dict[str, float | int]:
        result: dict[str, float | int] = {"request_count": len(items)}
        for name in ("ttft_s", "tpot_s", "latency_s"):
            values = [float(item[name]) for item in items]
            result[f"avg_{name}"] = sum(values) / len(values) if values else 0.0
            for label, rank in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
                result[f"{label}_{name}"] = _percentile(values, rank)
            result[f"max_{name}"] = max(values, default=0.0)
        return result

    return {
        "all": summarize(samples),
        "by_class": {
            name: summarize(items) for name, items in sorted(by_class.items())
        },
    }


def replay_scheduler_workload(
    engine,
    workload: SchedulerWorkload,
    *,
    prompt_factory: Callable[[SchedulerRequest], list[int]],
    sampling_params_factory: Callable[[SchedulerRequest], object],
    synchronize: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = perf_counter,
    max_engine_steps: int = 100_000,
) -> dict[str, object]:
    """Replay one logical-arrival trace through an initialized engine.

    Arrivals due at a logical step are admitted before scheduling that step.
    When no request is active, the logical clock advances directly to the next
    arrival without inventing empty model steps.
    """

    max_engine_steps = _plain_positive_int(max_engine_steps, "max_engine_steps")
    pending_index = 0
    logical_step = 0
    engine_steps = 0
    idle_fast_forwards = 0
    request_by_seq_id: dict[int, SchedulerRequest] = {}
    completion_step_by_seq_id: dict[int, int] = {}
    output_by_seq_id: dict[int, list[int]] = {}
    step_samples = []
    requests = workload.requests
    engine.metrics.reset()

    while pending_index < len(requests) or not engine.is_finished():
        if engine.is_finished() and pending_index < len(requests):
            next_arrival = requests[pending_index].arrival_step
            if next_arrival > logical_step:
                logical_step = next_arrival
                idle_fast_forwards += 1

        admitted = []
        while (
            pending_index < len(requests)
            and requests[pending_index].arrival_step <= logical_step
        ):
            request = requests[pending_index]
            seq_id = engine.add_request(
                prompt_factory(request),
                sampling_params_factory(request),
            )
            if seq_id in request_by_seq_id:
                raise RuntimeError(f"engine returned duplicate seq_id={seq_id}")
            request_by_seq_id[seq_id] = request
            admitted.append(request.request_id)
            pending_index += 1

        if engine.is_finished():
            continue
        if engine_steps >= max_engine_steps:
            raise RuntimeError(f"workload exceeded max_engine_steps={max_engine_steps}")

        synchronize()
        started_at = clock()
        outputs, num_tokens, prefill_tokens, decode_tokens = engine.step()
        synchronize()
        elapsed = clock() - started_at
        if elapsed < 0.0:
            raise RuntimeError("benchmark clock moved backwards")
        engine.metrics.record_step(
            num_tokens,
            elapsed,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        finished = []
        for seq_id, token_ids in outputs:
            if seq_id not in request_by_seq_id:
                raise RuntimeError(f"engine completed unknown seq_id={seq_id}")
            if seq_id in output_by_seq_id:
                raise RuntimeError(f"engine completed seq_id={seq_id} twice")
            output_by_seq_id[seq_id] = list(token_ids)
            completion_step_by_seq_id[seq_id] = logical_step
            finished.append(request_by_seq_id[seq_id].request_id)
        capacity = engine.scheduler.capacity_snapshot()
        step_samples.append(
            {
                "logical_step": logical_step,
                "elapsed_s": elapsed,
                "admitted_request_ids": admitted,
                "finished_request_ids": finished,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "capacity": capacity,
            }
        )
        engine_steps += 1
        logical_step += 1

    if len(output_by_seq_id) != len(requests):
        raise RuntimeError(
            f"engine completed {len(output_by_seq_id)}/{len(requests)} requests"
        )
    metric_by_seq_id = {
        int(sample["seq_id"]): sample for sample in engine.metrics.request_samples
    }
    if set(metric_by_seq_id) != set(request_by_seq_id):
        raise RuntimeError("request latency samples are incomplete")

    request_samples = []
    output_records = []
    for seq_id, request in request_by_seq_id.items():
        token_ids = output_by_seq_id[seq_id]
        if len(token_ids) != request.output_len:
            raise RuntimeError(
                f"request {request.request_id!r} produced {len(token_ids)}/"
                f"{request.output_len} tokens"
            )
        metric = metric_by_seq_id[seq_id]
        request_samples.append(
            {
                "request_id": request.request_id,
                "seq_id": seq_id,
                "workload_class": request.workload_class,
                "arrival_step": request.arrival_step,
                "completion_step": completion_step_by_seq_id[seq_id],
                "input_tokens": request.input_len,
                "requested_output_tokens": request.output_len,
                "output_tokens": len(token_ids),
                "preemption_count": int(metric.get("preemption_count", 0)),
                "preempted_token_progress": int(
                    metric.get("preempted_token_progress", 0)
                ),
                "ttft_s": metric["ttft_s"],
                "tpot_s": metric["tpot_s"],
                "latency_s": metric["latency_s"],
            }
        )
        output_records.append(
            {
                "request_id": request.request_id,
                "token_ids": token_ids,
            }
        )

    output_digest = sha256(
        json.dumps(
            output_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    return {
        "workload": workload.summary(),
        "engine_steps": engine_steps,
        "idle_fast_forwards": idle_fast_forwards,
        "request_samples": request_samples,
        "output_token_ids": {
            "digest": output_digest,
            "request_count": len(output_records),
            "token_count": sum(len(record["token_ids"]) for record in output_records),
        },
        "latency": _request_latency_summary(request_samples),
        "step_samples": step_samples,
        "engine_metrics": engine.metrics.to_dict(),
    }
