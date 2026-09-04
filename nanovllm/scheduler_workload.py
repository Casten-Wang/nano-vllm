"""Deterministic request-arrival traces for scheduler evaluation.

The offline engine cannot accept work while a synchronous model step is in
flight.  Scheduler experiments therefore use logical arrival steps rather than
pretending to measure a wall-clock request rate.  The same normalized trace can
be replayed against every policy and paired with an online serving benchmark
when a concurrent server path is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


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
    def from_dict(cls, item: object, index: int) -> "SchedulerRequest":
        if not isinstance(item, dict):
            raise ValueError(f"requests[{index}] must be an object")
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
            raise ValueError(
                f"requests[{index}].workload_class must not be empty"
            )
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
    ) -> "SchedulerWorkload":
        if not isinstance(payload, dict):
            raise ValueError("workload must be an object")
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
            "last_arrival_step": max(
                request.arrival_step for request in self.requests
            ),
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
    return SchedulerWorkload.from_dict({
        "version": TRACE_VERSION,
        "name": name,
        "requests": requests,
    })
