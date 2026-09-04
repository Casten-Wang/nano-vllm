"""Shared benchmark provenance and execution-stat validation helpers."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"checkpoint shard changed while hashing: {path}")
    return digest.hexdigest()


def checkpoint_manifest_metadata(
    model_path: str | Path,
    *,
    require_shards: bool = True,
    hash_shards: bool = False,
) -> dict:
    """Fingerprint checkpoint identity without reading tensor payloads."""

    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    config_sha256 = (
        _sha256_bytes(config_path.read_bytes()) if config_path.is_file() else None
    )
    index_path = root / "model.safetensors.index.json"
    index_sha256 = None
    if index_path.is_file():
        index_bytes = index_path.read_bytes()
        index_sha256 = _sha256_bytes(index_bytes)
        weight_map = json.loads(index_bytes)["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = sorted(path.name for path in root.glob("*.safetensors"))
    if not filenames:
        raise ValueError(f"no safetensors checkpoint files found in {root}")

    files = []
    all_content_addressed = True
    missing_shards = []
    for filename in filenames:
        path = root / filename
        if not path.is_file():
            if require_shards:
                raise ValueError(f"checkpoint shard is missing: {path}")
            missing_shards.append(filename)
            files.append(
                {
                    "name": filename,
                    "size_bytes": None,
                    "mtime_ns": None,
                    "content_id": None,
                    "present": False,
                }
            )
            all_content_addressed = False
            continue
        stat = path.stat()
        resolved_name = path.resolve().name
        content_id = (
            resolved_name.lower()
            if len(resolved_name) in (40, 64)
            and all(character in "0123456789abcdefABCDEF" for character in resolved_name)
            else None
        )
        content_sha256 = _sha256_file(path) if hash_shards else None
        all_content_addressed &= content_id is not None
        files.append(
            {
                "name": filename,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "content_id": content_id,
                "content_sha256": content_sha256,
                "present": True,
            }
        )
    identity_files = (
        [
            {
                "name": item["name"],
                "size_bytes": item["size_bytes"],
                "content_sha256": item["content_sha256"],
            }
            for item in files
        ]
        if hash_shards and not missing_shards
        else
        [
            {
                "name": item["name"],
                "size_bytes": item["size_bytes"],
                "content_id": item["content_id"],
            }
            for item in files
        ]
        if all_content_addressed and not missing_shards
        else files
    )
    identity = {
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "files": identity_files,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256",
        "digest": _sha256_bytes(encoded),
        "strength": (
            "index-only"
            if missing_shards
            else "sha256"
            if hash_shards
            else "content-addressed"
            if all_content_addressed
            else "metadata-only"
        ),
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shard_count": len(files),
        "present_shard_count": len(files) - len(missing_shards),
        "missing_shards": missing_shards,
        "total_size_bytes": sum(
            item["size_bytes"] or 0 for item in files
        ),
        "files": files,
    }


def token_ids_digest(outputs: list[dict]) -> dict:
    """Return a compact, deterministic fingerprint of generated token IDs."""

    token_ids = [
        [int(token_id) for token_id in output["token_ids"]]
        for output in outputs
    ]
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
        "sequence_lengths": [len(sequence) for sequence in token_ids],
    }


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _nvidia_smi_query() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _nvidia_smi_topology() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return output or None


def _cuda_collective_version(torch_module) -> list[int] | int | str | None:
    try:
        version = torch_module.cuda.nccl.version()
    except Exception:
        return None
    return _json_safe(version)


def kv_cache_storage_metadata(model_runner) -> dict:
    """Return the actual allocated KV data/scale storage for one runner."""

    kv_cache = model_runner.kv_cache
    kv_scale = getattr(model_runner, "kv_scale", None)
    data_bytes = kv_cache.numel() * kv_cache.element_size()
    scale_bytes = (
        kv_scale.numel() * kv_scale.element_size() if kv_scale is not None else 0
    )
    world_size = getattr(model_runner, "world_size", 1)
    local_total_bytes = data_bytes + scale_bytes
    return {
        "scope": "local_rank",
        "world_size": world_size,
        "data_dtype": str(kv_cache.dtype),
        "scale_dtype": str(kv_scale.dtype) if kv_scale is not None else None,
        "data_bytes": data_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": local_total_bytes,
        "total_mib": local_total_bytes / 1024 / 1024,
        "estimated_all_ranks_bytes": local_total_bytes * world_size,
        "estimated_all_ranks_mib": local_total_bytes * world_size / 1024 / 1024,
    }


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def model_config_metadata(hf_config) -> dict:
    """Return a JSON-serializable snapshot of a Transformers config."""

    return _json_safe(hf_config.to_dict())


def collect_benchmark_metadata(torch_module=None) -> dict:
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            torch_module = None

    cuda_available = bool(
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_available()
    )
    device = None
    device_capability = None
    device_count = 0
    cuda_devices = []
    cuda_version = None
    torch_version = None
    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", "unknown")
        cuda_version = getattr(torch_module.version, "cuda", None)
        if cuda_available:
            device = torch_module.cuda.get_device_name()
            device_capability = list(torch_module.cuda.get_device_capability())
            device_count = torch_module.cuda.device_count()
            cuda_devices = [
                {
                    "index": index,
                    "name": torch_module.cuda.get_device_name(index),
                    "capability": list(
                        torch_module.cuda.get_device_capability(index)
                    ),
                }
                for index in range(device_count)
            ]

    return {
        "commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["branch", "--show-current"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "command": list(sys.argv),
        "working_directory": os.getcwd(),
        "benchmark_timestamp": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "device": device,
        "device_capability": device_capability,
        "cuda_device_count": device_count,
        "cuda_devices": cuda_devices,
        "cuda_available": cuda_available,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "nccl_version": (
            _cuda_collective_version(torch_module)
            if cuda_available
            else None
        ),
        "transformers_version": _module_version("transformers"),
        "triton_version": _module_version("triton"),
        "flash_attn_version": _module_version("flash_attn"),
        "nvidia_smi_gpus": _nvidia_smi_query(),
        "nvidia_smi_topology": _nvidia_smi_topology(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_environment": {
            name: os.environ[name]
            for name in (
                "NCCL_ALGO",
                "NCCL_PROTO",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
            )
            if name in os.environ
        },
    }


def validate_execution_stats(
    execution_stats: dict,
    required_paths: list[str] | tuple[str, ...] = (),
) -> dict:
    model_counts = execution_stats.get("model_path_counts", {})
    attention_counts = execution_stats.get("attention_path_counts", {})
    state_access_counts = execution_stats.get("state_access_path_counts", {})
    dropped_signature_steps = execution_stats.get(
        "dropped_execution_signature_steps",
        0,
    )
    observed = set(model_counts) | set(attention_counts) | set(state_access_counts)
    missing = [path for path in required_paths if path not in observed]
    valid = bool(observed) and not missing and dropped_signature_steps == 0
    if not observed:
        reason = "no execution path was recorded"
    elif missing:
        reason = f"missing required paths: {', '.join(missing)}"
    elif dropped_signature_steps:
        reason = (
            "execution signature capacity was exceeded: "
            f"{dropped_signature_steps} step(s) were dropped"
        )
    else:
        reason = None
    return {
        "valid": valid,
        "required_paths": list(required_paths),
        "observed_paths": sorted(observed),
        "missing_paths": missing,
        "dropped_execution_signature_steps": dropped_signature_steps,
        "reason": reason,
    }


def validate_ranked_records(
    records: object,
    *,
    expected_world_size: int,
    record_name: str,
) -> list[dict]:
    """Return complete per-rank records in rank order."""

    if (
        not isinstance(expected_world_size, int)
        or isinstance(expected_world_size, bool)
        or expected_world_size <= 0
    ):
        raise ValueError("expected_world_size must be a positive integer")
    if not isinstance(record_name, str) or not record_name:
        raise ValueError("record_name must be a non-empty string")
    if not isinstance(records, list):
        raise ValueError(f"{record_name} must be a list")
    by_rank = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"each {record_name} record must be a dictionary")
        rank = record.get("rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < expected_world_size
            or rank in by_rank
        ):
            raise ValueError(f"{record_name} contain an invalid or duplicate rank")
        by_rank[rank] = record
    expected_ranks = set(range(expected_world_size))
    if set(by_rank) != expected_ranks:
        missing = sorted(expected_ranks - set(by_rank))
        raise ValueError(f"{record_name} are missing ranks: {missing}")
    return [by_rank[rank] for rank in range(expected_world_size)]


def validate_execution_stats_by_rank(
    execution_stats_by_rank: object,
    *,
    expected_world_size: int,
    required_paths: list[str] | tuple[str, ...] = (),
) -> dict:
    """Validate complete, uniquely identified execution evidence for every rank."""

    ranked_stats = validate_ranked_records(
        execution_stats_by_rank,
        expected_world_size=expected_world_size,
        record_name="execution stats",
    )
    validations = [
        {
            "rank": stats["rank"],
            **validate_execution_stats(stats, required_paths),
        }
        for stats in ranked_stats
    ]
    invalid_ranks = [item["rank"] for item in validations if not item["valid"]]
    return {
        "valid": not invalid_ranks,
        "invalid_ranks": invalid_ranks,
        "by_rank": validations,
    }


def validate_generation_completion(
    output_lengths: list[int],
    *,
    expected_num_seqs: int,
    expected_output_len: int,
    waiting_queue_len: int,
    running_queue_len: int,
) -> dict:
    errors = []
    if len(output_lengths) != expected_num_seqs:
        errors.append(
            f"completed {len(output_lengths)} requests, expected {expected_num_seqs}"
        )
    invalid_lengths = [
        length for length in output_lengths if length != expected_output_len
    ]
    if invalid_lengths:
        errors.append(
            f"{len(invalid_lengths)} outputs did not contain "
            f"{expected_output_len} tokens"
        )
    if waiting_queue_len or running_queue_len:
        errors.append(
            "scheduler queues are not empty: "
            f"waiting={waiting_queue_len}, running={running_queue_len}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "expected_num_seqs": expected_num_seqs,
        "actual_num_seqs": len(output_lengths),
        "expected_output_len": expected_output_len,
        "actual_output_tokens": sum(output_lengths),
        "output_length_min": min(output_lengths) if output_lengths else 0,
        "output_length_max": max(output_lengths) if output_lengths else 0,
        "waiting_queue_len": waiting_queue_len,
        "running_queue_len": running_queue_len,
    }
