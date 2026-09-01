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
    cuda_version = None
    torch_version = None
    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", "unknown")
        cuda_version = getattr(torch_module.version, "cuda", None)
        if cuda_available:
            device = torch_module.cuda.get_device_name()
            device_capability = list(torch_module.cuda.get_device_capability())
            device_count = torch_module.cuda.device_count()

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
        "cuda_available": cuda_available,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "transformers_version": _module_version("transformers"),
        "triton_version": _module_version("triton"),
        "flash_attn_version": _module_version("flash_attn"),
        "nvidia_smi_gpus": _nvidia_smi_query(),
    }


def validate_execution_stats(
    execution_stats: dict,
    required_paths: list[str] | tuple[str, ...] = (),
) -> dict:
    model_counts = execution_stats.get("model_path_counts", {})
    attention_counts = execution_stats.get("attention_path_counts", {})
    dropped_signature_steps = execution_stats.get(
        "dropped_execution_signature_steps",
        0,
    )
    observed = set(model_counts) | set(attention_counts)
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
