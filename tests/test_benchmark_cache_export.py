import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.cache_transfer import HostStagingBufferPool
from scripts.benchmark_cache_export import (
    _export,
    _measure,
    _payload_host_layout,
    _profile,
    make_source,
)


def test_measure_accepts_bounded_transient_staging(monkeypatch):
    pool = HostStagingBufferPool(max_cached_bytes=16)
    profile = {"components": {"total": 17}}

    def export_once(*_args, **_kwargs):
        lease = pool.acquire(17, pin_memory=False)
        return SimpleNamespace(release_host_staging=lease.release)

    monkeypatch.setattr(
        "scripts.benchmark_cache_export._export",
        export_once,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    result = _measure(
        None,
        profile,
        direct_host=True,
        warmup=1,
        repeats=2,
        host_staging_pool=pool,
    )

    assert result["host_staging_pool"]["valid"]
    assert result["host_staging_pool"]["storage_bytes"] == 0
    assert result["host_staging_pool"]["transient_allocation_count"] == 3


def write_preflight(path: Path, *, total_delta: int = 0) -> None:
    auto_components = {
        "kv": 2 * 10 * 512 * 256 * 2,
        "kv_scales": 0,
        "recurrent": 30 * 8 * 128 * 128 * 4,
        "convolution": 30 * (2 * 4 * 128 + 8 * 128) * 4 * 2,
    }
    auto_components["total"] = sum(auto_components.values()) + total_delta
    int8_components = {
        "kv": 2 * 10 * 512 * 256,
        "kv_scales": 2 * 10 * 512 * 2,
        "recurrent": 30 * 8 * 128 * 128 * 2,
        "convolution": 30 * (2 * 4 * 128 + 8 * 128) * 4 * 2,
    }
    int8_components["total"] = sum(int8_components.values())
    path.write_text(
        json.dumps(
            {
                "results": {
                    "tp4": {
                        "pd_transfer_allocated_tokens": 512,
                        "pd_transfer_context_tokens": 511,
                        "pd_transfer_components_per_sequence_by_dtype": {
                            "auto": {"float32": auto_components},
                            "int8": {"model": int8_components},
                        },
                    }
                }
            }
        )
    )


def test_profile_builds_exact_qwen35_cpu_source(tmp_path):
    path = tmp_path / "preflight.json"
    write_preflight(path)
    profile = _profile(path, 4, "auto", "float32")

    kv, scale, recurrent, convolution = make_source(
        profile,
        kv_dtype="auto",
        state_dtype="float32",
        device=torch.device("cpu"),
    )

    assert kv.shape == (2, 10, 2, 256, 1, 256)
    assert scale is None
    assert len(recurrent) == len(convolution) == 30
    assert sum(t.numel() * t.element_size() for t in recurrent) == profile["components"]["recurrent"]
    assert sum(t.numel() * t.element_size() for t in convolution) == profile["components"]["convolution"]

    payload = _export(
        (kv, scale, recurrent, convolution),
        profile,
        direct_host=True,
    )
    assert _payload_host_layout(payload) == {
        "tensor_count": 61,
        "storage_count": 1,
        "all_cpu": True,
        "all_pinned": False,
    }


def test_direct_host_export_reuses_a_released_pool_buffer(tmp_path):
    path = tmp_path / "preflight.json"
    write_preflight(path)
    profile = _profile(path, 4, "int8", "model")
    source = make_source(
        profile,
        kv_dtype="int8",
        state_dtype="model",
        device=torch.device("cpu"),
    )
    pool = HostStagingBufferPool()

    first = _export(
        source,
        profile,
        direct_host=True,
        host_staging_pool=pool,
    )
    first_ptr = first.kv_blocks.untyped_storage().data_ptr()
    first.release_host_staging()
    second = _export(
        source,
        profile,
        direct_host=True,
        host_staging_pool=pool,
    )

    assert second.kv_blocks.untyped_storage().data_ptr() == first_ptr
    assert pool.storage_stats()["reuse_count"] == 1
    second.release_host_staging()


def test_profile_builds_exact_int8_scale_and_model_state_source(tmp_path):
    path = tmp_path / "preflight.json"
    write_preflight(path)
    profile = _profile(path, 4, "int8", "model")

    kv, scale, recurrent, convolution = make_source(
        profile,
        kv_dtype="int8",
        state_dtype="model",
        device=torch.device("cpu"),
    )

    assert kv.dtype is torch.int8
    assert scale is not None and scale.shape == (2, 10, 2, 256, 1)
    assert all(tensor.dtype is torch.bfloat16 for tensor in recurrent)
    assert all(tensor.dtype is torch.bfloat16 for tensor in convolution)


def test_profile_rejects_inconsistent_component_total(tmp_path):
    path = tmp_path / "preflight.json"
    write_preflight(path, total_delta=1)

    with pytest.raises(ValueError, match="component total"):
        _profile(path, 4, "auto", "float32")
