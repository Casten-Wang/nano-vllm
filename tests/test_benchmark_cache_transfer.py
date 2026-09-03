import importlib.util
import json
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cache_transfer.py"
SPEC = importlib.util.spec_from_file_location("benchmark_cache_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cache_transfer_benchmark_records_raw_samples_and_limitations():
    payload = MODULE.make_payload(
        kv_bytes=4096,
        scale_bytes=512,
        recurrent_bytes=1024,
        convolution_bytes=512,
        kv_dtype=torch.int8,
    )

    result = MODULE.run_benchmark(
        payload,
        warmup=1,
        repeats=3,
        timeout_s=2.0,
    )

    assert result["scope"].startswith("single-rank synchronous TCP loopback")
    assert len(result["results"]["latency_ms_samples"]) == 3
    assert result["results"]["latency_ms_p50"] > 0
    assert result["results"]["effective_payload_gib_s_p50"] > 0
    assert result["workload"]["components_bytes"]["total"] == 6144
    assert result["workload"]["payload_frame_bytes_sent"] > 6144
    assert result["workload"]["receiver_ack_bytes"] == 1
    assert result["workload"]["framing_overhead_bytes"] > 0
    if torch.cuda.is_available():
        assert result["cuda_install"]["enabled"]
        assert result["cuda_install"]["valid"]
        assert result["cuda_install"]["measured_on_cuda"]
    else:
        assert result["cuda_install"] == {
            "enabled": False,
            "valid": True,
            "reason": "CUDA is unavailable",
        }
    assert any("not cross-node" in item for item in result["limitations"])


def test_cache_transfer_benchmark_loads_exact_preflight_components(tmp_path):
    path = tmp_path / "memory_preflight.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "tp4": {
                        "pd_transfer_components_per_sequence_by_dtype": {
                            "int8": {
                                "model": {
                                    "kv": 4096,
                                    "kv_scales": 512,
                                    "recurrent": 1024,
                                    "convolution": 512,
                                    "total": 6144,
                                }
                            }
                        }
                    }
                }
            }
        )
    )

    payload, profile = MODULE.payload_from_memory_preflight(
        path,
        tp_size=4,
        kv_dtype="int8",
        state_dtype="model",
    )

    assert MODULE.payload_bytes(payload)["total"] == 6144
    assert payload.kv_blocks.dtype is torch.int8
    assert payload.kv_scales.dtype is torch.float16
    assert profile["tp_size"] == 4
