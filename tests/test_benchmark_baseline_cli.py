from argparse import Namespace
import sys

from scripts import benchmark_baseline


def test_weight_quant_backend_is_forwarded_by_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_baseline.py", "--weight-quant-backend", "triton"],
    )

    args = benchmark_baseline.parse_args()

    assert args.weight_quant_backend == "triton"
    assert benchmark_baseline.default_result_prefix(args).endswith("wq-triton")


def test_decode_conv_backend_is_named_in_result_prefix(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_baseline.py",
            "--qwen35-decode-conv-backend",
            "channel_accumulate",
        ],
    )

    args = benchmark_baseline.parse_args()

    assert args.qwen35_decode_conv_backend == "channel_accumulate"
    assert benchmark_baseline.default_result_prefix(args).endswith(
        "conv-channel_accumulate"
    )


def test_default_result_prefix_omits_unselected_weight_quantization():
    args = Namespace(
        kv_cache_dtype="auto",
        kv_dequant_backend="fused",
        sliding_window_size=None,
        enable_dynamic_chunked_prefill=False,
        enable_decode_kv_reservation=False,
        enforce_eager=False,
        qwen35_decode_conv_backend="weighted",
        qwen35_moe_decode_backend="sorted",
        weight_quant_backend="auto",
    )

    assert benchmark_baseline.default_result_prefix(args) == "baseline"


def test_decode_kv_reservation_is_named_in_result_prefix():
    args = Namespace(
        kv_cache_dtype="auto",
        kv_dequant_backend="fused",
        sliding_window_size=None,
        enable_dynamic_chunked_prefill=True,
        enable_decode_kv_reservation=True,
        enforce_eager=False,
        qwen35_decode_conv_backend="weighted",
        qwen35_moe_decode_backend="sorted",
        weight_quant_backend="auto",
    )

    assert benchmark_baseline.default_result_prefix(args).endswith(
        "dynchunk_decode-kv-reserve"
    )
