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


def test_default_result_prefix_omits_unselected_weight_quantization():
    args = Namespace(
        kv_cache_dtype="auto",
        kv_dequant_backend="fused",
        sliding_window_size=None,
        enable_dynamic_chunked_prefill=False,
        enforce_eager=False,
        qwen35_moe_decode_backend="sorted",
        weight_quant_backend="auto",
    )

    assert benchmark_baseline.default_result_prefix(args) == "baseline"
