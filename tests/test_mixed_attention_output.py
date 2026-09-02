from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).parents[1]
CURRENT_CONTEXT = {}


def load_attention_module():
    names = (
        "flash_attn",
        "nanovllm",
        "nanovllm.engine",
        "nanovllm.engine.execution",
        "nanovllm.layers",
        "nanovllm.layers.kv_cache_quant",
        "nanovllm.layers.int8_fused_attention",
        "nanovllm.utils",
        "nanovllm.utils.context",
        "nanovllm.shape_trace",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in (
            "nanovllm",
            "nanovllm.engine",
            "nanovllm.layers",
            "nanovllm.utils",
        ):
            sys.modules[name] = types.ModuleType(name)
        flash = types.ModuleType("flash_attn")
        flash.flash_attn_varlen_func = object
        flash.flash_attn_with_kvcache = object
        sys.modules["flash_attn"] = flash
        execution = types.ModuleType("nanovllm.engine.execution")
        execution.select_int8_decode_attention_path = lambda **_kwargs: ""
        sys.modules[execution.__name__] = execution
        kv_quant = types.ModuleType("nanovllm.layers.kv_cache_quant")
        for name in (
            "dequant_packed_kvcache",
            "store_kvcache",
            "store_kvcache_int8",
            "store_kvcache_int8_range",
            "store_kvcache_range",
        ):
            setattr(kv_quant, name, lambda *_args, **_kwargs: None)
        sys.modules[kv_quant.__name__] = kv_quant
        fused = types.ModuleType("nanovllm.layers.int8_fused_attention")
        fused.fused_int8_decode_attention = object
        fused.partitioned_fused_int8_decode_attention = object
        sys.modules[fused.__name__] = fused
        context = types.ModuleType("nanovllm.utils.context")
        context.get_context = lambda: CURRENT_CONTEXT["value"]
        sys.modules[context.__name__] = context
        trace = types.ModuleType("nanovllm.shape_trace")
        trace.active_trace = lambda: None
        sys.modules[trace.__name__] = trace

        spec = spec_from_file_location(
            "attention_mixed_under_test",
            ROOT / "nanovllm/layers/attention.py",
        )
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


ATTENTION = load_attention_module()


def make_mixed_attention():
    layer = ATTENTION.Attention(2, 3, 3**-0.5, 1)
    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        decode_token_count=2,
        prefill_token_count=3,
        slot_mapping=torch.arange(5),
        decode_block_tables=None,
        decode_context_lens=torch.tensor([4, 5]),
        prefill_block_tables=None,
        prefill_max_seqlen_q=3,
        prefill_cu_seqlens_q=torch.tensor([0, 3]),
        prefill_max_seqlen_k=3,
        prefill_cu_seqlens_k=torch.tensor([0, 3]),
    )
    return layer


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_mixed_attention_reuses_query_only_in_inference(grad_enabled):
    layer = make_mixed_attention()
    query = torch.randn(5, 2, 3, requires_grad=grad_enabled)
    original = query.detach().clone()
    key = torch.randn(5, 1, 3)
    value = torch.randn(5, 1, 3)
    events = []

    def store(*args):
        events.append(("store", args[-2], args[-1]))

    def decode(q, *_args):
        events.append("decode")
        return q + 10

    def prefill(q, *_args):
        events.append("prefill")
        return q + 20

    grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with (
        grad_context,
        patch.object(ATTENTION, "store_kvcache_range", side_effect=store),
        patch.object(layer, "_flash_attn_with_kvcache", side_effect=decode),
        patch.object(layer, "_flash_attn_varlen", side_effect=prefill),
    ):
        output = layer._forward_mixed(query, key, value)

    expected = torch.cat((original[:2] + 10, original[2:] + 20))
    torch.testing.assert_close(output, expected)
    assert (output.data_ptr() == query.data_ptr()) is not grad_enabled
    assert events == [("store", 0, 2), "decode", ("store", 2, 5), "prefill"]
    if grad_enabled:
        output.sum().backward()
        assert query.grad is not None
