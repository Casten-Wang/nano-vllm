import importlib.util
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nanovllm" / "shape_trace.py"
SPEC = importlib.util.spec_from_file_location("shape_trace_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class ShapeTraceTest(unittest.TestCase):
    def make_context(self):
        return types.SimpleNamespace(
            is_prefill=False,
            is_mixed=False,
            max_seqlen_q=0,
            max_seqlen_k=0,
            max_context_len=17,
            sliding_window_size=None,
            decode_token_count=2,
            prefill_token_count=0,
            decode_max_context_len=17,
            prefill_max_seqlen_q=0,
            prefill_max_seqlen_k=0,
            slot_mapping=torch.tensor([4, 9], dtype=torch.int32),
            context_lens=torch.tensor([8, 17], dtype=torch.int32),
            block_tables=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
            dequant_block_ids=None,
            dequant_block_tables=None,
            decode_context_lens=None,
            decode_block_tables=None,
            decode_dequant_block_ids=None,
            decode_dequant_block_tables=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            prefill_cu_seqlens_q=None,
            prefill_cu_seqlens_k=None,
            prefill_block_tables=None,
            prefill_dequant_block_ids=None,
            prefill_dequant_block_tables=None,
        )

    def test_disabled_trace_does_not_record(self):
        with patch.dict(os.environ, {"NANOVLLM_SHAPE_TRACE": "0"}):
            trace = module.ShapeTrace()
        trace.record({"event": "should_not_be_saved"})
        self.assertFalse(trace.to_dict()["enabled"])
        self.assertEqual(trace.to_dict()["events"], [])
        self.assertEqual(trace.to_dict()["dropped_events"], 0)

    def test_tensor_metadata_contains_layout_and_small_values_only(self):
        small = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        metadata = module.tensor_metadata(
            "small",
            small,
            include_values=True,
            max_values=8,
        )
        self.assertEqual(metadata["shape"], [2, 3])
        self.assertEqual(metadata["stride"], [3, 1])
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["numel"], 6)
        self.assertEqual(metadata["bytes"], 24)
        self.assertEqual(metadata["values"], list(range(6)))

        large = torch.arange(10)
        large_metadata = module.tensor_metadata(
            "large",
            large,
            include_values=True,
            max_values=8,
        )
        self.assertNotIn("values", large_metadata)

    def test_trace_is_bounded_and_reports_dropped_events(self):
        with patch.dict(os.environ, {"NANOVLLM_SHAPE_TRACE": "1"}):
            trace = module.ShapeTrace(max_events=2)
        trace.record({"event": "one"})
        trace.record({"event": "two"})
        trace.record({"event": "three"})
        result = trace.to_dict()
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["dropped_events"], 1)

    def test_model_step_records_context_and_input_metadata(self):
        with patch.dict(os.environ, {"NANOVLLM_SHAPE_TRACE": "1"}):
            trace = module.ShapeTrace(max_events=4)
        trace.record_model_step(
            input_ids=torch.tensor([11, 12], dtype=torch.int64),
            positions=torch.tensor([7, 17], dtype=torch.int64),
            context=self.make_context(),
            model_path="decode_eager",
            attention_paths=("int8_fused_decode",),
            graph_bucket=None,
        )
        event = trace.to_dict()["events"][0]
        self.assertEqual(event["event"], "model_step_inputs")
        self.assertEqual(event["model_path"], "decode_eager")
        self.assertEqual(event["tensors"]["input_ids"]["shape"], [2])
        self.assertEqual(
            event["context"]["tensors"]["context_lens"]["values"],
            [8, 17],
        )
        self.assertEqual(
            event["context"]["tensors"]["block_tables"]["values"],
            [0, 2, 1, 3],
        )

    def test_attention_event_records_optional_scales(self):
        with patch.dict(os.environ, {"NANOVLLM_SHAPE_TRACE": "1"}):
            trace = module.ShapeTrace(max_events=4)
        context = self.make_context()
        trace.record_attention(
            layer_id=3,
            q=torch.zeros(2, 4, 8, dtype=torch.bfloat16),
            k=torch.zeros(2, 2, 8, dtype=torch.bfloat16),
            v=torch.zeros(2, 2, 8, dtype=torch.bfloat16),
            k_cache=torch.zeros(4, 256, 2, 8, dtype=torch.int8),
            v_cache=torch.zeros(4, 256, 2, 8, dtype=torch.int8),
            k_scale=torch.ones(4, 256, 2, dtype=torch.float16),
            v_scale=torch.ones(4, 256, 2, dtype=torch.float16),
            context=context,
        )
        event = trace.to_dict()["events"][0]
        self.assertEqual(event["layer_id"], 3)
        self.assertEqual(event["tensors"]["q"]["dtype"], "torch.bfloat16")
        self.assertEqual(event["tensors"]["k_cache"]["dtype"], "torch.int8")
        self.assertEqual(event["tensors"]["k_scale"]["dtype"], "torch.float16")


if __name__ == "__main__":
    unittest.main()
