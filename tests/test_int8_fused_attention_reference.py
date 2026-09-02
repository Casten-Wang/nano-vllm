import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    try:
        # Load the low-level kernel without importing nanovllm.__init__, whose
        # public LLM import has optional serving dependencies not needed here.
        managed_names = (
            "nanovllm",
            "nanovllm.engine",
            "nanovllm.engine.execution",
        )
        saved_modules = {name: sys.modules.get(name) for name in managed_names}
        try:
            nanovllm_pkg = types.ModuleType("nanovllm")
            nanovllm_pkg.__path__ = [str(ROOT / "nanovllm")]
            engine_pkg = types.ModuleType("nanovllm.engine")
            engine_pkg.__path__ = [str(ROOT / "nanovllm" / "engine")]
            sys.modules["nanovllm"] = nanovllm_pkg
            sys.modules["nanovllm.engine"] = engine_pkg
            execution_path = ROOT / "nanovllm" / "engine" / "execution.py"
            execution_spec = importlib.util.spec_from_file_location(
                "nanovllm.engine.execution",
                execution_path,
            )
            execution_module = importlib.util.module_from_spec(execution_spec)
            assert execution_spec.loader is not None
            sys.modules["nanovllm.engine.execution"] = execution_module
            execution_spec.loader.exec_module(execution_module)
            MODULE_PATH = ROOT / "nanovllm" / "layers" / "int8_fused_attention.py"
            SPEC = importlib.util.spec_from_file_location(
                "int8_fused_attention",
                MODULE_PATH,
            )
            int8_fused_attention = importlib.util.module_from_spec(SPEC)
            assert SPEC.loader is not None
            SPEC.loader.exec_module(int8_fused_attention)
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        fused_int8_decode_attention = int8_fused_attention.fused_int8_decode_attention
        fused_int8_decode_attention_latev = int8_fused_attention.fused_int8_decode_attention_latev
        fused_int8_decode_attention_v3 = int8_fused_attention.fused_int8_decode_attention_v3
        partitioned_fused_int8_decode_attention = int8_fused_attention.partitioned_fused_int8_decode_attention
        IMPORT_ERROR = None
    except ModuleNotFoundError as exc:
        fused_int8_decode_attention = None
        fused_int8_decode_attention_latev = None
        fused_int8_decode_attention_v3 = None
        partitioned_fused_int8_decode_attention = None
        IMPORT_ERROR = exc


@unittest.skipIf(torch is None or IMPORT_ERROR is not None, "torch/triton unavailable")
class PartitionedWorkspaceTest(unittest.TestCase):
    def test_buffer_pool_reuses_storage_across_shapes(self):
        pool = int8_fused_attention.PartitionedDecodeBufferPool()
        large_q = torch.empty(4, 3, 8, dtype=torch.float16)
        large_workspace, large_output = pool.acquire(large_q, 5, 8)
        workspace_storage = large_workspace[0].untyped_storage().data_ptr()
        output_storage = large_output.untyped_storage().data_ptr()

        small_q = torch.empty(2, 3, 8, dtype=torch.float16)
        small_workspace, small_output = pool.acquire(small_q, 2, 8)

        self.assertEqual(
            small_workspace[0].untyped_storage().data_ptr(),
            workspace_storage,
        )
        self.assertEqual(small_output.untyped_storage().data_ptr(), output_storage)
        self.assertEqual(small_output.shape, small_q.shape)
        int8_fused_attention.validate_partitioned_workspace(
            small_workspace,
            small_q,
            2,
            8,
        )
        int8_fused_attention.validate_partitioned_output(
            small_output,
            small_q,
            small_workspace,
        )

    def test_buffer_pool_grows_each_storage_independently(self):
        pool = int8_fused_attention.PartitionedDecodeBufferPool()
        q = torch.empty(1, 2, 4, dtype=torch.float16)
        first_workspace, first_output = pool.acquire(q, 1, 4)
        first_workspace_size = first_workspace[0].untyped_storage().nbytes()
        first_output_ptr = first_output.untyped_storage().data_ptr()

        larger_workspace, same_output = pool.acquire(q, 7, 8)

        self.assertGreater(
            larger_workspace[0].untyped_storage().nbytes(),
            first_workspace_size,
        )
        self.assertEqual(same_output.untyped_storage().data_ptr(), first_output_ptr)

        stats = pool.storage_stats()
        self.assertEqual(
            stats["total_bytes"],
            stats["workspace_bytes"] + stats["output_bytes"],
        )
        self.assertEqual(stats["output_bytes"], q.numel() * q.element_size())

    def test_partial_workspace_views_share_one_allocation(self):
        q = torch.empty(2, 3, 4)
        num_partitions = 5
        block_head_dim = 8
        partial_acc, partial_m, partial_l = (
            int8_fused_attention.allocate_partitioned_workspace(
                q,
                num_partitions,
                block_head_dim,
            )
        )

        storage = partial_acc.untyped_storage().data_ptr()
        self.assertEqual(partial_acc.untyped_storage().data_ptr(), storage)
        self.assertEqual(partial_m.untyped_storage().data_ptr(), storage)
        self.assertEqual(partial_l.untyped_storage().data_ptr(), storage)
        self.assertEqual(
            partial_acc.untyped_storage().nbytes(),
            (
                partial_acc.numel()
                + partial_m.numel()
                + partial_l.numel()
            )
            * partial_acc.element_size(),
        )

    def test_reusable_workspace_validation_accepts_exact_layout(self):
        q = torch.empty(2, 3, 4)
        workspace = int8_fused_attention.allocate_partitioned_workspace(q, 5, 8)

        actual = int8_fused_attention.validate_partitioned_workspace(
            workspace, q, 5, 8
        )

        self.assertIs(actual, workspace)

    def test_reusable_workspace_validation_rejects_wrong_shape(self):
        q = torch.empty(2, 3, 4)
        workspace = int8_fused_attention.allocate_partitioned_workspace(q, 5, 8)

        with self.assertRaisesRegex(ValueError, "workspace acc has shape"):
            int8_fused_attention.validate_partitioned_workspace(
                (workspace[0][:-1], workspace[1], workspace[2]), q, 5, 8
            )

    def test_reusable_workspace_validation_rejects_overlap(self):
        q = torch.empty(1, 1, 4)
        storage = torch.empty(12, dtype=torch.float32)
        acc = storage[:8].view(1, 1, 1, 8)
        overlapping_m = storage[:1].view(1, 1, 1)
        nonoverlapping_l = storage[8:9].view(1, 1, 1)

        with self.assertRaisesRegex(ValueError, "workspace acc overlaps m"):
            int8_fused_attention.validate_partitioned_workspace(
                (acc, overlapping_m, nonoverlapping_l), q, 1, 8
            )

    def test_reusable_output_validation_rejects_q_alias(self):
        q = torch.empty(1, 2, 4)
        workspace = int8_fused_attention.allocate_partitioned_workspace(q, 1, 4)

        with self.assertRaisesRegex(ValueError, "must not alias q"):
            int8_fused_attention.validate_partitioned_output(q, q, workspace)


def reference_int8_decode_attention(
    q,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    block_tables,
    context_lens,
    softmax_scale,
    sliding_window_size=None,
):
    """Small torch reference for the fused decode kernel.

    It intentionally mirrors the paged-cache addressing, GQA head mapping, and
    sliding-window mask used by the Triton kernel so GPU tests can compare the
    fused output against a simple implementation.
    """
    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    q_heads_per_kv_head = num_heads // num_kv_heads
    out = torch.empty_like(q)

    for seq_id in range(num_seqs):
        context_len = int(context_lens[seq_id].item())
        if sliding_window_size is None:
            window_start = 0
        else:
            window_start = max(0, context_len - sliding_window_size)
        positions = list(range(window_start, context_len))
        for q_head_id in range(num_heads):
            kv_head_id = q_head_id // q_heads_per_kv_head
            keys = []
            values = []
            for pos in positions:
                logical_block_id = pos // block_size
                token_offset = pos % block_size
                physical_block_id = int(block_tables[seq_id, logical_block_id].item())
                key = (
                    k_cache[physical_block_id, token_offset, kv_head_id].to(torch.float32)
                    * k_scale[physical_block_id, token_offset, kv_head_id].to(torch.float32)
                )
                value = (
                    v_cache[physical_block_id, token_offset, kv_head_id].to(torch.float32)
                    * v_scale[physical_block_id, token_offset, kv_head_id].to(torch.float32)
                )
                keys.append(key)
                values.append(value)

            key_tensor = torch.stack(keys, dim=0)
            value_tensor = torch.stack(values, dim=0)
            scores = torch.matmul(key_tensor, q[seq_id, q_head_id].to(torch.float32)) * softmax_scale
            probs = torch.softmax(scores, dim=0)
            out[seq_id, q_head_id] = torch.matmul(probs, value_tensor).to(q.dtype)

    return out


@unittest.skipIf(torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}")
@unittest.skipIf(torch is not None and not torch.cuda.is_available(), "CUDA is required for the Triton kernel")
class Int8FusedAttentionReferenceTest(unittest.TestCase):
    def make_case(self):
        torch.manual_seed(0)
        device = "cuda"
        num_blocks = 5
        block_size = 8
        num_seqs = 3
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float16)
        k_cache = torch.randint(
            -40,
            41,
            (num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=torch.int8,
        )
        v_cache = torch.randint(
            -40,
            41,
            (num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=torch.int8,
        )
        k_scale = (torch.rand(num_blocks, block_size, num_kv_heads, device=device, dtype=torch.float16) * 0.05 + 0.001)
        v_scale = (torch.rand(num_blocks, block_size, num_kv_heads, device=device, dtype=torch.float16) * 0.05 + 0.001)
        block_tables = torch.tensor(
            [
                [0, 2, 4],
                [1, 3, -1],
                [4, 0, 2],
            ],
            device=device,
            dtype=torch.int32,
        )
        context_lens = torch.tensor([19, 12, 21], device=device, dtype=torch.int32)
        return q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens

    def assert_fused_matches_reference(self, sliding_window_size):
        q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens = self.make_case()
        softmax_scale = q.shape[-1] ** -0.5
        actual = fused_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
            block_tokens=4,
        )
        expected = reference_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def assert_fused_v3_matches_reference(self, sliding_window_size):
        q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens = self.make_case()
        softmax_scale = q.shape[-1] ** -0.5
        actual = fused_int8_decode_attention_v3(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
            block_tokens=4,
        )
        expected = reference_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def assert_fused_latev_matches_reference(self, sliding_window_size):
        q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens = self.make_case()
        softmax_scale = q.shape[-1] ** -0.5
        actual = fused_int8_decode_attention_latev(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
            block_tokens=4,
        )
        expected = reference_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def assert_partitioned_fused_matches_reference(self, sliding_window_size):
        q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens = self.make_case()
        softmax_scale = q.shape[-1] ** -0.5
        actual = partitioned_fused_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
            block_tokens=4,
            partition_size=8,
            max_context_len=int(context_lens.max().item()),
        )
        expected = reference_int8_decode_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_tables,
            context_lens,
            softmax_scale,
            sliding_window_size=sliding_window_size,
        )
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_full_context_decode_matches_reference(self):
        self.assert_fused_matches_reference(sliding_window_size=None)

    def test_sliding_window_decode_matches_reference(self):
        self.assert_fused_matches_reference(sliding_window_size=7)

    def test_sliding_window_crosses_block_boundary(self):
        self.assert_fused_matches_reference(sliding_window_size=11)

    def test_v3_full_context_decode_matches_reference(self):
        self.assert_fused_v3_matches_reference(sliding_window_size=None)

    def test_v3_sliding_window_decode_matches_reference(self):
        self.assert_fused_v3_matches_reference(sliding_window_size=7)

    def test_v3_sliding_window_crosses_block_boundary(self):
        self.assert_fused_v3_matches_reference(sliding_window_size=11)

    def test_latev_full_context_decode_matches_reference(self):
        self.assert_fused_latev_matches_reference(sliding_window_size=None)

    def test_latev_sliding_window_decode_matches_reference(self):
        self.assert_fused_latev_matches_reference(sliding_window_size=7)

    def test_latev_sliding_window_crosses_block_boundary(self):
        self.assert_fused_latev_matches_reference(sliding_window_size=11)

    def test_partitioned_full_context_decode_matches_reference(self):
        self.assert_partitioned_fused_matches_reference(sliding_window_size=None)

    def test_partitioned_sliding_window_decode_matches_reference(self):
        self.assert_partitioned_fused_matches_reference(sliding_window_size=7)

    def test_partitioned_sliding_window_crosses_block_boundary(self):
        self.assert_partitioned_fused_matches_reference(sliding_window_size=11)


if __name__ == "__main__":
    unittest.main()
