import unittest

try:
    import torch

    from projects.triton_fa2.reference import (
        error_summary,
        repeat_kv_for_gqa,
        sdpa_reference,
        selected_sdpa_backend,
        torch_attention_reference,
    )
    from projects.triton_fa2.triton_fa2 import triton_flash_attention_forward

    IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc

try:
    from projects.triton_fa2.triton_fa2_v2 import (
        V2_AUTOTUNE_CONFIGS,
        _prune_v2_configs,
        get_last_v2_autotune_config,
        triton_flash_attention_forward_v2,
        triton_flash_attention_forward_v2_configured,
    )

    V2_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    V2_AUTOTUNE_CONFIGS = ()
    _prune_v2_configs = None
    get_last_v2_autotune_config = None
    triton_flash_attention_forward_v2 = None
    triton_flash_attention_forward_v2_configured = None
    V2_IMPORT_ERROR = exc

try:
    from projects.triton_fa2.benchmark import (
        from_dao_flash_attn_layout,
        load_dao_flash_attn,
        serialize_v2_configs,
        to_dao_flash_attn_layout,
    )

    BENCHMARK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    from_dao_flash_attn_layout = None
    load_dao_flash_attn = None
    serialize_v2_configs = None
    to_dao_flash_attn_layout = None
    BENCHMARK_IMPORT_ERROR = exc


class TritonFlashAttentionV2ConfigurationTest(unittest.TestCase):
    def test_v2_module_is_available(self):
        self.assertIsNone(V2_IMPORT_ERROR, f"V2 import failed: {V2_IMPORT_ERROR}")

    @unittest.skipIf(
        V2_IMPORT_ERROR is not None, f"missing V2 dependency: {V2_IMPORT_ERROR}"
    )
    def test_causal_pruning_matches_official_constraints(self):
        configs = _prune_v2_configs(
            list(V2_AUTOTUNE_CONFIGS),
            {},
            SEQ_LEN=64,
            HEAD_DIM=64,
            CAUSAL=True,
        )

        self.assertTrue(configs)
        self.assertTrue(all(config.kwargs["BLOCK_M"] <= 64 for config in configs))
        self.assertTrue(
            all(
                config.kwargs["BLOCK_M"] >= config.kwargs["BLOCK_N"]
                for config in configs
            )
        )
        self.assertTrue(all(config.kwargs["BLOCK_N"] <= 64 for config in configs))

    @unittest.skipIf(
        V2_IMPORT_ERROR is not None, f"missing V2 dependency: {V2_IMPORT_ERROR}"
    )
    def test_noncausal_pruning_keeps_legal_large_n_tiles(self):
        configs = _prune_v2_configs(
            list(V2_AUTOTUNE_CONFIGS),
            {},
            SEQ_LEN=256,
            HEAD_DIM=128,
            CAUSAL=False,
        )

        self.assertTrue(configs)
        self.assertTrue(any(config.kwargs["BLOCK_N"] == 128 for config in configs))
        self.assertTrue(any(config.kwargs["BLOCK_M"] == 64 for config in configs))

    @unittest.skipIf(
        V2_IMPORT_ERROR is not None, f"missing V2 dependency: {V2_IMPORT_ERROR}"
    )
    def test_causal_pruning_keeps_block_n_larger_than_head_dim(self):
        configs = _prune_v2_configs(
            list(V2_AUTOTUNE_CONFIGS),
            {},
            SEQ_LEN=128,
            HEAD_DIM=64,
            CAUSAL=True,
        )

        self.assertTrue(
            any(
                config.kwargs["BLOCK_M"] == 128
                and config.kwargs["BLOCK_N"] == 128
                for config in configs
            )
        )

    @unittest.skipIf(
        V2_IMPORT_ERROR is not None, f"missing V2 dependency: {V2_IMPORT_ERROR}"
    )
    def test_pruning_rejects_empty_candidate_set(self):
        with self.assertRaisesRegex(ValueError, "no valid V2 autotune"):
            _prune_v2_configs(
                list(V2_AUTOTUNE_CONFIGS),
                {},
                SEQ_LEN=32,
                HEAD_DIM=32,
                CAUSAL=True,
            )

    @unittest.skipIf(
        BENCHMARK_IMPORT_ERROR is not None,
        f"missing benchmark dependency: {BENCHMARK_IMPORT_ERROR}",
    )
    def test_dao_layout_round_trip(self):
        tensor = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        dao_layout = to_dao_flash_attn_layout(tensor)
        self.assertEqual(dao_layout.shape, (1, 3, 2, 4))
        self.assertTrue(torch.equal(from_dao_flash_attn_layout(dao_layout), tensor))

    @unittest.skipIf(
        BENCHMARK_IMPORT_ERROR is not None,
        f"missing benchmark dependency: {BENCHMARK_IMPORT_ERROR}",
    )
    def test_optional_dao_provider_has_explicit_status(self):
        provider, reason = load_dao_flash_attn()
        self.assertTrue((provider is None) ^ (reason is None))

    @unittest.skipIf(
        BENCHMARK_IMPORT_ERROR is not None,
        f"missing benchmark dependency: {BENCHMARK_IMPORT_ERROR}",
    )
    def test_v2_config_serialization_is_stable(self):
        configs = serialize_v2_configs()
        self.assertEqual(len(configs), len(V2_AUTOTUNE_CONFIGS))
        self.assertEqual(
            set(configs[0]),
            {"block_m", "block_n", "num_warps", "num_stages"},
        )
        self.assertEqual(configs[0]["block_m"], 64)
        self.assertEqual(configs[0]["block_n"], 32)


class TritonFlashAttentionReferenceTest(unittest.TestCase):
    @unittest.skipIf(
        torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}"
    )
    def test_repeat_kv_for_gqa_uses_contiguous_head_groups(self):
        q = torch.empty(1, 4, 2, 1)
        k = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
        v = k + 10.0

        repeated_k, repeated_v = repeat_kv_for_gqa(q, k, v)

        self.assertEqual(repeated_k.shape, (1, 4, 2, 1))
        self.assertTrue(torch.equal(repeated_k[:, 0], repeated_k[:, 1]))
        self.assertTrue(torch.equal(repeated_k[:, 2], repeated_k[:, 3]))
        self.assertTrue(torch.equal(repeated_v[:, 0], repeated_v[:, 1]))
        self.assertTrue(torch.equal(repeated_v[:, 2], repeated_v[:, 3]))

    @unittest.skipIf(
        torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}"
    )
    def test_torch_reference_supports_gqa(self):
        torch.manual_seed(0)
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn_like(k)

        actual = torch_attention_reference(q, k, v, causal=True)
        repeated_k, repeated_v = repeat_kv_for_gqa(q, k, v)
        expected = torch_attention_reference(q, repeated_k, repeated_v, causal=True)

        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipIf(
        torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}"
    )
    def test_rejects_invalid_gqa_ratio(self):
        q = torch.randn(1, 3, 8, 64)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn_like(k)

        with self.assertRaisesRegex(ValueError, "divisible"):
            repeat_kv_for_gqa(q, k, v)

    @unittest.skipIf(
        torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}"
    )
    def test_selected_sdpa_backend_returns_enum_name(self):
        q = torch.randn(1, 2, 8, 64)
        backend = selected_sdpa_backend(q, q, q, causal=True)

        self.assertIn(
            backend,
            {
                "MATH",
                "FLASH_ATTENTION",
                "EFFICIENT_ATTENTION",
                "CUDNN_ATTENTION",
                "OVERRIDEABLE",
            },
        )


@unittest.skipIf(
    torch is None or IMPORT_ERROR is not None, f"missing dependency: {IMPORT_ERROR}"
)
@unittest.skipIf(
    torch is not None and not torch.cuda.is_available(),
    "CUDA is required for the Triton kernel",
)
class TritonFlashAttentionForwardTest(unittest.TestCase):
    def assert_matches_reference(
        self,
        *,
        dtype,
        causal,
        seq_len,
        head_dim,
        num_q_heads=4,
        num_kv_heads=None,
        seed=0,
    ):
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
        torch.manual_seed(seed)
        q = torch.randn(1, num_q_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        actual = triton_flash_attention_forward(
            q,
            k,
            v,
            causal=causal,
            block_m=16,
            block_n=32,
            num_warps=4,
            num_stages=3,
            loop_num_stages=2,
        )
        expected = torch_attention_reference(q, k, v, causal=causal)
        summary = error_summary(actual, expected)
        max_abs_limit = 4e-2 if dtype is torch.bfloat16 else 3e-2
        mean_abs_limit = 5e-3 if dtype is torch.bfloat16 else 3e-3
        self.assertLess(summary["max_abs"], max_abs_limit)
        self.assertLess(summary["mean_abs"], mean_abs_limit)
        self.assertTrue(torch.isfinite(actual).all().item())

    def assert_v2_matches_reference(
        self,
        *,
        dtype,
        causal,
        seq_len,
        head_dim,
        num_q_heads=4,
        num_kv_heads=None,
        seed=0,
    ):
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
        torch.manual_seed(seed)
        q = torch.randn(1, num_q_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        actual = triton_flash_attention_forward_v2_configured(
            q,
            k,
            v,
            causal=causal,
            block_m=64,
            block_n=32,
            num_warps=4,
            num_stages=2,
        )
        expected = torch_attention_reference(q, k, v, causal=causal)
        summary = error_summary(actual, expected)
        max_abs_limit = 4e-2 if dtype is torch.bfloat16 else 3e-2
        mean_abs_limit = 5e-3 if dtype is torch.bfloat16 else 3e-3
        self.assertLess(summary["max_abs"], max_abs_limit)
        self.assertLess(summary["mean_abs"], mean_abs_limit)
        self.assertTrue(torch.isfinite(actual).all().item())

    def test_correctness_matrix_against_explicit_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (True, False):
                for head_dim in (64, 128):
                    for seq_len in (64, 97, 128, 256):
                        with self.subTest(
                            dtype=dtype,
                            causal=causal,
                            head_dim=head_dim,
                            seq_len=seq_len,
                        ):
                            self.assert_matches_reference(
                                dtype=dtype,
                                causal=causal,
                                seq_len=seq_len,
                                head_dim=head_dim,
                            )

    def test_v2_correctness_matrix_against_explicit_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (True, False):
                for head_dim in (64, 128):
                    for seq_len in (64, 65, 97, 129):
                        with self.subTest(
                            dtype=dtype,
                            causal=causal,
                            head_dim=head_dim,
                            seq_len=seq_len,
                        ):
                            self.assert_v2_matches_reference(
                                dtype=dtype,
                                causal=causal,
                                seq_len=seq_len,
                                head_dim=head_dim,
                            )

    def test_v2_gqa_correctness_matrix(self):
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (True, False):
                for seq_len in (65, 129):
                    with self.subTest(
                        dtype=dtype,
                        causal=causal,
                        seq_len=seq_len,
                    ):
                        self.assert_v2_matches_reference(
                            dtype=dtype,
                            causal=causal,
                            seq_len=seq_len,
                            head_dim=128,
                            num_q_heads=4,
                            num_kv_heads=2,
                            seed=1,
                        )

    def test_v2_causal_output_is_independent_of_future_kv(self):
        torch.manual_seed(7)
        seq_len = 129
        prefix_len = 65
        q = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        changed_k = k.clone()
        changed_v = v.clone()
        changed_k[:, :, prefix_len:] = (
            torch.randn_like(changed_k[:, :, prefix_len:]) * 100
        )
        changed_v[:, :, prefix_len:] = (
            torch.randn_like(changed_v[:, :, prefix_len:]) * 100
        )

        baseline = triton_flash_attention_forward_v2_configured(
            q, k, v, causal=True, block_m=64, block_n=32
        )
        changed = triton_flash_attention_forward_v2_configured(
            q, changed_k, changed_v, causal=True, block_m=64, block_n=32
        )
        self.assertTrue(
            torch.equal(baseline[:, :, :prefix_len], changed[:, :, :prefix_len])
        )

    def test_v2_agrees_with_v1(self):
        torch.manual_seed(8)
        q = torch.randn(1, 4, 129, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        v1 = triton_flash_attention_forward(
            q, k, v, causal=True, block_m=16, block_n=32
        )
        v2 = triton_flash_attention_forward_v2_configured(
            q, k, v, causal=True, block_m=64, block_n=32
        )
        summary = error_summary(v2, v1)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)

    def test_v2_autotuned_public_launcher(self):
        torch.manual_seed(9)
        q = torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = triton_flash_attention_forward_v2(q, k, v, causal=True)
        expected = torch_attention_reference(q, k, v, causal=True)
        summary = error_summary(actual, expected)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)
        config = get_last_v2_autotune_config()
        self.assertIsNotNone(config)
        self.assertEqual(
            set(config),
            {"block_m", "block_n", "num_warps", "num_stages"},
        )
        self.assertGreaterEqual(config["block_m"], config["block_n"])

    def test_v2_rejects_invalid_gqa_head_ratio(self):
        q = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        with self.assertRaisesRegex(ValueError, "divisible"):
            triton_flash_attention_forward_v2(q, k, v)

    def test_v2_rejects_unsupported_head_dim(self):
        q = torch.randn(1, 2, 64, 96, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with self.assertRaisesRegex(ValueError, "head_dim"):
            triton_flash_attention_forward_v2(q, k, v)

    def test_v2_rejects_invalid_causal_tile_relation(self):
        q = torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with self.assertRaisesRegex(ValueError, "block_m >= block_n"):
            triton_flash_attention_forward_v2_configured(
                q,
                k,
                v,
                causal=True,
                block_m=64,
                block_n=128,
            )

    def test_v2_accepts_sequence_tile_larger_than_head_dim(self):
        torch.manual_seed(10)
        q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        actual = triton_flash_attention_forward_v2_configured(
            q,
            k,
            v,
            causal=True,
            block_m=128,
            block_n=128,
            num_warps=4,
            num_stages=2,
        )
        expected = torch_attention_reference(q, k, v, causal=True)
        summary = error_summary(actual, expected)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)

    def test_gqa_correctness_matrix(self):
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (True, False):
                for head_dim in (64, 128):
                    for seq_len in (97, 128):
                        with self.subTest(
                            dtype=dtype,
                            causal=causal,
                            head_dim=head_dim,
                            seq_len=seq_len,
                        ):
                            self.assert_matches_reference(
                                dtype=dtype,
                                causal=causal,
                                seq_len=seq_len,
                                head_dim=head_dim,
                                num_q_heads=4,
                                num_kv_heads=2,
                                seed=1,
                            )

    def test_matches_default_sdpa_for_larger_causal_shape(self):
        torch.manual_seed(2)
        q = torch.randn(1, 4, 256, 128, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = triton_flash_attention_forward(
            q,
            k,
            v,
            causal=True,
            block_m=32,
            block_n=64,
            loop_num_stages=2,
        )
        expected = sdpa_reference(q, k, v, causal=True, backend="default")
        summary = error_summary(actual, expected)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)

    def test_causal_early_stop_matches_full_scan(self):
        torch.manual_seed(6)
        q = torch.randn(1, 2, 97, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        optimized = triton_flash_attention_forward(
            q,
            k,
            v,
            causal=True,
            causal_early_stop=True,
            block_m=16,
            block_n=32,
        )
        full_scan = triton_flash_attention_forward(
            q,
            k,
            v,
            causal=True,
            causal_early_stop=False,
            block_m=16,
            block_n=32,
        )
        summary = error_summary(optimized, full_scan)
        self.assertLess(summary["max_abs"], 3e-3)
        self.assertLess(summary["mean_abs"], 5e-4)

    def test_matches_default_sdpa_for_gqa(self):
        torch.manual_seed(3)
        q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        actual = triton_flash_attention_forward(q, k, v, causal=True)
        expected = sdpa_reference(q, k, v, causal=True, backend="default")
        summary = error_summary(actual, expected)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)

    def test_forced_flash_sdpa_when_supported(self):
        torch.manual_seed(4)
        q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        try:
            forced_flash = sdpa_reference(q, k, v, causal=True, backend="flash")
        except (RuntimeError, NotImplementedError) as exc:
            self.skipTest(f"forced FlashAttention SDPA unavailable: {exc}")
        default = sdpa_reference(q, k, v, causal=True, backend="default")
        summary = error_summary(forced_flash, default)
        self.assertLess(summary["max_abs"], 3e-2)
        self.assertLess(summary["mean_abs"], 3e-3)

    def test_causal_output_is_independent_of_future_kv(self):
        torch.manual_seed(5)
        seq_len = 96
        prefix_len = 48
        q = torch.randn(1, 4, seq_len, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        changed_k = k.clone()
        changed_v = v.clone()
        changed_k[:, :, prefix_len:] = (
            torch.randn_like(changed_k[:, :, prefix_len:]) * 100
        )
        changed_v[:, :, prefix_len:] = (
            torch.randn_like(changed_v[:, :, prefix_len:]) * 100
        )

        baseline = triton_flash_attention_forward(
            q, k, v, causal=True, block_m=16, block_n=32
        )
        changed = triton_flash_attention_forward(
            q,
            changed_k,
            changed_v,
            causal=True,
            block_m=16,
            block_n=32,
        )

        self.assertTrue(
            torch.equal(baseline[:, :, :prefix_len], changed[:, :, :prefix_len])
        )

    def test_rejects_unsupported_head_dim(self):
        q = torch.randn(1, 2, 64, 96, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with self.assertRaisesRegex(AssertionError, "head_dim"):
            triton_flash_attention_forward(q, k, v)

    def test_rejects_invalid_gqa_head_ratio(self):
        q = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        with self.assertRaisesRegex(AssertionError, "divisible"):
            triton_flash_attention_forward(q, k, v)


if __name__ == "__main__":
    unittest.main()
