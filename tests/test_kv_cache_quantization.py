import importlib.util
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
        module_path = ROOT / "nanovllm" / "layers" / "kv_cache_quant.py"
        spec = importlib.util.spec_from_file_location(
            "kv_cache_quant_under_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        store_kvcache_int8 = module.store_kvcache_int8
        IMPORT_ERROR = None
    except ModuleNotFoundError as exc:
        store_kvcache_int8 = None
        IMPORT_ERROR = exc


def quantize_reference(tensor):
    source = tensor.float()
    absmax = source.abs().amax(dim=-1)
    scale = torch.clamp(absmax / 127.0, min=1.0e-6)
    scaled = source / scale.unsqueeze(-1)
    rounded = torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        torch.ceil(scaled - 0.5),
    )
    quantized = rounded.clamp(-127, 127).to(torch.int8)
    return quantized, scale


@unittest.skipIf(
    torch is None or IMPORT_ERROR is not None,
    f"missing dependency: {IMPORT_ERROR}",
)
@unittest.skipIf(
    torch is not None and not torch.cuda.is_available(),
    "CUDA is required for the Triton KV store kernel",
)
class Int8KVCacheStoreTest(unittest.TestCase):
    def make_inputs(self):
        device = "cuda"
        key = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0],
                    [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
                ],
                [
                    [-3.0, -1.0, -0.25, 0.25, 0.5, 1.5, 3.0],
                    [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
                ],
            ],
            device=device,
            dtype=torch.float16,
        )
        value = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
                    [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5],
                ],
                [
                    [-1.0, -0.75, -0.5, -0.25, 0.25, 0.75, 1.0],
                    [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0],
                ],
            ],
            device=device,
            dtype=torch.float16,
        )
        return key, value

    def test_store_matches_round_away_from_zero_reference(self):
        key, value = self.make_inputs()
        num_blocks = 2
        block_size = 4
        num_kv_heads = key.size(1)
        head_dim = key.size(2)
        k_cache = torch.full(
            (num_blocks, block_size, num_kv_heads, head_dim),
            99,
            dtype=torch.int8,
            device="cuda",
        )
        v_cache = torch.full_like(k_cache, 99)
        k_scale = torch.full(
            (num_blocks, block_size, num_kv_heads),
            -1.0,
            dtype=torch.float16,
            device="cuda",
        )
        v_scale = torch.full_like(k_scale, -1.0)
        slot_mapping = torch.tensor([0, -1, 5], dtype=torch.int32, device="cuda")

        store_kvcache_int8(
            key,
            value,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            slot_mapping,
        )
        torch.cuda.synchronize()

        expected_k_q, expected_k_scale = quantize_reference(key)
        expected_v_q, expected_v_scale = quantize_reference(value)
        for token_index, slot in ((0, 0), (2, 5)):
            block_id, block_offset = divmod(slot, block_size)
            self.assertTrue(
                torch.equal(
                    k_cache[block_id, block_offset],
                    expected_k_q[token_index],
                )
            )
            self.assertTrue(
                torch.equal(
                    v_cache[block_id, block_offset],
                    expected_v_q[token_index],
                )
            )
            torch.testing.assert_close(
                k_scale[block_id, block_offset].float(),
                expected_k_scale[token_index],
                atol=2e-5,
                rtol=2e-3,
            )
            torch.testing.assert_close(
                v_scale[block_id, block_offset].float(),
                expected_v_scale[token_index],
                atol=2e-5,
                rtol=2e-3,
            )

        # slot_mapping=-1 must not write an arbitrary cache slot. Every slot
        # except the two explicitly mapped ones keeps its sentinel values.
        flat_k = k_cache.view(-1, num_kv_heads, head_dim)
        flat_v = v_cache.view(-1, num_kv_heads, head_dim)
        flat_ks = k_scale.view(-1, num_kv_heads)
        flat_vs = v_scale.view(-1, num_kv_heads)
        for slot in set(range(num_blocks * block_size)) - {0, 5}:
            self.assertTrue(torch.equal(flat_k[slot], torch.full_like(flat_k[slot], 99)))
            self.assertTrue(torch.equal(flat_v[slot], torch.full_like(flat_v[slot], 99)))
            self.assertTrue(torch.equal(flat_ks[slot], torch.full_like(flat_ks[slot], -1.0)))
            self.assertTrue(torch.equal(flat_vs[slot], torch.full_like(flat_vs[slot], -1.0)))

    def test_scales_are_fp16_independent_and_dequantization_is_bounded(self):
        key, value = self.make_inputs()
        num_blocks = 1
        block_size = 4
        num_kv_heads = key.size(1)
        head_dim = key.size(2)
        k_cache = torch.empty(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.int8,
            device="cuda",
        )
        v_cache = torch.empty_like(k_cache)
        k_scale = torch.empty(
            (num_blocks, block_size, num_kv_heads),
            dtype=torch.float16,
            device="cuda",
        )
        v_scale = torch.empty_like(k_scale)
        slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")

        store_kvcache_int8(
            key,
            value,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            slot_mapping,
        )
        torch.cuda.synchronize()

        self.assertEqual(k_scale.dtype, torch.float16)
        self.assertEqual(v_scale.dtype, torch.float16)
        self.assertFalse(torch.equal(k_scale[0, 1], v_scale[0, 1]))
        self.assertEqual(int(k_cache.abs().max().item()), 127)
        self.assertEqual(int(v_cache.abs().max().item()), 127)

        dequant_k = k_cache[0, :3].float() * k_scale[0, :3].float().unsqueeze(-1)
        dequant_v = v_cache[0, :3].float() * v_scale[0, :3].float().unsqueeze(-1)
        k_error = (dequant_k - key.float()).abs()
        v_error = (dequant_v - value.float()).abs()
        k_bound = k_scale[0, :3].float().unsqueeze(-1) * 0.55 + 2e-4
        v_bound = v_scale[0, :3].float().unsqueeze(-1) * 0.55 + 2e-4
        self.assertTrue((k_error <= k_bound).all().item())
        self.assertTrue((v_error <= v_bound).all().item())

        # Zero vectors use EPS rather than a zero scale and quantize exactly to
        # zero. This also covers a non-power-of-two head_dim of 7.
        self.assertTrue((k_scale[0, 0] > 0).all().item())
        self.assertTrue((v_scale[0, 0] > 0).all().item())
        self.assertTrue((k_cache[0, 0] == 0).all().item())
        self.assertTrue((v_cache[0, 0] == 0).all().item())


if __name__ == "__main__":
    unittest.main()
