import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_cudagraph_parity.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_cudagraph_parity_helpers",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class CudaGraphParityHelperTest(unittest.TestCase):
    def test_extract_decode_steps_excludes_prefill_shape_differences(self):
        artifact = {
            "logits_steps": [
                {"is_prefill": True, "shape": [1, 10], "logits": "prefill"},
                {"is_prefill": False, "shape": [3, 10], "logits": "decode-1"},
                {"is_prefill": False, "shape": [3, 10], "logits": "decode-2"},
            ]
        }

        self.assertEqual(
            module.extract_decode_steps(artifact),
            [
                {
                    "is_prefill": False,
                    "shape": [3, 10],
                    "logits": "decode-1",
                },
                {
                    "is_prefill": False,
                    "shape": [3, 10],
                    "logits": "decode-2",
                },
            ],
        )

    def test_scenario_lengths_cross_block_boundary_in_second_case(self):
        self.assertEqual(module.scenario_lengths(3, 0), [33, 65, 97])
        second = module.scenario_lengths(9, 1)
        self.assertIn(250, second)
        self.assertGreater(max(second), 256)


if __name__ == "__main__":
    unittest.main()
