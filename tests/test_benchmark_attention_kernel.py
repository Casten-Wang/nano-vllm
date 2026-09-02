from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import torch


ROOT = Path(__file__).parents[1]
flash_attn = types.ModuleType("flash_attn")
flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None)
flash_attn.flash_attn_with_kvcache = object
original_flash_attn = sys.modules.get("flash_attn")
try:
    sys.modules["flash_attn"] = flash_attn
    SPEC = spec_from_file_location(
        "benchmark_attention_kernel",
        ROOT / "scripts" / "benchmark_attention_kernel.py",
    )
    assert SPEC is not None and SPEC.loader is not None
    MODULE = module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    if original_flash_attn is None:
        sys.modules.pop("flash_attn", None)
    else:
        sys.modules["flash_attn"] = original_flash_attn


def manifest(
    include_packed_dequant_flash,
    *,
    context_len=8,
    sliding_window_size=None,
):
    args = SimpleNamespace(
        batch_size=1,
        context_len=context_len,
        block_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        dtype="float16",
        sliding_window_size=sliding_window_size,
        variants="v3",
        block_tokens="4",
        num_warps="4",
        num_stages="2",
        partition_sizes="8",
        include_partitioned=True,
        include_packed_dequant_flash=include_packed_dequant_flash,
        dequant_block_tokens=4,
    )
    num_blocks = (context_len + args.block_size - 1) // args.block_size
    q = torch.empty(1, 2, 4)
    k_reference = torch.empty(num_blocks, 8, 1, 4)
    k_int8 = torch.empty(num_blocks, 8, 1, 4, dtype=torch.int8)
    scale = torch.empty(num_blocks, 8, 1, dtype=torch.float16)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    context_lens = torch.full((1,), context_len, dtype=torch.int32)
    return MODULE.build_shape_manifest(
        args,
        q=q,
        k_reference=k_reference,
        v_reference=k_reference,
        k_int8=k_int8,
        v_int8=k_int8,
        k_scale=scale,
        v_scale=scale,
        block_tables=block_tables,
        context_lens=context_lens,
    )


def test_manifest_omits_disabled_packed_dequant_workspace_and_launch():
    result = manifest(False)

    assert result["workspace"]["packed_k_shape"] is None
    assert result["workspace"]["packed_k_dtype"] is None
    assert result["workspace"]["packed_v_shape"] is None
    assert "dequant_packed_kvcache" not in {
        launch["name"] for launch in result["kernel_launches"]
    }


def test_manifest_records_enabled_packed_dequant_workspace_and_launch():
    result = manifest(True)

    assert result["workspace"]["packed_k_shape"] == [1, 8, 1, 4]
    assert result["workspace"]["packed_k_dtype"] == "torch.float32"
    assert result["workspace"]["packed_v_shape"] == [1, 8, 1, 4]
    assert "dequant_packed_kvcache" in {
        launch["name"] for launch in result["kernel_launches"]
    }


def test_manifest_records_single_partitioned_workspace_allocation():
    result = manifest(True)

    workspace = result["workspace"]["partitioned"]["8"]
    assert workspace["allocation_count"] == 1
    assert workspace["shared_storage"]


def test_partitioned_workspace_respects_sliding_window():
    result = manifest(
        True,
        context_len=32,
        sliding_window_size=8,
    )

    workspace = result["workspace"]["partitioned"]["8"]
    assert workspace["partial_acc_shape"] == [1, 2, 1, 4]
