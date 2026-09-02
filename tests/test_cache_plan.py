from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def load_module(name: str, relative_path: str):
    spec = spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_spec_module = load_module(
    "nanovllm.models.model_spec",
    "nanovllm/models/model_spec.py",
)
cache_plan_module = load_module(
    "nanovllm.models.cache_plan",
    "nanovllm/models/cache_plan.py",
)


def qwen35_spec():
    layer_types = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ) * 10
    text_config = SimpleNamespace(
        num_hidden_layers=40,
        layer_types=layer_types,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    outer_config = SimpleNamespace(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text_config,
    )
    return model_spec_module.resolve_model_spec(outer_config)


def test_qwen35_tp1_cache_layout_matches_official_config():
    plan = cache_plan_module.plan_cache_memory(qwen35_spec(), 1)

    assert plan.local_kv_heads == 2
    assert plan.kv_head_replication == 1
    assert plan.kv_bytes_per_token == 20_480
    assert plan.int8_scale_bytes_per_token == 80
    assert plan.recurrent_bytes_per_sequence == 62_914_560
    assert plan.convolution_bytes_per_sequence == 1_966_080


def test_qwen35_tp4_accounts_for_kv_head_replication():
    plan = cache_plan_module.plan_cache_memory(qwen35_spec(), 4)

    assert plan.local_kv_heads == 1
    assert plan.kv_head_replication == 2
    assert plan.kv_bytes_per_token == 10_240
    assert plan.int8_scale_bytes_per_token == 40
    assert plan.recurrent_bytes_per_sequence == 15_728_640
    assert plan.convolution_bytes_per_sequence == 491_520
    assert plan.bytes_per_sequence(32_768) == 351_764_480


def test_qwen35_tp8_is_supported_with_replicated_kv_heads():
    plan = cache_plan_module.plan_cache_memory(qwen35_spec(), 8)

    assert plan.local_kv_heads == 1
    assert plan.kv_head_replication == 4
    assert plan.recurrent_bytes_per_sequence == 7_864_320


def test_model_dtype_recurrent_storage_halves_state_memory():
    fp32 = cache_plan_module.plan_cache_memory(qwen35_spec(), 4)
    bf16 = cache_plan_module.plan_cache_memory(
        qwen35_spec(),
        4,
        recurrent_dtype_bytes=2,
    )

    assert bf16.recurrent_bytes_per_sequence * 2 == fp32.recurrent_bytes_per_sequence


def test_invalid_tensor_parallel_size_is_rejected():
    try:
        cache_plan_module.plan_cache_memory(qwen35_spec(), 3)
    except ValueError as error:
        assert "attention heads cannot be sharded" in str(error)
    else:
        raise AssertionError("TP=3 should not be accepted")
