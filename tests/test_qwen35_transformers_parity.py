"""Tiny end-to-end parity check against the official Transformers model."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch


transformers = pytest.importorskip("transformers", minversion="5.2.0")
safetensors_torch = pytest.importorskip("safetensors.torch")
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as TransformersQwen35,
)


ROOT = Path(__file__).parents[1]
CURRENT_CONTEXT = {}


def _load(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_runtime():
    names = (
        "nanovllm",
        "nanovllm.layers",
        "nanovllm.models",
        "nanovllm.utils",
        "nanovllm.utils.context",
        "nanovllm.layers.activation",
        "nanovllm.layers.linear",
        "nanovllm.layers.embed_head",
        "nanovllm.layers.rotary_embedding",
        "nanovllm.layers.attention",
        "nanovllm.models.qwen35_moe",
        "nanovllm.models.qwen35_gated_delta",
        "nanovllm.models.qwen35_attention",
        "nanovllm.models.qwen35",
        "nanovllm.utils.loader",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:4]:
            sys.modules[name] = types.ModuleType(name)
        context_module = types.ModuleType("nanovllm.utils.context")
        context_module.get_context = lambda: CURRENT_CONTEXT["value"]
        sys.modules[context_module.__name__] = context_module
        _load("nanovllm.layers.activation", "nanovllm/layers/activation.py")
        linear = _load("nanovllm.layers.linear", "nanovllm/layers/linear.py")
        _load("nanovllm.layers.embed_head", "nanovllm/layers/embed_head.py")
        _load(
            "nanovllm.layers.rotary_embedding",
            "nanovllm/layers/rotary_embedding.py",
        )
        _load("nanovllm.models.qwen35_moe", "nanovllm/models/qwen35_moe.py")
        _load(
            "nanovllm.models.qwen35_gated_delta",
            "nanovllm/models/qwen35_gated_delta.py",
        )

        attention_module = types.ModuleType("nanovllm.layers.attention")

        class EagerAttention(torch.nn.Module):
            def __init__(self, num_heads, head_dim, scale, num_kv_heads):
                super().__init__()
                self.num_heads = num_heads
                self.num_kv_heads = num_kv_heads
                self.scale = scale
                self.k_cache = self.v_cache = torch.tensor([])
                self.keys_by_slot = {}
                self.values_by_slot = {}

            def forward(self, query, key, value):
                context = CURRENT_CONTEXT["value"]
                repeats = self.num_heads // self.num_kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
                output = torch.empty_like(query)

                slots = context.state_slots.tolist()
                if context.state_reset_mask is not None:
                    for slot, reset in zip(
                        slots,
                        context.state_reset_mask.tolist(),
                    ):
                        if reset:
                            self.keys_by_slot.pop(slot, None)
                            self.values_by_slot.pop(slot, None)

                decode_count = (
                    context.decode_token_count
                    if context.is_mixed
                    else (0 if context.is_prefill else query.shape[0])
                )
                for row in range(decode_count):
                    slot = slots[row]
                    previous_key = self.keys_by_slot.get(slot)
                    previous_value = self.values_by_slot.get(slot)
                    all_key = (
                        key[row : row + 1]
                        if previous_key is None
                        else torch.cat((previous_key, key[row : row + 1]))
                    )
                    all_value = (
                        value[row : row + 1]
                        if previous_value is None
                        else torch.cat((previous_value, value[row : row + 1]))
                    )
                    scores = torch.einsum(
                        "thd,shd->hts",
                        query[row : row + 1],
                        all_key,
                    ) * self.scale
                    output[row : row + 1] = torch.einsum(
                        "hts,shd->thd",
                        scores.softmax(dim=-1),
                        all_value,
                    )
                    self.keys_by_slot[slot] = all_key
                    self.values_by_slot[slot] = all_value

                for range_index, (start, end) in enumerate(
                    context.state_token_ranges
                ):
                    slot = slots[decode_count + range_index]
                    previous_key = self.keys_by_slot.get(slot)
                    previous_value = self.values_by_slot.get(slot)
                    prefix_length = 0 if previous_key is None else previous_key.shape[0]
                    all_key = (
                        key[start:end]
                        if previous_key is None
                        else torch.cat((previous_key, key[start:end]))
                    )
                    all_value = (
                        value[start:end]
                        if previous_value is None
                        else torch.cat((previous_value, value[start:end]))
                    )
                    scores = torch.einsum(
                        "thd,shd->hts",
                        query[start:end],
                        all_key,
                    ) * self.scale
                    query_positions = prefix_length + torch.arange(end - start)
                    key_positions = torch.arange(all_key.shape[0])
                    mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                    scores = scores.masked_fill(mask, float("-inf"))
                    output[start:end] = torch.einsum(
                        "hts,shd->thd",
                        scores.softmax(dim=-1),
                        all_value,
                    )
                    self.keys_by_slot[slot] = all_key
                    self.values_by_slot[slot] = all_value
                return output

        attention_module.Attention = EagerAttention
        sys.modules[attention_module.__name__] = attention_module
        _load(
            "nanovllm.models.qwen35_attention",
            "nanovllm/models/qwen35_attention.py",
        )
        local_model = _load(
            "nanovllm.models.qwen35",
            "nanovllm/models/qwen35.py",
        )
        loader = _load("nanovllm.utils.loader", "nanovllm/utils/loader.py")
        return local_model, loader, linear, context_module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


LOCAL, LOADER, LINEAR, CONTEXT_MODULE = _load_local_runtime()


def tiny_config():
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 0, 0],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=False,
        attention_bias=False,
        rms_norm_eps=1e-6,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def make_models(tmp_path, seed, recurrent_dtype=torch.float32):
    config = tiny_config()
    torch.manual_seed(seed)
    reference = TransformersQwen35(config).eval()
    with (
        patch.object(LINEAR.dist, "get_world_size", return_value=1),
        patch.object(LINEAR.dist, "get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        local = LOCAL.Qwen3_5MoeForCausalLM(config).eval()
        for module in local.modules():
            allocate = getattr(module, "allocate_state_cache", None)
            if allocate is not None:
                allocate(2, "cpu", recurrent_dtype=recurrent_dtype)
        safetensors_torch.save_file(
            {
                name: value.detach().contiguous()
                for name, value in reference.state_dict().items()
            },
            str(tmp_path / "model.safetensors"),
        )
        LOADER.load_model(local, str(tmp_path))
    return reference, local


def test_tiny_text_model_matches_transformers_end_to_end(tmp_path):
    reference, local = make_models(tmp_path, 29)

    tokens = torch.tensor([[1, 5, 7, 2, 9]])
    positions = torch.arange(tokens.shape[1])
    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        is_prefill=True,
        is_mixed=False,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=((0, tokens.shape[1]),),
        cu_seqlens_q=torch.tensor([0, tokens.shape[1]], dtype=torch.int32),
    )
    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        expected = reference.model(
            input_ids=tokens,
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        actual = local(tokens.squeeze(0), positions)
        expected_logits = reference.lm_head(expected)
        actual_logits = torch.nn.functional.linear(actual, local.lm_head.weight)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=2e-4,
        atol=2e-4,
    )


def test_prefill_then_decode_matches_transformers_full_recomputation(tmp_path):
    reference, local = make_models(tmp_path, 53)

    tokens = torch.tensor([[1, 5, 7, 2, 9]])
    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        is_prefill=True,
        is_mixed=False,
        decode_token_count=0,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=((0, 3),),
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
    )
    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        expected_prefix = reference.model(
            input_ids=tokens[:, :3],
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        actual_prefix = local(tokens[0, :3], torch.arange(3))
        torch.testing.assert_close(
            actual_prefix,
            expected_prefix,
            rtol=2e-4,
            atol=2e-4,
        )

        for token_index in range(3, tokens.shape[1]):
            CURRENT_CONTEXT["value"] = types.SimpleNamespace(
                is_prefill=False,
                is_mixed=False,
                decode_token_count=1,
                state_slots=torch.tensor([0], dtype=torch.int32),
                state_reset_mask=torch.tensor([False]),
                state_token_ranges=(),
                cu_seqlens_q=None,
            )
            actual = local(
                tokens[0, token_index : token_index + 1],
                torch.tensor([token_index]),
            )
            expected = reference.model(
                input_ids=tokens[:, : token_index + 1],
                use_cache=False,
            ).last_hidden_state[:, -1]
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(
                local.compute_logits(actual),
                reference.lm_head(expected),
                rtol=2e-4,
                atol=2e-4,
            )


def test_mixed_decode_and_prefill_matches_transformers(tmp_path):
    reference, local = make_models(tmp_path, 59)
    decode_tokens = torch.tensor([[1, 5, 7, 2]])
    prefill_tokens = torch.tensor([[3, 6]])

    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        is_prefill=True,
        is_mixed=False,
        decode_token_count=0,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=((0, 3),),
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
    )
    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        local(decode_tokens[0, :3], torch.arange(3))
        CURRENT_CONTEXT["value"] = types.SimpleNamespace(
            is_prefill=False,
            is_mixed=True,
            decode_token_count=1,
            state_slots=torch.tensor([0, 1], dtype=torch.int32),
            state_reset_mask=torch.tensor([False, True]),
            state_token_ranges=((1, 3),),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            prefill_cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        )
        mixed_tokens = torch.cat((decode_tokens[0, 3:], prefill_tokens[0]))
        mixed_positions = torch.tensor([3, 0, 1])
        actual = local(mixed_tokens, mixed_positions)
        expected_decode = reference.model(
            input_ids=decode_tokens,
            use_cache=False,
        ).last_hidden_state[:, -1]
        expected_prefill = reference.model(
            input_ids=prefill_tokens,
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        expected = torch.cat((expected_decode, expected_prefill))

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    sampled_expected = torch.stack((expected[0], expected[-1]))
    torch.testing.assert_close(
        local.compute_logits(actual),
        reference.lm_head(sampled_expected),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("recurrent_dtype", [torch.bfloat16, torch.float16])
def test_compressed_recurrent_storage_remains_close_over_multi_step_decode(
    tmp_path,
    recurrent_dtype,
):
    reference, local = make_models(
        tmp_path,
        61,
        recurrent_dtype=recurrent_dtype,
    )
    tokens = torch.tensor(
        [[1, 5, 7, 2, 9, 4, 6, 3, 8, 11, 13, 10, 12, 15, 14, 17, 16, 19]]
    )
    CURRENT_CONTEXT["value"] = types.SimpleNamespace(
        is_prefill=True,
        is_mixed=False,
        decode_token_count=0,
        state_slots=torch.tensor([0], dtype=torch.int32),
        state_reset_mask=torch.tensor([True]),
        state_token_ranges=((0, 2),),
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
    )
    max_hidden_error = 0.0
    max_logit_error = 0.0
    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        local(tokens[0, :2], torch.arange(2))
        for token_index in range(2, tokens.shape[1]):
            CURRENT_CONTEXT["value"] = types.SimpleNamespace(
                is_prefill=False,
                is_mixed=False,
                decode_token_count=1,
                state_slots=torch.tensor([0], dtype=torch.int32),
                state_reset_mask=torch.tensor([False]),
                state_token_ranges=(),
                cu_seqlens_q=None,
            )
            actual = local(
                tokens[0, token_index : token_index + 1],
                torch.tensor([token_index]),
            )
            expected = reference.model(
                input_ids=tokens[:, : token_index + 1],
                use_cache=False,
            ).last_hidden_state[:, -1]
            actual_logits = local.compute_logits(actual)
            expected_logits = reference.lm_head(expected)
            max_hidden_error = max(
                max_hidden_error,
                (actual - expected).abs().max().item(),
            )
            max_logit_error = max(
                max_logit_error,
                (actual_logits - expected_logits).abs().max().item(),
            )

    assert max_hidden_error < 2e-3
    assert max_logit_error < 2e-3


def test_reused_state_slot_does_not_leak_previous_request(tmp_path):
    reference, local = make_models(tmp_path, 67)
    previous_tokens = torch.tensor([1, 5, 7, 2])
    new_tokens = torch.tensor([[3, 6, 10]])

    def set_prefill_context(length, reset):
        CURRENT_CONTEXT["value"] = types.SimpleNamespace(
            is_prefill=True,
            is_mixed=False,
            decode_token_count=0,
            state_slots=torch.tensor([0], dtype=torch.int32),
            state_reset_mask=torch.tensor([reset]),
            state_token_ranges=((0, length),),
            cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32),
        )

    with (
        patch.dict(sys.modules, {"nanovllm.utils.context": CONTEXT_MODULE}),
        torch.inference_mode(),
    ):
        set_prefill_context(previous_tokens.numel(), True)
        local(previous_tokens, torch.arange(previous_tokens.numel()))

        set_prefill_context(new_tokens.shape[1], True)
        actual = local(new_tokens.squeeze(0), torch.arange(new_tokens.shape[1]))
        expected = reference.model(
            input_ids=new_tokens,
            use_cache=False,
        ).last_hidden_state.squeeze(0)
        actual_logits = local.compute_logits(actual)
        expected_logits = reference.lm_head(expected[-1:])

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=2e-4,
        atol=2e-4,
    )
