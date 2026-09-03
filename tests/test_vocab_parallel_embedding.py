from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nanovllm.layers import embed_head
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding


class TrackingSlice:
    def __init__(self, tensor):
        self.tensor = tensor
        self.requests = []

    def get_shape(self):
        return self.tensor.shape

    def __getitem__(self, key):
        self.requests.append(key)
        return self.tensor[key]


def make_embedding(rank=1, world_size=2):
    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=world_size),
    ):
        return VocabParallelEmbedding(8, 4)


def test_safetensors_loader_reads_only_local_vocabulary_partition():
    embedding = make_embedding()
    source = TrackingSlice(torch.arange(32).reshape(8, 4).float())

    embedding.safetensors_loader(embedding.weight, source)

    torch.testing.assert_close(embedding.weight, source.tensor[4:8])
    assert source.requests == [(slice(4, 8), slice(None))]


def test_safetensors_loader_rejects_invalid_vocabulary_shape():
    embedding = make_embedding()
    source = TrackingSlice(torch.empty(7, 4))

    with pytest.raises(ValueError, match="invalid vocabulary weight shape"):
        embedding.safetensors_loader(embedding.weight, source)


def make_lm_head(rank=0, world_size=2):
    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=world_size),
    ):
        return ParallelLMHead(8, 2)


def test_lm_head_gathers_tp_logits_into_one_vocab_buffer():
    head = make_lm_head()
    head.weight.data.copy_(torch.tensor([[1.0, 0.0]] * 4))
    hidden = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    remote_logits = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]
    )
    observed = {}

    def gather(local, gather_list, dst):
        observed["local_contiguous"] = local.is_contiguous()
        observed["destinations_contiguous"] = all(
            tensor.is_contiguous() for tensor in gather_list
        )
        gather_list[0].copy_(local)
        gather_list[1].copy_(remote_logits.transpose(0, 1))

    context = SimpleNamespace(is_mixed=False, is_prefill=False)
    with (
        torch.inference_mode(),
        patch.object(embed_head, "get_context", return_value=context),
        patch.object(embed_head.dist, "gather", side_effect=gather),
        patch.object(
            embed_head.torch,
            "cat",
            side_effect=AssertionError("TP logits must not be concatenated"),
        ),
    ):
        logits = head(hidden)

    expected_local = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    )
    torch.testing.assert_close(
        logits,
        torch.cat((expected_local, remote_logits), dim=-1),
    )
    assert observed == {
        "local_contiguous": True,
        "destinations_contiguous": True,
    }
    assert not logits.is_contiguous()
    assert head.tp_logits_storage_stats() == {
        "local_bytes": 8 * hidden.element_size(),
        "gathered_bytes": 16 * hidden.element_size(),
        "total_bytes": 24 * hidden.element_size(),
        "allocation_count": 2,
        "reuse_count": 0,
    }


def test_lm_head_reuses_tp_logits_storage_for_smaller_batches():
    head = make_lm_head()
    head.weight.data.copy_(torch.tensor([[1.0, 0.0]] * 4))
    context = SimpleNamespace(is_mixed=False, is_prefill=False)

    def gather(local, gather_list, dst):
        gather_list[0].copy_(local)
        gather_list[1].fill_(7.0)

    with (
        torch.inference_mode(),
        patch.object(embed_head, "get_context", return_value=context),
        patch.object(embed_head.dist, "gather", side_effect=gather),
    ):
        large = head(torch.tensor([[1.0, 0.0], [2.0, 0.0]])).clone()
        local_storage = head._tp_local_logits_buffer.data_ptr()
        gathered_storage = head._tp_gathered_logits_buffer.data_ptr()
        small = head(torch.tensor([[3.0, 0.0]])).clone()

    torch.testing.assert_close(
        large,
        torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 7.0],
             [2.0, 2.0, 2.0, 2.0, 7.0, 7.0, 7.0, 7.0]]
        ),
    )
    torch.testing.assert_close(
        small,
        torch.tensor([[3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0]]),
    )
    assert head._tp_local_logits_buffer.data_ptr() == local_storage
    assert head._tp_gathered_logits_buffer.data_ptr() == gathered_storage
    stats = head.tp_logits_storage_stats()
    assert stats["allocation_count"] == 2
    assert stats["reuse_count"] == 2


def test_lm_head_does_not_reuse_tp_logits_storage_with_autograd():
    head = make_lm_head()
    context = SimpleNamespace(is_mixed=False, is_prefill=False)

    with (
        torch.enable_grad(),
        patch.object(embed_head, "get_context", return_value=context),
        patch.object(embed_head.dist, "gather"),
    ):
        head(torch.randn(2, 2))

    assert head.tp_logits_storage_stats()["total_bytes"] == 0


def test_nonzero_lm_head_rank_does_not_allocate_gather_output():
    head = make_lm_head(rank=1)
    context = SimpleNamespace(is_mixed=False, is_prefill=False)
    observed = {}

    def gather(local, gather_list, dst):
        observed.update(
            local_shape=tuple(local.shape),
            gather_list=gather_list,
            dst=dst,
        )

    with (
        torch.inference_mode(),
        patch.object(embed_head, "get_context", return_value=context),
        patch.object(embed_head.dist, "gather", side_effect=gather),
    ):
        logits = head(torch.randn(3, 2))

    assert logits is None
    assert observed == {
        "local_shape": (4, 3),
        "gather_list": None,
        "dst": 0,
    }


@pytest.mark.parametrize(
    ("context", "indices"),
    [
        (
            SimpleNamespace(
                is_mixed=False,
                is_prefill=True,
                logits_indices=torch.tensor([1, 4]),
            ),
            torch.tensor([1, 4]),
        ),
        (
            SimpleNamespace(
                is_mixed=True,
                is_prefill=False,
                logits_indices=torch.tensor([0, 3, 5]),
            ),
            torch.tensor([0, 3, 5]),
        ),
    ],
)
def test_lm_head_uses_precomputed_logits_indices(context, indices):
    head = make_lm_head(world_size=1)
    hidden = torch.randn(6, 2)

    with (
        torch.inference_mode(),
        patch.object(embed_head, "get_context", return_value=context),
        patch.object(
            embed_head.torch,
            "arange",
            side_effect=AssertionError("logits indices must be precomputed"),
        ),
        patch.object(
            embed_head.torch,
            "cat",
            side_effect=AssertionError("logits indices must be precomputed"),
        ),
    ):
        actual = head(hidden)

    expected = torch.nn.functional.linear(hidden[indices], head.weight)
    torch.testing.assert_close(actual, expected)
