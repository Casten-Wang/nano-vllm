from unittest.mock import patch

import pytest
import torch

from nanovllm.layers.embed_head import VocabParallelEmbedding


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
