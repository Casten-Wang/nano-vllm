import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
        self.weight.safetensors_loader = self.safetensors_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def safetensors_loader(self, param: nn.Parameter, loaded_weight):
        shape = tuple(loaded_weight.get_shape())
        expected_shape = (self.num_embeddings, param.shape[1])
        if shape != expected_shape:
            raise ValueError(
                f"invalid vocabulary weight shape: {shape}; "
                f"expected {expected_shape}"
            )
        param.data.copy_(
            loaded_weight[self.vocab_start_idx : self.vocab_end_idx, :]
        )

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_mixed:
            decode_indices = torch.arange(context.decode_token_count, device=x.device)
            prefill_indices = context.decode_token_count + context.prefill_cu_seqlens_q[1:] - 1
            last_indices = torch.cat([decode_indices, prefill_indices]).to(torch.long)
            x = x[last_indices].contiguous()
        elif context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # Gather vocabulary-major shards into one allocation. Transposing
            # the result exposes the expected [batch, vocabulary] layout as a
            # view and avoids both a list of full-size shard buffers and the
            # second full-vocabulary allocation previously created by cat.
            local_logits = logits.transpose(0, 1).contiguous()
            if self.tp_rank == 0:
                gathered_logits = torch.empty(
                    self.num_embeddings,
                    logits.shape[0],
                    dtype=logits.dtype,
                    device=logits.device,
                )
                all_logits = list(
                    gathered_logits.split(
                        self.num_embeddings_per_partition,
                        dim=0,
                    )
                )
            else:
                gathered_logits = None
                all_logits = None
            dist.gather(local_logits, all_logits, 0)
            logits = (
                gathered_logits.transpose(0, 1)
                if gathered_logits is not None
                else None
            )
        return logits
