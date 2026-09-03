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
        self.register_buffer(
            "_tp_local_logits_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_gathered_logits_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_greedy_local_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_greedy_gather_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_top_k_values_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_top_k_ids_buffer",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_tp_top_k_flat_values_buffer",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_tp_top_k_flat_ids_buffer",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        self.tp_logits_allocation_count = 0
        self.tp_logits_reuse_count = 0
        self.tp_greedy_reduction_count = 0
        self.tp_greedy_candidate_bytes = 0
        self.tp_greedy_full_gather_avoided_bytes = 0
        self.tp_top_k_reduction_count = 0
        self.tp_top_k_candidate_bytes = 0
        self.tp_top_k_full_gather_avoided_bytes = 0

    def _tp_logits_buffer(
        self,
        name: str,
        required: int,
        reference: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        buffer = getattr(self, name)
        target_dtype = reference.dtype if dtype is None else dtype
        if (
            buffer.numel() < required
            or buffer.dtype != target_dtype
            or buffer.device != reference.device
        ):
            buffer = torch.empty(
                required,
                dtype=target_dtype,
                device=reference.device,
            )
            setattr(self, name, buffer)
            self.tp_logits_allocation_count += 1
        else:
            self.tp_logits_reuse_count += 1
        return buffer[:required]

    def tp_logits_storage_stats(self) -> dict[str, int]:
        local_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in (
                self._tp_local_logits_buffer,
                self._tp_greedy_local_buffer,
            )
        )
        gathered_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in (
                self._tp_gathered_logits_buffer,
                self._tp_greedy_gather_buffer,
                self._tp_top_k_values_buffer,
                self._tp_top_k_ids_buffer,
                self._tp_top_k_flat_values_buffer,
                self._tp_top_k_flat_ids_buffer,
            )
        )
        return {
            "local_bytes": local_bytes,
            "gathered_bytes": gathered_bytes,
            "total_bytes": local_bytes + gathered_bytes,
            "allocation_count": self.tp_logits_allocation_count,
            "reuse_count": self.tp_logits_reuse_count,
            "greedy_reduction_count": self.tp_greedy_reduction_count,
            "greedy_candidate_bytes": self.tp_greedy_candidate_bytes,
            "greedy_full_gather_avoided_bytes": (
                self.tp_greedy_full_gather_avoided_bytes
            ),
            "top_k_reduction_count": self.tp_top_k_reduction_count,
            "top_k_candidate_bytes": self.tp_top_k_candidate_bytes,
            "top_k_full_gather_avoided_bytes": (
                self.tp_top_k_full_gather_avoided_bytes
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        greedy: bool = False,
        top_k: int | None = None,
    ):
        if greedy and top_k is not None:
            raise ValueError("greedy and top_k reductions are mutually exclusive")
        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 0 < top_k <= self.num_embeddings
        ):
            raise ValueError(
                "top_k reduction width must be in the vocabulary range"
            )
        context = get_context()
        if context.is_mixed:
            last_indices = getattr(context, "logits_indices", None)
            if last_indices is None:
                decode_indices = torch.arange(
                    context.decode_token_count,
                    device=x.device,
                )
                prefill_indices = (
                    context.decode_token_count
                    + context.prefill_cu_seqlens_q[1:]
                    - 1
                )
                last_indices = torch.cat(
                    [decode_indices, prefill_indices]
                ).to(torch.long)
            x = x[last_indices].contiguous()
        elif context.is_prefill:
            last_indices = getattr(context, "logits_indices", None)
            if last_indices is None:
                last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if greedy:
            self.tp_greedy_reduction_count += 1
            local_values, local_ids = logits.max(dim=-1)
            if self.tp_size == 1:
                return local_ids.add(self.vocab_start_idx)
            # Float32 exactly represents every supported token id and lets one
            # small collective carry both the score and global vocabulary id.
            if torch.is_grad_enabled():
                local_candidates = torch.stack(
                    (
                        local_values.float(),
                        local_ids.add(self.vocab_start_idx).float(),
                    ),
                    dim=-1,
                )
            else:
                local_candidates = self._tp_logits_buffer(
                    "_tp_greedy_local_buffer",
                    logits.shape[0] * 2,
                    local_values,
                    dtype=torch.float32,
                ).view(logits.shape[0], 2)
                local_candidates[:, 0].copy_(local_values)
                local_candidates[:, 1].copy_(local_ids)
                local_candidates[:, 1].add_(self.vocab_start_idx)
            peer_count = self.tp_size - 1
            self.tp_greedy_candidate_bytes += (
                local_candidates.numel()
                * local_candidates.element_size()
                * peer_count
            )
            self.tp_greedy_full_gather_avoided_bytes += (
                logits.numel() * logits.element_size() * peer_count
            )
            if self.tp_rank == 0:
                if torch.is_grad_enabled():
                    gathered_candidates = torch.empty(
                        self.tp_size,
                        logits.shape[0],
                        2,
                        dtype=torch.float32,
                        device=logits.device,
                    )
                else:
                    gathered_candidates = self._tp_logits_buffer(
                        "_tp_greedy_gather_buffer",
                        self.tp_size * logits.shape[0] * 2,
                        local_candidates,
                    ).view(self.tp_size, logits.shape[0], 2)
                candidate_list = list(gathered_candidates.unbind(dim=0))
            else:
                gathered_candidates = None
                candidate_list = None
            dist.gather(local_candidates, candidate_list, 0)
            if gathered_candidates is None:
                return None
            winning_ranks = gathered_candidates[:, :, 0].argmax(dim=0)
            winning_ids = gathered_candidates[:, :, 1].gather(
                0,
                winning_ranks.unsqueeze(0),
            )
            return winning_ids.squeeze(0).to(torch.int64)
        if top_k is not None:
            self.tp_top_k_reduction_count += 1
            local_k = min(top_k, self.num_embeddings_per_partition)
            local_values, local_ids = torch.topk(logits, local_k, dim=-1)
            local_ids.add_(self.vocab_start_idx)
            if self.tp_size == 1:
                return local_values, local_ids
            if self.tp_rank == 0:
                shape = (self.tp_size, logits.shape[0], local_k)
                required = self.tp_size * logits.shape[0] * local_k
                if torch.is_grad_enabled():
                    gathered_values = torch.empty(
                        shape,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    gathered_ids = torch.empty(
                        shape,
                        dtype=torch.int64,
                        device=logits.device,
                    )
                else:
                    gathered_values = self._tp_logits_buffer(
                        "_tp_top_k_values_buffer",
                        required,
                        local_values,
                    ).view(shape)
                    gathered_ids = self._tp_logits_buffer(
                        "_tp_top_k_ids_buffer",
                        required,
                        local_ids,
                    ).view(shape)
                value_list = list(gathered_values.unbind(dim=0))
                id_list = list(gathered_ids.unbind(dim=0))
            else:
                gathered_values = None
                gathered_ids = None
                value_list = None
                id_list = None
            dist.gather(local_values, value_list, 0)
            dist.gather(local_ids, id_list, 0)
            peer_count = self.tp_size - 1
            self.tp_top_k_candidate_bytes += (
                local_values.numel() * local_values.element_size()
                + local_ids.numel() * local_ids.element_size()
            ) * peer_count
            self.tp_top_k_full_gather_avoided_bytes += (
                logits.numel() * logits.element_size() * peer_count
            )
            if gathered_values is None or gathered_ids is None:
                return None
            if torch.is_grad_enabled():
                all_values = gathered_values.permute(1, 0, 2).flatten(1)
                all_ids = gathered_ids.permute(1, 0, 2).flatten(1)
            else:
                flat_shape = (logits.shape[0], self.tp_size * local_k)
                all_values_storage = self._tp_logits_buffer(
                    "_tp_top_k_flat_values_buffer",
                    required,
                    local_values,
                ).view(logits.shape[0], self.tp_size, local_k)
                all_ids_storage = self._tp_logits_buffer(
                    "_tp_top_k_flat_ids_buffer",
                    required,
                    local_ids,
                ).view(logits.shape[0], self.tp_size, local_k)
                all_values_storage.copy_(gathered_values.permute(1, 0, 2))
                all_ids_storage.copy_(gathered_ids.permute(1, 0, 2))
                all_values = all_values_storage.view(flat_shape)
                all_ids = all_ids_storage.view(flat_shape)
            selected_values, selected_offsets = torch.topk(
                all_values,
                top_k,
                dim=-1,
            )
            selected_ids = all_ids.gather(1, selected_offsets)
            return selected_values, selected_ids
        if self.tp_size > 1:
            # Gather vocabulary-major shards into one allocation. Transposing
            # the result exposes the expected [batch, vocabulary] layout as a
            # view and avoids both a list of full-size shard buffers and the
            # second full-vocabulary allocation previously created by cat.
            if torch.is_grad_enabled():
                local_logits = logits.transpose(0, 1).contiguous()
            else:
                local_logits = self._tp_logits_buffer(
                    "_tp_local_logits_buffer",
                    logits.numel(),
                    logits,
                ).view(self.num_embeddings_per_partition, logits.shape[0])
                local_logits.copy_(logits.transpose(0, 1))
            if self.tp_rank == 0:
                if torch.is_grad_enabled():
                    gathered_logits = torch.empty(
                        self.num_embeddings,
                        logits.shape[0],
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                else:
                    gathered_logits = self._tp_logits_buffer(
                        "_tp_gathered_logits_buffer",
                        self.num_embeddings * logits.shape[0],
                        logits,
                    ).view(self.num_embeddings, logits.shape[0])
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
