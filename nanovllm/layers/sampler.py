from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SamplingBatchMetadata:
    batch_size: int
    sample_count: int
    vocab_size: int
    all_top_k_enabled: bool
    any_top_k_enabled: bool
    any_top_p_enabled: bool
    max_top_k: int
    sample_temperatures: tuple[float, ...]
    sample_top_ks: tuple[int, ...]
    sample_top_ps: tuple[float, ...]


def build_sampling_metadata(
    temperatures: list[float],
    top_ks: list[int],
    top_ps: list[float],
    vocab_size: int,
) -> SamplingBatchMetadata:
    if len(temperatures) != len(top_ks) or len(temperatures) != len(top_ps):
        raise ValueError("sampling parameter batch sizes must match")
    if vocab_size <= 0:
        raise ValueError("vocabulary size must be positive")
    if any(not isfinite(value) or value < 0.0 for value in temperatures):
        raise ValueError("temperatures must be finite and non-negative")
    if any(value != -1 and value <= 0 for value in top_ks):
        raise ValueError("top_k must be -1 or positive")
    if any(not isfinite(value) or not 0.0 < value <= 1.0 for value in top_ps):
        raise ValueError("top_p must be finite and in (0, 1]")
    sampled = [
        (temperature, top_k, top_p)
        for temperature, top_k, top_p in zip(temperatures, top_ks, top_ps)
        if temperature > 1e-10
    ]
    enabled_top_ks = [
        top_k for _, top_k, _ in sampled if 0 < top_k < vocab_size
    ]
    return SamplingBatchMetadata(
        batch_size=len(temperatures),
        sample_count=len(sampled),
        vocab_size=vocab_size,
        all_top_k_enabled=bool(sampled) and len(enabled_top_ks) == len(sampled),
        any_top_k_enabled=bool(enabled_top_ks),
        any_top_p_enabled=any(top_p < 1.0 for _, _, top_p in sampled),
        max_top_k=max(enabled_top_ks, default=0),
        sample_temperatures=tuple(value for value, _, _ in sampled),
        sample_top_ks=tuple(value for _, value, _ in sampled),
        sample_top_ps=tuple(value for _, _, value in sampled),
    )


def _sampling_ranks(
    logits: torch.Tensor,
    width: int,
    rank_buffer: torch.Tensor | None,
) -> torch.Tensor:
    if rank_buffer is None:
        return torch.arange(width, device=logits.device).unsqueeze(0)
    if (
        rank_buffer.ndim != 1
        or rank_buffer.numel() < width
        or rank_buffer.device != logits.device
    ):
        raise ValueError("sampling rank buffer is incompatible with logits")
    return rank_buffer[:width].unsqueeze(0)


def _exclusive_top_p_mask(
    cumulative_probs: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Mask tokens whose preceding probability mass already exceeds top-p."""

    remove = torch.empty_like(cumulative_probs, dtype=torch.bool)
    remove[:, 0] = False
    if cumulative_probs.size(1) > 1:
        torch.gt(
            cumulative_probs[:, :-1],
            top_ps.unsqueeze(1),
            out=remove[:, 1:],
        )
    return remove


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    metadata: SamplingBatchMetadata | None = None,
    rank_buffer: torch.Tensor | None = None,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Mask logits according to per-request top-k and top-p settings.

    The input shape is ``[batch, vocab_size]``. Each row belongs to one
    sequence in the current batch, and each row may have different sampling
    parameters. The function returns a new logits tensor where filtered-out
    tokens are set to ``-inf`` so their softmax probability becomes zero.

    This implementation intentionally uses regular torch tensor operations
    rather than CPU-only helpers. It works on CUDA tensors during real
    inference and also works on CPU tensors in small unit tests.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab_size]")
    if top_ks.ndim != 1 or top_ps.ndim != 1:
        raise ValueError("top_ks and top_ps must be one-dimensional")
    if logits.size(0) != top_ks.numel() or logits.size(0) != top_ps.numel():
        raise ValueError("sampling parameter batch sizes must match logits")
    if top_ks.device != logits.device or top_ps.device != logits.device:
        raise ValueError("sampling tensors must be on the same device as logits")
    if logits.size(1) == 0:
        raise ValueError("vocabulary dimension must be non-empty")
    if logits.size(0) == 0:
        return logits
    if inplace and logits.requires_grad:
        raise ValueError("in-place sampling filter is inference-only")
    vocab_size = logits.size(1)
    if metadata is None:
        if torch.any((top_ks != -1) & (top_ks <= 0)):
            raise ValueError("top_k must be -1 or positive")
        if torch.any(~torch.isfinite(top_ps)) or torch.any((top_ps <= 0) | (top_ps > 1)):
            raise ValueError("top_p must be finite and in (0, 1]")
        top_k_enabled = (top_ks > 0) & (top_ks < vocab_size)
        top_p_enabled = top_ps < 1.0
        all_top_k_enabled = bool(top_k_enabled.all())
        any_top_k_enabled = bool(top_k_enabled.any())
        any_top_p_enabled = bool(top_p_enabled.any())
        max_top_k = (
            int(top_ks[top_k_enabled].max().item())
            if any_top_k_enabled
            else 0
        )
    else:
        if metadata.sample_count != logits.size(0) or metadata.vocab_size != vocab_size:
            raise ValueError("sampling metadata does not match logits")
        top_k_enabled = (top_ks > 0) & (top_ks < vocab_size)
        all_top_k_enabled = metadata.all_top_k_enabled
        any_top_k_enabled = metadata.any_top_k_enabled
        any_top_p_enabled = metadata.any_top_p_enabled
        max_top_k = metadata.max_top_k
    if all_top_k_enabled:
        selected_logits, selected_indices = torch.topk(
            logits,
            max_top_k,
            dim=-1,
        )
        ranks = _sampling_ranks(logits, max_top_k, rank_buffer)
        top_k_keep = ranks < top_ks.unsqueeze(1)
        selected_logits.masked_fill_(~top_k_keep, float("-inf"))
        if any_top_p_enabled:
            selected_probs = torch.softmax(selected_logits, dim=-1)
            cumulative_probs = torch.cumsum(selected_probs, dim=-1)
            selected_remove = _exclusive_top_p_mask(cumulative_probs, top_ps)
            selected_logits.masked_fill_(selected_remove, float("-inf"))
        filtered_logits = logits if inplace else torch.full_like(
            logits,
            float("-inf"),
        )
        if inplace:
            filtered_logits.fill_(float("-inf"))
        filtered_logits.scatter_(
            dim=-1,
            index=selected_indices,
            src=selected_logits,
        )
        return filtered_logits
    if not any_top_p_enabled:
        if not any_top_k_enabled:
            return logits
        selected_logits, selected_indices = torch.topk(
            logits,
            max_top_k,
            dim=-1,
        )
        effective_top_ks = torch.where(
            top_k_enabled,
            top_ks,
            torch.full_like(top_ks, max_top_k),
        )
        ranks = _sampling_ranks(logits, max_top_k, rank_buffer)
        selected_logits.masked_fill_(
            ranks >= effective_top_ks.unsqueeze(1),
            float("-inf"),
        )
        filtered_logits = logits if inplace else torch.full_like(
            logits,
            float("-inf"),
        )
        if inplace:
            filtered_logits[top_k_enabled] = float("-inf")
        filtered_logits.scatter_(
            dim=-1,
            index=selected_indices,
            src=selected_logits,
        )
        if not all_top_k_enabled and not inplace:
            filtered_logits[~top_k_enabled] = logits[~top_k_enabled]
        return filtered_logits

    # Sort each request's vocabulary distribution from high logit to low logit.
    # Top-p needs this ordering to compute the cumulative probability mass.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    if any_top_k_enabled:
        # top_k == -1 means "disabled", so the effective k becomes vocab_size.
        # Otherwise clamp to vocab_size so oversized top_k values are harmless.
        full_vocab = torch.full_like(top_ks, vocab_size)
        effective_top_ks = torch.where(
            top_ks > 0,
            torch.minimum(top_ks, full_vocab),
            full_vocab,
        )

        # Apply top-k before calculating top-p probabilities. This is important:
        # top-p is defined over the distribution that remains after top-k.
        ranks = _sampling_ranks(logits, vocab_size, rank_buffer)
        top_k_keep = ranks < effective_top_ks.unsqueeze(1)
        probability_logits = sorted_logits.masked_fill(
            ~top_k_keep,
            float("-inf"),
        )
    else:
        # Pure top-p needs the full sorted distribution. Avoid materializing
        # all-true top-k masks, vocabulary ranks, and an unchanged logits copy.
        top_k_keep = None
        probability_logits = sorted_logits
    sorted_probs = torch.softmax(probability_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Compare each token against the mass before it. This keeps the first token
    # that crosses the threshold without cloning a full shifted boolean mask.
    sorted_remove = _exclusive_top_p_mask(cumulative_probs, top_ps)
    sorted_keep = (
        ~sorted_remove
        if top_k_keep is None
        else top_k_keep & ~sorted_remove
    )

    # Apply the mask in sorted space.
    filtered_sorted_logits = sorted_logits.masked_fill(~sorted_keep, float("-inf"))

    # Scatter back to the original vocabulary order expected by softmax.
    filtered_logits = logits if inplace else torch.empty_like(
        filtered_sorted_logits
    )
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
    return filtered_logits


def compact_top_k_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int,
    any_top_p_enabled: bool,
    rank_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build filtered FP32 logits only for selected top-k candidates."""

    selected_logits, selected_indices = torch.topk(
        logits,
        max_top_k,
        dim=-1,
    )
    selected_logits = filter_top_k_candidates(
        selected_logits,
        temperatures,
        top_ks,
        top_ps,
        any_top_p_enabled,
        rank_buffer,
    )
    return selected_logits, selected_indices


def filter_top_k_candidates(
    selected_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    any_top_p_enabled: bool,
    rank_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply per-request sampling parameters to preselected top-k logits."""

    if selected_logits.ndim != 2 or selected_logits.size(1) == 0:
        raise ValueError("selected logits must have shape [batch, candidates]")
    batch_size, candidate_count = selected_logits.shape
    if any(tensor.ndim != 1 for tensor in (temperatures, top_ks, top_ps)):
        raise ValueError("sampling parameter tensors must be one-dimensional")
    if any(
        tensor.numel() != batch_size
        for tensor in (temperatures, top_ks, top_ps)
    ):
        raise ValueError("sampling parameter batch sizes must match candidates")
    if any(
        tensor.device != selected_logits.device
        for tensor in (temperatures, top_ks, top_ps)
    ):
        raise ValueError("sampling tensors must be on the same device as candidates")
    selected_logits = selected_logits.float().div_(temperatures.unsqueeze(1))
    ranks = _sampling_ranks(selected_logits, candidate_count, rank_buffer)
    selected_logits.masked_fill_(
        ranks >= top_ks.unsqueeze(1),
        float("-inf"),
    )
    if any_top_p_enabled:
        selected_probs = torch.softmax(selected_logits, dim=-1)
        cumulative_probs = torch.cumsum(selected_probs, dim=-1)
        selected_remove = _exclusive_top_p_mask(cumulative_probs, top_ps)
        selected_logits.masked_fill_(selected_remove, float("-inf"))
    return selected_logits


def sample_top_k_candidates(
    selected_logits: torch.Tensor,
    selected_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    any_top_p_enabled: bool,
    rank_buffer: torch.Tensor | None = None,
    noise_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample exact token ids from already selected global top-k candidates."""

    if selected_indices.shape != selected_logits.shape:
        raise ValueError("selected token ids must match selected logits")
    if selected_indices.device != selected_logits.device:
        raise ValueError("selected token ids must share the logits device")
    selected_logits = filter_top_k_candidates(
        selected_logits,
        temperatures,
        top_ks,
        top_ps,
        any_top_p_enabled,
        rank_buffer,
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    # Filtering logits are dead after softmax. Reuse that equally shaped FP32
    # storage for exponential noise instead of retaining a second workspace.
    noise = selected_logits if noise_buffer is None else noise_buffer
    if noise.shape != probabilities.shape or noise.device != probabilities.device:
        raise ValueError("sampling noise buffer must match candidate probabilities")
    selected_offsets = probabilities.div_(
        noise.exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return selected_indices.gather(1, selected_offsets.unsqueeze(1)).squeeze(1)


def sample_top_k_compact(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int,
    any_top_p_enabled: bool,
    rank_buffer: torch.Tensor | None = None,
    noise_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from top-k candidates without materializing FP32 full logits."""

    selected_logits, selected_indices = compact_top_k_logits(
        logits,
        temperatures,
        top_ks,
        top_ps,
        max_top_k,
        any_top_p_enabled,
        rank_buffer,
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    noise = selected_logits if noise_buffer is None else noise_buffer
    selected_offsets = probabilities.div_(
        noise.exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return selected_indices.gather(1, selected_offsets.unsqueeze(1)).squeeze(1)


class Sampler(nn.Module):

    def __init__(
        self,
        max_sampling_rows: int = 32,
        max_compact_top_k: int = 256,
    ):
        super().__init__()
        if max_sampling_rows <= 0:
            raise ValueError("max_sampling_rows must be positive")
        if max_compact_top_k <= 0:
            raise ValueError("max_compact_top_k must be positive")
        self.max_sampling_rows = max_sampling_rows
        self.max_compact_top_k = max_compact_top_k
        self.reset_stats()
        self.register_buffer(
            "_rank_buffer",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    def reserve_runtime_buffers(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._ranks(vocab_size, self._rank_buffer.device)

    def _ranks(self, width: int, device: torch.device) -> torch.Tensor:
        if self._rank_buffer.device != device or self._rank_buffer.numel() < width:
            self._rank_buffer = torch.arange(width, device=device)
        return self._rank_buffer

    def storage_stats(self) -> dict[str, int]:
        return {
            "rank_buffer_bytes": (
                self._rank_buffer.numel() * self._rank_buffer.element_size()
            ),
            "noise_buffer_bytes": 0,
        }

    def reset_stats(self) -> None:
        self.full_sampling_call_count = 0
        self.full_sampling_row_count = 0
        self.full_sampling_chunk_count = 0
        self.max_full_sampling_chunk_rows = 0

    def runtime_stats(self) -> dict[str, int]:
        return {
            "full_sampling_call_count": self.full_sampling_call_count,
            "full_sampling_row_count": self.full_sampling_row_count,
            "full_sampling_chunk_count": self.full_sampling_chunk_count,
            "max_full_sampling_chunk_rows": self.max_full_sampling_chunk_rows,
            "configured_sampling_chunk_rows": self.max_sampling_rows,
        }

    def _record_full_sampling(self, rows: int, chunks: int) -> None:
        self.full_sampling_call_count += 1
        self.full_sampling_row_count += rows
        self.full_sampling_chunk_count += chunks
        self.max_full_sampling_chunk_rows = max(
            self.max_full_sampling_chunk_rows,
            min(rows, self.max_sampling_rows),
        )

    def _sample_full(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        metadata: SamplingBatchMetadata | None,
    ) -> torch.Tensor:
        sample_logits = logits.float()
        if sample_logits.requires_grad:
            sample_logits = sample_logits.div(temperatures.unsqueeze(dim=1))
        else:
            sample_logits.div_(temperatures.unsqueeze(dim=1))
        any_top_k_enabled = (
            metadata.any_top_k_enabled
            if metadata is not None
            else bool(((top_ks > 0) & (top_ks < logits.size(1))).any())
        )
        any_top_p_enabled = (
            metadata.any_top_p_enabled
            if metadata is not None
            else bool((top_ps < 1.0).any())
        )
        max_top_k = (
            metadata.max_top_k
            if metadata is not None
            else int(top_ks[(top_ks > 0) & (top_ks < logits.size(1))].max().item())
            if any_top_k_enabled
            else 0
        )
        sample_logits = apply_top_k_top_p(
            sample_logits,
            top_ks,
            top_ps,
            metadata,
            (
                self._ranks(
                    logits.size(1) if any_top_p_enabled else max_top_k,
                    logits.device,
                )
                if any_top_p_enabled or any_top_k_enabled
                else None
            ),
            inplace=not sample_logits.requires_grad,
        )
        probs = torch.softmax(sample_logits, dim=-1)
        return probs.div_(
            sample_logits.exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

    def _sample_full_chunked(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        metadata: SamplingBatchMetadata,
    ) -> torch.Tensor:
        chunk_count = (
            logits.size(0) + self.max_sampling_rows - 1
        ) // self.max_sampling_rows
        self._record_full_sampling(logits.size(0), chunk_count)
        if logits.size(0) <= self.max_sampling_rows:
            return self._sample_full(
                logits, temperatures, top_ks, top_ps, metadata
            )
        chunks = []
        for start in range(0, logits.size(0), self.max_sampling_rows):
            end = min(start + self.max_sampling_rows, logits.size(0))
            chunk_metadata = build_sampling_metadata(
                list(metadata.sample_temperatures[start:end]),
                list(metadata.sample_top_ks[start:end]),
                list(metadata.sample_top_ps[start:end]),
                metadata.vocab_size,
            )
            chunks.append(
                self._sample_full(
                    logits[start:end],
                    temperatures[start:end],
                    top_ks[start:end],
                    top_ps[start:end],
                    chunk_metadata,
                )
            )
        return torch.cat(chunks)

    @torch.inference_mode()
    def sample_top_k_candidates(
        self,
        selected_logits: torch.Tensor,
        selected_indices: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        metadata: SamplingBatchMetadata,
    ) -> torch.Tensor:
        if (
            metadata.sample_count != selected_logits.size(0)
            or not metadata.all_top_k_enabled
            or metadata.max_top_k != selected_logits.size(1)
        ):
            raise ValueError("sampling metadata does not match top-k candidates")
        return sample_top_k_candidates(
            selected_logits,
            selected_indices,
            temperatures,
            top_ks,
            top_ps,
            metadata.any_top_p_enabled,
            self._ranks(metadata.max_top_k, selected_logits.device),
            None,
        )

    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        metadata: SamplingBatchMetadata | None = None,
    ):
        """Sample one token for each sequence in the batch.

        ``temperature == 0`` enables greedy decoding for that sequence. Greedy
        rows bypass random sampling and return the maximum-logit token.

        Non-greedy rows apply temperature scaling, then top-k/top-p filtering,
        then use the same exponential-race sampling trick that nano-vLLM used
        originally. That trick is equivalent to sampling from ``probs`` but
        avoids calling ``torch.multinomial``.
        """
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, vocab_size]")
        if temperatures.ndim != 1 or temperatures.numel() != logits.size(0):
            raise ValueError("temperatures must have one value per logit row")
        if top_ks.ndim != 1 or top_ks.numel() != logits.size(0):
            raise ValueError("top_ks must have one value per logit row")
        if top_ps.ndim != 1 or top_ps.numel() != logits.size(0):
            raise ValueError("top_ps must have one value per logit row")
        if (
            temperatures.device != logits.device
            or top_ks.device != logits.device
            or top_ps.device != logits.device
        ):
            raise ValueError("sampling tensors must be on the same device as logits")
        if metadata is None:
            if torch.any(~torch.isfinite(temperatures)) or torch.any(temperatures < 0):
                raise ValueError("temperatures must be finite and non-negative")
        elif metadata.batch_size != logits.size(0) or metadata.vocab_size != logits.size(1):
            raise ValueError("sampling metadata does not match logits")

        greedy_mask = temperatures <= 1e-10
        all_greedy = (
            metadata.sample_count == 0
            if metadata is not None
            else bool(greedy_mask.all())
        )
        if all_greedy:
            return logits.argmax(dim=-1)

        # Only sampling rows need filtering, softmax, and random-number
        # generation. Greedy rows take the direct argmax fast path.
        all_sampling = (
            metadata.sample_count == logits.size(0)
            if metadata is not None
            else not bool(greedy_mask.any())
        )
        sample_mask = None if all_sampling else ~greedy_mask
        sample_source = (
            logits if all_sampling else logits[sample_mask]
        )
        sample_temperatures = (
            temperatures if all_sampling else temperatures[sample_mask]
        )
        sample_top_ks = top_ks if all_sampling else top_ks[sample_mask]
        sample_top_ps = top_ps if all_sampling else top_ps[sample_mask]
        if metadata is not None:
            all_top_k_enabled = metadata.all_top_k_enabled
            any_top_k_enabled = metadata.any_top_k_enabled
            any_top_p_enabled = metadata.any_top_p_enabled
            max_top_k = metadata.max_top_k
        else:
            sample_top_k_enabled = (
                (sample_top_ks > 0) & (sample_top_ks < logits.size(1))
            )
            all_top_k_enabled = bool(sample_top_k_enabled.all())
            any_top_k_enabled = bool(sample_top_k_enabled.any())
            any_top_p_enabled = bool((sample_top_ps < 1.0).any())
            max_top_k = (
                int(sample_top_ks[sample_top_k_enabled].max().item())
                if any_top_k_enabled
                else 0
            )
        if all_top_k_enabled and max_top_k <= self.max_compact_top_k:
            sample_tokens = sample_top_k_compact(
                sample_source,
                sample_temperatures,
                sample_top_ks,
                sample_top_ps,
                max_top_k,
                any_top_p_enabled,
                self._ranks(max_top_k, logits.device),
                None,
            )
            if all_sampling:
                return sample_tokens
            greedy_tokens = logits.argmax(dim=-1)
            greedy_tokens[sample_mask] = sample_tokens
            return greedy_tokens

        sample_tokens = (
            self._sample_full_chunked(
                sample_source,
                sample_temperatures,
                sample_top_ks,
                sample_top_ps,
                metadata,
            )
            if metadata is not None
            else self._sample_full(
                sample_source,
                sample_temperatures,
                sample_top_ks,
                sample_top_ps,
                None,
            )
        )
        if metadata is None:
            self._record_full_sampling(sample_source.size(0), 1)
        if all_sampling:
            return sample_tokens
        greedy_tokens = logits.argmax(dim=-1)
        # Argmax already returns fresh storage. Reuse it for sampled rows
        # instead of cloning the whole mixed-batch result.
        greedy_tokens[sample_mask] = sample_tokens
        return greedy_tokens
