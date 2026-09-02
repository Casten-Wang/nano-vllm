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
        (top_k, top_p)
        for temperature, top_k, top_p in zip(temperatures, top_ks, top_ps)
        if temperature > 1e-10
    ]
    enabled_top_ks = [top_k for top_k, _ in sampled if 0 < top_k < vocab_size]
    return SamplingBatchMetadata(
        batch_size=len(temperatures),
        sample_count=len(sampled),
        vocab_size=vocab_size,
        all_top_k_enabled=bool(sampled) and len(enabled_top_ks) == len(sampled),
        any_top_k_enabled=bool(enabled_top_ks),
        any_top_p_enabled=any(top_p < 1.0 for _, top_p in sampled),
        max_top_k=max(enabled_top_ks, default=0),
    )


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    metadata: SamplingBatchMetadata | None = None,
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
        ranks = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
        top_k_keep = ranks < top_ks.unsqueeze(1)
        selected_logits.masked_fill_(~top_k_keep, float("-inf"))
        if any_top_p_enabled:
            selected_probs = torch.softmax(selected_logits, dim=-1)
            cumulative_probs = torch.cumsum(selected_probs, dim=-1)
            selected_remove = cumulative_probs > top_ps.unsqueeze(1)
            selected_remove[:, 1:] = selected_remove[:, :-1].clone()
            selected_remove[:, 0] = False
            selected_logits.masked_fill_(selected_remove, float("-inf"))
        filtered_logits = torch.full_like(logits, float("-inf"))
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
        ranks = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
        selected_logits.masked_fill_(
            ranks >= effective_top_ks.unsqueeze(1),
            float("-inf"),
        )
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(
            dim=-1,
            index=selected_indices,
            src=selected_logits,
        )
        if not all_top_k_enabled:
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
        ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
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
    sorted_remove = cumulative_probs > top_ps.unsqueeze(1)

    # Shift the removal mask right by one position so the first token that
    # crosses the top-p threshold is still kept. This is the common nucleus
    # sampling behavior and also guarantees at least one token survives.
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False
    sorted_keep = (
        ~sorted_remove
        if top_k_keep is None
        else top_k_keep & ~sorted_remove
    )

    # Apply the mask in sorted space.
    filtered_sorted_logits = sorted_logits.masked_fill(~sorted_keep, float("-inf"))

    # Scatter back to the original vocabulary order expected by softmax.
    filtered_logits = torch.empty_like(filtered_sorted_logits)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
    return filtered_logits


def compact_top_k_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int,
    any_top_p_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build filtered FP32 logits only for selected top-k candidates."""

    selected_logits, selected_indices = torch.topk(
        logits,
        max_top_k,
        dim=-1,
    )
    selected_logits = selected_logits.float().div_(temperatures.unsqueeze(1))
    ranks = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
    selected_logits.masked_fill_(
        ranks >= top_ks.unsqueeze(1),
        float("-inf"),
    )
    if any_top_p_enabled:
        selected_probs = torch.softmax(selected_logits, dim=-1)
        cumulative_probs = torch.cumsum(selected_probs, dim=-1)
        selected_remove = cumulative_probs > top_ps.unsqueeze(1)
        selected_remove[:, 1:] = selected_remove[:, :-1].clone()
        selected_remove[:, 0] = False
        selected_logits.masked_fill_(selected_remove, float("-inf"))
    return selected_logits, selected_indices


def sample_top_k_compact(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int,
    any_top_p_enabled: bool,
) -> torch.Tensor:
    """Sample from top-k candidates without materializing FP32 full logits."""

    selected_logits, selected_indices = compact_top_k_logits(
        logits,
        temperatures,
        top_ks,
        top_ps,
        max_top_k,
        any_top_p_enabled,
    )
    probabilities = torch.softmax(selected_logits, dim=-1)
    selected_offsets = probabilities.div_(
        torch.empty_like(probabilities).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return selected_indices.gather(1, selected_offsets.unsqueeze(1)).squeeze(1)


class Sampler(nn.Module):

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
            any_top_p_enabled = metadata.any_top_p_enabled
            max_top_k = metadata.max_top_k
        else:
            sample_top_k_enabled = (
                (sample_top_ks > 0) & (sample_top_ks < logits.size(1))
            )
            all_top_k_enabled = bool(sample_top_k_enabled.all())
            any_top_p_enabled = bool((sample_top_ps < 1.0).any())
            max_top_k = (
                int(sample_top_ks.max().item())
                if all_top_k_enabled
                else 0
            )
        if all_top_k_enabled:
            sample_tokens = sample_top_k_compact(
                sample_source,
                sample_temperatures,
                sample_top_ks,
                sample_top_ps,
                max_top_k,
                any_top_p_enabled,
            )
            if all_sampling:
                return sample_tokens
            greedy_tokens = logits.argmax(dim=-1)
            greedy_tokens[sample_mask] = sample_tokens
            return greedy_tokens

        sample_logits = sample_source.float().div(
            sample_temperatures.unsqueeze(dim=1)
        )
        sample_logits = apply_top_k_top_p(
            sample_logits,
            sample_top_ks,
            sample_top_ps,
            metadata,
        )
        probs = torch.softmax(sample_logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        if all_sampling:
            return sample_tokens
        greedy_tokens = logits.argmax(dim=-1)
        # Argmax already returns fresh storage. Reuse it for sampled rows
        # instead of cloning the whole mixed-batch result.
        greedy_tokens[sample_mask] = sample_tokens
        return greedy_tokens
