import torch
from torch import nn


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
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
    if torch.any((top_ks != -1) & (top_ks <= 0)):
        raise ValueError("top_k must be -1 or positive")
    if torch.any(~torch.isfinite(top_ps)) or torch.any((top_ps <= 0) | (top_ps > 1)):
        raise ValueError("top_p must be finite and in (0, 1]")

    vocab_size = logits.size(1)
    top_k_enabled = (top_ks > 0) & (top_ks < vocab_size)
    top_p_enabled = top_ps < 1.0
    if bool(top_k_enabled.all()):
        max_top_k = int(top_ks.max().item())
        selected_logits, selected_indices = torch.topk(
            logits,
            max_top_k,
            dim=-1,
        )
        ranks = torch.arange(max_top_k, device=logits.device).unsqueeze(0)
        top_k_keep = ranks < top_ks.unsqueeze(1)
        selected_logits.masked_fill_(~top_k_keep, float("-inf"))
        if bool(top_p_enabled.any()):
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
    if not bool(top_p_enabled.any()):
        if not bool(top_k_enabled.any()):
            return logits
        max_top_k = int(top_ks[top_k_enabled].max().item())
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
        if bool((~top_k_enabled).any()):
            filtered_logits[~top_k_enabled] = logits[~top_k_enabled]
        return filtered_logits

    # Sort each request's vocabulary distribution from high logit to low logit.
    # Top-p needs this ordering to compute the cumulative probability mass.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # top_k == -1 means "disabled", so the effective k becomes vocab_size.
    # Otherwise clamp to vocab_size so oversized top_k values are harmless.
    full_vocab = torch.full_like(top_ks, vocab_size)
    effective_top_ks = torch.where(top_ks > 0, torch.minimum(top_ks, full_vocab), full_vocab)

    # rank 0 is the largest logit, rank 1 the second largest, etc.
    # A token survives top-k iff its sorted rank is smaller than k.
    ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    top_k_keep = ranks < effective_top_ks.unsqueeze(1)

    # Apply top-k before calculating top-p probabilities. This is important:
    # top-p is defined over the distribution that remains after top-k
    # filtering, not over the original full vocabulary distribution.
    top_k_logits = sorted_logits.masked_fill(~top_k_keep, float("-inf"))
    sorted_probs = torch.softmax(top_k_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_ps.unsqueeze(1)

    # Shift the removal mask right by one position so the first token that
    # crosses the top-p threshold is still kept. This is the common nucleus
    # sampling behavior and also guarantees at least one token survives.
    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
    sorted_remove[:, 0] = False
    sorted_keep = top_k_keep & ~sorted_remove

    # Apply the mask in sorted space.
    filtered_sorted_logits = sorted_logits.masked_fill(~sorted_keep, float("-inf"))

    # Scatter back to the original vocabulary order expected by softmax.
    filtered_logits = torch.empty_like(filtered_sorted_logits)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
    return filtered_logits


class Sampler(nn.Module):

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
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
        if torch.any(~torch.isfinite(temperatures)) or torch.any(temperatures < 0):
            raise ValueError("temperatures must be finite and non-negative")

        logits = logits.float()

        greedy_mask = temperatures <= 1e-10
        greedy_tokens = logits.argmax(dim=-1)
        if bool(greedy_mask.all()):
            return greedy_tokens

        # Only sampling rows need filtering, softmax, and random-number
        # generation. Greedy rows take the direct argmax fast path.
        sample_mask = ~greedy_mask
        sample_logits = logits[sample_mask].div(
            temperatures[sample_mask].unsqueeze(dim=1)
        )
        sample_logits = apply_top_k_top_p(
            sample_logits,
            top_ks[sample_mask],
            top_ps[sample_mask],
        )
        probs = torch.softmax(sample_logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        output = greedy_tokens.clone()
        output[sample_mask] = sample_tokens
        return output
