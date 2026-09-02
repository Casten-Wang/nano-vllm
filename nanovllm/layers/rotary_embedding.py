from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace_output: bool = False,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary dimension exceeds attention head dimension")
    x_rotary, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rotary.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    if inplace_output and not torch.is_grad_enabled():
        half_rotary_dim = rotary_dim // 2
        x[..., :half_rotary_dim].copy_(y1)
        x[..., half_rotary_dim:rotary_dim].copy_(y2)
        return x
    rotated = torch.cat((y1, y2), dim=-1).to(x.dtype)
    return torch.cat((rotated, x_pass), dim=-1)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
            raise ValueError(
                "rotary_dim must be positive, even, and no larger than head_size"
            )
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        inplace_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(
            query,
            cos,
            sin,
            inplace_output=inplace_output,
        )
        key = apply_rotary_emb(
            key,
            cos,
            sin,
            inplace_output=inplace_output,
        )
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
