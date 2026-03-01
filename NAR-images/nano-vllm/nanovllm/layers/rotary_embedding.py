from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


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
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class RotaryEmbedding2D(nn.Module):
    
    def __init__(
        self,
        head_size: int,
        grid_size: int,
        base: float = 10000,
        cls_token_num: int = 1,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.grid_size = grid_size
        half_dim = head_size // 2
        
        freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float) / half_dim))
        t = torch.arange(grid_size, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        freqs_grid = torch.concat([
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ], dim=-1)
        cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
        cache = cache_grid.flatten(0, 1)
        cond_cache = torch.cat([torch.zeros(cls_token_num, half_dim, 2), cache])
        self.register_buffer("freqs_cis", cond_cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis[positions]
        cos = freqs_cis[..., 0].unsqueeze(1).unsqueeze(2)
        sin = freqs_cis[..., 1].unsqueeze(1).unsqueeze(2)
        
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


@lru_cache(1)
def get_rope_2d(
    head_size: int,
    grid_size: int,
    base: float = 10000,
    cls_token_num: int = 1,
):
    rotary_emb = RotaryEmbedding2D(head_size, grid_size, base, cls_token_num)
    return rotary_emb
