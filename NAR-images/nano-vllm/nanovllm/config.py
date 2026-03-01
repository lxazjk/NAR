import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Any
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    
    model_type: str = "qwen3"
    use_proximity_mask: bool = False
    block_size: int = 256
    cls_token_num: int = 1
    grid_size: int = 16
    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    medusa_attention_num: int = 1
    cfg_scale: float = 1.0
    cfg_interval: int = -1
    gpt_ckpt: str | None = None
    n_layer: int = 12
    n_head: int = 12
    dim: int = 768

    def __post_init__(self):
        if os.path.isdir(self.model):
            self.hf_config = AutoConfig.from_pretrained(self.model)
            if hasattr(self.hf_config, 'max_position_embeddings'):
                self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        
        if self.model_type == "nar":
            self.grid_size = int(self.block_size ** 0.5)
            assert self.grid_size * self.grid_size == self.block_size
            # NAR's effective maximum sequence length is (condition tokens + image tokens).
            # Keep it consistent with precomputed RoPE/proximity mask to avoid OOB in warmup.
            self.max_model_len = self.cls_token_num + self.block_size


@dataclass
class NARConfig:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    medusa_attention_num: int = 1
    torch_dtype: Any = field(default_factory=lambda: torch.bfloat16)
    use_qk_norm: bool = False
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = cls()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num: int = 1):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])
    return cond_cache


def setup_proximity_mask(mask: torch.Tensor, block_size: int):
    mask[:, :] = 0
    cur_token, previous_token = [], []
    H, W = int(block_size ** 0.5), int(block_size ** 0.5)
    for c in range(H + W - 1):
        cur_token = []
        for h in range(H):
            w = c - h
            if 0 <= w < W:
                token_id = (h * W + w)
                cur_token.append(token_id)
                previous_token.append(token_id)
        for id in cur_token:
            mask[id, previous_token] = 1


def create_proximity_mask(
    max_seq_length: int,
    cls_token_num: int,
    block_size: int,
    batch_size: int = 1,
    device: str = "cuda",
    dtype = torch.bool,
):
    proximity_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=dtype, device=device))
    low = cls_token_num
    high = cls_token_num + block_size
    setup_proximity_mask(proximity_mask[low: high, low: high], block_size)
    proximity_mask = proximity_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return proximity_mask
