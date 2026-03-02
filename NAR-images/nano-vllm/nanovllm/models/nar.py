import torch
from torch import nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import NARAttention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.rotary_embedding import get_rope_2d


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num: int = 1):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float)[: (half_dim // 2)] / half_dim))
    t = torch.arange(grid_size, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, half_dim, 2), cache])
    return cond_cache


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
    torch_dtype: torch.dtype = torch.bfloat16


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train=False, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class NARMLP(nn.Module):
    def __init__(self, config: NARConfig):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = (hidden_dim + config.multiple_of - 1) // config.multiple_of * config.multiple_of
        
        self.gate_up_proj = MergedColumnParallelLinear(
            config.dim,
            [hidden_dim] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            hidden_dim,
            config.dim,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class NARDecoderLayer(nn.Module):
    def __init__(self, config: NARConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = NARAttention(
            hidden_size=config.dim,
            num_heads=config.n_head,
            num_kv_heads=config.n_kv_head or config.n_head,
            head_dim=config.dim // config.n_head,
            rms_norm_eps=config.norm_eps,
            rope_theta=config.rope_base,
            # use_qk_norm=config.use_qk_norm,
        )
        self.mlp = NARMLP(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        freqs_cis: torch.Tensor | None = None,
        proximity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, freqs_cis, proximity_mask)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class NARModel(nn.Module):
    def __init__(self, config: NARConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.dim)
        
        if config.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif config.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        self.layers = nn.ModuleList([NARDecoderLayer(config, i) for i in range(config.n_layer + config.medusa_attention_num)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.medusa_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        grid_size = int(config.block_size ** 0.5)
        assert grid_size * grid_size == config.block_size
        self.grid_size = grid_size
        
        head_dim = config.dim // config.n_head
        freqs_cis = precompute_freqs_cis_2d(grid_size, head_dim, config.rope_base, config.cls_token_num)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cond_embeddings: torch.Tensor | None = None,
        proximity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond_embeddings is not None:
            token_embeddings = self.embed_tokens(input_ids)
            if token_embeddings.dim() == 2:
                token_embeddings = token_embeddings.unsqueeze(1)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
        else:
            token_embeddings = self.embed_tokens(input_ids)
            # When input_ids is a flattened 1D tensor (varlen prefill), keep a consistent 3D shape.
            if token_embeddings.dim() == 2:
                token_embeddings = token_embeddings.unsqueeze(1)

            # Inference convenience: allow encoding class labels as "cls tokens" in input_ids
            # (positions < cls_token_num), and embed them with cls_embedding rather than the
            # visual token embedding table.
            if (
                positions is not None
                and self.config.model_type == "c2i"
                and hasattr(self, "cls_embedding")
                and hasattr(self.cls_embedding, "embedding_table")
            ):
                # Handle the common varlen path: input_ids: [N], positions: [N]
                if input_ids.dim() == 1 and positions.dim() == 1 and token_embeddings.dim() == 3 and token_embeddings.shape[1] == 1:
                    cond_mask = positions < self.config.cls_token_num
                    if torch.any(cond_mask):
                        cond_ids = input_ids[cond_mask].to(token_embeddings.device)
                        cond_emb = self.cls_embedding.embedding_table(cond_ids)
                        token_embeddings[cond_mask, 0, :] = cond_emb.to(token_embeddings.dtype)
                # Batched path: input_ids: [B, S], positions: [B, S]
                elif input_ids.dim() == 2 and positions.dim() == 2 and token_embeddings.dim() == 3:
                    cond_mask = positions < self.config.cls_token_num
                    if torch.any(cond_mask):
                        cond_ids = input_ids[cond_mask].to(token_embeddings.device)
                        cond_emb = self.cls_embedding.embedding_table(cond_ids)
                        token_embeddings[cond_mask] = cond_emb.to(token_embeddings.dtype)
        
        hidden_states = self.tok_dropout(token_embeddings)
        residual = None
        # Always prefer the precomputed 2D RoPE cache for NAR.
        # positions can be [B, L] or flattened [N] (varlen prefill).
        freqs_cis = None

        if positions is not None:
            pos = positions.to(self.freqs_cis.device)
            if pos.dim() == 1:
                freqs_cis = self.freqs_cis[pos].to(hidden_states.device)
            elif pos.dim() == 2:
                # positions across batch are identical in NAR decoding
                freqs_cis = self.freqs_cis[pos[0]].to(hidden_states.device)
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual, freqs_cis,
        proximity_mask)
            if i == self.config.n_layer - 1:
                h_at_n = hidden_states
                residual_at_n = residual

        # logitsR: use output at n_layer
        h_at_n, _ = self.norm(h_at_n, residual_at_n)
        # logitsB: use output after all layers (including medusa layers)
        medusa_hidden, _ = self.medusa_norm(hidden_states, residual)

        return h_at_n, medusa_hidden


class NARForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: NARConfig):
        super().__init__()
        self.config = config
        self.model = NARModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.dim)
        self.medusa_head = ParallelLMHead(config.vocab_size, config.dim)
        self.grid_size = config.block_size ** 0.5
        
        nn.init.constant_(self.lm_head.weight.data, 0)
        nn.init.constant_(self.medusa_head.weight.data, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cond_embeddings: torch.Tensor | None = None,
        proximity_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, medusa_hidden = self.model(input_ids, positions, cond_embeddings, proximity_mask)
        logitsR = self.lm_head(hidden_states)
        logitsB = self.medusa_head(medusa_hidden)
        return logitsR, logitsB

    def compute_logits(
        self,
        logits_tuple: tuple[torch.Tensor, torch.Tensor],
        accept_first_last: bool = True,
    ) -> torch.Tensor:
        logitsR, logitsB = logits_tuple
        
        if accept_first_last:
            first = logitsR[:, 0, :]
            last = logitsB[:, -1, :]
            middle = (logitsB[:, :-1, :] + logitsR[:, 1:, :]) / 2
            logits = torch.cat([first[:, None, :], middle, last[:, None, :]], dim=1)
        else:
            logits = (logitsB[:, :-1, :] + logitsR[:, 1:, :]) / 2
        
        return logits
