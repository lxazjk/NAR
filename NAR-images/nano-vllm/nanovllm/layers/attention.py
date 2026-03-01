import torch
from torch import nn
import triton
import triton.language as tl
from typing import Optional

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    offs = tl.arange(0, BLOCK)
    m = offs < D
    key_offsets = idx * key_stride + offs
    value_offsets = idx * value_stride + offs
    key = tl.load(key_ptr + key_offsets, mask=m, other=0)
    value = tl.load(value_ptr + value_offsets, mask=m, other=0)
    cache_offsets = slot * D + offs
    tl.store(k_cache_ptr + cache_offsets, key, mask=m)
    tl.store(v_cache_ptr + cache_offsets, value, mask=m)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # Triton tl.arange requires power-of-2 range; pad with a masked store.
    BLOCK = triton.next_power_of_2(D)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D, BLOCK)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o


class NARAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000,
    ):
        super().__init__()
        tp_size = 1
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                tp_size = dist.get_world_size()
        except:
            pass
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.hidden_size = hidden_size

        self.qkv_proj = nn.Linear(hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        
        self.k_cache = self.v_cache = torch.tensor([])
        self._flex_attention_compiled = None
        self._block_mask_cache = {}

    def _get_flex_attention(self):
        if self._flex_attention_compiled is None and FLEX_ATTENTION_AVAILABLE:
            # `torch.compile(flex_attention)` is fragile across torch/triton versions.
            # Prefer eager for correctness; callers can add their own compilation if desired.
            self._flex_attention_compiled = flex_attention
        return self._flex_attention_compiled

    def _get_block_mask(
        self,
        proximity_mask: torch.Tensor,
        input_pos: torch.Tensor,
        q_len: int,
        kv_len: int,
        num_heads: int,
        device: torch.device,
    ):
        """Build (and cache) a FlexAttention BlockMask.

        FlexAttention's block mask is defined over *relative* q/kv indices [0..Q_LEN), [0..KV_LEN).
        Our proximity mask is defined over *absolute* positions in the full sequence.
        We bridge them via `input_pos` (absolute positions for each query token).
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        # Cache key: shapes + device + a cheap fingerprint of input_pos.
        # Note: input_pos changes every step for NAR diagonal decoding, so this cache is best-effort.
        try:
            pos_fp = (int(input_pos.numel()), int(input_pos.reshape(-1)[0].item()), int(input_pos.reshape(-1)[-1].item()))
        except Exception:
            pos_fp = (int(input_pos.numel()), -1, -1)
        cache_key = (q_len, kv_len, int(proximity_mask.shape[0]), num_heads, str(device), pos_fp)
        cached = self._block_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        # mask_mod signature: (q_idx, kv_idx, b, h) -> bool
        # q_idx/kv_idx/b/h are tensors of indices.
        q_len_i = int(q_len)
        kv_len_i = int(kv_len)

        def mask_mod(q_idx, kv_idx, b, h):
            # create_block_mask may invoke mask_mod on padded indices within a block.
            # Clamp indices for safe gather, then mask out invalid lanes.
            q_safe = q_idx.clamp_max(q_len_i - 1)
            kv_safe = kv_idx.clamp_max(kv_len_i - 1)
            valid = (q_idx < q_len_i) & (kv_idx < kv_len_i)

            # Map relative q indices to absolute positions.
            # input_pos: [B, Q_LEN]
            abs_q = input_pos[b, q_safe]
            # proximity_mask: [B, KV_LEN, KV_LEN] over absolute positions.
            allowed = proximity_mask[b, abs_q, kv_safe]
            return allowed & valid

        block_mask = create_block_mask(
            mask_mod,
            B=int(proximity_mask.shape[0]),
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=str(device),
            BLOCK_SIZE=128,
            _compile=False,
        )
        self._block_mask_cache[cache_key] = block_mask
        return block_mask

    @staticmethod
    def _paged_kv_to_contiguous(
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor:
        """Materialize per-sequence KV from a paged cache.

        kv_cache: [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        block_tables: [B, NUM_BLOCKS_PER_SEQ]
        returns: [B, KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        """
        # Gather blocks then flatten. Some callers may pad block_tables with -1; clamp for safety.
        bt = block_tables.clamp_min(0)
        blocks = kv_cache.index_select(0, bt.reshape(-1)).view(
            block_tables.shape[0], block_tables.shape[1], kv_cache.shape[1], kv_cache.shape[2], kv_cache.shape[3]
        )
        flat = blocks.flatten(1, 2)  # [B, NUM_BLOCKS_PER_SEQ*BLOCK_SIZE, ...]
        return flat[:, :kv_len]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        proximity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        if freqs_cis is not None:
            q, k = self._apply_rotary_emb(freqs_cis, q, k)
        elif positions is not None:
            q, k = self._apply_rotary_emb_from_positions(positions, q, k)
        
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        has_kv_cache = k_cache.numel() and v_cache.numel() and context.slot_mapping is not None
        
        if has_kv_cache:
            # store_kvcache expects [N, num_heads, head_dim]
            k_store = k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
            v_store = v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous()
            store_kvcache(k_store, v_store, k_cache, v_cache, context.slot_mapping)
        
        use_flash = FLASH_ATTN_AVAILABLE and context.cu_seqlens_q is not None and context.cu_seqlens_k is not None

        # FlexAttention path: supports NAR proximity mask. Use it regardless of prefill/decode mode.
        use_flex = proximity_mask is not None and FLEX_ATTENTION_AVAILABLE
        if use_flex:
            flex_attn = self._get_flex_attention()
            q_t = q.transpose(1, 2)  # [B, H, Q, D]

            kv_len = int(proximity_mask.shape[-1])
            eff = getattr(context, "effective_kv_len", 0)
            if eff:
                kv_len = min(kv_len, int(eff))
            if has_kv_cache and context.block_tables is not None:
                k_full = self._paged_kv_to_contiguous(k_cache, context.block_tables, kv_len)
                v_full = self._paged_kv_to_contiguous(v_cache, context.block_tables, kv_len)
            else:
                k_full = k
                v_full = v

            k_t = k_full.transpose(1, 2)  # [B, KV_H, KV, D]
            v_t = v_full.transpose(1, 2)
            if self.num_kv_heads < self.num_heads:
                k_t = k_t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v_t = v_t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

            if context.input_pos is None:
                raise RuntimeError("FlexAttention for NAR requires context.input_pos (absolute query positions)")
            input_pos = context.input_pos
            if input_pos.dim() == 1:
                input_pos = input_pos.view(bsz, -1)

            q_len_i = int(q_t.shape[-2])
            kv_len_i = int(k_t.shape[-2])

            # Avoid create_block_mask() here: it is relatively brittle and can trigger
            # device-side asserts for highly irregular masks. Instead, use score_mod.
            def score_mod(score, b, h, q_idx, kv_idx):
                # Clamp indices for safety (padding within blocks).
                q_safe = q_idx.clamp_max(q_len_i - 1)
                kv_safe = kv_idx.clamp_max(kv_len_i - 1)
                valid = (q_idx < q_len_i) & (kv_idx < kv_len_i)
                abs_q = input_pos[b, q_safe]
                allowed = proximity_mask[b, abs_q, kv_safe] & valid
                return score.masked_fill(~allowed, float("-inf"))

            o = flex_attn(q_t, k_t, v_t, score_mod=score_mod, scale=self.scaling)
            o = o.transpose(1, 2).contiguous()
        elif context.is_prefill or not has_kv_cache:
            if use_flash:
                q_flat = q.view(-1, self.num_heads, self.head_dim)
                k_flat = k.view(-1, self.num_kv_heads, self.head_dim)
                v_flat = v.view(-1, self.num_kv_heads, self.head_dim)
                
                o = flash_attn_varlen_func(
                    q_flat, k_flat, v_flat,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    softmax_scale=self.scaling,
                    causal=True,
                )
                o = o.view(bsz, seqlen, self.num_heads * self.head_dim)
            else:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                
                if self.num_kv_heads < self.num_heads:
                    k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                
                if proximity_mask is not None:
                    # Support both additive masks (float) and boolean allow-masks.
                    if proximity_mask.dtype == torch.bool:
                        # proximity_mask is defined over absolute positions in the *full* sequence.
                        # `positions` already contains absolute positions for this (sub)sequence.
                        abs_pos = positions
                        if abs_pos is None:
                            raise RuntimeError("proximity_mask (bool) requires `positions` for absolute indexing")
                        if abs_pos.dim() == 1:
                            abs_pos = abs_pos.view(bsz, -1)
                        b_idx = torch.arange(bsz, device=proximity_mask.device)[:, None, None]
                        allowed = proximity_mask[b_idx, abs_pos[:, :, None], abs_pos[:, None, :]]  # [B, Q, K]
                        attn_weights = attn_weights.masked_fill(~allowed.unsqueeze(1), float("-inf"))
                    else:
                        attn_weights = attn_weights + proximity_mask
                
                attn_weights = torch.softmax(attn_weights, dim=-1)
                o = torch.matmul(attn_weights, v)
                o = o.transpose(1, 2).contiguous()
        else:
            if FLASH_ATTN_AVAILABLE and context.context_lens is not None and context.block_tables is not None:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scaling,
                    causal=True,
                )
                o = o.squeeze(1)
            else:
                q = q.transpose(1, 2)
                if self.num_kv_heads < self.num_heads:
                    k_cache = k_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_cache = v_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                
                attn_weights = torch.matmul(q, k_cache.transpose(-2, -1)) * self.scaling
                attn_weights = torch.softmax(attn_weights, dim=-1)
                o = torch.matmul(attn_weights, v_cache)
                o = o.transpose(1, 2).contiguous()
        
        output = self.o_proj(o.view(bsz, seqlen, -1))
        return output

    def _apply_rotary_emb(
        self,
        freqs_cis: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if freqs_cis is None:
            return query, key
        
        bsz, seqlen, num_heads, head_dim = query.shape
        
        xshaped = query.float().reshape(bsz, seqlen, num_heads, -1, 2)
        freqs_cis_expanded = freqs_cis.unsqueeze(0).unsqueeze(2)
        
        q_out = torch.stack([
            xshaped[..., 0] * freqs_cis_expanded[..., 0] - xshaped[..., 1] * freqs_cis_expanded[..., 1],
            xshaped[..., 1] * freqs_cis_expanded[..., 0] + xshaped[..., 0] * freqs_cis_expanded[..., 1],
        ], dim=-1)
        q_out = q_out.flatten(3).type_as(query)
        
        xshaped_k = key.float().reshape(bsz, seqlen, num_heads, -1, 2)
        k_out = torch.stack([
            xshaped_k[..., 0] * freqs_cis_expanded[..., 0] - xshaped_k[..., 1] * freqs_cis_expanded[..., 1],
            xshaped_k[..., 1] * freqs_cis_expanded[..., 0] + xshaped_k[..., 0] * freqs_cis_expanded[..., 1],
        ], dim=-1)
        k_out = k_out.flatten(3).type_as(key)
        
        return q_out, k_out

    def _apply_rotary_emb_from_positions(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            return query, key
        
        bsz, seqlen, num_heads, head_dim = query.shape
        # Rotary frequencies should be computed over `head_dim` (not head_dim//2).
        # This matches the standard RoPE construction used in the original NAR implementation.
        n_elem = head_dim
        freqs = 1.0 / (10000 ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=query.device) / n_elem))
        t = positions.float().flatten()
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
        
        xshaped = query.float().reshape(bsz, seqlen, num_heads, -1, 2)
        freqs_cis_expanded = cache.view(bsz, seqlen, 1, -1, 2)
        
        q_out = torch.stack([
            xshaped[..., 0] * freqs_cis_expanded[..., 0] - xshaped[..., 1] * freqs_cis_expanded[..., 1],
            xshaped[..., 1] * freqs_cis_expanded[..., 0] + xshaped[..., 0] * freqs_cis_expanded[..., 1],
        ], dim=-1)
        q_out = q_out.flatten(3).type_as(query)
        
        xshaped_k = key.float().reshape(bsz, seqlen, num_heads, -1, 2)
        k_out = torch.stack([
            xshaped_k[..., 0] * freqs_cis_expanded[..., 0] - xshaped_k[..., 1] * freqs_cis_expanded[..., 1],
            xshaped_k[..., 1] * freqs_cis_expanded[..., 0] + xshaped_k[..., 0] * freqs_cis_expanded[..., 1],
        ], dim=-1)
        k_out = k_out.flatten(3).type_as(key)
        
        return q_out, k_out


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
        
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        if residual is not None:
            return self.weight * hidden_states, residual
        return self.weight * hidden_states
