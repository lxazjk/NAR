from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    proximity_mask: torch.Tensor | None = None
    input_pos: torch.Tensor | None = None
    use_proximity_mask: bool = False
    # Optional: for FlexAttention path, shrink KV_LEN to reduce compute/materialization.
    # 0 means "not set".
    effective_kv_len: int = 0
    # When True, NARAttention uses contiguous KV cache + F.scaled_dot_product_attention
    # instead of paged KV cache + FlexAttention.
    use_contiguous_cache: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    proximity_mask=None,
    input_pos=None,
    use_proximity_mask=False,
    effective_kv_len=0,
    use_contiguous_cache=False,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        proximity_mask,
        input_pos,
        use_proximity_mask,
        effective_kv_len,
        use_contiguous_cache,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
