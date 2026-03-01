from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from nanovllm.config import create_proximity_mask
from nanovllm.models.nar import NARForCausalLM, NARConfig
from nanovllm.sampling_params import NARSamplingParams
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_nar_model_from_gpt


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    # logits: [B, V]
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Sample tokens from logits.

    logits: [B, L, V]
    returns: [B, L]
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    _, seqlen, _ = logits.shape
    out = []
    for i in range(seqlen):
        one = logits[:, i, :].float() / max(float(temperature), 1e-5)
        if top_k > 0 or top_p < 1.0:
            one = _top_k_top_p_filtering(one, top_k=top_k, top_p=top_p)
        probs = torch.softmax(one, dim=-1)
        out.append(torch.multinomial(probs, num_samples=1))
    return torch.cat(out, dim=-1)


@dataclass
class NARPagedConfig:
    # KV cache is stored in fixed-size blocks (paged KV). Default 256 aligns with nano-vllm.
    kvcache_block_size: int = 256
    # When proximity mask is enabled, use FlexAttention.
    use_proximity_mask: bool = True


class NARPagedLLM:
    """NAR token generator using paged KV cache + FlexAttention (for proximity mask).

    This keeps the original NAR diagonal decoding scheme, but stores KV in blocks
    (paged KV) and feeds a FlexAttention block mask derived from the proximity mask.
    """

    def __init__(
        self,
        gpt_ckpt: str,
        *,
        block_size: int = 256,
        cls_token_num: int = 1,
        num_classes: int = 1000,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        paged: Optional[NARPagedConfig] = None,
        # GPT-B defaults
        dim: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        medusa_attention_num: int = 1,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.gpt_ckpt = gpt_ckpt
        self.block_size = int(block_size)
        self.cls_token_num = int(cls_token_num)
        self.num_classes = int(num_classes)

        if paged is None:
            paged = NARPagedConfig()
        self.paged = paged
        self.kvcache_block_size = int(paged.kvcache_block_size)

        grid = int(math.isqrt(self.block_size))
        if grid * grid != self.block_size:
            raise ValueError(f"block_size must be a perfect square, got {self.block_size}")
        self.grid_size = grid

        cfg = NARConfig(
            dim=dim,
            n_layer=n_layer,
            n_head=n_head,
            vocab_size=16384,
            block_size=self.block_size,
            cls_token_num=self.cls_token_num,
            num_classes=self.num_classes,
            caption_dim=2048,
            class_dropout_prob=0.1,
            model_type="c2i",
            medusa_attention_num=medusa_attention_num,
        )

        torch.set_default_device(str(self.device))
        torch.set_default_dtype(self.dtype)
        self.model = NARForCausalLM(cfg).to(device=self.device, dtype=self.dtype)
        load_nar_model_from_gpt(self.model, self.gpt_ckpt)
        self.model.eval()
        torch.set_default_device("cpu")

        self._num_attn_layers = sum(1 for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
        if self._num_attn_layers <= 0:
            raise RuntimeError("Failed to find attention layers with k_cache/v_cache")

    def _allocate_kv_cache(self, batch_size_cfg: int, max_seq_len: int):
        num_kv_heads = self.model.config.n_kv_head or self.model.config.n_head
        head_dim = self.model.config.dim // self.model.config.n_head
        num_blocks_per_seq = (max_seq_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        num_total_blocks = batch_size_cfg * num_blocks_per_seq
        kv_cache = torch.empty(
            2,
            self._num_attn_layers,
            num_total_blocks,
            self.kvcache_block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        layer_id = 0
        for m in self.model.modules():
            if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                m.k_cache = kv_cache[0, layer_id]
                m.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        return kv_cache, num_blocks_per_seq

    def _build_block_tables(self, batch_size_cfg: int, num_blocks_per_seq: int) -> torch.Tensor:
        base = torch.arange(batch_size_cfg, device=self.device, dtype=torch.int32)[:, None] * num_blocks_per_seq
        offs = torch.arange(num_blocks_per_seq, device=self.device, dtype=torch.int32)[None, :]
        return base + offs

    def _slot_mapping_from_positions(self, block_tables: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, L] int64
        pos_i32 = positions.to(torch.int32)
        block_idx_i32 = torch.div(pos_i32, self.kvcache_block_size, rounding_mode="floor")
        offset = pos_i32 - block_idx_i32 * self.kvcache_block_size
        block_id = torch.gather(block_tables, 1, block_idx_i32.to(torch.int64))
        slot = block_id * self.kvcache_block_size + offset
        return slot.reshape(-1)

    @torch.no_grad()
    def generate_image(
        self,
        condition: torch.Tensor,
        sampling_params: Optional[NARSamplingParams] = None,
        use_tqdm: bool = True,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = NARSamplingParams(block_size=self.block_size, cls_token_num=self.cls_token_num, model_type="c2i")

        cond = condition.to(device=self.device)
        if cond.dtype != torch.long:
            cond = cond.long()

        cfg_scale = float(sampling_params.cfg_scale)
        batch_size = int(cond.shape[0])
        batch_size_cfg = batch_size * 2 if cfg_scale > 1.0 else batch_size

        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null], dim=0)
        else:
            cond_combined = cond

        max_seq_len = self.cls_token_num + self.block_size
        _, num_blocks_per_seq = self._allocate_kv_cache(batch_size_cfg, max_seq_len)
        block_tables = self._build_block_tables(batch_size_cfg, num_blocks_per_seq)

        proximity_mask = None
        if self.paged.use_proximity_mask:
            proximity_mask = create_proximity_mask(
                max_seq_length=max_seq_len,
                cls_token_num=self.cls_token_num,
                block_size=self.block_size,
                batch_size=batch_size_cfg,
                device=str(self.device),
                dtype=torch.bool,
            )

        # Prefill: cls token(s) at absolute position 0
        cls_pos = torch.zeros((batch_size_cfg, 1), device=self.device, dtype=torch.long)
        cls_ids = cond_combined.view(batch_size_cfg, 1)
        slot_mapping = self._slot_mapping_from_positions(block_tables, cls_pos)
        eff_kv = int(cls_pos.max().item()) + 1
        set_context(
            False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            proximity_mask=proximity_mask,
            input_pos=cls_pos,
            use_proximity_mask=self.paged.use_proximity_mask,
            max_seqlen_k=max_seq_len,
            effective_kv_len=eff_kv,
        )
        logitsR, logitsB = self.model(cls_ids, cls_pos, proximity_mask=proximity_mask)
        reset_context()
        prefill_logits = (logitsR[:, -1, :] + logitsB[:, -1, :]) / 2.0  # [B_cfg, V]

        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(prefill_logits, batch_size, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            logits = logits[:, None, :]
        else:
            logits = prefill_logits[:, None, :]

        first_token = _sample_logits(
            logits,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
        cur_token = first_token
        cur_token_cfg = torch.cat([cur_token, cur_token], dim=0) if cfg_scale > 1.0 else cur_token

        new_tokens = [[] for _ in range(self.grid_size)]
        new_tokens[0].append(cur_token)

        input_pos_list = [self.cls_token_num]
        iterations = 2 * self.grid_size - 1
        generated_token_num = 1
        cfg_flag = True

        it_range = range(1, iterations)
        if use_tqdm:
            from tqdm import tqdm
            it_range = tqdm(it_range, desc="NAR(paged)", dynamic_ncols=True)

        for itera in it_range:
            accept_token_num = itera + 1 if itera < self.grid_size else iterations - itera
            if sampling_params.cfg_interval > -1 and generated_token_num > sampling_params.cfg_interval:
                cfg_flag = False
            accept_first_last = itera < self.grid_size

            pos = torch.tensor(input_pos_list, device=self.device, dtype=torch.long)[None, :].repeat(batch_size_cfg, 1)
            slot_mapping = self._slot_mapping_from_positions(block_tables, pos)
            eff_kv = int(pos.max().item()) + 1
            set_context(
                False,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                proximity_mask=proximity_mask,
                input_pos=pos,
                use_proximity_mask=self.paged.use_proximity_mask,
                max_seqlen_k=max_seq_len,
                effective_kv_len=eff_kv,
            )
            logitsR, logitsB = self.model(cur_token_cfg, pos, proximity_mask=proximity_mask)
            reset_context()

            logits_all = self.model.compute_logits((logitsR, logitsB), accept_first_last=accept_first_last)

            if cfg_scale > 1.0:
                cond_logits, uncond_logits = torch.split(logits_all, batch_size, dim=0)
                if cfg_flag:
                    logits_all = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits_all = cond_logits

            next_token = _sample_logits(
                logits_all,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
            )

            cur_token = next_token
            cur_token_cfg = torch.cat([cur_token, cur_token], dim=0) if cfg_scale > 1.0 else cur_token

            next_pos = []
            for i in range(self.grid_size):
                j = itera - i
                if 0 <= j < self.grid_size:
                    next_pos.append(self.cls_token_num + i * self.grid_size + j)
            input_pos_list = next_pos

            i = 0
            for token_arr in new_tokens:
                if i >= cur_token.shape[1]:
                    break
                if len(token_arr) < self.grid_size:
                    token_arr.append(cur_token[:, i].view(-1, 1))
                    i += 1

            generated_token_num += accept_token_num

        out = torch.cat([torch.cat(token_arr, dim=-1) for token_arr in new_tokens], dim=-1)
        return out


class NARContinuousBatcher:
    """简化版 continuous batching：允许新请求在对角 decode 过程中插入。

    约束：
    - 以“对角 step”为调度粒度；同一 step 的请求会被打成一个 batch。
    - 每个请求占用两份 KV（cond/uncond）以支持 CFG（cfg_scale>1）。
    - KV 采用 paged blocks，block_tables 在初始化时为每个序列预分配（不做 block sharing）。
    """

    def __init__(
        self,
        base: NARPagedLLM,
        *,
        max_num_requests: int,
        sampling_params: NARSamplingParams,
    ):
        self.base = base
        self.sampling_params = sampling_params
        self.cfg_scale = float(sampling_params.cfg_scale)
        self.max_num_requests = int(max_num_requests)
        self.batch_size_cfg = self.max_num_requests * (2 if self.cfg_scale > 1.0 else 1)

        self.max_seq_len = self.base.cls_token_num + self.base.block_size
        _, num_blocks_per_seq = self.base._allocate_kv_cache(self.batch_size_cfg, self.max_seq_len)
        self._global_block_tables = self.base._build_block_tables(self.batch_size_cfg, num_blocks_per_seq)

        self._proximity_mask = None
        if self.base.paged.use_proximity_mask:
            self._proximity_mask = create_proximity_mask(
                max_seq_length=self.max_seq_len,
                cls_token_num=self.base.cls_token_num,
                block_size=self.base.block_size,
                batch_size=self.batch_size_cfg,
                device=str(self.base.device),
                dtype=torch.bool,
            )

        # Request slots
        self._free_slots = list(range(self.max_num_requests))
        self._active: dict[int, dict] = {}

        self._iterations = 2 * self.base.grid_size - 1

    def add_requests(self, labels: torch.Tensor) -> list[int]:
        """添加一批新请求，返回 request_ids（内部 slot id）。"""
        labels = labels.to(device=self.base.device)
        if labels.dtype != torch.long:
            labels = labels.long()

        if labels.numel() == 0:
            return []
        if labels.numel() > len(self._free_slots):
            raise RuntimeError("not enough free slots for new requests")

        slots = [self._free_slots.pop(0) for _ in range(int(labels.numel()))]

        # Prefill cls token(s) for all new requests in one batch.
        if self.cfg_scale > 1.0:
            labels_null = torch.ones_like(labels) * self.base.num_classes
            labels_cfg = torch.cat([labels, labels_null], dim=0)
            rows = torch.tensor([2 * s for s in slots] + [2 * s + 1 for s in slots], device=self.base.device, dtype=torch.long)
        else:
            labels_cfg = labels
            rows = torch.tensor(slots, device=self.base.device, dtype=torch.long)

        block_tables = self._global_block_tables.index_select(0, rows.to(torch.int64))
        prox = self._proximity_mask.index_select(0, rows.to(torch.int64)) if self._proximity_mask is not None else None

        cls_pos = torch.zeros((rows.numel(), 1), device=self.base.device, dtype=torch.long)
        cls_ids = labels_cfg.view(rows.numel(), 1)
        slot_mapping = self.base._slot_mapping_from_positions(block_tables, cls_pos)
        eff_kv = int(cls_pos.max().item()) + 1
        set_context(
            False,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            proximity_mask=prox,
            input_pos=cls_pos,
            use_proximity_mask=self.base.paged.use_proximity_mask,
            max_seqlen_k=self.max_seq_len,
            effective_kv_len=eff_kv,
        )
        logitsR, logitsB = self.base.model(cls_ids, cls_pos, proximity_mask=prox)
        reset_context()
        prefill_logits = (logitsR[:, -1, :] + logitsB[:, -1, :]) / 2.0

        if self.cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(prefill_logits, len(slots), dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale
        else:
            logits = prefill_logits
        first_token = _sample_logits(
            logits[:, None, :],
            temperature=self.sampling_params.temperature,
            top_k=self.sampling_params.top_k,
            top_p=self.sampling_params.top_p,
        )  # [B_new, 1]

        for i, s in enumerate(slots):
            # init state
            self._active[s] = {
                "step": 1,  # next decode iteration index (matches NARPagedLLM loop)
                "input_pos": [self.base.cls_token_num],
                "cur_token": first_token[i : i + 1],  # [1, 1]
                "slashed": [[first_token[i : i + 1]]]+[[] for _ in range(self.base.grid_size-1)],
            }
        return slots

    def is_finished(self) -> bool:
        return len(self._active) == 0

    @torch.no_grad()
    def step(self):
        """推进所有活跃请求一个对角 step（按 step 分组）。"""
        if not self._active:
            return []

        finished = []
        # group by step
        groups: dict[int, list[int]] = {}
        for rid, st in self._active.items():
            groups.setdefault(int(st["step"]), []).append(rid)

        for itera, rids in sorted(groups.items()):
            if itera >= self._iterations:
                # already done
                for rid in rids:
                    finished.append(rid)
                continue

            accept_first_last = itera < self.base.grid_size
            # current diagonal length == len(input_pos)
            input_pos_list = self._active[rids[0]]["input_pos"]
            cur_len = len(input_pos_list)

            # Build a batch for this group
            cur_tokens = torch.cat([self._active[r]["cur_token"] for r in rids], dim=0)  # [B, L]

            if self.cfg_scale > 1.0:
                # duplicate tokens for cond/uncond
                cur_tokens_cfg = torch.cat([cur_tokens, cur_tokens], dim=0)
                rows = torch.tensor([2 * r for r in rids] + [2 * r + 1 for r in rids], device=self.base.device, dtype=torch.long)
            else:
                cur_tokens_cfg = cur_tokens
                rows = torch.tensor(rids, device=self.base.device, dtype=torch.long)

            block_tables = self._global_block_tables.index_select(0, rows.to(torch.int64))
            prox = self._proximity_mask.index_select(0, rows.to(torch.int64)) if self._proximity_mask is not None else None

            pos = torch.tensor(input_pos_list, device=self.base.device, dtype=torch.long)[None, :].repeat(rows.numel(), 1)
            slot_mapping = self.base._slot_mapping_from_positions(block_tables, pos)
            eff_kv = int(pos.max().item()) + 1
            set_context(
                False,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                proximity_mask=prox,
                input_pos=pos,
                use_proximity_mask=self.base.paged.use_proximity_mask,
                max_seqlen_k=self.max_seq_len,
                effective_kv_len=eff_kv,
            )
            logitsR, logitsB = self.base.model(cur_tokens_cfg, pos, proximity_mask=prox)
            reset_context()

            logits_all = self.base.model.compute_logits((logitsR, logitsB), accept_first_last=accept_first_last)
            if self.cfg_scale > 1.0:
                cond_logits, uncond_logits = torch.split(logits_all, len(rids), dim=0)
                logits_all = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale

            next_token = _sample_logits(
                logits_all,
                temperature=self.sampling_params.temperature,
                top_k=self.sampling_params.top_k,
                top_p=self.sampling_params.top_p,
            )

            # update per request
            for i, rid in enumerate(rids):
                st = self._active[rid]
                st["cur_token"] = next_token[i : i + 1]

                # append into slashes
                j = 0
                for token_arr in st["slashed"]:
                    if j >= st["cur_token"].shape[1]:
                        break
                    if len(token_arr) < self.base.grid_size:
                        token_arr.append(st["cur_token"][:, j].view(-1, 1))
                        j += 1

                # next positions
                next_pos = []
                for ii in range(self.base.grid_size):
                    jj = itera - ii
                    if 0 <= jj < self.base.grid_size:
                        next_pos.append(self.base.cls_token_num + ii * self.base.grid_size + jj)
                st["input_pos"] = next_pos
                st["step"] = itera + 1

                if st["step"] >= self._iterations:
                    finished.append(rid)

        # finalize finished
        outputs = []
        for rid in finished:
            st = self._active.pop(rid, None)
            if st is None:
                continue
            out = torch.cat([torch.cat(arr, dim=-1) for arr in st["slashed"]], dim=-1)
            outputs.append((rid, out))
            self._free_slots.append(rid)
        self._free_slots.sort()
        return outputs
