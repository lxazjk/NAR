"""Public entrypoints.

This repository vendors nano-vllm as a local dependency. The original nano-vllm
engine is optimized for text LMs. For NAR image generation in this repo, we
provide a lightweight wrapper that runs the repo's NAR Transformer with a
"vLLM-style" diagonal decoding loop.

Why: the upstream-like NAR engine path in this vendored nano-vllm is incomplete
for this project (NAR diagonal decoding / CFG / conditioning). This wrapper is
used by benchmarks to compare baseline vs accelerated decoding.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import NARSamplingParams


class LLM(LLMEngine):
    pass


class NARLLM:
    def __init__(
        self,
        model: str,
        *,
        gpt_ckpt: Optional[str] = None,
        gpt_model: Optional[str] = None,
        model_type: str = "nar",
        block_size: int = 256,
        cls_token_num: int = 1,
        num_classes: int = 1000,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **_: object,
    ):
        # `model` is kept for API compatibility with nano-vllm; in this repo we treat it as ckpt path.
        self.gpt_ckpt = gpt_ckpt or model
        self.block_size = block_size
        self.cls_token_num = cls_token_num
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.dtype = dtype

        if gpt_model is None:
            # Best-effort inference from common configs (matches scripts in this repo)
            gpt_model = "GPT-B"

        # Make repo imports work even if not installed.
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from autoregressive.models.gpt import GPT_models

        # Build NAR Transformer from the main repo implementation
        self.model = GPT_models[gpt_model](
            vocab_size=16384,
            block_size=self.block_size,
            num_classes=self.num_classes,
            cls_token_num=self.cls_token_num,
            model_type="c2i",
        ).to(device=self.device, dtype=self.dtype)

        checkpoint = torch.load(self.gpt_ckpt, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "module" in checkpoint:
            state = checkpoint["module"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def generate_image(
        self,
        condition: torch.Tensor,
        sampling_params: Optional[NARSamplingParams] = None,
        use_tqdm: bool = True,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = NARSamplingParams(block_size=self.block_size, cls_token_num=self.cls_token_num)

        # condition: [B] class labels
        cond = condition.to(device=self.device)
        if cond.dtype != torch.long:
            cond = cond.long()

        # Use a vLLM-style diagonal decode loop (same dependency pattern as
        # `autoregressive/models/generate.py`'s diagonal decoding).
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from tqdm import tqdm

        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size

        batch_size = cond.shape[0]

        cfg_scale = float(sampling_params.cfg_scale)
        cfg_interval = int(sampling_params.cfg_interval)

        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond

        T = self.cls_token_num
        T_new = T + self.block_size
        max_batch_size_cfg = batch_size * 2 if cfg_scale > 1.0 else batch_size
        # Ensure internal caches / proximity mask are allocated on the correct device.
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=max_batch_size_cfg,
                max_seq_length=T_new,
                dtype=self.model.tok_embeddings.weight.dtype,
            )

        # Prefill: sample the first (top-left) token
        input_pos = torch.arange(0, T, device=self.device)
        logits, _ = self.model(None, cond_combined, input_pos=input_pos)
        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, batch_size, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        next_token = _sample_logits(
            logits,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        # Diagonal decode
        input_pos = torch.tensor([T], device=self.device, dtype=torch.int)
        bias = input_pos.item()
        new_tokens = [[] for _ in range(grid_size)]
        new_tokens[0].append(next_token)

        cfg_flag = True
        iterations = 2 * grid_size - 1
        generated_token_num = 1

        it_range = range(1, iterations)
        if use_tqdm:
            it_range = tqdm(it_range, desc="Generating", dynamic_ncols=True)

        for itera in it_range:
            with sdpa_kernel(SDPBackend.MATH):
                accept_token_num = itera + 1 if itera < grid_size else iterations - itera
                if cfg_interval > -1 and generated_token_num > cfg_interval:
                    cfg_flag = False

                accept_first_last = itera < grid_size
                if cfg_scale > 1.0:
                    x_combined = torch.cat([next_token, next_token])
                    logits, _ = self.model(
                        x_combined,
                        cond_idx=None,
                        input_pos=input_pos,
                        accept_first_last=accept_first_last,
                    )
                    cond_logits, uncond_logits = torch.split(logits, batch_size, dim=0)
                    logits = (
                        uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                        if cfg_flag
                        else cond_logits
                    )
                else:
                    logits, _ = self.model(
                        next_token,
                        cond_idx=None,
                        input_pos=input_pos,
                        accept_first_last=accept_first_last,
                    )

                next_token = _sample_logits(
                    logits,
                    temperature=sampling_params.temperature,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                )

                input_pos_list = []
                for i in range(grid_size):
                    j = itera - i
                    if 0 <= j < grid_size:
                        input_pos_list.append(bias + i * grid_size + j)
                input_pos = torch.tensor(input_pos_list, device=self.device)

                i = 0
                for token_arr in new_tokens:
                    if i >= next_token.shape[1]:
                        break
                    if len(token_arr) < grid_size:
                        token_arr.append(next_token[:, i].view(-1, 1))
                        i += 1

                generated_token_num += accept_token_num

        new_tokens_tensor = torch.cat([torch.cat(token_arr, dim=-1) for token_arr in new_tokens], dim=-1)
        return new_tokens_tensor


def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    # logits: [B, L, V]
    idx = []
    for i in range(logits.shape[1]):
        one_logits = logits[:, i, :] / max(float(temperature), 1e-5)
        if top_k > 0 or top_p < 1.0:
            one_logits = _top_k_top_p_filtering(one_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(one_logits, dim=-1)
        one_idx = torch.multinomial(probs, num_samples=1)
        idx.append(one_idx.view(-1, 1))
    return torch.cat(idx, dim=-1)
