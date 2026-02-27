"""
Multi-request benchmark comparing original vs vLLM-style inference for NAR models.
Tests continuous batching and parallel request handling.
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import argparse
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nano-vllm"))
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models

from nanovllm import NARPagedLLM, NARContinuousBatcher, NARSamplingParams


@dataclass
class Request:
    request_id: int
    class_label: int
    status: str = "waiting"  # waiting, running, finished
    generated_tokens: Optional[torch.Tensor] = None


def top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0, 
                          filter_value: float = -float("Inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits: bool = True):
    idx = []
    for i in range(logits.shape[1]):
        one_logits = logits[:, i, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            one_logits = top_k_top_p_filtering(one_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(one_logits, dim=-1)
        if sample_logits:
            one_idx = torch.multinomial(probs, num_samples=1)
        else:
            _, one_idx = torch.topk(probs, k=1, dim=-1)
        idx.append(one_idx.view(-1, 1))
    return torch.cat(idx, dim=-1)


class OriginalInference:
    """Original implementation - processes requests sequentially."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def generate_single(self, cond_idx: torch.Tensor, max_new_tokens: int, 
                        cfg_scale: float, **sampling_kwargs):
        """Generate for a single request."""
        from autoregressive.models.generate import generate
        return generate(self.model, cond_idx, max_new_tokens, cfg_scale=cfg_scale, **sampling_kwargs)
    
    def generate_batch(self, requests: List[Request], max_new_tokens: int,
                       cfg_scale: float, **sampling_kwargs):
        """Process requests sequentially."""
        results = []
        for req in requests:
            cond_idx = torch.tensor([req.class_label], device=self.device)
            tokens = self.generate_single(cond_idx, max_new_tokens, cfg_scale, **sampling_kwargs)
            results.append(tokens)
        return results


class vLLMStyleInference:
    """批量推理实现（非 continuous batching）。

    说明：原始脚本里“vLLM-style”示例实现不完整，容易在对角解码时出现 token/position 对不齐。
    这里用仓库自带的 `autoregressive.models.generate.generate()` 作为稳定的 batch 基线：
    - 一次性把 N 个 request 打成一个 batch
    - 仍使用原始模型的 KVCache + proximity_mask
    - 不做 request 插队/动态并批（continuous batching 由 `--backend nano-vllm` 覆盖）
    """

    def __init__(self, model, device, block_size: int = 16):
        self.model = model
        self.device = device
        self.block_size = block_size

    def generate_batch(self, requests: List[Request], max_new_tokens: int, cfg_scale: float, **sampling_kwargs):
        from autoregressive.models.generate import generate

        cond_indices = torch.tensor([r.class_label for r in requests], device=self.device)
        tokens = generate(
            self.model,
            cond_indices,
            max_new_tokens,
            cfg_scale=cfg_scale,
            **sampling_kwargs,
        )
        return [tokens[i : i + 1] for i in range(tokens.shape[0])]


def run_benchmark(args, device):
    """Run benchmark comparing original vs vLLM-style inference."""
    
    print(f"\n{'='*70}")
    print(f"Multi-Request Benchmark: {args.num_requests} concurrent requests")
    print(f"{'='*70}")
    
    # NOTE: GPT's `block_size` here means number of visual tokens (H*W), not the grid edge length.
    grid = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=grid * grid,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=getattr(torch, args.precision))
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    gpt_model.load_state_dict(checkpoint, strict=False)
    gpt_model.eval()
    del checkpoint
    print("Model loaded")
    
    import random
    random.seed(args.seed)
    class_labels = [random.randint(0, args.num_classes-1) for _ in range(args.num_requests)]
    
    requests = [Request(request_id=i, class_label=label) for i, label in enumerate(class_labels)]
    
    max_new_tokens = (args.image_size // args.downsample_size) ** 2
    
    print(f"\n{'='*50}")
    print("Original Implementation (Sequential)")
    print(f"{'='*50}")
    
    original_inference = OriginalInference(gpt_model, device)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    
    original_results = original_inference.generate_batch(
        requests, max_new_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True
    )
    
    torch.cuda.synchronize()
    original_time = time.time() - t1
    original_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Time: {original_time:.2f}s")
    print(f"Throughput: {args.num_requests / original_time:.3f} requests/s")
    print(f"Peak Memory: {original_memory:.2f} GB")
    
    print(f"\n{'='*50}")
    print("vLLM-Style Implementation (Batched)")
    print(f"{'='*50}")
    
    vllm_inference = vLLMStyleInference(gpt_model, device)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    
    vllm_results = vllm_inference.generate_batch(
        requests, max_new_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True
    )
    
    torch.cuda.synchronize()
    vllm_time = time.time() - t1
    vllm_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Time: {vllm_time:.2f}s")
    print(f"Throughput: {args.num_requests / vllm_time:.3f} requests/s")
    print(f"Peak Memory: {vllm_memory:.2f} GB")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Original':<15} {'vLLM-Style':<15} {'Speedup':<10}")
    print(f"{'-'*65}")
    print(f"{'Time (s)':<25} {original_time:<15.2f} {vllm_time:<15.2f} {original_time/vllm_time:<10.2f}x")
    print(f"{'Throughput (req/s)':<25} {args.num_requests/original_time:<15.3f} {args.num_requests/vllm_time:<15.3f} {vllm_time/original_time:<10.2f}x")
    print(f"{'Peak Memory (GB)':<25} {original_memory:<15.2f} {vllm_memory:<15.2f} {original_memory/vllm_memory:<10.2f}x")
    
    return {
        'original_time': original_time,
        'vllm_time': vllm_time,
        'speedup': original_time / vllm_time,
        'original_memory': original_memory,
        'vllm_memory': vllm_memory,
    }


def run_nanovllm_continuous(args, device):
    """使用 nano-vllm 风格的 paged KV + FlexAttention，并模拟 continuous batching."""
    assert args.image_size == 256 and args.downsample_size == 16, "当前基准只覆盖 256/16 (block_size=256)"

    gpt_configs = {
        'GPT-B': {'n_layer': 12, 'n_head': 12, 'dim': 768, 'medusa_attention_num': 1},
        'GPT-M': {'n_layer': 18, 'n_head': 16, 'dim': 1024, 'medusa_attention_num': 1},
        'GPT-L': {'n_layer': 24, 'n_head': 16, 'dim': 1024, 'medusa_attention_num': 1},
        'GPT-XL': {'n_layer': 36, 'n_head': 20, 'dim': 1280, 'medusa_attention_num': 1},
    }
    cfg = gpt_configs.get(args.gpt_model, gpt_configs['GPT-B'])

    base = NARPagedLLM(
        args.gpt_ckpt,
        block_size=256,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        device=device,
        dtype=getattr(torch, args.precision),
        dim=cfg['dim'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        medusa_attention_num=cfg['medusa_attention_num'],
    )

    sp = NARSamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        cfg_interval=-1,
        block_size=256,
        cls_token_num=args.cls_token_num,
        model_type='c2i',
    )

    batcher = NARContinuousBatcher(base, max_num_requests=args.num_requests, sampling_params=sp)

    import random
    random.seed(args.seed)
    labels = [random.randint(0, args.num_classes - 1) for _ in range(args.num_requests)]
    labels = torch.tensor(labels, device=device, dtype=torch.long)

    # Simulate arrivals: first half now, second half after a few steps.
    split = max(1, args.num_requests // 2)
    t0 = time.time()
    batcher.add_requests(labels[:split])

    inserted = split
    outputs = []
    step_idx = 0
    while len(outputs) < args.num_requests:
        step_idx += 1
        if inserted < args.num_requests and step_idx == 5:
            batcher.add_requests(labels[inserted:])
            inserted = args.num_requests
        out = batcher.step()
        outputs.extend(out)

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[nano-vllm][continuous] finished={len(outputs)} time={dt:.2f}s throughput={len(outputs)/dt:.3f} req/s")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="original",
        choices=["original", "nano-vllm"],
        help="original: 旧脚本；nano-vllm: paged KV + FlexAttention + continuous batching(简化)",
    )
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default="pretrained_models/c2i_B_256.pt")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bfloat16', choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-requests", type=int, default=8, help="Number of concurrent requests")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backend == "nano-vllm":
        run_nanovllm_continuous(args, device)
    else:
        run_benchmark(args, device)


if __name__ == "__main__":
    main()
