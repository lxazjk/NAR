import os
import sys
import time
import torch
import argparse
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Allow importing local nano-vllm implementation without pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


def top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0, 
                          filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
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
    idx = torch.cat(idx, dim=-1)
    return idx


def benchmark_original(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    times = []
    for i in range(args.num_runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
        )
        torch.cuda.synchronize()
        sampling_time = time.time() - t1
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        times.append(sampling_time)
        print(f"Original Run {i+1}: gpt sampling takes {sampling_time:.2f} seconds, peak memory: {peak_memory:.2f} GB")
    
    avg_time = sum(times) / len(times)
    print(f"\nOriginal Average sampling time: {avg_time:.2f} seconds")
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    save_image(samples, "sample_original.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_original.png")
    
    return avg_time, index_sample


def benchmark_vllm(args):
    from nanovllm import NARLLM, NARSamplingParams
    from nanovllm.config import Config
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2
    
    gpt_configs = {
        'GPT-B': {'n_layer': 12, 'n_head': 12, 'dim': 768},
        'GPT-M': {'n_layer': 18, 'n_head': 16, 'dim': 1024},
        'GPT-L': {'n_layer': 24, 'n_head': 16, 'dim': 1024},
        'GPT-XL': {'n_layer': 36, 'n_head': 20, 'dim': 1280},
    }
    
    model_config = gpt_configs.get(args.gpt_model, {'n_layer': 12, 'n_head': 12, 'dim': 768})
    
    llm = NARLLM(
        args.gpt_ckpt,
        model_type="nar",
        use_proximity_mask=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        block_size=block_size,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        gpt_ckpt=args.gpt_ckpt,
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        dim=model_config['dim'],
    )
    
    sampling_params = NARSamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        block_size=block_size,
        model_type='c2i',
    )
    
    class_labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279])
    
    times = []
    for i in range(args.num_runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        generated_tokens = llm.generate_image(
            condition=class_labels,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        torch.cuda.synchronize()
        sampling_time = time.time() - t1
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        times.append(sampling_time)
        print(f"vLLM Run {i+1}: gpt sampling takes {sampling_time:.2f} seconds, peak memory: {peak_memory:.2f} GB")
    
    avg_time = sum(times) / len(times)
    print(f"\nvLLM Average sampling time: {avg_time:.2f} seconds")
    
    return avg_time, generated_tokens


def main(args):
    print("=" * 50)
    print("Benchmarking Original Implementation")
    print("=" * 50)
    original_time, original_tokens = benchmark_original(args)
    
    print("\n" + "=" * 50)
    print("Benchmarking nano-vLLM Implementation")
    print("=" * 50)
    vllm_time, vllm_tokens = benchmark_vllm(args)

    # Decode and save nano-vLLM result for visual comparison
    latent_size = args.image_size // args.downsample_size
    qzshape = [len(vllm_tokens), args.codebook_embed_dim, latent_size, latent_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    samples = vq_model.decode_code(vllm_tokens.to(device), qzshape)
    save_image(samples, "sample_nanovllm.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("image is saved to sample_nanovllm.png")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Original Average Time: {original_time:.2f} seconds")
    print(f"nano-vLLM Average Time: {vllm_time:.2f} seconds")
    speedup = original_time / vllm_time if vllm_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    # Token-level match is only meaningful under identical sampling; keep as a rough sanity check.
    try:
        token_match = (original_tokens.to(vllm_tokens.device) == vllm_tokens).float().mean().item() * 100
        print(f"Token Match Rate: {token_match:.2f}%")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-runs", type=int, default=3)
    args = parser.parse_args()
    main(args)
