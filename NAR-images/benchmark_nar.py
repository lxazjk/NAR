#!/usr/bin/env python3
"""
Benchmark script for NAR image generation.

Compares two backends:
  1. Original (autoregressive.models.gpt + generate)
  2. nano-vllm (NARPagedLLM with paged KV + FlexAttention)

Usage:
  # Quick test (256 samples, skip FID)
  python benchmark_nar.py --num-samples 256 --skip-fid

  # Full FID evaluation (50k samples)
  python benchmark_nar.py --num-samples 50000

  # Run only one backend
  python benchmark_nar.py --backend original --num-samples 256
  python benchmark_nar.py --backend nanovllm --num-samples 256
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "nano-vllm"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Prevent weight re-init when building models
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


# ---------------------------------------------------------------------------
# Model configs (matching checkpoint)
# ---------------------------------------------------------------------------
GPT_CONFIGS = {
    "GPT-B":    {"n_layer": 12, "n_head": 12, "dim": 768},
    "GPT-M":    {"n_layer": 18, "n_head": 16, "dim": 1024},
    "GPT-L":    {"n_layer": 24, "n_head": 16, "dim": 1024},
    "GPT-XL":   {"n_layer": 36, "n_head": 20, "dim": 1280},
    "GPT-XXL":  {"n_layer": 48, "n_head": 24, "dim": 1536},
    "GPT-XXXL": {"n_layer": 48, "n_head": 40, "dim": 2560},
}


def create_npz_from_folder(sample_dir, num):
    """Build a single .npz file from a folder of .png samples."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz"):
        path = os.path.join(sample_dir, f"{i:06d}.png")
        img = Image.open(path)
        samples.append(np.asarray(img).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved NPZ: {npz_path}  shape={samples.shape}")
    return npz_path


# ============================================================================
# Backend: Original
# ============================================================================
def run_original(args, vq_model, device, precision):
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate

    cfg = GPT_CONFIGS[args.gpt_model]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2

    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type="c2i",
    ).to(device=device, dtype=precision)

    ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    gpt_model.load_state_dict(sd, strict=False)
    gpt_model.eval()
    del ckpt
    print(f"[Original] Loaded GPT-L: dim={cfg['dim']}, layers={cfg['n_layer']}")

    sample_dir = os.path.join(args.output_dir, "original")
    os.makedirs(sample_dir, exist_ok=True)

    n = args.batch_size
    total = args.num_samples
    iterations = math.ceil(total / n)

    torch.manual_seed(args.seed)
    torch.cuda.synchronize()
    t_start = time.time()
    generated = 0

    for it in tqdm(range(iterations), desc="[Original] Generating"):
        cur_n = min(n, total - generated)
        c_indices = torch.randint(0, args.num_classes, (cur_n,), device=device)
        qzshape = [cur_n, args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate(
            gpt_model, c_indices, block_size,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
        )

        samples = vq_model.decode_code(index_sample, qzshape)
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()

        for i, sample in enumerate(samples):
            idx = generated + i
            if idx < total:
                Image.fromarray(sample).save(os.path.join(sample_dir, f"{idx:06d}.png"))
        generated += cur_n

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    print(f"[Original] Generated {total} samples in {elapsed:.2f}s  ({total/elapsed:.2f} img/s)")

    return sample_dir, elapsed


# ============================================================================
# Backend: nano-vllm
# ============================================================================
def run_nanovllm(args, vq_model, device, precision):
    from nanovllm.inference.nar_vllm import NARPagedLLM
    from nanovllm.sampling_params import NARSamplingParams

    cfg = GPT_CONFIGS[args.gpt_model]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2

    llm = NARPagedLLM(
        args.gpt_ckpt,
        block_size=block_size,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        device=device,
        dtype=precision,
        dim=cfg["dim"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        medusa_attention_num=args.medusa_attention_num,
    )
    print(f"[nano-vllm] Loaded NARPagedLLM: dim={cfg['dim']}, layers={cfg['n_layer']}")

    sampling_params = NARSamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        block_size=block_size,
        cls_token_num=args.cls_token_num,
        model_type="c2i",
    )

    sample_dir = os.path.join(args.output_dir, "nanovllm")
    os.makedirs(sample_dir, exist_ok=True)

    n = args.batch_size
    total = args.num_samples
    iterations = math.ceil(total / n)

    torch.manual_seed(args.seed)
    torch.cuda.synchronize()
    t_start = time.time()
    generated = 0

    for it in tqdm(range(iterations), desc="[nano-vllm] Generating"):
        cur_n = min(n, total - generated)
        c_indices = torch.randint(0, args.num_classes, (cur_n,), device=device)
        qzshape = [cur_n, args.codebook_embed_dim, latent_size, latent_size]

        try:
            index_sample = llm.generate_image(
                condition=c_indices,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except (RuntimeError, torch.cuda.CudaError) as e:
            print(f"\n[nano-vllm] CUDA error at iter {it} (generated={generated}): {e}")
            print("[nano-vllm] Resetting CUDA and retrying...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Re-try once
            try:
                index_sample = llm.generate_image(
                    condition=c_indices,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception as e2:
                print(f"[nano-vllm] Retry also failed: {e2}. Stopping early at {generated} samples.")
                break

        samples = vq_model.decode_code(index_sample.to(device), qzshape)
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()

        for i, sample in enumerate(samples):
            idx = generated + i
            if idx < total:
                Image.fromarray(sample).save(os.path.join(sample_dir, f"{idx:06d}.png"))
        generated += cur_n

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    print(f"[nano-vllm] Generated {total} samples in {elapsed:.2f}s  ({total/elapsed:.2f} img/s)")

    return sample_dir, elapsed


# ============================================================================
# FID / sFID Evaluation (c2i evaluator)
# ============================================================================
def run_fid_evaluation(sample_npz, ref_npz):
    """Run FID/sFID using the c2i TF-based evaluator."""
    import tensorflow._api.v2.compat.v1 as tf
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluations", "c2i"))
    from evaluator import Evaluator, FIDStatistics

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    evaluator = Evaluator(sess)
    print("Warming up TF InceptionV3...")
    evaluator.warmup()

    # Reference stats
    ref_obj = np.load(ref_npz)
    has_stats = all(k in ref_obj for k in ["mu", "sigma", "mu_s", "sigma_s"])
    if has_stats:
        print("Using precomputed reference stats.")
        ref_stats = FIDStatistics(ref_obj["mu"], ref_obj["sigma"])
        ref_stats_spatial = FIDStatistics(ref_obj["mu_s"], ref_obj["sigma_s"])
    else:
        print("Computing reference activations...")
        ref_acts = evaluator.read_activations(ref_npz)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_npz, ref_acts)

    # Sample stats
    print("Computing sample activations...")
    sample_acts = evaluator.read_activations(sample_npz)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_npz, sample_acts)

    IS = evaluator.compute_inception_score(sample_acts[0])
    FID = sample_stats.frechet_distance(ref_stats)
    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

    sess.close()
    return {"IS": IS, "FID": FID, "sFID": sFID}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="NAR benchmark: Original vs nano-vllm")
    # Backend
    parser.add_argument("--backend", type=str, default="both",
                        choices=["original", "nanovllm", "both"],
                        help="Which backend(s) to benchmark")
    # Model
    parser.add_argument("--gpt-model", type=str, default="GPT-L",
                        choices=list(GPT_CONFIGS.keys()))
    parser.add_argument("--gpt-ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "pretrained_models", "c2i_L_256.pt"))
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str,
                        default=os.path.join(PROJECT_ROOT, "pretrained_models", "vq_ds16_c2i.pt"))
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--medusa-attention-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["none", "fp16", "bf16"])
    # Generation
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-size-eval", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.75)
    parser.add_argument("--cfg-interval", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    # Eval
    parser.add_argument("--ref-npz", type=str,
                        default=os.path.join(PROJECT_ROOT, "pretrained_models",
                                             "VIRTUAL_imagenet256_labeled.npz"))
    parser.add_argument("--skip-fid", action="store_true",
                        help="Skip FID evaluation (useful for speed-only benchmarks)")
    # Output
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "benchmark_results"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load VQ Model (shared) ----
    from tokenizer.tokenizer_image.vq_model import VQ_models
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(ckpt["model"])
    del ckpt
    print("VQ model loaded.")

    results = {}

    # ---- Run backends ----
    backends = []
    if args.backend in ("original", "both"):
        backends.append(("original", run_original))
    if args.backend in ("nanovllm", "both"):
        backends.append(("nanovllm", run_nanovllm))

    for name, run_fn in backends:
        print(f"\n{'='*60}")
        print(f"  Backend: {name}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            sample_dir, elapsed = run_fn(args, vq_model, device, precision)

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        result = {
            "time_s": elapsed,
            "throughput_img_s": args.num_samples / elapsed,
            "peak_gpu_gb": peak_mem,
        }

        # Build NPZ and evaluate FID
        if not args.skip_fid and args.num_samples >= 256:
            npz_path = create_npz_from_folder(sample_dir, args.num_samples)
            print(f"Running FID/sFID evaluation...")
            fid_results = run_fid_evaluation(npz_path, args.ref_npz)
            result.update(fid_results)
            print(f"  IS:  {fid_results['IS']:.4f}")
            print(f"  FID: {fid_results['FID']:.4f}")
            print(f"  sFID: {fid_results['sFID']:.4f}")
        else:
            print("Skipping FID evaluation.")

        results[name] = result
        print(f"\n[{name}] Time: {elapsed:.2f}s | Throughput: {result['throughput_img_s']:.2f} img/s | Peak GPU: {peak_mem:.2f} GB")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.gpt_model} | Samples: {args.num_samples} | Batch: {args.batch_size}")
    print(f"CFG: {args.cfg_scale} | top_k: {args.top_k} | temp: {args.temperature}")
    print()

    header = f"{'Backend':<12} {'Time(s)':>10} {'Img/s':>10} {'PeakGPU':>10}"
    if not args.skip_fid:
        header += f" {'FID':>10} {'sFID':>10} {'IS':>10}"
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        line = f"{name:<12} {res['time_s']:>10.2f} {res['throughput_img_s']:>10.2f} {res['peak_gpu_gb']:>10.2f}"
        if "FID" in res:
            line += f" {res['FID']:>10.4f} {res['sFID']:>10.4f} {res['IS']:>10.4f}"
        print(line)

    if "original" in results and "nanovllm" in results:
        speedup = results["original"]["time_s"] / max(results["nanovllm"]["time_s"], 1e-9)
        print(f"\nSpeedup (nano-vllm vs original): {speedup:.2f}x")

    # Save results
    results_path = os.path.join(args.output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Model: {args.gpt_model}\n")
        f.write(f"Samples: {args.num_samples}\n")
        f.write(f"Batch: {args.batch_size}\n")
        f.write(f"CFG: {args.cfg_scale}\n")
        for name, res in results.items():
            f.write(f"\n[{name}]\n")
            for k, v in res.items():
                f.write(f"  {k}: {v}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
