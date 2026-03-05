#!/usr/bin/env python3
"""
Run FID/sFID evaluation on a folder of generated .png images.

Usage:
  python eval_fid.py <sample_dir> [--ref-npz <path>] [--num-samples <N>]

Example:
  python eval_fid.py benchmark_results/original_5k/original --num-samples 5000
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "evaluations", "c2i"))


def create_npz_from_folder(sample_dir, num):
    """Build a single .npz from a folder of .png samples."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz"):
        path = os.path.join(sample_dir, f"{i:06d}.png")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, stopping at {i}")
            break
        samples.append(np.asarray(Image.open(path)).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved NPZ: {npz_path}  shape={samples.shape}")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID/sFID on generated images")
    parser.add_argument("sample_dir", type=str, help="Folder of .png images (000000.png, 000001.png, ...)")
    parser.add_argument("--ref-npz", type=str,
                        default=os.path.join(PROJECT_ROOT, "pretrained_models", "VIRTUAL_imagenet256_labeled.npz"))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--skip-pr", action="store_true", default=True,
                        help="Skip Precision/Recall (expensive O(N^2) computation)")
    args = parser.parse_args()

    # Build NPZ from sample folder
    npz_path = f"{args.sample_dir}.npz"
    if not os.path.exists(npz_path):
        npz_path = create_npz_from_folder(args.sample_dir, args.num_samples)
    else:
        print(f"Using existing NPZ: {npz_path}")

    # Import TF evaluator
    import tensorflow._api.v2.compat.v1 as tf
    from evaluator import Evaluator, FIDStatistics

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    evaluator = Evaluator(sess)

    print("Warming up TF InceptionV3...")
    evaluator.warmup()

    # Reference
    ref_obj = np.load(args.ref_npz)
    has_stats = all(k in ref_obj for k in ["mu", "sigma", "mu_s", "sigma_s"])
    if has_stats:
        print("Using precomputed reference stats.")
        ref_stats = FIDStatistics(ref_obj["mu"], ref_obj["sigma"])
        ref_stats_spatial = FIDStatistics(ref_obj["mu_s"], ref_obj["sigma_s"])
        ref_acts = None
    else:
        print("Computing reference activations (this may take a while)...")
        ref_acts = evaluator.read_activations(args.ref_npz)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_npz, ref_acts)

    # Sample
    print("Computing sample activations...")
    t0 = time.time()
    sample_acts = evaluator.read_activations(npz_path)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(npz_path, sample_acts)
    t1 = time.time()
    print(f"  Activation extraction took {t1 - t0:.1f}s")

    # Metrics
    IS = evaluator.compute_inception_score(sample_acts[0])
    FID = sample_stats.frechet_distance(ref_stats)
    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

    print(f"\n{'='*40}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"  Samples:         {args.num_samples}")
    print(f"  Inception Score: {IS:.4f}")
    print(f"  FID:             {FID:.4f}")
    print(f"  sFID:            {sFID:.4f}")

    if not args.skip_pr and ref_acts is not None:
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        print(f"  Precision:       {prec:.4f}")
        print(f"  Recall:          {recall:.4f}")

    # Save results
    txt_path = npz_path.replace(".npz", "_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Samples: {args.num_samples}\n")
        f.write(f"Inception Score: {IS}\n")
        f.write(f"FID: {FID}\n")
        f.write(f"sFID: {sFID}\n")
    print(f"\nResults saved to {txt_path}")

    sess.close()


if __name__ == "__main__":
    main()
