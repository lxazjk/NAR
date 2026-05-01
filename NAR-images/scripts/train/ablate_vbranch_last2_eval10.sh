#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  autoregressive/train/train_c2i_nar.py \
  --code-path ./imagenet_code_c2i_flip_ten_crop \
  --cloud-save-path ./cloud_ckpt_vbranch_last2_hvgate_reg0p1 \
  --dataset imagenet_code --image-size 256 --downsample-size 16 \
  --gpt-model GPT-L --gpt-type c2i \
  --init-ckpt ./pretrained_models/c2i_L_256.pt \
  --epochs 30 --steps-per-epoch 2500 \
  --global-batch-size 512 --num-workers 24 --log-every 100 --log-loss-every 500 \
  --kd-weight 0 \
  --mask-schedule static_proximity \
  --hv-gate --gate-collapse-weight 0.1 \
  --medusa-attention-num 1 \
  --vertical-start-layer 22 \
  --fid-every 10 \
  --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
  --fid-num-samples 50000 --fid-batch-size 64 \
  --fid-cfg-scale 2.0 \
  --fid-sample-dir ./samples_vbranch_last2_hvgate_reg0p1_eval10 \
  --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
  --wandb-project c2i_nar \
  --wandb-name vbranch_last2_hvgate_reg0p1_eval10_gptl \
  --no-compile
