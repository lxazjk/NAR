  torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    autoregressive/train/train_c2i_nar.py \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --cloud-save-path ./cloud_ckpt \
    --dataset imagenet_code --image-size 256 --downsample-size 16 \
    --gpt-model GPT-L --gpt-type c2i \
    --init-ckpt ./pretrained_models/c2i_L_256.pt \
    --teacher-ckpt ./pretrained_models/c2i_L_256.pt \
    --epochs 30 --steps-per-epoch 2500 \
    --global-batch-size 512 --num-workers 24 --log-every 100 --log-loss-every 500 \
    --kd-weight 0.1 --kd-temperature 2.0 --ce-weight 1.0 \
    --kd-start-step 200 --kd-every 1 --kd-prob 0.5 \
    --mask-schedule static_proximity \
    --fid-ref ./pretrained_models/VIRTUAL_imagenet256_labeled.npz \
    --fid-num-samples 50000 --fid-batch-size 64 \
    --fid-sample-dir samples_c2i \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --wandb-project c2i_nar \
    --wandb-name init_teacher_distill_30x2500_gptl \
    --no-compile