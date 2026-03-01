cd /opt/tiger/tmp/NAR/NAR-images                                                                                                                                                                                                                                     
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  autoregressive/train/train_c2i_nar.py \
  --code-path /opt/tiger/tmp/NAR/NAR-images/imagenet_code_c2i_flip_ten_crop \
  --cloud-save-path ./ckpt_eval_step5000 \
  --dataset imagenet_code --image-size 256 --downsample-size 16 \
  --gpt-model GPT-L --gpt-type c2i \
  --init-ckpt /opt/tiger/tmp/NAR/NAR-images/pretrained_models/c2i_L_256.pt \
  --teacher-ckpt /opt/tiger/tmp/NAR/NAR-images/pretrained_models/c2i_L_256.pt \
  --kd-weight 0.1 --kd-temperature 2.0 --ce-weight 1.0 \
  --kd-start-step 200 --kd-every 1 --kd-prob 0.5 \
  --mask-schedule static_proximity \
  --global-batch-size 512 --num-workers 24 --log-every 100 --ckpt-every 2500 \
  --max-steps 2500 \
  --fid-every 2500 \
  --fid-ref /opt/tiger/tmp/NAR/NAR-images/pretrained_models/VIRTUAL_imagenet256_labeled.npz \
  --fid-num-samples 50000 \
  --fid-batch-size 256 \
  --fid-sample-dir samples_kd_tuned_kdw0p1_t2_p0p5 --fid-skip-evaluator \
  --vq-ckpt /opt/tiger/tmp/NAR/NAR-images/pretrained_models/vq_ds16_c2i.pt \
  --no-compile
