    cd /opt/tiger/tmp/NAR/NAR-images/LlamaGen && \
    torchrun --standalone --nproc_per_node=8 autoregressive/sample/sample_c2i_ddp.py \
    --gpt-model GPT-L --gpt-type c2i \
    --gpt-ckpt /opt/tiger/tmp/NAR/NAR-images/pretrained_models/c2i_L_256.pt \
    --vq-model VQ-16 \
    --vq-ckpt /opt/tiger/tmp/NAR/NAR-images/pretrained_models/vq_ds16_c2i.pt \
    --image-size 256 --image-size-eval 256 --downsample-size 16 \
    --num-fid-samples 50000 --per-proc-batch-size 256 \
    --cfg-scale 1.5 --top-k 0 --top-p 1.0 --temperature 1.0 \
    --no-compile \
    --sample-dir /opt/tiger/tmp/NAR/NAR-images/samples_ar_eval_small   


    cd /opt/tiger/tmp/NAR/NAR-images && \                                                                                                                                                                                                                                     
    python3 evaluations/c2i/evaluator.py \
    /opt/tiger/tmp/NAR/NAR-images/pretrained_models/VIRTUAL_imagenet256_labeled.npz \
    /opt/tiger/tmp/NAR/NAR-images/samples_ar_eval_small/GPT-L-c2i_L_256-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.5-seed-0.npz                                                                                                                          
      