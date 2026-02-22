import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, '/opt/tiger/tmp/NAR/NAR-images')

from tokenizer.tokenizer_image.vq_model import VQ_models
from LlamaGen.autoregressive.models.gpt import GPT_models as AR_GPT_models
from LlamaGen.autoregressive.models.generate import generate as ar_generate


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'
    
    torch.manual_seed(args.seed + rank)
    
    if rank == 0:
        print("Loading VQ model...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(vq_ckpt["model"])
    vq_model.eval()
    
    if rank == 0:
        print("Loading AR (LlamaGen) model...")
    latent_size = args.image_size // args.downsample_size
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    
    model = AR_GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type='c2i',
    ).to(device=device, dtype=precision)
    
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    samples_per_gpu = args.num_samples // world_size
    if rank < args.num_samples % world_size:
        samples_per_gpu += 1
    
    if rank == 0:
        print(f"Total samples: {args.num_samples}, GPUs: {world_size}, Samples per GPU: {samples_per_gpu}")
        print(f"cfg_scale={args.cfg_scale}, top_k={args.top_k}")
    
    all_samples = []
    num_batches = (samples_per_gpu + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches), desc=f"GPU {rank}", disable=rank != 0):
        cur_batch = min(args.batch_size, samples_per_gpu - i * args.batch_size)
        
        class_labels = torch.randint(0, args.num_classes, (cur_batch,), device=device)
        
        with torch.no_grad():
            index_sample = ar_generate(
                model, class_labels, latent_size ** 2,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True,
            )
            
            qzshape = [cur_batch, args.codebook_embed_dim, latent_size, latent_size]
            samples = vq_model.decode_code(index_sample, qzshape)
            samples = (samples + 1) / 2 * 255
            samples = samples.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            all_samples.append(samples)
    
    if len(all_samples) > 0:
        local_samples = np.concatenate(all_samples, axis=0)
    else:
        local_samples = np.zeros((0, args.image_size, args.image_size, 3), dtype=np.uint8)
    
    gathered_samples = [None] * world_size
    dist.all_gather_object(gathered_samples, local_samples)
    
    if rank == 0:
        all_samples = np.concatenate(gathered_samples, axis=0)[:args.num_samples]
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "ar_samples_50k.npz")
        np.savez(output_path, arr_0=all_samples)
        print(f"\nGenerated {len(all_samples)} samples")
        print(f"Samples saved to {output_path}")
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default="pretrained_models/c2i_L_256.pt")
    parser.add_argument("--vq-ckpt", type=str, default="pretrained_models/vq_ds16_c2i.pt")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./eval_output_full")
    args = parser.parse_args()
    main(args)
