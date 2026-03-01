import os
import sys
import time
import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer.tokenizer_image.vq_model import VQ_models
from nanovllm.sampling_params import NARSamplingParams
from nanovllm.inference.nar_vllm import NARPagedLLM


GPT_CONFIGS = {
    "GPT-B": {"n_layer": 12, "n_head": 12, "dim": 768},
    "GPT-M": {"n_layer": 18, "n_head": 16, "dim": 1024},
    "GPT-L": {"n_layer": 24, "n_head": 16, "dim": 1024},
    "GPT-XL": {"n_layer": 36, "n_head": 20, "dim": 1280},
    "GPT-XXL": {"n_layer": 48, "n_head": 24, "dim": 1536},
    "GPT-XXXL": {"n_layer": 48, "n_head": 40, "dim": 2560},
}


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load image tokenizer
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print("image tokenizer is loaded")

    if args.gpt_type != "c2i":
        raise ValueError("NARPagedLLM currently supports only c2i checkpoints.")

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2
    gpt_cfg = GPT_CONFIGS.get(args.gpt_model, GPT_CONFIGS["GPT-B"])
    dim = args.dim if args.dim is not None else gpt_cfg["dim"]
    n_layer = args.n_layer if args.n_layer is not None else gpt_cfg["n_layer"]
    n_head = args.n_head if args.n_head is not None else gpt_cfg["n_head"]

    # create and load NAR model
    llm = NARPagedLLM(
        args.gpt_ckpt,
        block_size=block_size,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        device=device,
        dtype=precision,
        dim=dim,
        n_layer=n_layer,
        n_head=n_head,
        medusa_attention_num=args.medusa_attention_num,
    )
    print("gpt model is loaded")

    sampling_params = NARSamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        block_size=block_size,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    )

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    t1 = time.time()
    index_sample = llm.generate_image(
        condition=c_indices,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")

    t2 = time.time()
    samples = vq_model.decode_code(index_sample.to(device), qzshape)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    save_image(samples, f"sample_{args.gpt_type}.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=str, default=None, required=True)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_CONFIGS.keys()), default="GPT-B")
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    # GPT model hyper-parameters (optional override)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--medusa-attention-num", type=int, default=1)

    args = parser.parse_args()
    main(args)
