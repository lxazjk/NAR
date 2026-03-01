# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import argparse
import inspect
import math
import os
import random
import time
import subprocess
import contextlib
from datetime import timedelta
from copy import deepcopy
from glob import glob
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Enable TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models as NAR_GPT_models
from autoregressive.models.generate import generate
from tokenizer.tokenizer_image.vq_model import VQ_models
from LlamaGen.autoregressive.models.gpt import GPT_models as AR_GPT_models
from autoregressive.utils.mask import build_masks, pick_mask


#################################################################################
#                           Checkpoint / Init Utils                            #
#################################################################################
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "module" in ckpt:
            return ckpt["module"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    return ckpt


def normalize_state_dict(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_checkpoint(path, map_location="cpu"):
    if path is None:
        return None
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def init_student_from_teacher(student, teacher_state, logger=None):
    student_state = student.state_dict()
    new_state = {}
    loaded = 0
    loaded_numel = 0
    param_keys = set(name for name, _ in student.named_parameters())

    for key, val in teacher_state.items():
        if key in student_state and student_state[key].shape == val.shape:
            new_state[key] = val
            loaded += 1
            if key in param_keys:
                loaded_numel += int(val.numel())

    # Reuse output/norm for medusa heads if present
    if "medusa_output.weight" in student_state and "output.weight" in teacher_state:
        if student_state["medusa_output.weight"].shape == teacher_state["output.weight"].shape:
            new_state["medusa_output.weight"] = teacher_state["output.weight"].clone()
    if "medusa_norm.weight" in student_state and "norm.weight" in teacher_state:
        if student_state["medusa_norm.weight"].shape == teacher_state["norm.weight"].shape:
            new_state["medusa_norm.weight"] = teacher_state["norm.weight"].clone()

    # Reuse last teacher layer for extra medusa layers
    base_layer = student.n_layer - 1
    extra_layers = len(student.layers) - student.n_layer
    if extra_layers > 0:
        base_prefix = f"layers.{base_layer}."
        for extra_id in range(student.n_layer, len(student.layers)):
            target_prefix = f"layers.{extra_id}."
            for key, val in teacher_state.items():
                if key.startswith(base_prefix):
                    target_key = target_prefix + key[len(base_prefix):]
                    if target_key in student_state and student_state[target_key].shape == val.shape:
                        new_state[target_key] = val.clone()

    missing, unexpected = student.load_state_dict(new_state, strict=False)
    if logger is not None:
        total_numel = sum(int(p.numel()) for p in student.parameters())
        ratio = (loaded_numel / total_numel) if total_numel > 0 else 0.0
        logger.info(
            f"Loaded {loaded} teacher keys into student. Missing keys: {len(missing)}, unexpected: {len(unexpected)}. "
            f"Param init coverage: {ratio*100:.2f}%"
        )
        if ratio < 0.10:
            logger.warning(
                "Init coverage < 10%: this often means gpt-model/ckpt mismatch (e.g., GPT-B init with GPT-L ckpt), "
                "or different key naming. Training may behave like from-scratch."
            )


def _parse_step_from_ckpt(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.isdigit():
        return int(name)
    return -1


def resolve_resume_ckpt(args, logger):
    if args.gpt_ckpt:
        return args.gpt_ckpt
    if not args.auto_resume:
        return None

    search_dirs = []
    if args.resume_local_dir:
        search_dirs.append(os.path.join(args.resume_local_dir, "checkpoints"))
    if args.resume_cloud_dir:
        search_dirs.append(os.path.join(args.resume_cloud_dir, "checkpoints"))

    candidates = []
    for ckpt_dir in search_dirs:
        if ckpt_dir and os.path.isdir(ckpt_dir):
            candidates.extend(glob(os.path.join(ckpt_dir, "*.pt")))

    candidates = [c for c in candidates if _parse_step_from_ckpt(c) >= 0]
    if not candidates:
        if logger is not None:
            logger.warning("auto-resume enabled but no valid checkpoint found.")
        return None

    best = max(candidates, key=_parse_step_from_ckpt)
    if logger is not None:
        logger.info(f"Auto-resume selected checkpoint: {best}")
    return best


#################################################################################
#                               FID Evaluation                                 #
#################################################################################
def create_npz_from_samples(sample_dir, num):
    from PIL import Image
    import numpy as np

    samples = []
    for i in range(num):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    return npz_path


@torch.no_grad()
def run_fid_eval(args, model, vq_model, device, step, logger):
    if args.fid_ref is None:
        logger.info("FID eval skipped: --fid-ref not provided.")
        return
    os.makedirs(args.fid_sample_dir, exist_ok=True)

    model.eval()
    vq_model.eval()

    latent_size = args.image_size // args.downsample_size
    total = 0
    sample_dir = os.path.join(args.fid_sample_dir, f"fid_step_{step:07d}")
    os.makedirs(sample_dir, exist_ok=True)

    num_iters = math.ceil(args.fid_num_samples / args.fid_batch_size)
    for _ in range(num_iters):
        n = min(args.fid_batch_size, args.fid_num_samples - total)
        if n <= 0:
            break
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [n, args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate(
            model, c_indices, latent_size ** 2,
            cfg_scale=args.fid_cfg_scale, cfg_interval=args.fid_cfg_interval,
            temperature=args.fid_temperature, top_k=args.fid_top_k,
            top_p=args.fid_top_p, sample_logits=True,
        )
        samples = vq_model.decode_code(index_sample, qzshape)
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1)
        samples = samples.to("cpu", dtype=torch.uint8).numpy()

        from PIL import Image
        for i, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{sample_dir}/{total + i:06d}.png")
        total += n

    npz_path = create_npz_from_samples(sample_dir, args.fid_num_samples)
    logger.info(f"Saved FID samples to {npz_path}")

    if getattr(args, "fid_skip_evaluator", False):
        logger.info("Skipping FID evaluator (--fid-skip-evaluator enabled).")
        model.train()
        return

    evaluator_path = os.path.join(ROOT, "evaluations", "c2i", "evaluator.py")
    cmd = ["python3", evaluator_path, args.fid_ref, npz_path]
    logger.info(f"Running FID evaluator: {' '.join(cmd)}")
    try:
        # Run evaluator on CPU to avoid TF grabbing GPU memory and crashing training.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        subprocess.run(cmd, check=False, cwd=ROOT, env=env)
    except Exception as e:
        logger.info(f"FID evaluator failed: {e}")

    model.train()


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.gpt_model.replace("/", "-")
        experiment_index = None
        if args.resume_local_dir:
            experiment_dir = args.resume_local_dir
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment directory at {experiment_dir}")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        if args.resume_cloud_dir:
            cloud_results_dir = args.resume_cloud_dir
            cloud_checkpoint_dir = f"{cloud_results_dir}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Resuming cloud experiment directory at {cloud_checkpoint_dir}")
        else:
            if experiment_index is None:
                experiment_index = len(glob(f"{args.results_dir}/*"))
            time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
            cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)

    # Use a dedicated Gloo process group for long-running synchronization points
    # (e.g., FID evaluation / checkpointing on rank-0). This avoids NCCL watchdog
    # timeouts on `dist.barrier()` while rank-0 is busy doing heavy CPU/GPU work.
    control_group = None
    try:
        control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))
    except Exception as e:
        if rank == 0:
            logger.info(f"Failed to create Gloo control group, falling back to NCCL barriers: {e}")

    def control_barrier():
        if control_group is not None:
            dist.barrier(group=control_group)
        else:
            dist.barrier()

    def control_all_reduce_min(t: torch.Tensor):
        if control_group is not None:
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=control_group)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup student model (NAR)
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    student = NAR_GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        medusa_attention_num=args.medusa_attention_num,
        hv_mix=getattr(args, "hv_mix", False),
        hv_mix_init=getattr(args, "hv_mix_init", 0.5),
    ).to(device)
    logger.info(f"Student GPT Parameters: {sum(p.numel() for p in student.parameters()):,}")

    if args.ema:
        ema = deepcopy(student).to(device)
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Load init / resume
    train_steps = 0
    start_epoch = 0
    args.gpt_ckpt = resolve_resume_ckpt(args, logger if rank == 0 else None)
    resume_ckpt = load_checkpoint(args.gpt_ckpt, map_location="cpu") if args.gpt_ckpt else None

    # Apply LoRA before loading resume ckpt so LoRA params are present
    if resume_ckpt is not None and args.use_lora:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        apply_lora(student, targets, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_train_base, logger)

    if resume_ckpt is not None:
        ckpt_state = normalize_state_dict(extract_state_dict(resume_ckpt))
        student.load_state_dict(ckpt_state, strict=False)
        if args.ema and isinstance(resume_ckpt, dict) and "ema" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema"])
        if isinstance(resume_ckpt, dict) and "steps" in resume_ckpt:
            train_steps = resume_ckpt["steps"]
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    else:
        if args.init_ckpt is None:
            args.init_ckpt = args.teacher_ckpt
        if args.init_ckpt:
            teacher_ckpt = load_checkpoint(args.init_ckpt, map_location="cpu")
            teacher_state = normalize_state_dict(extract_state_dict(teacher_ckpt))
            init_student_from_teacher(student, teacher_state, logger)
            del teacher_ckpt
        if args.ema:
            update_ema(ema, student, decay=0)

        # Apply LoRA after teacher init so base weights are reused
        if args.use_lora:
            targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
            apply_lora(student, targets, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_train_base, logger)

    # Setup optimizer
    optimizer = create_optimizer(student, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    milestones = [int(args.epochs * 0.5), int(args.epochs * 2 / 3), int(args.epochs * 5 / 6)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    if resume_ckpt is not None and isinstance(resume_ckpt, dict):
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])

    # Setup data
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    subset_info = ""
    if getattr(args, 'train_max_samples', -1) and args.train_max_samples > 0:
        subset_info = f", subset={args.train_max_samples} (shuffle={getattr(args, 'train_subset_shuffle', False)})"
    logger.info(
        f"Dataset contains {len(dataset):,} images ({args.code_path}) {flip_info} flip augmentation and {aug_info} crop augmentation{subset_info}"
    )
    if resume_ckpt is not None and train_steps > 0:
        steps_per_epoch = int(len(dataset) / args.global_batch_size)
        start_epoch = int(train_steps / max(steps_per_epoch, 1))

    # Setup teacher model for distillation
    teacher = None
    if args.kd_weight > 0:
        assert args.teacher_ckpt is not None, "Teacher ckpt must be provided when kd_weight > 0."
        teacher = AR_GPT_models[args.teacher_gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=0.0,
            token_dropout_p=0.0,
        ).to(device)
        teacher_ckpt = load_checkpoint(args.teacher_ckpt, map_location="cpu")
        teacher_state = normalize_state_dict(extract_state_dict(teacher_ckpt))
        teacher.load_state_dict(teacher_state, strict=False)
        teacher.eval()
        requires_grad(teacher, False)
        del teacher_ckpt
        logger.info("Teacher model loaded for logits distillation.")

    # Compile model if requested
    if not args.no_compile:
        logger.info("compiling the student model... (may take several minutes)")
        student = torch.compile(student)

    # Setup masks
    local_bs = int(args.global_batch_size // dist.get_world_size())
    mask_causal, mask_proximity, mask_union, delta_indices, removal_indices = build_masks(
        student._orig_mod if not args.no_compile else student,
        local_bs,
        device,
        seed=(args.global_seed + rank),
    )
    rng = random.Random(args.global_seed + rank)

    # Wrap with DDP
    student = DDP(student.to(device), device_ids=[args.gpu])
    student.train()
    if args.ema:
        ema.eval()

    # Setup FID eval
    vq_model = None
    if rank == 0 and (args.fid_every > 0 or args.eval_on_early_stop):
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
        ).to(device)
        vq_ckpt = load_checkpoint(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(vq_ckpt["model"] if isinstance(vq_ckpt, dict) and "model" in vq_ckpt else vq_ckpt)
        vq_model.eval()
        del vq_ckpt
        logger.info("VQ model loaded for FID evaluation.")

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # Training loop
    log_steps = 0
    running_loss = 0.0
    running_ce = 0.0
    running_kd = 0.0
    start_time = time.time()
    start_time_all = start_time
    accum_steps = max(1, args.gradient_accumulation_steps)
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    # Early stop state (rank0 decides, then we sync a stop flag)
    early_stopper = None
    early_stop_triggered = False
    early_stop_reason = ""
    early_stop_step = -1
    if rank == 0 and args.early_stop:
        early_stopper = EarlyStopper(
            mode=args.early_stop_mode,
            patience_checks=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
            ema_decay=args.early_stop_ema_decay,
            threshold=(None if args.early_stop_threshold < 0 else args.early_stop_threshold),
            logger=logger,
        )

    logger.info(f"Training for {args.epochs} epochs...")
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kd = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            mask, mask_prob, prox_prob = pick_mask(
                mask_causal,
                mask_proximity,
                mask_union,
                args.mask_schedule,
                train_steps,
                args.mask_anneal_steps,
                rng,
                delta_indices=delta_indices,
                removal_indices=removal_indices,
                switch_step=getattr(args, "mask_switch_step", 2000),
                prox_steps=getattr(args, "mask_prox_steps", 500),
            )

            # Optional: schedule-controlled HV mixing.
            if getattr(args, "hv_mix", False):
                core = student.module
                if (not args.no_compile) and hasattr(core, "_orig_mod"):
                    core = core._orig_mod
                # Target schedule: right_w goes from hv_mix_init -> hv_mix_target over hv_mix_anneal_steps.
                # User intent example: make vertical head dominant early => right_w small early.
                a0 = float(getattr(args, "hv_mix_init", 0.5))
                aT = float(getattr(args, "hv_mix_target", 0.5))
                steps = int(getattr(args, "hv_mix_anneal_steps", 0))
                if steps > 0:
                    p = min(1.0, train_steps / steps)
                    target = a0 + (aT - a0) * p
                else:
                    target = aT
                # Blend schedule: start fully following target, then hand over to learnable weight.
                blend_steps = int(getattr(args, "hv_mix_blend_steps", 0))
                if blend_steps > 0:
                    blend = 1.0 - min(1.0, train_steps / blend_steps)
                else:
                    blend = 0.0
                if hasattr(core, "hv_mix_target"):
                    core.hv_mix_target = target
                if hasattr(core, "hv_mix_blend"):
                    core.hv_mix_blend = blend

            sync = ((micro_step + 1) % accum_steps == 0)
            context = student.no_sync() if not sync else contextlib.nullcontext()
            with context:
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    student_logits, ce_loss = student(cond_idx=c_indices, idx=z_indices, targets=z_indices, mask=mask)
                    kd_loss = torch.tensor(0.0, device=device)
                    do_kd = False
                    if teacher is not None and args.kd_weight > 0:
                        opt_step = train_steps  # optimizer-step index (same across ranks)
                        if opt_step >= args.kd_start_step and (args.kd_every <= 1 or (opt_step % args.kd_every == 0)):
                            if args.kd_prob >= 1.0:
                                do_kd = True
                            elif args.kd_prob > 0.0:
                                # Deterministic across ranks to avoid stragglers in DDP.
                                do_kd = random.Random(args.global_seed + opt_step).random() < args.kd_prob

                    if do_kd:
                        with torch.no_grad():
                            seq_len = z_indices.size(1) - 1 + teacher.cls_token_num
                            input_pos = torch.arange(seq_len, device=device)
                            teacher_logits, _ = teacher(
                                cond_idx=c_indices,
                                idx=z_indices[:, :-1],
                                targets=None,
                                input_pos=input_pos,
                            )
                        t = args.kd_temperature
                        kd_loss = F.kl_div(
                            F.log_softmax(student_logits.float() / t, dim=-1),
                            F.softmax(teacher_logits.float() / t, dim=-1),
                            reduction="batchmean",
                        ) * (t * t)
                    loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss
                    loss = loss / accum_steps
                scaler.scale(loss).backward()

            accum_loss += (args.ce_weight * ce_loss.item() + args.kd_weight * kd_loss.item())
            accum_ce += ce_loss.item()
            accum_kd += kd_loss.item()
            micro_step += 1

            if sync:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if args.ema:
                    update_ema(ema, student.module._orig_mod if not args.no_compile else student.module)

                # Per-optimizer-step metrics before we reset accumulation buffers.
                step_loss_val = accum_loss / accum_steps
                step_ce_val = accum_ce / accum_steps
                step_kd_val = (accum_kd / accum_steps) if teacher is not None else 0.0

                # Logging
                running_loss += step_loss_val
                running_ce += step_ce_val
                running_kd += step_kd_val
                accum_loss = 0.0
                accum_ce = 0.0
                accum_kd = 0.0
                log_steps += 1
                train_steps += 1

                if args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps:
                    stop_training = True

                # Convergence / early-stop check (do it on a fixed cadence to keep all ranks in sync)
                if args.early_stop and (train_steps >= args.early_stop_warmup_steps) and (train_steps % args.early_stop_check_every == 0):
                    # Compute averaged metrics across ranks (cheap because it's infrequent)
                    cur_loss = torch.tensor(step_loss_val, device=device)
                    cur_ce = torch.tensor(step_ce_val, device=device)
                    cur_kd = torch.tensor(step_kd_val, device=device)
                    dist.all_reduce(cur_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cur_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cur_kd, op=dist.ReduceOp.SUM)
                    cur_loss = (cur_loss.item() / dist.get_world_size())
                    cur_ce = (cur_ce.item() / dist.get_world_size())
                    cur_kd = (cur_kd.item() / dist.get_world_size())

                    if rank == 0 and early_stopper is not None:
                        metric_map = {"loss": cur_loss, "ce": cur_ce, "kd": cur_kd}
                        mval = metric_map.get(args.early_stop_metric, cur_ce)
                        should_stop, reason = early_stopper.update(
                            mval,
                            step=train_steps,
                            wall_time_s=(time.time() - start_time_all),
                        )
                        if should_stop:
                            stop_training = True
                            early_stop_triggered = True
                            early_stop_reason = reason
                            early_stop_step = int(train_steps)
                            logger.info(
                                f"EarlyStop triggered at step={train_steps} metric={args.early_stop_metric} "
                                f"ema_best={early_stopper.best:.6f}@step{early_stopper.best_step} "
                                f"elapsed_s={time.time() - start_time_all:.1f} reason={reason}"
                            )

                # Make stop decision consistent across ranks to avoid hanging.
                stop_flag = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int32)
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                stop_training = bool(stop_flag.item())
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_ce = torch.tensor(running_ce / log_steps, device=device)
                    avg_kd = torch.tensor(running_kd / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_kd, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    avg_ce = avg_ce.item() / dist.get_world_size()
                    avg_kd = avg_kd.item() / dist.get_world_size()
                    hv_info = ""
                    if getattr(args, "hv_mix", False):
                        core = student.module
                        if (not args.no_compile) and hasattr(core, "_orig_mod"):
                            core = core._orig_mod
                        if hasattr(core, "get_hv_right_weight"):
                            hv_info = f", HVW: {core.get_hv_right_weight():.3f}"
                    prox_info = ""
                    if prox_prob is not None:
                        prox_info = f", ProxP: {prox_prob:.2f}"
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KD: {avg_kd:.4f}, "
                        f"MaskP: {mask_prob:.2f}{prox_info}{hv_info}, Steps/Sec: {steps_per_sec:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    running_loss = 0.0
                    running_ce = 0.0
                    running_kd = 0.0
                    log_steps = 0
                    start_time = time.time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = student.module._orig_mod.state_dict()
                    else:
                        model_weight = student.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": train_steps,
                        "args": args,
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                control_barrier()

            # FID evaluation
            if args.fid_every > 0 and train_steps % args.fid_every == 0:
                fid_ok = torch.tensor(1, device=device, dtype=torch.int32)
                if rank == 0:
                    eval_model = ema if (args.ema and args.fid_use_ema) else (student.module._orig_mod if not args.no_compile else student.module)
                    try:
                        run_fid_eval(args, eval_model, vq_model, device, train_steps, logger)
                    except Exception as e:
                        fid_ok.fill_(0)
                        logger.exception(f"FID eval failed at step={train_steps}: {e}")

                # Sync FID status across ranks to avoid one-rank crash causing a hang.
                control_all_reduce_min(fid_ok)
                if fid_ok.item() == 0:
                    if rank == 0:
                        logger.info(f"FID eval failed; --fid-fail-action={args.fid_fail_action}.")
                    if args.fid_fail_action == "stop":
                        stop_training = True
                control_barrier()

            if stop_training:
                break
        scheduler.step()
        if stop_training:
            if rank == 0 and (args.max_steps is not None and args.max_steps > 0 and train_steps >= args.max_steps):
                logger.info(f"Reached --max-steps={args.max_steps}, stopping early.")
            break

    # Optional: save a checkpoint right when early-stop triggers (even if ckpt_every is large)
    es_flag = torch.tensor(1 if early_stop_triggered else 0, device=device, dtype=torch.int32)
    dist.all_reduce(es_flag, op=dist.ReduceOp.MAX)
    early_stop_triggered = bool(es_flag.item())

    if early_stop_triggered and args.save_on_early_stop and rank == 0:
        if not args.no_compile:
            model_weight = student.module._orig_mod.state_dict()
        else:
            model_weight = student.module.state_dict()
        checkpoint = {
            "model": model_weight,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "steps": train_steps,
            "args": args,
        }
        if args.ema:
            checkpoint["ema"] = ema.state_dict()
        if not args.no_local_save:
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved early-stop checkpoint to {checkpoint_path}")
        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved early-stop checkpoint in cloud to {cloud_checkpoint_path}")

    # Optional: run evaluation immediately after early-stop (rank0 runs, others wait)
    if early_stop_triggered and args.eval_on_early_stop:
        control_barrier()
        fid_ok = torch.tensor(1, device=device, dtype=torch.int32)
        if rank == 0:
            if args.fid_ref is None:
                logger.info("eval-on-early-stop enabled but --fid-ref is not provided; skipping.")
            else:
                eval_model = ema if (args.ema and args.fid_use_ema) else (student.module._orig_mod if not args.no_compile else student.module)
                try:
                    run_fid_eval(args, eval_model, vq_model, device, train_steps, logger)
                except Exception as e:
                    fid_ok.fill_(0)
                    logger.exception(f"eval-on-early-stop FID failed at step={train_steps}: {e}")

        control_all_reduce_min(fid_ok)
        if fid_ok.item() == 0 and args.fid_fail_action == "stop":
            if rank == 0:
                logger.info("eval-on-early-stop FID failed; stopping.")
        control_barrier()

    # Save last checkpoint
    if rank == 0:
        if not args.no_compile:
            model_weight = student.module._orig_mod.state_dict()
        else:
            model_weight = student.module.state_dict()
        checkpoint = {"model": model_weight}
        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/last_version.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

    student.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help="cloud disk path")
    parser.add_argument("--no-local-save", action='store_true', help="no save checkpoints to local path")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    # dataset subset (for fast convergence / ablation iterations)
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=-1,
        help="If >0, only use first N samples (optionally shuffled) from imagenet_code to speed up experiments.",
    )
    parser.add_argument(
        "--train-subset-seed",
        type=int,
        default=0,
        help="Seed used when --train-max-samples is set and --train-subset-shuffle is enabled.",
    )
    parser.add_argument(
        "--train-subset-shuffle",
        action='store_true',
        help="Shuffle ids before taking first --train-max-samples (deterministic by --train-subset-seed).",
    )

    # model
    parser.add_argument("--gpt-model", type=str, choices=list(NAR_GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="resume checkpoint")
    parser.add_argument("--auto-resume", action='store_true', help="auto pick latest checkpoint from resume dirs")
    parser.add_argument("--resume-local-dir", type=str, default=None, help="existing local experiment dir, e.g., results/005-GPT-L")
    parser.add_argument("--resume-cloud-dir", type=str, default=None, help="existing cloud experiment dir, e.g., ./ckpt/2026-02-23-17-00-55/005-GPT-L")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--medusa-attention-num", type=int, default=1)
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")

    # distillation
    parser.add_argument("--teacher-gpt-model", type=str, choices=list(AR_GPT_models.keys()), default="GPT-L")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="teacher ckpt path (LlamaGen) for logits distillation")
    parser.add_argument("--init-ckpt", type=str, default=None, help="init student from this ckpt, default to teacher ckpt")
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--kd-prob",
        type=float,
        default=1.0,
        help="Probability to apply KD on an optimizer step. <1 reduces teacher forward compute.",
    )
    parser.add_argument(
        "--kd-every",
        type=int,
        default=1,
        help="Apply KD every N optimizer steps (1 means every step).",
    )
    parser.add_argument(
        "--kd-start-step",
        type=int,
        default=0,
        help="Start applying KD from this optimizer step (warmup without KD).",
    )
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)

    # lora
    parser.add_argument("--use-lora", action='store_true')
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    # Match actual module names in this repo's GPT implementation:
    # layers.{i}.attention.(wqkv|wo) and layers.{i}.feed_forward.(w1|w2|w3)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="attention.wqkv,attention.wo,feed_forward.w1,feed_forward.w2,feed_forward.w3",
    )
    parser.add_argument("--lora-train-base", action='store_true', help="also train base weights when using LoRA")

    # optimization
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    # mask schedule
    parser.add_argument(
        "--mask-schedule",
        type=str,
        default="linear",
        choices=["static_causal", "static_proximity", "static_union", "linear", "progressive", "shrink", "shrink_to_proximity", "linear_then_proximity"],
    )
    parser.add_argument("--mask-anneal-steps", type=int, default=20000)
    parser.add_argument(
        "--mask-switch-step",
        type=int,
        default=2000,
        help="Anneal steps used by --mask-schedule=linear_then_proximity (UNION->PROXIMITY length).",
    )
    parser.add_argument(
        "--mask-prox-steps",
        type=int,
        default=500,
        help="Second phase length (UNION->PROXIMITY) used by --mask-schedule=linear_then_proximity.",
    )

    # Right/Below logits mixing (learnable alpha)
    parser.add_argument("--hv-mix", action='store_true', help="enable learnable mixing between right/below logits")
    parser.add_argument("--hv-mix-init", type=float, default=0.5, help="initial right(head) weight in [0,1]")
    parser.add_argument("--hv-mix-target", type=float, default=0.5, help="target right(head) weight in [0,1]")
    parser.add_argument(
        "--hv-mix-anneal-steps",
        type=int,
        default=0,
        help="If >0, linearly anneal target from --hv-mix-init to --hv-mix-target over this many steps.",
    )
    parser.add_argument(
        "--hv-mix-blend-steps",
        type=int,
        default=0,
        help="If >0, blend target->learned weight over this many steps (starts 1.0 then decays to 0.0).",
    )

    # runtime control
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimizer steps (i.e., after gradient accumulation sync). -1 means no limit.",
    )

    # convergence / early stop
    parser.add_argument("--early-stop", action='store_true', help="Enable convergence-based early stopping")
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="ce",
        choices=["loss", "ce", "kd"],
        help="Which metric to monitor for convergence (lower is better).",
    )
    parser.add_argument(
        "--early-stop-mode",
        type=str,
        default="plateau",
        choices=["plateau", "threshold", "both"],
        help="Stop on plateau, threshold, or both.",
    )
    parser.add_argument(
        "--early-stop-warmup-steps",
        type=int,
        default=500,
        help="Do not consider early stop before this optimizer step.",
    )
    parser.add_argument(
        "--early-stop-check-every",
        type=int,
        default=50,
        help="Check convergence every N optimizer steps (must be the same on all ranks).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Plateau patience in number of checks (so effective patience in steps is patience*check_every).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum EMA improvement to be considered progress.",
    )
    parser.add_argument(
        "--early-stop-ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for smoothing the monitored metric. Larger = smoother.",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=-1.0,
        help="If >=0, stop when EMA(metric) <= threshold. -1 disables threshold stop.",
    )

    # early-stop side effects
    parser.add_argument(
        "--eval-on-early-stop",
        action='store_true',
        help="If set, run FID evaluation once immediately after early-stop (rank0 only). Requires --fid-ref and --vq-ckpt.",
    )
    parser.add_argument(
        "--save-on-early-stop",
        action='store_true',
        help="If set, save a step checkpoint when early-stop triggers (in addition to last_version.pt).",
    )

    # fid evaluation
    parser.add_argument("--fid-every", type=int, default=0)
    parser.add_argument("--fid-ref", type=str, default=None)
    parser.add_argument("--fid-num-samples", type=int, default=50000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-sample-dir", type=str, default="samples")
    parser.add_argument(
        "--fid-fail-action",
        type=str,
        default="stop",
        choices=["stop", "skip"],
        help="What to do if rank0 FID evaluation raises an exception. 'stop' stops training; 'skip' logs and continues.",
    )
    parser.add_argument("--fid-use-ema", action='store_true')
    parser.add_argument(
        "--fid-skip-evaluator",
        action='store_true',
        help="Only generate FID sample .npz during training; do not run evaluations/c2i/evaluator.py (run it manually later).",
    )
    parser.add_argument("--fid-top-k", type=int, default=0)
    parser.add_argument("--fid-top-p", type=float, default=1.0)
    parser.add_argument("--fid-temperature", type=float, default=1.0)
    parser.add_argument("--fid-cfg-scale", type=float, default=1.5)
    parser.add_argument("--fid-cfg-interval", type=float, default=-1)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)

    args = parser.parse_args()
    main(args)
