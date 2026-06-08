# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

from datetime import datetime
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PETdataset_a100 import PETDataset
from torchvision import transforms
import numpy as np

from PIL import Image
from copy import deepcopy
from glob import glob
from time import time

from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
import os
# based on your environment to make it online/offline
os.environ["WANDB_MODE"]="offline"
from tqdm import tqdm
from models_uvit_acc_224 import DiT
from diffusion import create_diffusion
from autoencoderkl import AutoencoderKL
import wandb

from collections import OrderedDict

# not used
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])

        new_state_dict[name] = v
    keys=[]
    for k,v in new_state_dict.items():
        if k.startswith('x_embedder'):
            continue
        if k.startswith('c_embedder'):
            continue
        if k.startswith('xc_embedder'):
            continue
        if k.startswith('final_layer'):
            continue
        keys.append(k)
    new_state_dict = {k:new_state_dict[k] for k in keys}
    return new_state_dict




#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider app
        #  lying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{datetime.now()}"# Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(
            project="MaM_DiT",
            entity="aging",
            name=f'{experiment_index:03d}--{str(datetime.now())[:10]}',
            config={
                "lr":args.lr,
                "model_depth": args.depth,
                "model_hidden_size": args.hidden_size,
                "model_patch_size": args.patch_size,
                "model_num_head": args.num_head,
                "batchsize": args.global_batch_size,
                "epochs": args.epochs,
                "seed": args.global_seed,
                "numworkers": args.num_workers
            }
        )
    else:
        logger = create_logger(None)

    # Create model
    scaler = GradScaler()
    model = DiT(depth=args.depth,
            in_channels=32,
            input_size=(args.input_size1,args.input_size2,args.input_size3),
            hidden_size=args.hidden_size,
            patch_size=args.patch_size,
            num_heads=args.num_head,
            class_dropout_prob=0.2,
            num_classes=3,
            learn_sigma=False,
            enable_flashattn = args.enable_flashattn,
            enable_layernorm_kernel =args.enable_layernorm_kernel,
            enable_modulate_kernel = args.enable_modulate_kernel
            )
    # add your DiT state dict
    state_dict = torch.load('/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/LDM/008-2024-05-23 05:11:02.298972-basic2_MK/checkpoints/0012000.pt',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    ema.load_state_dict(state_dict['ema'])
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(16,64, 96, 32),
            latent_channels=32,
            num_res_blocks=2,
            norm_num_groups=8,
            attention_levels=(False,False, False, True),
        )
    # add your vae state dict
    state_dict = torch.load('/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/Autoencoder/pretrain_224_5/generator-99.pth',
                            map_location=torch.device('cpu'))
    vae.load_state_dict(state_dict)
    requires_grad(vae, False)
    vae.to(device)
    vae.eval()

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    dataset = PETDataset(mode='train',tasktype='mk')
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for idx, item in enumerate(tqdm(loader)):
            opt.zero_grad()
            ori_x = item['pet'].to(device)
            c = item['mri'].to(device)
            y = item['tasktype'].to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x =vae.encode_stage_2_inputs(ori_x)
                c =vae.encode_stage_2_inputs(c)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            with autocast():
                loss_dict = diffusion.training_losses(model,vae, ori_x,x,c, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank==0:
                    wandb.log({"loss":avg_loss})
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--input_size1", type=int, default=24)
    parser.add_argument("--input_size2", type=int, default=28)
    parser.add_argument("--input_size3", type=int, default=24)
    parser.add_argument("--num_head", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument('--enable_flashattn', action='store_true')
    parser.add_argument('--enable_layernorm_kernel', action='store_true')
    parser.add_argument('--enable_modulate_kernel', action='store_true')

    args = parser.parse_args()
    main(args)
