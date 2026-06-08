import argparse
import datetime
import json
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter

import wandb
from generative.losses import PatchAdversarialLoss, PerceptualLoss, Modalityloss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

# Import your custom modules
import util.misc as misc
from PETdataset import PETDataset
from vae_train import train_one_epoch

os.environ["WANDB_MODE"] = "offline"


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def get_args_parser():
    parser = argparse.ArgumentParser('PETLDM_main-training', add_help=False)

    # Batch & Epoch parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    parser.add_argument('--name', default='422_pretrained_swin_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Lower lr bound')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='Epochs to warmup LR')

    # Dataset & I/O parameters
    parser.add_argument('--output_dir', default='./output_dir', help='Path where to save checkpoints')
    parser.add_argument('--log_dir', default='./output_dir', help='Path where to save tensorboard logs')
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='Url used to set up distributed training')

    # Loss weights
    parser.add_argument('--kl_weight', default=1.0, type=float, help='Weight of KL loss')
    parser.add_argument('--perceptual_weight', default=1.0, type=float, help='Weight of perceptual loss')
    parser.add_argument('--modality_weight', default=1.0, type=float, help='Weight of modality loss')
    parser.add_argument('--adv_weight', default=1.0, type=float, help='Weight of adversarial loss')
    parser.add_argument('--warm_up_n_epochs', default=5, type=int, help='Epochs before applying adversarial loss')

    return parser


def main(args):
    print(f'Job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(f"{args}".replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Initialize Dataset and DataLoader
    dataset_train = PETDataset(mode='train', tasktype='all')
    print(f"Dataset length: {len(dataset_train)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Initialize Loggers
    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        wandb.init(
            project="PET_LDM",
            entity="aging",
            name=args.name,
            config=vars(args)
        )

    # Initialize Models
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(16, 64, 96, 32),
        latent_channels=32,
        num_res_blocks=2,
        norm_num_groups=8,
        attention_levels=(False, False, False, True),
        use_flash_attention=True,
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=4,
        num_channels=4,
        in_channels=1,
        out_channels=1
    ).to(device)

    # Initialize EMA model
    ema = deepcopy(autoencoder).to(device)
    requires_grad(ema, False)

    # Note: Wrap with DDP here if distributed training is enabled
    autoencoder_without_ddp = autoencoder
    discriminator_without_ddp = discriminator

    # Initialize Losses
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(
        device)
    loss_modality = Modalityloss().to(device)  # Assuming this is defined elsewhere

    # Initialize Optimizers
    optimizer_g = torch.optim.AdamW(autoencoder_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(discriminator_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=args.weight_decay)

    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # Start Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    update_ema(ema, autoencoder, decay=0)

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            autoencoder, discriminator, data_loader_train,
            optimizer_g, optimizer_d, device,
            epoch, scaler_g, scaler_d, l1_loss, adv_loss, loss_perceptual, loss_modality,
            log_writer=log_writer,
            args=args
        )

        update_ema(ema, autoencoder)

        # Save checkpoints
        if args.output_dir and ((epoch + 1) % 2 == 0 or epoch + 1 == args.epochs):
            torch.save(autoencoder_without_ddp.state_dict(), os.path.join(args.output_dir, f'generator-{epoch}.pth'))
            torch.save(discriminator_without_ddp.state_dict(),
                       os.path.join(args.output_dir, f'discriminator-{epoch}.pth'))
            torch.save(ema.state_dict(), os.path.join(args.output_dir, f'ema-{epoch}.pth'))

        # Log statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # Properly construct unique logging and output directories based on the run name
    if args.output_dir:
        args.log_dir = os.path.join(args.output_dir, args.name)
        args.output_dir = os.path.join(args.output_dir, args.name)
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)