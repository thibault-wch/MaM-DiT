import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler  # Added AMP imports
import util.misc as misc
import util.lr_sched as lr_sched
import wandb
import random


def KL_loss(z_mu, z_sigma, eps=1e-8):
    # Add epsilon to prevent log(0) and support arbitrary dimensions
    var = z_sigma.pow(2) + eps
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + var - torch.log(var) - 1, dim=list(range(1, z_mu.dim())))
    return kl_loss.mean()


def train_one_epoch(autoencoder,
                    discriminator,
                    data_loader,
                    optimizer_g,
                    optimizer_d,
                    device, epoch,
                    scaler_g,
                    scaler_d,
                    l1_loss=None,
                    adv_loss=None,
                    loss_perceptual=None,
                    loss_modality=None,
                    log_writer=None,
                    args=None):
    autoencoder.train(True)
    discriminator.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust learning rate per iteration
        lr_sched.adjust_learning_rate(optimizer_g, data_iter_step / len(data_loader) + epoch, args)
        lr_sched.adjust_learning_rate(optimizer_d, data_iter_step / len(data_loader) + epoch, args)

        # ========================
        # Data Preparation
        # ========================
        # Broadcast random value from rank 0 to ensure DDP synchronization
        random_tensor = torch.rand(1, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(random_tensor, src=0)
        random_num = random_tensor.item()

        # Prepare inputs and initialize labels with proper tensor types
        if random_num >= 0.5:
            images = batch["mri"].to(device)
            labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)
        else:
            images = batch["pet"].to(device)
            if isinstance(batch["tasktype"], torch.Tensor):
                labels = batch["tasktype"].clone().detach().to(device) + 1
            else:
                labels = torch.as_tensor(batch["tasktype"], dtype=torch.long, device=device) + 1

        # ========================
        # Generator
        # ========================
        optimizer_g.zero_grad(set_to_none=True)

        # 1. Wrap the forward pass in autocast
        with autocast():
            reconstruction, z_mu, z_sigma = autoencoder(images)

            kl_loss = KL_loss(z_mu, z_sigma)
            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float()).mean()

            loss_g = recons_loss + args.kl_weight * kl_loss + args.perceptual_weight * p_loss

            if epoch >= args.warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += args.adv_weight * generator_loss

                m_loss = loss_modality(reconstruction.float(), labels).mean()
                loss_g += args.modality_weight * m_loss

                m_loss_value = m_loss.item()
                metric_logger.update(loss_m=m_loss_value)
                m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)

                generator_loss_value = generator_loss.item()
                metric_logger.update(loss_gene=generator_loss_value)
                generator_loss_value_reduce = misc.all_reduce_mean(generator_loss_value)

        # 2. Scale the loss, backpropagate, and step using scaler_g
        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # ========================
        # Discriminator
        # ========================
        if epoch >= args.warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)

            # 3. Wrap discriminator forward pass in autocast
            with autocast():
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = args.adv_weight * discriminator_loss

            # 4. Scale the loss, backpropagate, and step using scaler_d
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            loss_d_value = loss_d.item()
            metric_logger.update(loss_d=loss_d_value)
            loss_d_value_reduce = misc.all_reduce_mean(loss_d_value)

        # ========================
        # Logging
        # ========================
        loss_g_value = loss_g.item()
        kl_loss_value = kl_loss.item()
        recons_loss_value = recons_loss.item()
        p_loss_value = p_loss.item()
        lr = optimizer_g.param_groups[0]["lr"]

        metric_logger.update(loss_g=loss_g_value)
        metric_logger.update(loss_kl=kl_loss_value)
        metric_logger.update(loss_recons=recons_loss_value)
        metric_logger.update(loss_p=p_loss_value)
        metric_logger.update(lr=lr)

        loss_g_value_reduce = misc.all_reduce_mean(loss_g_value)
        kl_loss_value_reduce = misc.all_reduce_mean(kl_loss_value)
        recons_loss_value_reduce = misc.all_reduce_mean(recons_loss_value)
        p_loss_value_reduce = misc.all_reduce_mean(p_loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_g', loss_g_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Dynamically build wandb dict to avoid logging None/0 during warm-up
            wandb_dict = {
                'lr': lr,
                'train_loss_g': loss_g_value_reduce,
                'train_kl_loss': kl_loss_value_reduce,
                'train_recons_loss': recons_loss_value_reduce,
                'train_p_loss': p_loss_value_reduce,
            }

            if epoch >= args.warm_up_n_epochs:
                log_writer.add_scalar('train_loss_d', loss_d_value_reduce, epoch_1000x)
                wandb_dict.update({
                    'train_loss_d': loss_d_value_reduce,
                    'train_generator_loss': generator_loss_value_reduce,
                    'train_m_loss': m_loss_value_reduce
                })

            wandb.log(wandb_dict)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}