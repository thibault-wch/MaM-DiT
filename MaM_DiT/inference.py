import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import SimpleITK as sitk
import monai.transforms as mtransforms

from models_uvit_acc_224 import DiT
from autoencoderkl import AutoencoderKL
from diffusion import create_diffusion


def main():
    # 1. Enable TF32 to significantly speed up inference on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ================= Configuration Area =================
    TARGET_TRACER = 'av45'  # Switch the target tracer type here: 'av45' or 'fdg' or 'mk'

    # Label mapping dictionary (av45 corresponds to 0, fdg corresponds to 1, mk corresponds to 2)
    TRACER_LABEL_MAP = {'av45': 0, 'fdg': 1, 'mk': 2}
    label_id = TRACER_LABEL_MAP[TARGET_TRACER]

    mri_dir = '/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/Results/External/original_mri/'
    base_out_dir = '/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/Results/External/generated/'

    # Automatically create the output directory for the corresponding tracer
    out_dir = os.path.join(base_out_dir, TARGET_TRACER)
    os.makedirs(out_dir, exist_ok=True)

    n = 1
    cfg_scale = 1.2
    # ====================================================

    # 2. Model Definition
    model = DiT(depth=16,
                in_channels=32,
                input_size=(24, 28, 24),
                hidden_size=1536,
                patch_size=2,
                num_heads=24,
                class_dropout_prob=0.0,
                num_classes=3,
                learn_sigma=False,
                enable_flashattn=True,
                enable_layernorm_kernel=True,
                enable_modulate_kernel=True,
                dtype=torch.float16)

    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(16, 64, 96, 32),
        latent_channels=32,
        num_res_blocks=2,
        norm_num_groups=8,
        attention_levels=(False, False, False, True),
        use_flash_attention=True)

    # 3. Load Weights
    print("Loading VAE weights...")
    vae_state_dict = torch.load(
        '/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/Autoencoder/pretrain_224_5/generator-99.pth')
    vae.load_state_dict(vae_state_dict)
    vae.cuda()
    vae.eval()

    print("Loading DiT weights...")
    model_state_dict = torch.load(
        '/cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/LDM/007-2024-09-18 06:56:12.088330/checkpoints/0000300.pt',
        map_location='cpu')['ema']
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    # 4. Data Preprocessing
    mri_transform = mtransforms.Compose([
        mtransforms.LoadImage(image_only=True),
        mtransforms.EnsureChannelFirst(),
        mtransforms.SqueezeDim(),
        mtransforms.EnsureChannelFirst(),
        mtransforms.EnsureType(),
        mtransforms.ScaleIntensityRangePercentiles(lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True, relative=False),
        mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192))
    ])

    # 5. Inference Preparation
    diffusion = create_diffusion("50")

    # Pre-build label conditions
    y_null = torch.tensor([3] * n).cuda()
    y_true = torch.tensor([label_id] * n).cuda()
    y = torch.cat([y_true, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # 6. Inference Loop
    print(f"Starting inference for {TARGET_TRACER.upper()}...")
    for dataname in tqdm(os.listdir(mri_dir), desc=f"Generating {TARGET_TRACER.upper()}"):
        # Generate independent initial noise for each iteration
        z = torch.randn(1, 32, 24, 28, 24).cuda()
        z = torch.cat([z, z], 0)

        # Extract MRI conditional features
        mri_path = os.path.join(mri_dir, dataname)
        con = mri_transform(mri_path).unsqueeze(0).cuda()

        with torch.no_grad():
            z_conc = vae.encode_stage_2_inputs(con)
        z_conc = torch.cat([z_conc, z_conc], 0)

        # Sample generation
        with torch.no_grad():
            with autocast():
                samples = diffusion.ddim_sample_loop(
                    model.forward_with_cfg,
                    z.shape,
                    z,
                    c=z_conc,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device='cuda:0'
                )
            decode_image = vae.decode_stage_2_outputs(samples[0].unsqueeze(0))

        # SimpleITK safe conversion and saving
        img_array = decode_image[0][0].cpu().numpy()
        img_array = (img_array + 1) * 1.3

        out = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(out, os.path.join(out_dir, dataname))

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()