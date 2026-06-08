export CUDA_VISIBLE_DEVICES=0
python  ./vae_main.py \
		--batch_size 1 \
		--num_workers 4 \
		--accum_iter 1 \
		--epochs 1000 \
		--warmup_epochs 10 \
		--warm_up_n_epochs 10 \
		--lr 2e-5  \
		--pin_mem \
		--seed 42 \
		--adv_weight 0.1 \
		--modality_weight 0 \
    --perceptual_weight 0.1 \
    --kl_weight  1e-6 \
		--name unified_vae \
		--log_dir /cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/Autoencoder \
		--output_dir /cpfs01/projects-HDD/cfff-7abceac4e328_HDD/dhm_41310/chwang/Logs/PETLDM/Autoencoder