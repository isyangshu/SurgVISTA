# base: pretrain_videomae_base_patch16_224, decoder_depth: 4
# large: pretrain_videomae_large_patch16_224, decoder_depth: 12
OUTPUT_DIR="/project/mmendoscope/SurgSSL_output"
DATA_ROOT="/scratch/mmendoscope/pretraining/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 12322 \
pretrain_SurgVISTA/run_videomae_pretraining.py \
--data_root ${DATA_ROOT} \
--model pretrain_videomae_base_patch16_224 \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--pretrained_datasets Cholec80 \
--mask_type tube \
--mask_ratio 0.9 \
--opt adamw \
--opt_betas 0.9 0.95 \
--decoder_depth 4 \
--batch_size 96 \
--num_frames 16 \
--save_ckpt_freq 20 \
--sampling_rate 4 \
--lr 1.5e-4 \
--min_lr 1e-5 \
--drop_path 0.0 --warmup_epochs 40 --epochs 401 \
--auto_resume \
--num_workers 12