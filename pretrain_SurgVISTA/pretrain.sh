OUTPUT_DIR="/home/syangcw/SurgVISTA"
DATA_ROOT="/home/syangcw/pretraining"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 12332 \
pretrain_SurgVISTA/run_pretraining.py \
--data_root ${DATA_ROOT} \
--model pretrain_masked_video_student_base_patch16_224 \
--log_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--pretrained_datasets Cholec80 M2CAI16-workflow HeiChole PitVis PSI-AVA AutoLaparo BernBypass70 StrasBypass70 GenSurgery \
--image_teacher_model surgery_teacher_vit_large_patch16 \
--distillation_target_dim 1024 \
--distill_loss_func SmoothL1 \
--image_teacher_model_ckpt_path '/home/syangcw/SurgVISTA/pretrain_params/vit_large_patch16_224_surgery.pth' \
--mask_type tube \
--mask_ratio 0.85 \
--opt adamw \
--opt_betas 0.9 0.95 \
--decoder_depth_image 2 \
--decoder_depth_video 4  \
--image_teacher_loss_weight 0.05 \
--video_reconstruction_loss_weight 1.0 \
--feat_decoder_embed_dim 384 \
--feat_decoder_num_heads 6 \
--batch_size 64 \
--update_freq 4 \
--num_frames 16 \
--save_ckpt_freq 10 \
--sampling_rate 4 \
--lr 1.5e-4 \
--min_lr 1e-4 \
--drop_path 0.1 --warmup_epochs 40 --epochs 201 \
--auto_resume \
--num_workers 10