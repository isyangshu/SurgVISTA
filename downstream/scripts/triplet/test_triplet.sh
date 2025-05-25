# --output_dir /project/mmendoscope/downstream_uni/CholecT50/Challenge \
# --log_dir /project/mmendoscope/downstream_uni/CholecT50/Challenge \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 12328 \
downstream_triplet/run_triplet_training.py \
--batch_size 16 \
--epochs 50 \
--save_ckpt_freq 10 \
--model  unified_base_image \
--pretrained_data imagenet1k \
--pretrained_method supervised \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /project/medimgfmod/Video/Prostate21 \
--eval_data_path /project/medimgfmod/Video/Prostate21 \
--eval \
--finetune /project/mmendoscope/Natural_Comparison/Prostate21/unified_base_image_imagenet1k_supervised_Prostate21_0.0005_0.75_online_key_frame_frame8_Fixed_Stride_8/checkpoint-best/mp_rank_00_model_states.pt \
--nb_classes 89 \
--data_strategy online \
--output_mode key_frame \
--num_frames 8 \
--sampling_rate 8 \
--data_set Prostate21 \
--data_fps 1fps \
--output_dir /project/mmendoscope/Natural_Comparison/Prostate21 \
--log_dir /project/mmendoscope/Natural_Comparison/Prostate21 \
--num_workers 16 \
--dist_eval \
--enable_deepspeed \
--no_auto_resume