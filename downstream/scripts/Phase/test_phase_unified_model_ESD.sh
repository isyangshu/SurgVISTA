param1_list=("k400") 
param2_list=("mvd")
param3_list=("unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12328 \
    downstream_phase/run_phase_training_val.py \
    --batch_size 4 \
    --epochs 50 \
    --save_ckpt_freq 10 \
    --model  "${param3_list[$i]}" \
    --pretrained_data "${param1_list[$i]}" \
    --pretrained_method "${param2_list[$i]}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 5e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --data_path /project/medimgfmod/Video/ESD57/ \
    --eval_data_path /project/medimgfmod/Video/ESD57/ \
    --nb_classes 8 \
    --data_strategy online \
    --output_mode key_frame \
    --eval \
    --finetune /project/mmendoscope/Natural_Comparison/ESD57/unified_base_st_k400_mvd_ESD57_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best/mp_rank_00_model_states.pt \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set ESD57 \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/Natural_Comparison/ESD57 \
    --log_dir /project/mmendoscope/Natural_Comparison/ESD57 \
    --num_workers 4 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done