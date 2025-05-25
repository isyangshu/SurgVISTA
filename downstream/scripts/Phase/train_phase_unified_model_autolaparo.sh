param1_list=("settingE") 
param2_list=("videomae-st")
param3_list=("unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12322 \
    downstream_phase/run_phase_training_val.py \
    --batch_size 32 \
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
    --data_path /scratch/mmendoscope/downstream/AutoLaparo \
    --eval_data_path /scratch/mmendoscope/downstream/AutoLaparo \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set AutoLaparo \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/Natural_Comparison/AutoLaparo \
    --log_dir /project/mmendoscope/Natural_Comparison/AutoLaparo \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done