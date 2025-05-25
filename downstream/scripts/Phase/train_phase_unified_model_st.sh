param1_list=("settingA" "settingA") 
param2_list=("videomae-st-075" "videomae-st-07")
param3_list=("unified_base_st" "unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12332 \
    downstream_phase/run_phase_training.py \
    --batch_size 16 \
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
    --data_path /scratch/mmendoscope/downstream/cholec80 \
    --eval_data_path /scratch/mmendoscope/downstream/cholec80 \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 8 \
    --data_set Cholec80 \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/downstream_uni \
    --log_dir /project/mmendoscope/downstream_uni \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume \
    --cut_black
done