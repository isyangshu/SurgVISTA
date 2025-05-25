param1_list=("k400" "ssv2" "hybrid" "k710" "k400" "settingE") 
param2_list=("videomae" "videomae" "videomae" "umt" "mvd" "videomae-st")
param3_list=("unified_base" "unified_base" "unified_base" "unified_base" "unified_base_st" "unified_base_st")

# param1_list=("settingE") 
# param2_list=("videomae-st")
# param3_list=("unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12332 \
    downstream_triplet/run_triplet_training.py \
    --batch_size 32 \
    --epochs 20 \
    --save_ckpt_freq 10 \
    --model  "${param3_list[$i]}" \
    --pretrained_data "${param1_list[$i]}" \
    --pretrained_method "${param2_list[$i]}" \
    --lr 5e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --data_path /project/medimgfmod/Video/Prostate21 \
    --eval_data_path /project/medimgfmod/Video/Prostate21 \
    --nb_classes 89 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set Prostate21 \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/Natural_Comparison/Prostate21 \
    --log_dir /project/mmendoscope/Natural_Comparison/Prostate21 \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done