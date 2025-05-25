# param1_list=("k400" "ssv2" "hybrid" "pt_hybrid_ft_k710" "k710" "k400" "settingE") 
# param2_list=("videomae" "videomae" "videomae" "videomaev2" "umt" "mvd" "videomae-st")
# param3_list=("unified_base" "unified_base" "unified_base" "unified_base" "unified_base" "unified_base_st" "unified_base_st")

param1_list=("settingA" "settingB" "settingC" "settingD") 
param2_list=("videomae" "videomae" "videomae" "videomae")
param3_list=("unified_base" "unified_base" "unified_base" "unified_base")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12336 \
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
    --data_path /project/medimgfmod/syangcw/downstream/AutoLaparo \
    --eval_data_path /project/medimgfmod/syangcw/downstream/AutoLaparo \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set AutoLaparo \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/SurgSSL/KD/AutoLaparo \
    --log_dir /project/mmendoscope/SurgSSL/KD/AutoLaparo \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done