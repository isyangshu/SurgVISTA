# param1_list=("k400" "ssv2" "hybrid" "pt_hybrid_ft_k710" "k710" "k400" "settingD" "settingE") 
# param2_list=("videomae" "videomae" "videomae" "videomaev2" "umt" "mvd" "videomae-st" "videomae-st")
# param3_list=("unified_base" "unified_base" "unified_base" "unified_base" "unified_base" "unified_base_st" "unified_base_st" "unified_base_st")

param1_list=("settingA" "settingB" "settingC" "settingD" "settingE") 
param2_list=("videomae-st" "videomae-st" "videomae-st" "videomae-st" "videomae-st")
param3_list=("unified_base_st" "unified_base_st" "unified_base_st" "unified_base_st" "unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12336 \
    downstream_clip/run_clip_training.py \
    --batch_size 32 \
    --epochs 500 \
    --save_ckpt_freq 100 \
    --model  "${param3_list[$i]}" \
    --pretrained_data "${param1_list[$i]}" \
    --pretrained_method "${param2_list[$i]}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 5e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 50 \
    --data_path /project/smartlab2021/syangcw/SurgicalActions160 \
    --eval_data_path /project/smartlab2021/syangcw/SurgicalActions160 \
    --nb_classes 16 \
    --num_frames 16 \
    --data_set SurgicalActions160 \
    --output_dir /project/smartlab2021/syangcw/SurgSSL/SurgicalActions160 \
    --log_dir /project/smartlab2021/syangcw/SurgSSL/SurgicalActions160 \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done