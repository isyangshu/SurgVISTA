# Cholec80-CVS nb_classes 6 num_frames 16 lr 5e-4
# SurgicalActions160 nb_classes 16 num_frames 16 lr 1e-4
param1_list=("k400" "ssv2" "hybrid" "k710" "k400" "SurgVISTA") 
param2_list=("videomae" "videomae" "videomae" "umt" "mvd" "SurgVISTA")
param3_list=("unified_base" "unified_base" "unified_base" "unified_base" "unified_base_st" "unified_base_st")


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
    --data_path /path/to/your/dataset \
    --eval_data_path /path/to/your/dataset \
    --nb_classes 6 \
    --num_frames 16 \
    --data_set Cholec80-CVS \
    --output_dir /save/path/to/your/dataset \
    --log_dir /save/path/to/your/dataset \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done