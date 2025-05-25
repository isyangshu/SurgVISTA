# param1_list=("imagenet1k" "imagenet1k" "imagenet1k" "wit400m") 
# param2_list=("supervised" "mae" "dino" "clip")

param1_list=("wit400m") 
param2_list=("clip")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12336 \
    downstream_clip/run_clip_training.py \
    --batch_size 32 \
    --epochs 500 \
    --save_ckpt_freq 100 \
    --model  unified_base_image \
    --pretrained_data "${param1_list[$i]}" \
    --pretrained_method "${param2_list[$i]}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 1e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 50 \
    --data_path /project/smartlab2021/syangcw/SurgicalActions160 \
    --eval_data_path /project/smartlab2021/syangcw/SurgicalActions160 \
    --nb_classes 16 \
    --num_frames 8 \
    --data_set SurgicalActions160 \
    --output_dir /project/smartlab2021/syangcw/SurgSSL/SurgicalActions160_ \
    --log_dir /project/smartlab2021/syangcw/SurgSSL/SurgicalActions160_ \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done