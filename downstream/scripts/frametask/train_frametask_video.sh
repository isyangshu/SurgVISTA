param1_list=("k400") 
param2_list=("mvd")
param3_list=("unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12330 \
    downstream_csv/run_csv_training.py \
    --batch_size 16 \
    --epochs 20 \
    --save_ckpt_freq 10 \
    --model  "${param3_list[$i]}" \
    --pretrained_data "${param1_list[$i]}" \
    --pretrained_method "${param2_list[$i]}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 5e-5 \
    --layer_decay 0.75 \
    --warmup_epochs 2 \
    --data_path /path/to/your/dataset \
    --eval_data_path /path/to/your/dataset \
    --nb_classes 4 \
    --num_frames 8 \
    --sampling_rate 4 \
    --data_set Endoscapes \
    --output_dir /save/path/to/your/Endoscapes \
    --log_dir /save/path/to/your/Endoscapes \
    --num_workers 16 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done