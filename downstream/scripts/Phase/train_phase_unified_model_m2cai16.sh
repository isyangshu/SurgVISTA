param1_list=("settingD") 
param2_list=("videomae-st")
param3_list=("unified_base_st")

for i in "${!param1_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 12322 \
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
    --lr 1e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --data_path /scratch/mmendoscope/pretraining/M2CAI16-workflow \
    --eval_data_path /scratch/mmendoscope/pretraining/M2CAI16-workflow \
    --nb_classes 8 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 24 \
    --sampling_rate 4 \
    --data_set M2CAI16 \
    --data_fps 1fps \
    --output_dir /project/mmendoscope/downstream_uni/M2CAI16 \
    --log_dir /project/mmendoscope/downstream_uni/M2CAI16 \
    --num_workers 12 \
    --dist_eval \
    --enable_deepspeed \
    --no_auto_resume
done