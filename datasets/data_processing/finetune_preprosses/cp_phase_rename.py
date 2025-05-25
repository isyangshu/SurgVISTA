import os
import shutil

# 源文件路径夹
src_dir = '/scratch/mmendoscope/downstream/m2cai16-workflow/'
dst_dir = '/scratch/mmendoscope/pretraining/M2CAI16-workflow/'

video_count = 1
for split_path in ["train_dataset", "test_dataset"]:
    data_path = os.path.join(src_dir, split_path)
    video_list = sorted([i for i in os.listdir(data_path) if i.endswith(".mp4")])
    phase_list = sorted([i for i in os.listdir(data_path) if i.endswith(".txt") and "timestamp" not in i and "pred" not in i])
    for video_name, phase_name in zip(video_list, phase_list):
        src_file_video = os.path.join(data_path, video_name)
        src_file_phase = os.path.join(data_path, phase_name)
        dst_video_name = "video_" + "{:02d}".format(video_count) + ".mp4"
        dst_phase_name = "video_phase_" + "{:02d}".format(video_count) + ".txt"
        video_count +=1
        dst_file_video = os.path.join(dst_dir, "videos", dst_video_name)
        dst_file_phase = os.path.join(dst_dir, "phases", dst_phase_name)
        
        # 复制文件
        shutil.copy2(src_file_video, dst_file_video)
        print(f'Copied {src_file_video} to {dst_file_video}')
        shutil.copy2(src_file_phase, dst_file_phase)
        print(f'Copied {src_file_phase} to {dst_file_phase}')