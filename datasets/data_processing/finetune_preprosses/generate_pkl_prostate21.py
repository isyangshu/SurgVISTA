import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import json


def main():
    split = "test"  # Change this to 'train', 'val', or 'test' as needed
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames", split))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])
    pkl = dict()
    unique_id = 0
    FRAME_NUMBERS = 0

    for video_id in VIDEO_NAMES:
        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", split, video_id)
        frames_list = os.listdir(video_path)
        frames_list = sorted([int(x.split('.')[0]) for x in frames_list if "jpg" in x])

        # 打开Label文件
        triplet_path = os.path.join(ROOT_DIR, 'annos', split, video_id + '.json')
        with open(triplet_path, 'r') as triplet_file:
            triplet_data = json.load(triplet_file)
        annos = triplet_data['annotations']

        assert len(frames_list) == len(annos.keys())
        frame_infos = list()
        for frame_id in tqdm(frames_list):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id
            info['video_id'] = video_id
            info['frames'] = len(frames_list)
            triplet_info = annos[str(frame_id)]
            triplet_id = []
            for triplet in triplet_info:
                triplet_id.append(triplet[0])
            triplet_id = sorted([int(i) for i in triplet_id])
            info['triplet_gt'] = triplet_id
            frame_infos.append(info)
            unique_id += 1
    
        pkl[video_id] = frame_infos
        FRAME_NUMBERS += len(frames_list)

    save_dir = os.path.join(ROOT_DIR, 'labels_pkl', split)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, '1fps'+split+'.pickle'), 'wb') as file:
        pickle.dump(pkl, file)


    print('Frams', FRAME_NUMBERS, unique_id)

if __name__ == "__main__":
    main()