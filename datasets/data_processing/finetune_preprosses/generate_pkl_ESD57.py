import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import pandas as pd
import math
from random import shuffle

def main():
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES])
    
    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0
    # Shuffle the video names
    shuffle(VIDEO_NAMES)

    # Split into train, val, and test sets
    train_split = int(len(VIDEO_NAMES) * 0.7)
    val_split = int(len(VIDEO_NAMES) * 0.8)

    TRAIN_NAME = VIDEO_NAMES[:train_split]  # 39
    VAL_NAME = VIDEO_NAMES[train_split:val_split]  # 6
    TEST_NAME = VIDEO_NAMES[val_split:]  # 12

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    for video_name in VIDEO_NAMES:
        if video_name in TRAIN_NAME:
            unique_id = unique_id_train
            vid_id = int(video_name)
        elif video_name in VAL_NAME:
            unique_id = unique_id_val
            vid_id = int(video_name)
        elif video_name in TEST_NAME:
            unique_id = unique_id_test
            vid_id = int(video_name)

        frame_path = os.path.join(ROOT_DIR, 'frames', video_name)
        frames_list = os.listdir(frame_path)
        frames_list = sorted(frames_list, key=lambda x: int(os.path.splitext(x)[0]))
        
        phase_path = os.path.join(ROOT_DIR, 'phases', video_name + '.txt')
        data_dict = dict()
        with open(phase_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                frame, phase = line.strip().split()
                data_dict[int(frame)] = int(phase)
        phases_list = list(data_dict.keys())

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, len(phases_list))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['video_id'] = video_name
            phase = data_dict[frame_id]
            info['phase_gt'] = int(phase)
            info['fps'] = 1
            info['original_frames'] = len(phases_list)
            info['frames'] = frame_id_ + 1
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        assert info['original_frames'] == info['frames']

        if video_name in TRAIN_NAME:
            train_pkl[video_name] = frame_infos
            TRAIN_FRAME_NUMBERS += len(phases_list)
            unique_id_train = unique_id
        elif video_name in VAL_NAME:
            val_pkl[video_name] = frame_infos
            VAL_FRAME_NUMBERS += len(phases_list)
            unique_id_val = unique_id
        elif video_name in TEST_NAME:
            test_pkl[video_name] = frame_infos
            TEST_FRAME_NUMBERS += len(phases_list)
            unique_id_test = unique_id

    train_save_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()