import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import math
import random

def main():
    ROOT_DIR = "/scratch/mmendoscope/downstream/Cataract101/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    VIDEO_NUMS = sorted([int(i) for i in VIDEO_NAMES])
    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0
    numbers = list(range(1, 102))

    # 从 1-101 中随机抽取 60 个数字
    TRAIN_ID = random.sample(numbers, 60)

    # 从剩余的数字中随机抽取 13 个数字
    remaining_numbers = [num for num in numbers if num not in TRAIN_ID]
    VAL_ID = random.sample(remaining_numbers, 13)

    # 保留最后剩下的 28 个数字
    TEST_ID = [num for num in numbers if num not in TRAIN_ID and num not in VAL_ID]

    TRAIN_NUMBERS = sorted([VIDEO_NUMS[i-1] for i in TRAIN_ID])
    VAL_NUMBERS = sorted([VIDEO_NUMS[i-1] for i in VAL_ID])
    TEST_NUMBERS = sorted([VIDEO_NUMS[i-1] for i in TEST_ID])

    train_file = open(os.path.join(ROOT_DIR, 'labels_pkl/train/1fpstrain.pickle'), 'rb')
    val_file = open(os.path.join(ROOT_DIR, 'labels_pkl/val/1fpsval.pickle'), 'rb')
    test_file = open(os.path.join(ROOT_DIR, 'labels_pkl/test/1fpstest.pickle'), 'rb')
    train_info = pickle.load(train_file)
    val_info = pickle.load(val_file)
    test_info = pickle.load(test_file)

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()

    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0
    total_info = {**train_info, **val_info, **test_info}
    total_count = 0
    for vid_id, frame_infos in total_info.items():
        total_count += len(frame_infos)
        if int(vid_id) in TRAIN_NUMBERS:
            new_frame_infos = []
            for frame_info in frame_infos:
                frame_info['unique_id'] = unique_id_train
                unique_id_train += 1
                new_frame_infos.append(frame_info)
            train_pkl[vid_id] = new_frame_infos
            TRAIN_FRAME_NUMBERS += len(new_frame_infos)
        elif int(vid_id) in VAL_NUMBERS:
            new_frame_infos = []
            for frame_info in frame_infos:
                frame_info['unique_id'] = unique_id_val
                unique_id_val += 1
                new_frame_infos.append(frame_info)
            val_pkl[vid_id] = new_frame_infos
            VAL_FRAME_NUMBERS += len(frame_infos)
        elif int(vid_id) in TEST_NUMBERS:
            new_frame_infos = []
            for frame_info in frame_infos:
                frame_info['unique_id'] = unique_id_test
                unique_id_test += 1
                new_frame_infos.append(frame_info)
            test_pkl[vid_id] = new_frame_infos
            TEST_FRAME_NUMBERS += len(frame_infos)
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels_fold3', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels_fold3', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_fold3', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)
    print(total_count)
    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()

    # file = open('/scratch/mmendoscope/downstream/Cataract101/labels/test/1fpstest.pickle', 'rb')
    # info = pickle.load(file)
    # total_num = 0
    # print(info.keys())
    # map_key = dict()
    # for ind, i in enumerate(info.keys()):
    #     map_key[i] = ind+10
    # print(map_key)