# https://www.synapse.org/Synapse:syn21680292/wiki/601561
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import pandas as pd
import math
import re
from collections import defaultdict

def main():
    ROOT_DIR = "/scratch/syangcw/Endoscapes/"
    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0
    with open(os.path.join(ROOT_DIR, 'train_vids.txt'), 'r') as file1, \
         open(os.path.join(ROOT_DIR, 'val_vids.txt'), 'r') as file2, \
         open(os.path.join(ROOT_DIR, 'test_vids.txt'), 'r') as file3:
        content1 = file1.read()
        content2 = file2.read()
        content3 = file3.read()
        def sci_to_int(sci_str):
            return int(float(re.sub(r'[^\d\.\-eE]', '', sci_str)))

        TRAIN_NAME = [sci_to_int(num) for num in content1.split()]
        VAL_NAME = [sci_to_int(num) for num in content2.split()]
        TEST_NAME = [sci_to_int(num) for num in content3.split()]
        print(len(TRAIN_NAME), len(VAL_NAME), len(TEST_NAME))
    VIDEO_NAMES = TRAIN_NAME + VAL_NAME + TEST_NAME
    
    image_list_train = os.listdir(os.path.join(ROOT_DIR, 'train'))
    train_dict = defaultdict(list)
    for image_name in image_list_train:
        if image_name == '.DS_Store' or "annotation" in image_name:
            continue
        vid = image_name.split('_')[0]
        train_dict[vid].append(image_name)

    image_list_val = os.listdir(os.path.join(ROOT_DIR, 'val'))
    val_dict = defaultdict(list)
    for image_name in image_list_val:
        if image_name == '.DS_Store' or "annotation" in image_name:
            continue
        vid = image_name.split('_')[0]
        val_dict[vid].append(image_name)

    image_list_test = os.listdir(os.path.join(ROOT_DIR, 'test'))
    test_dict = defaultdict(list)
    for image_name in image_list_test:
        if image_name == '.DS_Store' or "annotation" in image_name:
            continue
        vid = image_name.split('_')[0]
        test_dict[vid].append(image_name)

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    labels_count = []
    for video_name in VIDEO_NAMES:
        if video_name in TRAIN_NAME:
            unique_id = unique_id_train
            vid_id = video_name
            frames = sorted(train_dict[str(video_name)])
        elif video_name in VAL_NAME:
            unique_id = unique_id_val
            vid_id = video_name
            frames = sorted(val_dict[str(video_name)])
        elif video_name in TEST_NAME:
            unique_id = unique_id_test
            vid_id = video_name
            frames = sorted(test_dict[str(video_name)])

        with open(os.path.join(ROOT_DIR, 'labels', str(video_name) + '.txt'), 'r') as file1:
            labels = file1.readlines()
            label_dict = dict()
            for label in labels:
                label = label.split()
                label_dict[label[0]] = label[1]
                if video_name in VAL_NAME:
                    labels_count.append(label[1])

        frame_infos = list()
        for frame_id in tqdm(range(len(frames))):
            info = dict()
            frame_name = frames[frame_id]
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id
            info['original_frame_id'] = frame_name
            info['video_id'] = video_name
            
            csv_score = label_dict[frame_name.replace('.jpg', '')]

            info['csv_gt'] = csv_score
            info['fps'] = 0.2
            info['frames'] = len(frames)
            frame_infos.append(info)
            unique_id += 1

        if video_name in TRAIN_NAME:
            train_pkl[str(video_name)] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames)
            unique_id_train = unique_id
        elif video_name in VAL_NAME:
            val_pkl[str(video_name)] = frame_infos
            VAL_FRAME_NUMBERS += len(frames)
            unique_id_val = unique_id
        elif video_name in TEST_NAME:
            test_pkl[str(video_name)] = frame_infos
            TEST_FRAME_NUMBERS += len(frames)
            unique_id_test = unique_id

    # train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'train')
    # os.makedirs(train_save_dir, exist_ok=True)
    # with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
    #     pickle.dump(train_pkl, file)

    # val_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'val')
    # os.makedirs(val_save_dir, exist_ok=True)
    # with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
    #     pickle.dump(val_pkl, file)

    # test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'test')
    # os.makedirs(test_save_dir, exist_ok=True)
    # with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
    #     pickle.dump(test_pkl, file)

    # print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    # print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    # print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 
    # print(len(labels_count))
    # print('Labels Count', pd.Series(labels_count).value_counts())

if __name__ == '__main__':
    # main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    file = open('/scratch/syangcw/Endoscapes/labels_pkl/train/1fpstrain.pickle', 'rb')
    info = pickle.load(file)
    print(len(info.keys()))
    count_train = []
    for key in info.keys():
        for length in info[key]:
            count_train.append(length['csv_gt'])
    count_train_series = pd.Series(count_train)
    print(count_train_series.value_counts())

    file = open('/scratch/syangcw/Endoscapes/labels_pkl/val/1fpsval.pickle', 'rb')
    info = pickle.load(file)
    print(len(info.keys()))
    count_val = []
    for key in info.keys():
        for length in info[key]:
            count_val.append(length['csv_gt'])
    count_val_series = pd.Series(count_val)
    print(count_val_series.value_counts())

    file = open('/scratch/syangcw/Endoscapes/labels_pkl/test/1fpstest.pickle', 'rb')
    info = pickle.load(file)
    print(len(info.keys()))
    count_test = []
    for key in info.keys():
        for length in info[key]:
            count_test.append(length['csv_gt'])
    count_test_series = pd.Series(count_test)
    print(count_test_series.value_counts())