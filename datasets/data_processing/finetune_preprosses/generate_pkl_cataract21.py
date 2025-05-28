import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import math
import random
import csv

def main():
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    VIDEO_NUMS = sorted([int(i[5:]) for i in VIDEO_NAMES])
    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    TRAIN_NUMBERS = [f for f in range(1, 18)]
    TEST_NUMBERS = [f for f in range(18, 22)]

    train_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0

    id2phase = {0: "not_initialized", 1: "Incision", 2: "Viscoelasticum", 3: "Rhexis", 
                4: "Hydrodissektion", 5: "Phako", 6: "Irrigation-Aspiration", 7: "Kapselpolishing",
                8: "Linsenimplantation", 9: "Visco-Absaugung", 10: "Tonisieren", 11: "Antibiotikum"}
    
    phase2id = {"not_initialized": 0, "Incision": 1, "Viscoelasticum": 2, "Rhexis": 3, 
                "Hydrodissektion": 4, "Phako": 5, "Irrigation-Aspiration": 6, "Kapselpolishing": 7,
                "Linsenimplantation": 8, "Visco-Absaugung": 9, "Tonisieren": 10, "Antibiotikum": 10}

    for video_name in VIDEO_NAMES:
        video_id = video_name[5:]
        vid_id = int(video_id)
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test
        # 打开视频文件
        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, './videos/' + video_name + '.mp4'))
        # 帧率(frames per second)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps != 25:
            print(video_name, 'not at 25fps', fps)
        # 总帧数(frames)
        frame_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 打开Label文件
        phase_path = os.path.join(ROOT_DIR, 'labels', video_name + '.csv')
        with open(phase_path, mode='r') as infile:
            reader = csv.reader(infile)
            phase_dict = {rows[0]: rows[1] for rows in reader}

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, frame_length, 25)):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['original_frame_id'] = frame_id
            info['video_id'] = video_id
            info['video_name'] = video_name
            info['frames'] = (frame_length // 25) if frame_length % 25 == 0 else (frame_length // 25) + 1
            phase_id = int(phase2id[phase_dict[str(frame_id)]])
            info['phase_gt'] = phase_id
            info['phase_name'] = phase_dict[str(frame_id)]
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1

        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frame_infos)
            unique_id_train = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frame_infos)
            unique_id_test = unique_id
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl_', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels_pkl_', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl_', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)


    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()