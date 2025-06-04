import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import pandas as pd
import math

def main():
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'videos/micro'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])
    TRAIN_NAME = ['train01.mp4', 'train02.mp4', 'train03.mp4', 'train04.mp4', 'train05.mp4', 'train06.mp4', 'train07.mp4', 'train08.mp4', 'train09.mp4', 'train10.mp4', 'train11.mp4', 'train12.mp4', 'train13.mp4', 'train14.mp4', 'train15.mp4', 'train16.mp4', 'train17.mp4', 'train18.mp4', 'train19.mp4', 'train20.mp4', 'train21.mp4', 'train22.mp4', 'train23.mp4', 'train24.mp4', 'train25.mp4']
    VAL_NAME = ['test01.mp4', 'test07.mp4', 'test14.mp4', 'test16.mp4', 'test19.mp4']
    TEST_NAME = ['test02.mp4', 'test03.mp4', 'test04.mp4', 'test05.mp4', 'test06.mp4', 'test08.mp4', 'test09.mp4', 'test10.mp4', 'test11.mp4', 'test12.mp4', 'test13.mp4', 'test15.mp4', 'test17.mp4', 'test18.mp4', 'test20.mp4', 'test21.mp4', 'test22.mp4', 'test23.mp4', 'test24.mp4', 'test25.mp4']
    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    phase_dict = {
                    0: "Idle",
                    1: "Toric Marking",
                    2: "Implant Ejection",
                    3: "Incision",
                    4: "Viscodilatation",
                    5: "Capsulorhexis",
                    6: "Hydrodissection",
                    7: "Nucleus Breaking",
                    8: "Phacoemulsification",
                    9: "Vitrectomy",
                    10: "Irrigation/Aspiration",
                    11: "Preparing Implant",
                    12: "Manual Aspiration",
                    13: "Implantation",
                    14: "Positioning",
                    15: "OVD Aspiration",
                    16: "Suturing",
                    17: "Sealing Control",
                    18: "Wound Hydratation"
                }

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
            vid_id = int(video_name.replace('.mp4', '').replace("train", ""))
        elif video_name in VAL_NAME:
            unique_id = unique_id_val
            vid_id = int(video_name.replace('.mp4', '').replace("test", ""))
        elif video_name in TEST_NAME:
            unique_id = unique_id_test
            vid_id = int(video_name.replace('.mp4', '').replace("test", ""))

        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, 'videos/micro', video_name))
        fps = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_name in TRAIN_NAME:
            phase_path = os.path.join(ROOT_DIR, 'ground_truth/CATARACTS_2020/train_gt', video_name.replace('.mp4', '.csv'))
        elif video_name in VAL_NAME:
            phase_path = os.path.join(ROOT_DIR, 'ground_truth/CATARACTS_2020/dev_gt', video_name.replace('.mp4', '.csv'))
        elif video_name in TEST_NAME:
            phase_path = os.path.join(ROOT_DIR, 'ground_truth/CATARACTS_2020/test_gt', video_name.replace('.mp4', '.csv'))

        phase_file = pd.read_csv(phase_path, header=0)
        data_dict = dict(zip(phase_file.iloc[:, 0]-1, phase_file.iloc[:, 1]))

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, int(frames), int(fps))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            # info['original_frame_id'] = frame_id
            info['video_id'] = video_name.replace('.mp4', '')
            if frame_id >= len(data_dict):
                continue
            else:
                phase = data_dict[frame_id]

            phase_id = phase
            info['phase_gt'] = phase_id
            info['fps'] = 1
            info['original_frames'] = int(frames)
            info['frames'] = frame_id_ + 1
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        print(info['frames'])
        phase_path = os.path.join(ROOT_DIR, 'frames', video_name.replace('.mp4', ''))
        frame_length = len(os.listdir(phase_path))
        print(frame_length)

        if video_name in TRAIN_NAME:
            train_pkl[video_name] = frame_infos
            TRAIN_FRAME_NUMBERS += frames
            unique_id_train = unique_id
        elif video_name in VAL_NAME:
            val_pkl[video_name] = frame_infos
            VAL_FRAME_NUMBERS += frames
            unique_id_val = unique_id
        elif video_name in TEST_NAME:
            test_pkl[video_name] = frame_infos
            TEST_FRAME_NUMBERS += frames
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