import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm


def main():
    ROOT_DIR = "/project/mmendoscope/surgical_video/CholecT50"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "VID" in x])
    PRETRAIN_COUNT = 0
    pretrain_pkl = dict()

    unique_id = 0

    for video_name in VIDEO_NAMES:
        vid_id = int(video_name.replace("VID", ""))
        
        video_path = os.path.join(ROOT_DIR, "frames", video_name)
        frames_list = os.listdir(video_path)
        frames_list = sorted([x for x in frames_list if "png" in x])
        frames = len(frames_list)

        frame_infos = list()
        for frame_id in tqdm(range(0, int(frames))):
            info = dict()
            info["dataset"] = "CholecT50"
            info["unique_id"] = unique_id
            info["frame_id"] = frame_id
            info["video_name"] = video_name
            info["video_id"] = vid_id
            info["fps"] = 1
            info["frames"] = int(frames)
            frame_infos.append(info)
            unique_id += 1

        pretrain_pkl[vid_id] = frame_infos
        PRETRAIN_COUNT += frames

    pretrain_save_dir = os.path.join(ROOT_DIR, 'labels')
    os.makedirs(pretrain_save_dir, exist_ok=True)
    with open(os.path.join(pretrain_save_dir, 'pretrain.pickle'), 'wb') as file:
        pickle.dump(pretrain_pkl, file)

    print('PRETRAIN Frams', PRETRAIN_COUNT, unique_id)


if __name__ == "__main__":
    # main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    train_file = open('/scratch/mmendoscope/downstream/CholecT50/labels_pkl_challenge/train/1fpstrain.pickle', 'rb')
    train_info = pickle.load(train_file)
    print(train_info.keys())
    print(len(train_info.keys()))

    val_file = open('/scratch/mmendoscope/downstream/CholecT50/labels_pkl_challenge/val/1fpsval.pickle', 'rb')
    val_info = pickle.load(val_file)
    print(val_info.keys())
    print(len(val_info.keys()))

    test_file = open('/scratch/mmendoscope/downstream/CholecT50/labels_pkl_challenge/test/1fpstest.pickle', 'rb')
    test_info = pickle.load(test_file)
    print(test_info.keys())
    print(len(test_info.keys()))

    # train_path = '/scratch/mmendoscope/downstream/CholecT50/labels_pkl_challenge/train/1fpstrain.pickle'
    # val_path = "/scratch/mmendoscope/downstream/CholecT50/labels_pkl_challenge/val/1fpsval.pickle"

    # combined_train_info = {**train_info, **val_info}

    # with open(train_path, 'wb') as train_file:
    #     pickle.dump(combined_train_info, train_file)

    # with open(val_path, 'wb') as val_file:
    #     pickle.dump(test_info, val_file)

    # print(f'Combined train info saved to {train_path}')
    # print(f'Test info saved to {val_path}')