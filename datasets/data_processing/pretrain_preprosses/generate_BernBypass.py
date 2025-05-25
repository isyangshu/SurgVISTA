import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm


def main():
    ROOT_DIR = "/project/medimgfmod/syangcw/pretraining/BernBypass70"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "BBP" in x])
    PRETRAIN_COUNT = 0
    pretrain_pkl = dict()

    unique_id = 0

    for video_name in VIDEO_NAMES:
        vid_id = int(video_name.replace("BBP", ""))
        
        video_path = os.path.join(ROOT_DIR, "frames", video_name)
        frames_list = os.listdir(video_path)
        if ".DS_Store" in frames_list:
            frames_list.remove(".DS_Store")
        frames = len(frames_list)

        frame_infos = list()
        for frame_id in tqdm(range(0, int(frames))):
            info = dict()
            info["dataset"] = "BernBypass70"
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
    main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    # file = open('/project/medimgfmod/syangcw/pretraining/BernBypass70/labels/pretrain.pickle', 'rb')
    # info = pickle.load(file)
    # print(info.keys())
    # print(info[53])
    # total_num = 0
    # for index in info.keys():
    #     num = len(info[index])
    #     total_num += num
    #     info_final = info[index][-1]
    #     if info_final['frame_id'] != info_final['frames']-1:
    #         print(info_final)
    #         print('!!!!!!!!!!!!!!!!!')
    #     total_num += num