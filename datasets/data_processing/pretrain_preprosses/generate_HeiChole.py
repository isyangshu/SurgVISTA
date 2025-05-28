import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm


def main():
    ROOT_DIR = "/data/to/your/HeiChole"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "Hei-Chole" in x])
    PRETRAIN_COUNT = 0
    pretrain_pkl = dict()

    unique_id = 0

    for video_name in VIDEO_NAMES:
        vid_id = int(video_name.replace("Hei-Chole", ""))
        
        video_path = os.path.join(ROOT_DIR, "frames", video_name)
        frames_list = os.listdir(video_path)
        if ".DS_Store" in frames_list:
            frames_list.remove(".DS_Store")
        frames = len(frames_list)

        frame_infos = list()
        for frame_id in tqdm(range(0, int(frames))):
            info = dict()
            info["dataset"] = "HeiChole"
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