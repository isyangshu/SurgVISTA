import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm


def main():
    ROOT_DIR = "/project/medimgfmod/syangcw/pretraining/GenSurgery/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    PRETRAIN_COUNT = 0
    PRETRAIN_COUNT_VIDEO = 0

    pretrain_pkl = dict()
    short_video_count_30 = 0
    short_video_count_60 = 0
    short_video_count_90 = 0
    short_video_count_120 = 0
    unique_id = 0
    video_types = {'cholestectomy', 'colectomy', 'splenectomy', 'appendectomy', 
                   'IBS', 'rectopexy', 'hernia', 'sigmoidectomy','davinci_instructions',
                   'gastrojejunostomy', 'cardiomytomy', 'rectal_prolapse',
                   'robotic_chole', 'liver_resection', 'esophagectomy',  'fundoplication', 
                   'colon_cancer', 'colorectal_disease', 'video_atlas', 'anal_surgery', 
                   'superior_mesenteric', 'gastrectomy', 'unsorted', 'hemicolectomy', 
                   'heller_myotomy', 'rectal_cancer', 'ladds_procedure'}
    for video_name in VIDEO_NAMES:
        video_path = os.path.join(ROOT_DIR, "frames", video_name)
        video_type = None
        for i in video_types:
            if i in video_name:
                video_type = i
                break
        if video_type is None:
            print(f"Video {video_name} not in video_types")
        
        video_path = os.path.join(ROOT_DIR, "frames", video_name)
        frames_list = os.listdir(video_path)
        if ".DS_Store" in frames_list:
            frames_list.remove(".DS_Store")
    #     if len(frames_list) < 30:
    #         short_video_count_30 += 1
    #     if len(frames_list) < 60:
    #         short_video_count_60 += 1
    #     if len(frames_list) < 90:
    #         short_video_count_90 += 1
    #     if len(frames_list) < 120:
    #         short_video_count_120 += 1
    # # 30: 36; 60: 122; 90: 229; 120: 315
    # print(short_video_count_30, short_video_count_60, short_video_count_90, short_video_count_120)
        frames_length = len(frames_list)
        if frames_length < 60:
            continue
        frames_list = sorted([x for x in frames_list if "png" in x])
        frame_infos = list()
        PRETRAIN_COUNT_VIDEO += 1
        for frame_id in tqdm(range(0, int(frames_length))):
            info = dict()
            info["dataset"] = "GenSurgery"
            info['type'] = video_type
            info["unique_id"] = unique_id
            info["frame_id"] = frame_id
            info["video_name"] = video_name
            info['video_id'] = video_name
            info["fps"] = 1
            info["frames"] = int(frames_length)
            frame_infos.append(info)
            unique_id += 1

        pretrain_pkl[video_name] = frame_infos
        PRETRAIN_COUNT += frames_length

    pretrain_save_dir = os.path.join(ROOT_DIR, 'labels')
    os.makedirs(pretrain_save_dir, exist_ok=True)
    with open(os.path.join(pretrain_save_dir, 'pretrain.pickle'), 'wb') as file:
        pickle.dump(pretrain_pkl, file)

    print('PRETRAIN Frams', PRETRAIN_COUNT, PRETRAIN_COUNT_VIDEO)
    # >60: 3376 videos: 2349618 frames
    # >90: 3269 videos: 2341801 frames
    # >120: 3183 videos: 2332828 frames

if __name__ == "__main__":
    main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    # file = open('/project/mmendoscope/surgical_video/Cholec80/labels/pretrain.pickle', 'rb')
    # info = pickle.load(file)
    # print(info.keys())
    # video_name = list()
    # # total_num = 0
    # for index in info.keys():
    #     num = len(info[index])
    # #     total_num += num
    #     info_final = info[index][-1]
    #     video_name.append(info_final['video_name'])
    # #     if info_final['frame_id'] != info_final['frames']-1:
    # #         print(info_final)
    # #         print('!!!!!!!!!!!!!!!!!')
    # #     total_num += num
    # print(video_name)