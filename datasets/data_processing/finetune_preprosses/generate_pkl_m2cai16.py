import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

def main():
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'videos'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])
    TRAIN_NUMBERS = np.arange(1,28).tolist()
    TEST_NUMBERS = np.arange(28,42).tolist()
    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0

    phase2id = {'TrocarPlacement': 0, 'Preparation': 1, 'CalotTriangleDissection': 2, 'ClippingCutting': 3, 'GallbladderDissection': 4, 
                'GallbladderPackaging': 5, 'CleaningCoagulation': 6, 'GallbladderRetraction': 7}

    for video_name in VIDEO_NAMES:
        print(video_name)
        video_id = video_name.replace('.mp4', '')
        vid_id = int(video_name.replace('.mp4', '').replace("video_", ""))
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, 'videos', video_name))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        phase_path = os.path.join(ROOT_DIR, 'phases', video_name.replace('.mp4', '.txt')).replace("video", "video_phase")
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, int(frames), 25)):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['original_frame_id'] = frame_id
            info['video_id'] = video_id
            if frame_id >= len(phase_results):
                print(video_id)
                continue
            else:
                phase = phase_results[frame_id].strip().split()
                assert int(phase[0]) == frame_id
            phase_id = phase2id[phase[1]]
            info['phase_gt'] = phase_id
            info['phase_name'] = phase[1]
            info['fps'] = 1
            info['original_frames'] = int(frames)
            info['frames'] = frame_id_ + 1
            info['phase_name'] = phase[1]
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        # print(len(frame_infos) == int(frame_infos[-1]['frames']) == int(frame_infos[-1]["frame_id"])+1)
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += frames
            unique_id_train = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += frames
            unique_id_test = unique_id

    train_save_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()