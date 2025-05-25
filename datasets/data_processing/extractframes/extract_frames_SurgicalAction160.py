import numpy as np
import os
import cv2
from tqdm import tqdm

ROOT_DIR = "/project/smartlab2021/syangcw/SurgicalActions160"
VIDEO_Label_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = []
for VIDEO_Label_NAME in VIDEO_Label_NAMES:
    VIDEOS = os.listdir(os.path.join(ROOT_DIR, "videos", VIDEO_Label_NAME))
    for VIDEO in VIDEOS:
        VIDEO_NAMES.append(os.path.join(VIDEO_Label_NAME, VIDEO))

FRAME_NUMBERS = 0

for video_name in VIDEO_NAMES:
    print(video_name)
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("fps", fps)
    success=True
    count=0
    save_dir = './frames/' + video_name.split('/')[1].replace('.mp4', '') +'/'
    save_dir = os.path.join(ROOT_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    while success is True:
        success,image = vidcap.read()
        if success:
            # if count % fps == 0:
            cv2.imwrite(save_dir + str(int(count)).zfill(5) + '.png', image)
            count+=1
    vidcap.release()
    # cv2.destroyAllWindows()
    print(count)
    FRAME_NUMBERS += count

print('Total Frams', FRAME_NUMBERS)
