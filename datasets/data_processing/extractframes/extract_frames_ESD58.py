import numpy as np
import os
import cv2
from tqdm import tqdm
import math

ROOT_DIR = "/project/medimgfmod/Video/ESD58"
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])
# video_ids = ["17", "19", "21", "22", "23", "24", "26", "29", "30", "37", "40", "51", "53"]
video_ids = ["26"]
FRAME_NUMBERS = 0
for video_name in VIDEO_NAMES:
    print('------------------------------------')
    video_id = video_name.split('.')[0]
    if video_id not in video_ids:
        continue
    else:
        print(video_name)
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    success=True
    count=0
    save_dir = '/project/medimgfmod/Video/ESD58/images/' + video_name.replace('.mp4', '') +'/'
    os.makedirs(save_dir, exist_ok=True)
    while success is True:
        success,image = vidcap.read()
        if success:
            if count % fps == 0:
                cv2.imwrite(save_dir + str(int(count//fps)).zfill(5) + '.png', image)
            count+=1
    vidcap.release()
    print(count)
    FRAME_NUMBERS += count

print('Total Frams', FRAME_NUMBERS)
