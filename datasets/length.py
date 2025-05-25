import os
import cv2
import numpy as np
import pandas as pd

folder_A = "/project/mmendoscope/surgical_video/Cholec80/frames/test/"
folder_B = "/project/mmendoscope/surgical_video/M2CAI16-workflow/frames/"

subfolders_A = {name: len(os.listdir(os.path.join(folder_A, name))) for name in sorted(os.listdir(folder_A)) if os.path.isdir(os.path.join(folder_A, name))}
subfolders_B = {name: len(os.listdir(os.path.join(folder_B, name))) for name in sorted(os.listdir(folder_B)) if os.path.isdir(os.path.join(folder_B, name))}
print(subfolders_A)
print(subfolders_B)
for name, count in subfolders_A.items():
    for name2, count2 in subfolders_B.items():
        if count == count2:
            print(f"Matching subfolder: {name} ↔ {name2} | Frame Count: {count}")