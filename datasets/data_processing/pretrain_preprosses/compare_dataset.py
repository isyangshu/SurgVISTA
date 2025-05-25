import os
from tqdm import tqdm
import cv2
if __name__ == "__main__":
    dir1 = "/project/medimgfmod/syangcw/pretraining/AutoLaparo/frames"
    # dir2 = "/project/medimgfmod/syangcw/Cholec80/frames/test"
    dir2 = "/scratch/mmendoscope/pretraining/AutoLaparo/frames"

    file_1 = os.listdir(dir1)
    file_2 = os.listdir(dir2)

    print(f"Number of files in {dir1}: {len(set(file_1))}")
    print(f"Number of files in {dir2}: {len(set(file_2))}")
    
    for i in file_1:
        if i not in file_2:
            print(f"File {i} in {dir1} but not in {dir2}")
    
    for i in file_2:
        if i not in file_1:
            print(f"File {i} in {dir2} but not in {dir1}")

    for i in tqdm(file_1):
        image_1 = sorted(os.listdir(f"{dir1}/{i}"))
        image_2 = sorted(os.listdir(f"{dir2}/{i}"))
        if len(image_1) != len(image_2):
            print(f"Mismatch in {i}: {len(image_1)} images in {dir1}, {len(image_2)} images in {dir2}")
        if len(image_1) == 0:
            print(f"No images in {i}")
            continue
        if cv2.imread(os.path.join(dir1, i, image_1[0])).shape != cv2.imread(os.path.join(dir2, i, image_2[0])).shape:
            print(f"Shape mismatch in {i}")