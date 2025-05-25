import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import math
import random
import csv

if __name__ == '__main__':
    file = open('/project/mmendoscope/surgical_video/PmLR50/labels/train/1fpstrain.pickle', 'rb')
    info = pickle.load(file)
    total_num = 0
    print(info.keys())
    print(len(info.keys())) 
    map_key = dict()
    count = 0
    for ind, i in enumerate(info.keys()):
        count += len(info[i])
    print(count)
    for ind, i in enumerate(info.keys()):
        print(info[i][0])
        print(info[i][-1])

    file = open('/project/mmendoscope/surgical_video/PmLR50/labels/val/1fpsval.pickle', 'rb')
    info = pickle.load(file)
    total_num = 0
    print(info.keys())
    print(len(info.keys())) 
    map_key = dict()
    count = 0
    for ind, i in enumerate(info.keys()):
        count += len(info[i])
    print(count)

    file = open('/project/mmendoscope/surgical_video/PmLR50/labels/test/1fpstest.pickle', 'rb')
    info = pickle.load(file)
    total_num = 0
    print(info.keys())
    print(len(info.keys())) 
    map_key = dict()
    count = 0
    for ind, i in enumerate(info.keys()):
        count += len(info[i])
    print(count)