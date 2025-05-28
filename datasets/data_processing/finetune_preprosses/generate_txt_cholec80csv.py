import os
import random
from collections import Counter

def main():
    # Path to the directory containing the 160 subfolders
    ROOT_DIR = "path/to/your/dataset"  # Change this to your dataset path

    # Get the list of all subfolders
    subfolders = sorted([f.name for f in os.scandir(base_dir) if f.is_dir()])
    label_dir = '/path/to/your/dataset/labels.csv'
    label_file = open(label_dir, 'r')
    lines = label_file.readlines()[1:]

    label_dict = dict()
    label_count = set()
    for line in lines:
        line = line.strip().split(',')
        video_path_name = line[0] + '_' + line[1]
        label = int(line[-1])
        label_dict[video_path_name] = label
        label_count.add(label)
    print("Label:", sorted(label_count))

    train_set = []
    test_set = []
    for folder in subfolders:
        video_path_name = folder
        label = label_dict[video_path_name]
        if label == 6:
            continue
        video_id = int(video_path_name.split('_')[0].replace('video', ''))
        if video_id <= 40:
            train_set.append(str(label) + '-' + folder)
        else:
            test_set.append(str(label) + '-' + folder)
    print(len(train_set), len(test_set))
    # Write the training subfolders to train.txt
    with open('Cholec80-CVS/train.txt', 'w') as train_file:
        for folder in train_set:
            train_file.write(f"{folder}\n")

    # Write the testing subfolders to test.txt
    with open('Cholec80-CVS/test.txt', 'w') as test_file:
        for folder in test_set:
            test_file.write(f"{folder}\n")

    print("Train and test files have been generated.")

if __name__ == '__main__':
    main()