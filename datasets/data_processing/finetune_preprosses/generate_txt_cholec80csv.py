import os
import random
from collections import Counter

def main():
    # Path to the directory containing the 160 subfolders
    base_dir = '/scratch/syangcw/Cholec80-CSV/videos'

    # Get the list of all subfolders
    subfolders = sorted([f.name for f in os.scandir(base_dir) if f.is_dir()])
    label_dir = '/scratch/syangcw/Cholec80-CSV/labels.csv'
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
    with open('/scratch/syangcw/Cholec80-CSV/train.txt', 'w') as train_file:
        for folder in train_set:
            train_file.write(f"{folder}\n")

    # Write the testing subfolders to test.txt
    with open('/scratch/syangcw/Cholec80-CSV/test.txt', 'w') as test_file:
        for folder in test_set:
            test_file.write(f"{folder}\n")

    print("Train and test files have been generated.")

if __name__ == '__main__':
    # main()
    # with open('/scratch/syangcw/Cholec80-CSV/train.txt', 'r') as train_file:
    #     train_set = train_file.readlines()
    #     train_labels = [int(folder.split('-')[0]) for folder in train_set]
    # train_label_counts = Counter(train_labels)
    # print(train_label_counts)
    # with open('/scratch/syangcw/Cholec80-CSV/test.txt', 'r') as test_file:
    #     test_set = test_file.readlines()
    #     test_labels = [int(folder.split('-')[0]) for folder in test_set]
    # test_label_counts = Counter(test_labels)
    # print(test_label_counts)

    with open('/scratch/syangcw/Cholec80-CSV/train.txt', 'r') as train_file:
        train_set = train_file.readlines()
        count = 0
        print(len(train_set))
        for i in train_set:
            name = i.split('-')[-1].strip()
            img_list = os.listdir(os.path.join('/scratch/syangcw/Cholec80-CSV/videos', name.strip()))
            count += len(img_list)
        print(count)

    with open('/scratch/syangcw/Cholec80-CSV/test.txt', 'r') as test_file:
        test_set = test_file.readlines()
        print(len(test_set))
        count_ = 0
        for i in test_set:
            name = i.split('-')[-1].strip()
            img_list = os.listdir(os.path.join('/scratch/syangcw/Cholec80-CSV/videos', name.strip()))
            count_ += len(img_list)
        print(count_)
print(len(train_set) + len(test_set))
print(count + count_)
