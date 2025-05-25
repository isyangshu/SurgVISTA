import os
import random

# # Path to the directory containing the 160 subfolders
# base_dir = '/project/smartlab2021/syangcw/SurgicalActions160/frames'

# # Get the list of all subfolders
# subfolders = [f.name for f in os.scandir(base_dir) if f.is_dir()]

# # Shuffle the list of subfolders
# random.shuffle(subfolders)

# # Calculate the split index
# split_index = int(len(subfolders) * 0.7)

# # Split the subfolders into training and testing sets
# train_subfolders = subfolders[:split_index]
# test_subfolders = subfolders[split_index:]
# print(len(train_subfolders), len(test_subfolders))
# # Write the training subfolders to train.txt
# with open('/project/smartlab2021/syangcw/SurgicalActions160/train.txt', 'w') as train_file:
#     for folder in train_subfolders:
#         train_file.write(f"{folder}\n")

# # Write the testing subfolders to test.txt
# with open('/project/smartlab2021/syangcw/SurgicalActions160/test.txt', 'w') as test_file:
#     for folder in test_subfolders:
#         test_file.write(f"{folder}\n")

# print("Train and test files have been generated.")


with open('/project/smartlab2021/syangcw/SurgicalActions160/train.txt', 'r') as train_file:
    train_id = train_file.readlines()
    print(len(train_id))
    count = 0
    for i in train_id:
        img_list = os.listdir(os.path.join('/project/smartlab2021/syangcw/SurgicalActions160/frames', i.strip()))
        count += len(img_list)
    print(count)

# Write the testing subfolders to test.txt
with open('/project/smartlab2021/syangcw/SurgicalActions160/test.txt', 'r') as test_file:
    test_id = test_file.readlines()
    count = 0
    for i in test_id:
        img_list = os.listdir(os.path.join('/project/smartlab2021/syangcw/SurgicalActions160/frames', i.strip()))
        count += len(img_list)
    print(count)