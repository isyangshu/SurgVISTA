import os
import random

# Path to the directory containing the 160 subfolders
base_dir = "path/to/your/dataset" # Change this to your dataset path

# Get the list of all subfolders
subfolders = [f.name for f in os.scandir(base_dir) if f.is_dir()]

# Shuffle the list of subfolders
random.shuffle(subfolders)

# Calculate the split index
split_index = int(len(subfolders) * 0.7)

# Split the subfolders into training and testing sets
train_subfolders = subfolders[:split_index]
test_subfolders = subfolders[split_index:]
print(len(train_subfolders), len(test_subfolders))
# Write the training subfolders to train.txt
with open('SurgicalActions160/train.txt', 'w') as train_file:
    for folder in train_subfolders:
        train_file.write(f"{folder}\n")

# Write the testing subfolders to test.txt
with open('SurgicalActions160/test.txt', 'w') as test_file:
    for folder in test_subfolders:
        test_file.write(f"{folder}\n")

print("Train and test files have been generated.")
