import pickle
import os

# Define the paths to the three pkl files
path1 = '/project/medimgfmod/syangcw/downstream/cholec80/labels/train/1fpstrain.pickle'
path2 = '/project/medimgfmod/syangcw/downstream/cholec80/labels/val/1fpsval.pickle'
path3 = '/project/medimgfmod/syangcw/downstream/cholec80/labels/test/1fpstest_27.pickle'

# Function to load a pkl file
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load the three pkl files
data1 = load_pkl(path1)
data2 = load_pkl(path2)
data3 = load_pkl(path3)
print(len(data1), len(data2), len(data3))
# Merge the dictionaries
merged_data = {**data1, **data2, **data3}
print(len(merged_data))

print(data3.keys())

# # List of test IDs to remove
# test_ids_to_remove = ['video73', 'video77', 'video78', 'video79', 'video80']

# # Remove the specified keys from data3
# for test_id in test_ids_to_remove:
#     if test_id in data3:
#         del data3[test_id]

# print(f"Remaining keys in data3: {data3.keys()}")
# print(len(data1), len(data2), len(data3))
# # Count the number of elements in the list for each key in each dataset
# all_count = 0
# def count_elements(data, dataset_name):
#     print(f"Counts for {dataset_name}:")
#     Count = 0
#     for key, value in data.items():
#         Count += len(value)
#     print(Count)

# count_elements(data1, "data1")
# count_elements(data2, "data2")
# count_elements(data3, "data3")
# # # Save the merged dictionary to a new pkl file
# output_path = '/project/medimgfmod/syangcw/downstream/cholec80/labels/test/1fpstest_27.pickle'
# with open(output_path, 'wb') as f:
#     pickle.dump(data3, f)

# print(f"Merged dictionary saved to {output_path}")