import os

def count_images_in_subdirs(directory):
    subdir_image_counts = {}
    for subdir, _, files in os.walk(directory):
        image_count = sum(1 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
        subdir_image_counts[subdir] = image_count
    return subdir_image_counts

def compare_image_counts(dir1, dir2):
    counts1 = count_images_in_subdirs(dir1)
    counts2 = count_images_in_subdirs(dir2)
    
    all_subdirs = set(counts1.keys()).union(set(counts2.keys()))
    
    for subdir in all_subdirs:
        count1 = counts1.get(subdir, 0)
        count2 = counts2.get(subdir, 0)
        if count1 != count2:
            print(f"Mismatch in {subdir}: {count1} images in {dir1}, {count2} images in {dir2}")
        else:
            print(f"Match in {subdir}: {count1} images")

if __name__ == "__main__":
    dir1 = "/project/medimgfmod/syangcw/GenSurgery"
    dir2 = "/scratch/mmendoscope/pretraining/GenSurgery"
    compare_image_counts(dir1, dir2)