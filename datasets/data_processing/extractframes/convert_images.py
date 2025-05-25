import os
import shutil
import filecmp
from PIL import Image
from tqdm import tqdm

# def compare_folders(folder1, folder2):
#     # 比较文件列表
#     filelist = os.listdir(folder1)
#     if ".DS_Store" in filelist:
#         filelist.remove(".DS_Store")

#     folders_equal = filecmp.cmpfiles(
#         folder1, folder2, filelist, shallow=False
#     )
#     print(len(folders_equal[1]))
#     if folders_equal[0]:  # 文件列表完全一样
#         # 逐个比较文件内容
#         for folder, _, files in os.walk(folder1):
#             for file in files:
#                 file1_path = os.path.join(folder, file)
#                 file2_path = os.path.join(folder2, os.path.relpath(file1_path, folder1))
#                 if not filecmp.cmp(file1_path, file2_path, shallow=False):
#                     return False

#         return True  # 文件内容也完全一样

#     return False  # 文件列表不一样


def compare_folders(folder1, folder2):
    # 比较文件列表
    filelist = os.listdir(folder1)
    if ".DS_Store" in filelist:
        filelist.remove(".DS_Store")

    for file in filelist:
        file1_path = os.path.join(folder1, file)
        file2_path = os.path.join(folder2, file)
        if not filecmp.cmp(file1_path, file2_path, shallow=False):
            return False

        return True  # 文件内容也完全一样

    return False  # 文件列表不一样


def rename(folder_path):
    # 指定文件夹路径: folder_path

    # 遍历文件夹A下的所有文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                # 获取原始文件名
                old_name = os.path.join(root, file)

                # 删除第一位的0，得到新的文件名
                new_name = os.path.join(root, file[1:])

                # 重命名文件
                shutil.move(old_name, new_name)


def reordering(folder_path):
    # 初始化计数器

    # 遍历文件夹A下的所有文件夹
    for root, dirs, files in os.walk(folder_path):
        counter = 0
        files = sorted(files)
        for file in files:
            if file.endswith(".png"):
                # 构建新的文件名
                new_name = os.path.join(root, f"{counter:05d}.png")

                # 原始文件路径
                old_path = os.path.join(root, file)

                # 重命名文件
                shutil.move(old_path, new_name)

                # 更新计数器
                counter += 1


def counting(folder_path):
    # 初始化计数器
    counter = 0
    # 遍历文件夹A下的所有文件夹
    videos = sorted(os.listdir(folder_path))
    for video in videos:
        if video == ".DS_Store":
            continue
        files = os.listdir(os.path.join(folder_path, video))
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        counter += len(files)
    print(counter)


def comparing(folder_A, folder_B):
    # 获取文件夹 A 中每个子文件夹的图像数量
    image_counts_A = {}
    for subfolder_A in os.listdir(folder_A):
        if subfolder_A == ".DS_Store":
            continue
        subfolder_A_path = os.path.join(folder_A, subfolder_A)
        for sub_subfolder_A in os.listdir(subfolder_A_path):
            if sub_subfolder_A == ".DS_Store":
                continue
            sub_subfolder_A_path = os.path.join(subfolder_A_path, sub_subfolder_A)
            if os.path.isdir(sub_subfolder_A_path):
                image_counts_A[subfolder_A + "/" + sub_subfolder_A] = len(
                    os.listdir(sub_subfolder_A_path)
                )
    # image_counts_A = {}
    # for folder_A_item in os.listdir(folder_A):
    #     folder_A_item_path = os.path.join(folder_A, folder_A_item)
    #     if os.path.isdir(folder_A_item_path):
    #         image_counts_A[folder_A_item] = len(os.listdir(folder_A_item_path))
    print(image_counts_A)

    # 获取文件夹 B 中每个文件夹的图像数量
    image_counts_B = {}
    for folder_B_item in os.listdir(folder_B):
        folder_B_item_path = os.path.join(folder_B, folder_B_item)
        if os.path.isdir(folder_B_item_path):
            image_counts_B[folder_B_item] = len(os.listdir(folder_B_item_path))
    print(image_counts_B)

    # 比较图像数量并进行匹配
    matches = []
    for subfolder_A, count_A in image_counts_A.items():
        for subfolder_B, count_B in image_counts_B.items():
            if count_A == count_B:
                matches.append([subfolder_A, subfolder_B])
    print(matches)

    for k in matches:
        subfolder_A, subfolder_B = k
        print(subfolder_A, subfolder_B)
        subfolder_A_path = os.path.join(folder_A, subfolder_A)
        subfolder_B_path = os.path.join(folder_B, subfolder_B)
        print(subfolder_A_path, subfolder_B_path)
        if compare_folders(subfolder_A_path, subfolder_B_path):
            print("两个文件夹的所有图像完全一样")
        else:
            print("两个文件夹的图像不完全一样")


def deleting(folder_path):
    sub_folder_path = sorted(os.listdir(folder_path))
    sub_folder_path.remove(".DS_Store")
    for sub_folder in sub_folder_path:
        sub_folder = os.path.join(folder_path, sub_folder)
        files = os.listdir(sub_folder)
        print(sub_folder, len(files))
        remove_files = []
        for file_name in tqdm(files):
            # 检查文件名是否符合特定规则
            if file_name.count(".") == 2:
                remove_files.append(file_name)
                # 删除文件
                file_path = os.path.join(sub_folder, file_name)
                os.remove(file_path)
        print(len(remove_files), len(files) // 2)


def is_monochrome(image):
    """判断图像是否为纯色"""
    # 转换为灰度图像
    gray_image = image.convert('L')
    # 获取图像的像素值列表
    pixels = list(gray_image.getdata())
    # 计算像素值差异超过阈值的像素数量
    diff_count = sum(1 for pixel in pixels if abs(pixel - pixels[0]) == 0)
    # 如果直方图中只有一个非零的像素计数，则为纯色图像
    return diff_count == len(pixels)


def deleting_null(folder_path):
    sub_folder_path = sorted(os.listdir(folder_path))
    sub_folder_path.remove(".DS_Store")
    # sub_folder_path = ["BBP04"]
    for sub_folder in sub_folder_path:
        sub_folder = os.path.join(folder_path, sub_folder)
        files = sorted(os.listdir(sub_folder))
        print(sub_folder, len(files))
        remove_files = []
        for file_name in tqdm(files):
            # 检查文件是否符合特定规则
            file_path = os.path.join(sub_folder, file_name)
            image = Image.open(file_path)
            if is_monochrome(image):
                remove_files.append(file_name)
                os.remove(file_path)
        print(len(remove_files))
        print(remove_files)


def dividing(fold_path):
    ROOT_DIR = "/Users/yangshu/Downloads/AutoLaparo"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

# deleting_null("/Users/yangshu/Documents/StrasBypass70/frames")
# counting("/Users/yangshu/Documents/StrasBypass70/frames")
# reordering("/Users/yangshu/Documents/StrasBypass70/frames")
# counting("/Users/yangshu/Documents/StrasBypass70/frames")
dividing("/Users/yangshu/Downloads/endoscapes/")