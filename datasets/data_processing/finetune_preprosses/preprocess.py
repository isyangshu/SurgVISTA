import pandas as pd
import os

# 读取CSV文件
file_path = '/scratch/syangcw/Endoscapes/all_metadata.csv'
df = pd.read_csv(file_path)
print(len(df))
grouped = df.groupby(df.columns[6])
for name, group in grouped:
    print(name)
    print(len(group))
# # 按照第一列的信息分组
# grouped = df.groupby(df.columns[0])

# # 创建一个文件夹，如果存在则不处理
# output_dir = '/scratch/syangcw/Endoscapes/labels'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 打印每个组的信息
# max = 0
# for name, group in grouped:
#     group_info = {}
#     print(f"Group name: {name}")
#     for index, row in group.iterrows():
#         info = row.to_dict()
#         vid = info['vid']
#         frame = info['frame']
#         vid_frame = f"{vid}_{frame}"
#         cvs = info['avg_cvs']
#         cvs = eval(cvs)
#         cvs_norm = 0
#         for i, cv in enumerate(cvs):
#             if cv == 0:
#                 cvs_norm += 0
#             elif cv == 0.3333333333333333 or cv == 0.6666666666666666 or cv == 1:
#                 cvs_norm += 1
#             else:
#                 print(f"Error: {cv}")
#         if max <= cvs_norm:
#             max = cvs_norm
#         group_info[vid_frame] = cvs_norm

#     output_file = os.path.join(output_dir, f"{name}.txt")
#     with open(output_file, 'w') as f:
#         for key, value in group_info.items():
#             f.write(f"{key}\t{value}\n")
# print(max)