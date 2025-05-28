import os

frame_count = 0
save_dir = '/path/to/your/video_frames_directory'  # Change this to your directory path
VIDEO_NAMES = os.listdir(save_dir)
print(len(VIDEO_NAMES))
for video_name in VIDEO_NAMES:
    print('------------------------------------')
    print(video_name)
    frames = os.listdir(os.path.join(save_dir, video_name))
    print(len(frames))
    print('------------------------------------')
    frame_count += len(frames)
print('Total Frames:', frame_count)