import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from joblib import Parallel, delayed


def load_frames(folder):
    """Load grayscale frames from folder, sorted by filename."""
    frames = []
    filenames = sorted(os.listdir(folder))
    for fname in tqdm(filenames):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append((fname, img))
    return frames


def compute_ssim_pair(name1, img1, name2, img2, threshold, resize_shape):
    img1_resized = cv2.resize(img1, resize_shape)
    img2_resized = cv2.resize(img2, resize_shape)
    score = ssim(img1_resized, img2_resized)
    if score >= threshold:
        return (name1, name2, score)
    return None

def match_until_threshold(frames1, frames2, threshold=0.95, match_limit=10, resize_shape=(256, 256), n_jobs=12):
    """Match frames between two lists with inner-loop parallelized over B."""
    matches = []
    for name1, img1 in tqdm(frames1, desc="Scanning Folder A"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_ssim_pair)(name1, img1, name2, img2, threshold, resize_shape)
            for name2, img2 in frames2
        )
        for result in results:
            if result is not None:
                matches.append(result)
                print(f"✅ Match: {result[0]} ↔ {result[1]} | SSIM: {result[2]:.4f}")
                if len(matches) >= match_limit:
                    print(f"\n🎯 Early stop: {match_limit} matches reached.")
                    return matches
    return matches


def save_matches_to_csv(matches, output_path="early_matches.csv"):
    """Save match results to CSV file."""
    df = pd.DataFrame(matches, columns=["Frame_A", "Frame_B", "SSIM"])
    df.to_csv(output_path, index=False)
    print(f"\n📄 Matches saved to: {output_path}")

# 78 (02)
# for video_id in ["video_02", "video_04", "video_08", "video_09", "video_11", "video_12", "video_14", "video_15", "video_16", "video_18", "video_19", "video_22", "video_23", "video_24", "video_26"]:
for video_id in ["video_15", "video_16", "video_18", "video_19", "video_22", "video_23", "video_24", "video_26"]:
# for video_id in ["video_24", "video_26"]:
    # ===== 🛠️ 用户自定义参数 =====
    folder_A = "/project/mmendoscope/surgical_video/Cholec80/frames/test/video79"
    folder_B = "/project/mmendoscope/surgical_video/M2CAI16-workflow/frames/" + video_id
    threshold = 0.95
    match_limit = 10
    resize_shape = (128, 128)

    # ===== ✅ 程序执行 =====
    if os.path.exists(folder_A) and os.path.exists(folder_B):
        print("📂 Loading frame folders...")
        print(folder_A)
        print(folder_B)
        frames_a = load_frames(folder_A)
        frames_b = load_frames(folder_B)

        print(f"🔍 Start comparing with SSIM threshold {threshold} and match limit {match_limit}...")
        matches = match_until_threshold(frames_a, frames_b, threshold=threshold, match_limit=match_limit, resize_shape=resize_shape)

        print(f"\n🔎 Total matches found: {len(matches)}")
        save_matches_to_csv(matches)
    else:
        print("Error: One or both folder paths are invalid. Please check `folder")