import os
from PIL import Image
from tqdm import tqdm

def resize_dataset(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, dirs, files in os.walk(input_folder):
        rel_path = os.path.relpath(subdir, input_folder)
        output_subdir = os.path.join(output_folder, rel_path)
        print(output_subdir)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        for file in tqdm(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)
                img = Image.open(input_path)
                img = img.resize(target_size, resample=Image.BILINEAR)

                img.save(output_path)
        print(f'Processed: {subdir} -> {output_subdir}')
        print(len(os.listdir(subdir)), len(os.listdir(output_subdir)))

if __name__ == "__main__":
    input_folder = '/path/to/your/dataset/frames'
    output_folder = '/path/to/your/dataset/frames_resized'
    resize_dataset(input_folder, output_folder, (512, 320))
