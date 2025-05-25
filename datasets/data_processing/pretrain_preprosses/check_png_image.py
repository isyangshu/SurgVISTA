import pickle
import os
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

files = ['/scratch/mmendoscope/pretraining/GenSurgery/labels/pretrain.pickle']

# def check_image(image_info):
#     image_path = os.path.join('/scratch/mmendoscope/pretraining/GenSurgery/frames', image_info['video_name'], str(image_info['frame_id']).zfill(5)+'.png')
#     if not os.path.exists(image_path):
#         print(f"File not found: {image_path}")
#     else:
#         try:
#             img = Image.open(image_path)
#             img.verify()  # Verify that it is, in fact, an image
#         except (IOError, SyntaxError) as e:
#             print(f"Bad file: {image_path}")

def process_file(file):
    video_types = {'cholestectomy', 'colectomy', 'splenectomy', 'appendectomy', 
                   'IBS', 'rectopexy', 'hernia', 'sigmoidectomy','davinci_instructions',
                   'gastrojejunostomy', 'cardiomytomy', 'rectal_prolapse',
                   'robotic_chole', 'liver_resection', 'esophagectomy',  'fundoplication', 
                   'colon_cancer', 'colorectal_disease', 'video_atlas', 'anal_surgery', 
                   'superior_mesenteric', 'gastrectomy', 'unsorted', 'hemicolectomy', 
                   'heller_myotomy', 'rectal_cancer', 'ladds_procedure'}
    
    with open(file, 'rb') as f:
        infos = pickle.load(f)
        print(len(infos))
        frame_count = 0
        type_count = {key: 0 for key in video_types}
        type_count_frames = {key: 0 for key in video_types} 
        for key in infos.keys():
            for video_type in video_types:
                if video_type in key:
                    type_count[video_type] += 1
                    type_count_frames[video_type] += len(infos[key])
                    break
            frame_count += len(infos[key])
        print(frame_count)
        print(type_count)

if __name__ == '__main__':
    
    # anal_surgery
    for file in files:
        print(f"Processing {file}")
        process_file(file)
        