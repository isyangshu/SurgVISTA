import pandas as pd
from torch.utils.data import Dataset
import pprint, pickle
from pathlib import Path
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
import os
import torch

class CATARACT101FeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode  # frame/video
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.cholec_root_dir = Path(self.hparams.data_root +"/Cataract101")  # videos splitted in images
        self.transformations = self.__get_transformations()
        weights = [1.0] * 10
        self.class_weights = np.asarray(weights)
        self.label_col = "class"
        self.df = {}
        train_pkl = pd.read_pickle(self.cholec_root_dir / "labels_pkl/train/1fpstrain_fold1.pickle")
        val_pkl = pd.read_pickle(self.cholec_root_dir / "labels_pkl/val/1fpsval_fold1.pickle")
        test_pkl = pd.read_pickle(self.cholec_root_dir / "labels_pkl/test/1fpstest_fold1.pickle")

        count = 1
        for split, pkl in zip(["train", "val", "test"], [train_pkl, val_pkl, test_pkl]):
            self.df[split] = {}
            for i in pkl.keys():
                self.df[split][str(count)] = []
                for j in pkl[i]:
                    j['video_name'] = j['video_id']
                    j['video_id'] = count
                    self.df[split][str(count)].append(j)
                count += 1
       
        self.df["all"] = {**self.df["train"], **self.df["val"], **self.df["test"]}
 
        self.vids_for_training = [i for i in range(1, 51)]
        self.vids_for_val = [i for i in range(51, 61)]
        self.vids_for_test = [i for i in range(61, 101)]

        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.df["test"] = self.df["all"]
            self.vids_for_test = [i for i in range(1, 101)]
    
        # self.df["train"] = {list(self.df["train"].keys())[0]: self.df["train"][list(self.df["train"].keys())[0]]}
        # self.df["val"] = {list(self.df["val"].keys())[0]: self.df["val"][list(self.df["val"].keys())[0]]}
        # self.df["test"] = {list(self.df["test"].keys())[0]: self.df["test"][list(self.df["test"].keys())[0]]}
        # self.vids_for_training = [i for i in range(1, 2)]
        # self.vids_for_val = [i for i in range(1, 2)]
        # self.vids_for_test = [i for i in range(1, 2)]

        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"]),
            "all": len(self.df["all"])
        }
        self.data = {}

        if self.dataset_mode == "img":
            for split in ["train", "val", "test"]:
                self.data[split] = Dataset_from_Dataframe(
                    self.df[split],
                    self.transformations[split],
                    img_root=self.cholec_root_dir / "frames",
                    split=split)

        if self.dataset_mode == "vid":
            for split in ["train", "val", "test"]:
                self.data[split] = Dataset_from_Dataframe_video_based(
                    self.df[split],
                    self.transformations[split],
                    img_root=self.cholec_root_dir / "frames",
                    split=split)

    def __get_transformations(self):
        # norm_mean = [0.3456, 0.2281, 0.2233]
        # norm_std = [0.2528, 0.2135, 0.2104]
        norm_mean=[0.485, 0.456, 0.406]
        norm_std=[0.229, 0.224, 0.225]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    def median_frequency_weights(
            self, file_list):  ## do only once and define weights in class
        frequency = [0, 0, 0, 0, 0, 0, 0]
        for i in file_list:
            frequency[int(i[1])] += 1
        median = np.median(frequency)
        weights = [median / j for j in frequency]
        return weights

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cataract101_specific_args = parser.add_argument_group(
            title='Cataract101 specific args options')
        cataract101_specific_args.add_argument(
            "--dataset_mode",
            default='img',
            choices=[
                'vid', 'img'
            ])
        cataract101_specific_args.add_argument("--test_extract", action="store_true")
        return parser


class Dataset_from_Dataframe_video_based(Dataset):
    """simple datagenerator from pandas dataframe"""
    def __init__(self,
                 df,
                 transform,
                 img_root="",
                 split="train",
                 clip_len=16,
                 frame_sample_rate=4):
        self.df = df
        self.transform = transform
        self.img_root = img_root
        self.split = split
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.samples = self._make_dataset(self.df)
        print(f"Dataset {split} has {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_infos = self.samples[index]
        img_list = img_infos['img_path']
        videos_x = torch.zeros([len(img_list), 3, 224, 224], dtype=torch.float)
        label = torch.tensor(img_infos['phase_gt'], dtype=torch.int)
        f_video = self.load_cholec_video(img_list)
        if self.transform:
            for i in range(len(f_video)):
                videos_x[i] = self.transform(image=f_video[i])["image"]
        add_label = [self.samples[index]["video_id"], self.samples[index]["original_frame_id"], self.samples[index]["video_name"], img_list]
        videos_x = videos_x.transpose(0, 1)
        return videos_x, label, add_label

    def load_cholec_video(self, img_list):
        f_video = []
        for i in range(len(img_list)):
            im = Image.open(img_list[i])
            f_video.append(np.asarray(im, dtype=np.uint8))
        f_video = np.asarray(f_video)
        return f_video
    
    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            data_ = data
            for index, line_info in enumerate(data):
                frame_index_list = list()
                for i, _ in enumerate(range(0, self.clip_len)):
                    frame_index_list.append(index)
                    if index - self.frame_sample_rate >= 0:
                        index -= self.frame_sample_rate
                frame_index_list = frame_index_list[::-1]
                image_index_list = list()
                for i in frame_index_list:
                    img_path = os.path.join(self.img_root, line_info["video_name"], str(data_[i]["original_frame_id"]).zfill(5) + ".png")
                    # if os.path.exists(img_path):
                    #     image_index_list.append(img_path)
                    # else:
                    #     raise ValueError(f"Image {img_path} does not exist")
                    image_index_list.append(img_path)
                line_info["img_path"] = image_index_list
                frames.append(line_info)
        return frames


class Dataset_from_Dataframe(Dataset):
    def __init__(self, df, transform, img_root="", split="train"):
        self.df = df
        self.transform = transform
        self.img_root = img_root
        self.split = split
        self.samples = self._make_dataset(self.df)
        print(f"Dataset {split} has {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def load_from_path(self, index):
        img_path = self.samples[index]["img_path"]
        X = Image.open(img_path)
        X_array = np.array(X)
        phase = self.samples[index]["phase_gt"]
        return X_array, phase

    def __getitem__(self, index):
        X_array, phase = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]
        label = torch.tensor(phase, dtype=torch.int)
        X = X.type(torch.FloatTensor)
        add_label = [self.samples[index]["video_id"], self.samples[index]["original_frame_id"], self.samples[index]["video_name"]]
        return X, label, add_label
    
    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            for line_info in data:
                img_path = os.path.join(self.img_root, line_info["video_name"], str(line_info["original_frame_id"]).zfill(5) + ".png")
                line_info["img_path"] = img_path
                if os.path.exists(img_path):
                    frames.append(line_info)
                else:
                    raise ValueError(f"Image {img_path} does not exist")
        return frames

