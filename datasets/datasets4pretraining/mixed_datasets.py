import os
import numpy as np
from numpy.lib.function_base import disp
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("/home/syangcw/SurgVISTA")
from datasets.transforms.random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset
import datasets.transforms.video_transforms as video_transforms
import datasets.transforms.volume_transforms as volume_transforms
import pickle
import random

Unedited_Datasets = ["Cholec80", "HeiChole", "CholecT50", "AutoLaparo", "PitVis", "PSI-AVA", "M2CAI16-tool", "M2CAI16-workflow", "StrasBypass70", "BernBypass70", "GenSurgery"]

# For video distillation
class VideoDistillation(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(
        self,
        root,
        datasets,
        is_color=True,
        modality="rgb",
        num_frames=1,
        sampling_rate=1,
        transform=None,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=1,
    ):
        super(VideoDistillation, self).__init__()
        self.root = root
        self.datasets = datasets
        self.is_color = is_color
        self.modality = modality
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.skip_length = self.num_frames * self.sampling_rate
        self.temporal_jitter = temporal_jitter
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.unedited_frames = 0
        self.clip_frames = 0

        self.data_samples = []
        self.data_settings = []
        self.pre_train_count = 0

        if not self.lazy_init:
            for dataset in datasets:
                if dataset in Unedited_Datasets:
                    anno_setting = os.path.join(self.root, dataset, "labels", "pretrain.pickle")
                    clips = self._make_dataset(anno_setting)
                    if len(clips) == 0:
                        raise (
                            RuntimeError(
                                "Found 0 video clips in subfolders of: "
                                + dataset
                                + "\n"
                                "Check your data directory (opt.data-dir)."
                            )
                        )
                    self.data_settings.append(anno_setting)
                    self.data_samples += clips
                    self.unedited_frames += len(clips)
                    print(dataset,":", len(clips))
                else:
                    raise (
                        RuntimeError(
                            "Found 0 video clips in subfolders of: " + dataset + "\n"
                            "Check your data directory."
                        )
                    )
        self.pre_train_count = self.unedited_frames + self.clip_frames
        print("The datasets are used for pretraining:", self.data_settings)
        print("Total pretrain number:  ", self.pre_train_count)

    def __getitem__(self, index):
        if index < self.unedited_frames:
            frame_infos = self.data_samples[index]
            video_id, frame_id, frames = (
                frame_infos["video_id"],
                frame_infos["frame_id"],
                frame_infos["frames"],
            )
            images, sampled_list = self._video_batch_loader(
                frames, frame_id, video_id, index
            )  # T H W C
        else:
            raise IndexError("Index out of range for unedited frames")

        process_data_0, process_data_1, mask = self.transform((images, None))  # T*C,H,W
        process_data_0 = process_data_0.view(
            (self.num_frames, 3) + process_data_0.size()[-2:]
        ).transpose(
            0, 1
        )  # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data_1 = process_data_1.view(
            (self.num_frames, 3) + process_data_1.size()[-2:]
        ).transpose(
            0, 1
        )  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data_0, process_data_1, mask)

    def __len__(self):
        return len(self.data_samples)

    def _make_dataset(self, anno_setting):
        if not os.path.exists(anno_setting):
            raise (
                RuntimeError(
                    "Pickle file %s doesn't exist. Check datasets for pretraining."
                    % (anno_setting)
                )
            )
        infos = pickle.load(open(anno_setting, "rb"))
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            for line_info in data:
                if line_info["dataset"] == "Cholec80":
                    img_path = os.path.join(
                        self.root,
                        line_info["dataset"],
                        "frames/train",
                        line_info["video_name"],
                        str(line_info["frame_id"]).zfill(5) + ".png",
                    )
                else:
                    img_path = os.path.join(
                        self.root,
                        line_info["dataset"],
                        "frames",
                        line_info["video_name"],
                        str(line_info["frame_id"]).zfill(5) + ".png",
                    )
                line_info["unique_id"] += self.unedited_frames
                line_info["img_path"] = img_path
                frames.append(line_info)
        return frames

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments
            )
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments
                )
            )
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step
            )
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(
        self, directory, video_reader, duration, indices, skip_offsets
    ):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB")
                for vid, _ in enumerate(frame_id_list)
            ]
        except:
            raise RuntimeError(
                "Error occured in reading frames {} from video {} of duration {}.".format(
                    frame_id_list, directory, duration
                )
            )
        return sampled_list

    def _video_batch_loader(self, duration, indice, video_id, index):
        offset_value = index - indice
        frame_sample_rate = self.sampling_rate
        sampled_list = []
        frame_id_list = []
        for i, _ in enumerate(range(0, self.num_frames)):
            frame_id = indice
            frame_id_list.append(frame_id)
            if self.sampling_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.sampling_rate == 0:
                frame_sample_rate = 2**i
            elif self.sampling_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i
            if indice - frame_sample_rate >= 0:
                indice -= frame_sample_rate
        sampled_list = sorted([i + offset_value for i in frame_id_list])
        sampled_image_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.data_samples[image_index]["img_path"])
                path = self.data_samples[image_index]["img_path"]
                image_data = Image.open(path)
                # image_data.show()
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.frames[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = sampled_image_list
        # video_data = np.stack(sampled_image_list)

        return video_data, image_name_list
    
# For video masked pretraining
class VideoMaskedPretraining(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(
        self,
        root,
        datasets,
        num_frames=1,
        sampling_rate=1,
        transform=None,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=1,
    ):
        super(VideoMaskedPretraining, self).__init__()
        self.root = root
        self.datasets = datasets
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.skip_length = self.num_frames * self.sampling_rate
        self.temporal_jitter = temporal_jitter
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.unedited_frames = 0
        self.clip_frames = 0

        self.data_samples = []
        self.data_settings = []
        self.pre_train_count = 0

        if not self.lazy_init:
            for dataset in datasets:
                if dataset in Unedited_Datasets:
                    anno_setting = os.path.join(self.root, dataset, "labels", "pretrain.pickle")
                    clips = self._make_dataset(anno_setting)
                    if len(clips) == 0:
                        raise (
                            RuntimeError(
                                "Found 0 video clips in subfolders of: "
                                + dataset
                                + "\n"
                                "Check your data directory (opt.data-dir)."
                            )
                        )
                    self.data_settings.append(anno_setting)
                    self.data_samples += clips
                    self.unedited_frames += len(clips)
                else:
                    raise (
                        RuntimeError(
                            "Found 0 video clips in subfolders of: " + dataset + "\n"
                            "Check your data directory."
                        )
                    )
        self.pre_train_count = self.unedited_frames + self.clip_frames
        print("The datasets are used for pretraining:", self.data_settings)
        print("Total pretrain number:  ", self.pre_train_count)

    def __getitem__(self, index):
        if index < self.unedited_frames:
            frame_infos = self.data_samples[index]
            video_id, frame_id, frames = (
                frame_infos["video_id"],
                frame_infos["frame_id"],
                frame_infos["frames"],
            )
            images, sampled_list = self._video_batch_loader(
                frames, frame_id, video_id, index
            )  # T H W C
        else:
            raise IndexError("Index out of range for unedited frames")

        process_data, mask = self.transform((images, None))  # T*C,H,W
        process_data = process_data.view((self.num_frames, 3) + process_data.size()[-2:]).transpose(0, 1)
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)

    def __len__(self):
        return len(self.data_samples)

    def _make_dataset(self, anno_setting):
        if not os.path.exists(anno_setting):
            raise (
                RuntimeError(
                    "Pickle file %s doesn't exist. Check datasets for pretraining."
                    % (anno_setting)
                )
            )
        infos = pickle.load(open(anno_setting, "rb"))
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            for line_info in data:
                if line_info["dataset"] == "Cholec80":
                    img_path = os.path.join(
                        self.root,
                        line_info["dataset"],
                        "frames/train",
                        line_info["video_name"],
                        str(line_info["frame_id"]).zfill(5) + ".png",
                    )
                else:
                    img_path = os.path.join(
                        self.root,
                        line_info["dataset"],
                        "frames",
                        line_info["video_name"],
                        str(line_info["frame_id"]).zfill(5) + ".png",
                    )
                line_info["unique_id"] += self.unedited_frames
                line_info["img_path"] = img_path
                frames.append(line_info)
        return frames

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments
            )
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments
                )
            )
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step
            )
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(
        self, directory, video_reader, duration, indices, skip_offsets
    ):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB")
                for vid, _ in enumerate(frame_id_list)
            ]
        except:
            raise RuntimeError(
                "Error occured in reading frames {} from video {} of duration {}.".format(
                    frame_id_list, directory, duration
                )
            )
        return sampled_list

    def _video_batch_loader(self, duration, indice, video_id, index):
        offset_value = index - indice
        frame_sample_rate = self.sampling_rate
        sampled_list = []
        frame_id_list = []
        for i, _ in enumerate(range(0, self.num_frames)):
            frame_id = indice
            frame_id_list.append(frame_id)
            if self.sampling_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.sampling_rate == 0:
                frame_sample_rate = 2**i
            elif self.sampling_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i
            if indice - frame_sample_rate >= 0:
                indice -= frame_sample_rate
        sampled_list = sorted([i + offset_value for i in frame_id_list])
        sampled_image_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.data_samples[image_index]["img_path"])
                path = self.data_samples[image_index]["img_path"]
                image_data = Image.open(path)
                # image_data.show()
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.frames[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = sampled_image_list
        # video_data = np.stack(sampled_image_list)

        return video_data, image_name_list


def get_random_sampling_rate(max_sampling_rate, min_sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if max_sampling_rate > 0:
        assert max_sampling_rate >= min_sampling_rate
        return (
            np.random.randint(min_sampling_rate, max_sampling_rate)
            if min_sampling_rate < max_sampling_rate
            else max_sampling_rate
        )
    else:
        return min_sampling_rate


def round_integer(x, factor):
    remainder = x % factor
    if remainder > 0:
        x = x + factor - remainder
    return x


if __name__ == "__main__":
    from datasets.args import get_args
    from datasets.datasets4pretraining.datasets_pretraining import DataAugmentationForVideoDistillation

    args = get_args()
    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // 16,
        args.input_size // 16,
    )
    transform = DataAugmentationForVideoDistillation(args, num_frames=args.num_frames)
    dataset = VideoDistillation(
        root=args.data_root,
        datasets=args.datasets,
        is_color=True,
        modality="rgb",
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
    )
    print("Data Aug = %s" % str(transform))
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=2)
    for k in data_loader_train:
        images, images_teacher, label = k
        print(images.shape)
        break
