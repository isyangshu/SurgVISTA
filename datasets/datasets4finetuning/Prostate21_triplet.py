import os
import cv2
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
import pickle
from PIL import Image
from torchvision import transforms
from datasets.transforms.random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset

import datasets.transforms.video_transforms as video_transforms
import datasets.transforms.volume_transforms as volume_transforms


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def to_binary(label_list):
    outputs = []
    for label in label_list:
        label = torch.tensor(label).bool().int()
        outputs.append(label)
    return torch.tensor(outputs)

class TripletDataset_Prostate21(Dataset):
    """Load video phase recognition dataset."""

    def __init__(
        self,
        anno_path="data/Prostate21/labels_pkl/train/1fpstrain.pickle",
        data_path="data/Prostate21/",
        mode="train",  # val/test
        data_strategy="online",  # offline
        output_mode="key_frame",  # all_frame
        clip_len=16,
        frame_sample_rate=4,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
        crop_size=224,
        short_side_size=256,
        long_side_size=448,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        args=None,
    ):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.data_strategy = data_strategy
        self.output_mode = output_mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.long_side_size = long_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.args = args

        self.frame_span = self.clip_len * self.frame_sample_rate

        # Augment
        self.aug = False
        self.rand_erase = False
        if self.mode in ["train"]:
            self.aug = True
            if self.args.reprob > 0:  # default: 0.25
                self.rand_erase = True
        self.infos = pickle.load(open(self.anno_path, "rb"))
        self.dataset_samples = self._make_dataset(self.infos)

        if mode == "train":
            pass

        elif mode == "val":
            self.data_transform = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        (self.short_side_size, self.short_side_size),
                        interpolation="bilinear",
                    ),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == "test":
            self.data_resize = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        size=(short_side_size, short_side_size),
                        interpolation="bilinear",
                    ),
                ]
            )
            self.data_transform = video_transforms.Compose(
                [
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, index):
        if self.mode == "train":
            args = self.args
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames, min_frame_index = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
                frames_info["min_frame_index"],
            )

            if self.data_strategy == "online":
                buffer, triplet_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, min_frame_index
                )  # T H W C
            else:
                raise NotImplementedError("Data Strategy [{}] is not implemented".format(self.data_strategy))
        

            buffer = self._aug_frame(buffer, args)

            if self.output_mode == "key_frame":
                return (
                    buffer,
                    triplet_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                )
            else:
                raise NotImplementedError("Output Mode [{}] is not implemented".format(self.output_mode))
        
        elif self.mode == "val":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames, min_frame_index = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
                frames_info["min_frame_index"],
            )
            if self.data_strategy == "online":
                buffer, triplet_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, min_frame_index
                )  # T H W C
            else:
                raise NotImplementedError("Data Strategy [{}] is not implemented".format(self.data_strategy))

            buffer = self.data_transform(buffer)

            if self.output_mode == "key_frame":   
                return (
                    buffer,
                    triplet_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                )
            else:
                raise NotImplementedError("Output Mode [{}] is not implemented".format(self.output_mode))

        elif self.mode == "test":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames, min_frame_index = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
                frames_info["min_frame_index"],
            )
            if self.data_strategy == "online":
                buffer, triplet_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, min_frame_index
                )  # T H W C
            else:
                raise NotImplementedError("Data Strategy [{}] is not implemented".format(self.data_strategy))

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            buffer = self.data_transform(buffer)

            if self.output_mode == "key_frame":
                return (
                    buffer,
                    triplet_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                )
            else:
                raise NotImplementedError("Output Mode [{}] is not implemented".format(self.output_mode))
        else:
            raise NameError("mode {} unkown".format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.7, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            min_frame_index = infos[video_id][0]["unique_id"]
            data = infos[video_id]
            for line_info in data:
                img_path = os.path.join(
                    self.data_path,
                    "frames_resized",
                    self.mode,
                    line_info["video_id"],
                    str(line_info["frame_id"])  + ".jpg",
                )
                line_info["img_path"] = img_path
                line_info['min_frame_index'] = min_frame_index
                frames.append(line_info)
        return frames

    def _video_batch_loader(self, duration, indice, video_id, index, min_frame_index):
        # index当前帧的全局索引
        # frame_id当前帧的局部索引
        # min_frame_index当前视频的最小帧索引
        frame_index = index
        frame_sample_rate = self.frame_sample_rate
        frame_index_list = []
        for i, _ in enumerate(range(0, self.clip_len)):
            frame_index_list.append(frame_index)
            if self.frame_sample_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                frame_sample_rate = 2**i
            elif self.frame_sample_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i
            if frame_index - frame_sample_rate >= min_frame_index:
                frame_index -= frame_sample_rate
        sampled_list = sorted([i for i in frame_index_list])
        sampled_image_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                img_path = self.dataset_samples[image_index]["img_path"]
                image_name_list.append(img_path)
                image_data = Image.open(img_path)

                sampled_image_list.append(image_data)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of frame {}".format(
                        frame_id_list[num],
                        video_id,
                        image_index,
                    )
                )
        # for img, name in zip(sampled_image_list, image_name_list):
        #     print(name, img.size)
        video_data = np.stack(sampled_image_list)
        triplet_label = self.dataset_samples[index]["triplet_gt"]
        triplet_label_ = np.zeros([89])
        for i in triplet_label:
            triplet_label_[i] += 1

        return video_data, to_binary(triplet_label_), sampled_list

    def __len__(self):
        return len(self.dataset_samples)


def build_dataset(is_train, test_mode, fps):
    dataset = TripletDataset_Prostate21(
        anno_path="/project/medimgfmod/Video/Prostate21/labels_pkl/test/1fpstest.pickle",
        data_path='/project/medimgfmod/Video/Prostate21/',
        mode='test',
        data_strategy="online",
        output_mode="key_frame",
        clip_len=8,
        frame_sample_rate=4,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
        keep_aspect_ratio=True,
        crop_size=224,
        short_side_size=224,
        new_height=256,
        new_width=320,
    )

    return dataset


if __name__ == "__main__":
    dataset = build_dataset(
        is_train=False, test_mode=True, fps="1fps"
    )
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=2)
    for k in data_loader_train:
        images, gt, names = k
        break
        
