import os
from torchvision import transforms
from datasets.transforms.transforms import *
import datasets.transforms.video_transforms as video_transforms
from datasets.transforms.masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
from datasets.datasets4pretraining.mixed_datasets import VideoDistillation, VideoMaskedPretraining

class DataAugmentationForVideoDistillation(object):
    def __init__(self, args, num_frames=None):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleTwoResizedCrop(
            args.input_size, args.teacher_input_size, [1, .875, .75, .66]
        )
        self.transform = transforms.Compose([
            Stack(roll=False),  # roll的意思代表将图像序列反序构建
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        window_size = args.window_size if num_frames is None else (num_frames // args.tubelet_size, args.window_size[1], args.window_size[2])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                window_size, args.mask_ratio
            )

    def __call__(self, images):
        # print(type(images[0]), len(images[0]), images[0][0].size, type(images[0][0]))
        # images: list of PIL images
        process_data_0, process_data_1, labels = self.train_augmentation(images)
        # print(len(process_data_0), process_data_0[0].size)
        # print(len(process_data_1), process_data_1[0].size)
        process_data_0, _ = self.transform((process_data_0, labels))
        process_data_1, _ = self.transform((process_data_1, labels))
        return process_data_0, process_data_1, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoDistillation,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_distillation_dataset(args, num_frames=None):
    if num_frames is None:
        num_frames = args.num_frames
    transform = DataAugmentationForVideoDistillation(args, num_frames=num_frames)
    print("Pretrained Datasets", args.pretrained_datasets)
    dataset = VideoDistillation(
        root=args.data_root,
        datasets=args.pretrained_datasets,
        is_color=True,
        modality='rgb',
        num_frames=num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False,
        num_sample=args.num_sample,
    )
    print("Data Aug = %s" % str(transform))
    return dataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMaskedPretraining(
        root=args.data_root,
        datasets=args.pretrained_datasets,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        lazy_init=False
    )
    print("Data Aug = %s" % str(transform))
    return dataset