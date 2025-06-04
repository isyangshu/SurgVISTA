import os
import sys
sys.path.append("/home/syangcw/SurgVISTA")

from datasets.transforms import *

from datasets.datasets4finetuning.Cholec80_phase import PhaseDataset_Cholec80
from datasets.datasets4finetuning.AutoLaparo_phase import PhaseDataset_AutoLaparo
from datasets.datasets4finetuning.Cataract101_phase import PhaseDataset_Cataract101
from datasets.datasets4finetuning.M2CAI16_phase import PhaseDataset_M2CAI16
from datasets.datasets4finetuning.CATARACTS_phase import PhaseDataset_CATARACTS
from datasets.datasets4finetuning.Cataract21_phase import PhaseDataset_Cataract21
from datasets.datasets4finetuning.PmLR50_phase import PhaseDataset_PmLR50
from datasets.datasets4finetuning.ESD57_phase import PhaseDataset_ESD57
from datasets.datasets4finetuning.SurgicalAction160_action import CLIPDataset_SurgicalAction160
from datasets.datasets4finetuning.Cholec80_CVS import CLIPDataset_Cholec80CVS
from datasets.datasets4finetuning.Endoscapes_CVS import CVSDataset_Endoscapes
from datasets.datasets4finetuning.CholecT50_triplet import TripletDataset_CholecT50
from datasets.datasets4finetuning.Prostate21_triplet import TripletDataset_Prostate21

def build_dataset(is_train, test_mode, fps, args):
    """Load video phase recognition dataset."""
    if args.data_set == "Cholec80":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test_35.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", "test", fps + "test_35.pickle")

        dataset = PhaseDataset_Cholec80(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7
    
    elif args.data_set == "AutoLaparo":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val.pickle")

        dataset = PhaseDataset_AutoLaparo(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7

    elif "Cataract101" in args.data_set:
        dataset, fold = args.data_set.split("_")
        assert dataset == "Cataract101"
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train_" + fold + ".pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test_"+ fold + ".pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val_" + fold + ".pickle")

        dataset = PhaseDataset_Cataract101(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 10
        
    elif args.data_set == "M2CAI16":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", "test", fps + "test.pickle")
        
        dataset = PhaseDataset_M2CAI16(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 8

    elif args.data_set == "CATARACTS":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", mode, fps + "val.pickle")

        dataset = PhaseDataset_CATARACTS(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 19

    elif args.data_set == "Cataract21":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train" + ".pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test" + ".pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val" + ".pickle")
        dataset = PhaseDataset_Cataract21(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            cut_black=args.cut_black,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 11

    elif args.data_set == "PmLR50":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", mode, fps + "val.pickle")
        dataset = PhaseDataset_PmLR50(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 5

    elif args.data_set == "SurgicalActions160":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "train.txt"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "test.txt"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "test.txt")
        dataset = CLIPDataset_SurgicalAction160(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 16

    elif args.data_set == "Cholec80-CVS":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "train.txt"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "test.txt"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "val.txt")

        dataset = CLIPDataset_Cholec80CVS(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 6

    elif args.data_set == "Endoscapes":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, "1fpstrain.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, "1fpstest.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, "1fpsval.pickle")
        dataset = CVSDataset_Endoscapes(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 4

    elif args.data_set == "CholecT50":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val.pickle")
        dataset = TripletDataset_CholecT50(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            long_side_size=args.long_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 100

    elif args.data_set == "Prostate21":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val.pickle")
        dataset = TripletDataset_Prostate21(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            long_side_size=args.long_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 89

    elif args.data_set == "ESD57":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", mode, fps + "val.pickle")

        dataset = PhaseDataset_ESD57(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 8

    else:
        print("Error")

    assert nb_classes == args.nb_classes
    print("%s - %s : Number of the class = %d" % (mode, fps, args.nb_classes))

    return dataset, nb_classes
