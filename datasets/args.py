import argparse

def get_args():
    parser = argparse.ArgumentParser("Surgery pre-training script", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--save_ckpt_freq", default=50, type=int)
    parser.add_argument("--update_freq", default=1, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="pretrain_masked_video_student_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--decoder_depth", default=2, type=int, help="depth of decoder")

    parser.add_argument(
        "--image_teacher_model",
        default="surgery_teacher_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of teacher model",
    )
    parser.add_argument("--image_teacher_model_ckpt_path", default="", type=str)
    parser.add_argument("--distillation_target_dim", default=768, type=int)
    parser.add_argument(
        "--distill_loss_func",
        default="SmoothL1",
        choices=["L1", "L2", "SmoothL1"],
        type=str,
    )
    parser.add_argument("--image_teacher_loss_weight", default=1.0, type=float)

    parser.add_argument(
        "--video_teacher_model",
        default="pretrain_videomae_teacher_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of teacher model",
    )
    parser.add_argument("--video_teacher_model_ckpt_path", default="", type=str)
    parser.add_argument("--video_distillation_target_dim", default=768, type=int)
    parser.add_argument(
        "--video_distill_loss_func",
        default="SmoothL1",
        choices=["L1", "L2", "SmoothL1"],
        type=str,
    )
    parser.add_argument("--video_teacher_loss_weight", default=1.0, type=float)
    parser.add_argument("--video_teacher_drop_path", default=0.0, type=float)

    parser.add_argument(
        "--teacher_input_size",
        default=224,
        type=int,
        help="videos input size for backbone",
    )
    parser.add_argument(
        "--video_teacher_input_size",
        default=224,
        type=int,
        help="videos input size for backbone",
    )

    parser.add_argument("--feat_decoder_embed_dim", default=None, type=int)
    parser.add_argument("--feat_decoder_num_heads", default=None, type=int)

    parser.add_argument("--norm_feature", action="store_true", default=False)

    parser.add_argument("--tubelet_size", default=2, type=int)

    parser.add_argument(
        "--mask_type",
        default="tube",
        choices=["random", "tube"],
        type=str,
        help="masked strategy of video tokens/patches",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="ratio of the visual tokens/patches need be masked",
    )

    parser.add_argument(
        "--input_size", default=224, type=int, help="videos input size for backbone"
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument("--use_checkpoint", action="store_true", default=False)

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        metavar="LR",
        help="learning rate (default: 1.5e-4)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=40,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.6,
        metavar="PCT",
        help="Color jitter factor (default: 0.6)",
    )
    parser.add_argument(
        "--color_jitter_hue",
        type=float,
        default=0.15,
        metavar="PCT",
        help="Color jitter Hue factor (default: 0.15)",
    )
    parser.add_argument(
        "--gray_scale",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Gray scale factor (default: 0.2)",
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )
    parser.add_argument(
        "--num_sample", type=int, default=1, help="Repeated_aug (default: 1)"
    )

    # Dataset parameters
    parser.add_argument("--data_root", default="/scratch/mmendoscope/pretraining", type=str, help="dataset path root")
    parser.add_argument(
        "--datasets",
        default=["Cholec80","GenSurgery"],
        type=list,
        help="path of dataset file list",
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--load_model", default=None, help="init from checkpoint")

    parser.add_argument("--use_cls_token", action="store_true", default=False)
    parser.add_argument(
        "--time_stride_loss",
        action="store_true",
        default=True,
        help="predict one frame per temporal stride if true",
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser.parse_args()

def get_args_finetuning():
    parser = argparse.ArgumentParser(
        "SurgVISTA fine-tuning and evaluation script for video phase recognition",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--pretrained_data",
        default="k400",
        type=str,
        metavar="parameters",
        help="Name of model to train",
    )
    parser.add_argument(
        "--pretrained_method",
        default="timesformer",
        type=str,
        metavar="parameters",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="videos input size")

    parser.add_argument(
        "--fc_drop_rate",
        type=float,
        default=0.5,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Attention dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=(0.9, 0.999),
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4/1e-3)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports, default 5",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m7-n4-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0, default 0.8.",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0, default 1.0.",
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/scratch/mmendoscope/downstream/AutoLaparo",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_data_path",
        default="/scratch/mmendoscope/downstream/cholec80",
        type=str,
        help="dataset path for evaluation",
    )
    parser.add_argument(
        "--nb_classes", default=7, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )

    parser.add_argument(
        "--data_strategy", type=str, default="online"
    )  # online/offline
    parser.add_argument(
        "--output_mode", type=str, default="key_frame"
    )  # key_frame/all_frame
    parser.add_argument("--cut_black", action="store_true")  # True/False
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--sampling_rate", type=int, default=4
    )  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
    parser.add_argument(
        "--data_set",
        default="AutoLaparo",
        choices=["Cholec80", "AutoLaparo", "Cataract101"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--data_fps",
        default="1fps",
        choices=["", "5fps", "1fps"],
        type=str,
        help="dataset",
    )

    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)

    return parser.parse_args()