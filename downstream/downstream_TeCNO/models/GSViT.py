import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
from datetime import datetime

import sys, os
from pathlib import Path
from timm.models.layers import trunc_normal_
from timm.models import create_model
from models.GSViT_utils import build

class GSViT(nn.Module):
    def __init__(self, hparams):
        super(GSViT, self).__init__()

        self.model = create_model(
            hparams.GSViT_model,
            num_classes=hparams.nb_classes,
            pretrained=hparams.ckpt,
            distillation=False,
            fuse=False,
        )
        
        print('GSViT model loaded.')

        self.fc_phase = nn.Linear(hparams.nb_classes, hparams.out_features)

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        return out_stem, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        GSViT_specific_args = parser.add_argument_group(
            title='GSViT specific args options')

        # Model parameters
        GSViT_specific_args.add_argument("--GSViT_model",
                                            default="EfficientViT_M5",
                                            choices=["EfficientViT_M0", "EfficientViT_M1", "EfficientViT_M2", "EfficientViT_M3", "EfficientViT_M4", "EfficientViT_M5"],
                                            help="EfficientViT architecture to use")


        # Finetuning params
        GSViT_specific_args.add_argument('--ckpt', default='surgical_pretrain_params/GSViT.pkl',
                            help='Start training from an MAE checkpoint')

        GSViT_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=128)

        # Last layer parameters
        GSViT_specific_args.add_argument('--nb_classes', default=2048, type=int,
                            help='number of the features in the output of the last layer (head)') # set it to 0 to remove the last layer
        return parser

#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x