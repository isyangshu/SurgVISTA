import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime
from torchvision.models import ResNet50_Weights


class ResNet50Model(nn.Module):
    def __init__(self, hparams):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(2048, hparams.out_features)

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        return out_stem, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
