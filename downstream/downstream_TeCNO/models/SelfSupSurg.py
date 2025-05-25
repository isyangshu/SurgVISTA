import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime
from collections import OrderedDict as odict
from torchvision.models import ResNet50_Weights


class SelfSupSurg(nn.Module):
    def __init__(self, hparams, pretrained_path = "/home/syangcw/SurgSSL/downstream/downstream_TeCNO/surgical_pretrain_params/model_final_checkpoint_dino_surg.torch"):
        super(SelfSupSurg, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(2048, hparams.out_features)
        self.pretrained_path = pretrained_path
        self.load_ckpt(self.pretrained_path)

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        return out_stem, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        selfsupsurgmodel_specific_args = parser.add_argument_group(
            title='SelfSupSurg specific args options')
        selfsupsurgmodel_specific_args.add_argument("--pretrained", action="store_true", help="pretrained by SelfSupSurg")
        selfsupsurgmodel_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser
    
    def load_ckpt(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location="cpu")
        if "classy_state_dict" in checkpoint:
            state_dict = checkpoint["classy_state_dict"]["base_model"]["model"]["trunk"]
            state_dict = odict(
                {k.replace("_feature_blocks.", ""): v for k, v in state_dict.items()}
            )
            m, v = self.model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained model loaded. missing keys: {m}, invalid keys:{v}")
        else:
            model_dict = self.model.state_dict()
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # to exclude weights not present in ckpt
            state_dict_new = odict({})

            for k, v in state_dict.items():
                state_dict_new[k] = state_dict[k]

            model_dict.update(state_dict_new)
            m, v = self.model.load_state_dict(model_dict)
            print(f"Pretrained model loaded. missing keys: {m}, invalid keys:{v}")

#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    model = SelfSupSurg(None)
    model_original = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Compare parameters of model and model_original
    def compare_models(model1, model2):
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2 or not torch.equal(param1, param2):
                print(f"Difference found in layer: {name1}")
                return False
        return True

    if compare_models(model.model, model_original):
        print("The models have the same parameters.")
    else:
        print("The models have different parameters.")