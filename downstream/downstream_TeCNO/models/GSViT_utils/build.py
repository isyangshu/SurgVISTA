'''
Build the EfficientViT model family
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/syangcw/SurgVISTA/downstream/downstream_TeCNO")
from models.GSViT_utils.efficientvit import EfficientViT
from timm.models.registry import register_model

EfficientViT_m0 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

def show(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    n_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("*" * 200)
    print("*" * 200)
    print("")
    print("MAE Model = %s" % str(model))
    print("")
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number of trainable params (M): %.2f' % (n_parameters_train / 1.e6))
    print("")
    print("*" * 200)
    print("*" * 200)
    
    # Printing non-trainable parameters.
    print("")
    print("Listing all parameters which have requires_grad == False:")
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"\t{name}")

    print("")
    print("*" * 200)
    print("*" * 200)
    print("")

    print('GSViT model loaded.')

@register_model
def EfficientViT_M0(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m0):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m1):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    show(model)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m2):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    show(model)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m3):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    show(model)
    if fuse:
        replace_batchnorm(model)
    return model

@register_model
def EfficientViT_M4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m4):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    show(model)
    if fuse:
        replace_batchnorm(model)
    
    return model

@register_model
def EfficientViT_M5(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m5):
    model = EfficientViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = model.state_dict()
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("evit.", "blocks") if "evit." in key else key
            checkpoint_model[new_key] = value
        checkpoint = checkpoint_model
        checkpoint_model = dict()
        for key, value in checkpoint.items():
            new_key = key.replace("blocks0", "patch_embed") if "blocks0" in key else key
            checkpoint_model[new_key] = value
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    show(model)
    if fuse:
        replace_batchnorm(model)
    return model

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

if __name__ == "__main__":
    model = EfficientViT_M5(num_classes=2048, pretrained="/home/syangcw/SurgVISTA/downstream/downstream_TeCNO/surgical_pretrain_params/GSViT.pkl")