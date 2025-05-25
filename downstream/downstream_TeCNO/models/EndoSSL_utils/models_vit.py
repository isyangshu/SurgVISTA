import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
from typing import Tuple
import functools
import math
import matplotlib.pyplot as plt


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, W

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        in_chans=3,
        depth=12,
        num_heads=12,
        num_classes=7,
        drop_path_rate=0.2,
        qkv_bias=False,
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.qkv_bias = qkv_bias
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        ## Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=dpr[i],
                    qkv_bias=self.qkv_bias,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "norm1",
            "norm2",
            "norm",
        }

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, W = self.patch_embed(x)  # B, HW, C
        x = x + self.pos_embed.to(x.dtype)  # B, HW, C  ---> 2, 196, 768
        x = torch.cat(
            [
                self.cls_token.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def vit_large_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs,
    )

    return model

if __name__ == "__main__":
    model = vit_large_patch16()
    input = torch.randn(2, 3, 224, 224)

    output = model(input)