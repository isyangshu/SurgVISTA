import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/syangcw/SurgVISTA/downstream")
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math
import torch.utils.checkpoint as checkpoint


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale=None):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    if scale is not None:
        pos = pos * scale
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False, scale_t=None):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t, scale=scale_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return torch.FloatTensor(pos_embed).unsqueeze(0)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        qkv_divided_bias=False,
        qkv_divided=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv_divided = qkv_divided
        self.qkv_bias = qkv_bias
        self.qkv_divided_bias = qkv_divided_bias
        if not self.qkv_divided:
            if qkv_bias:
                self.qkv = nn.Linear(dim, all_head_dim * 3, bias=True)
                self.q_bias = None
                self.v_bias = None
            else:
                self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
                if qkv_divided_bias:
                    self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                    self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
                else:
                    self.q_bias = None
                    self.v_bias = None
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.k = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if not self.qkv_divided:
            if self.qkv_bias:
                qkv = F.linear(input=x, weight=self.qkv.weight, bias=self.qkv.bias)
            else:
                qkv_bias=None
                if self.q_bias is not None:
                    qkv_bias = torch.cat(
                        (
                            self.q_bias,
                            torch.zeros_like(self.v_bias, requires_grad=False),
                            self.v_bias,
                        )
                    )
                qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = (qkv[0], qkv[1], qkv[2],)  # make torchscript happy (cannot use tensor as tuple)
        else:
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
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        qkv_divided_bias=False,
        qkv_divided=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            qkv_divided_bias=qkv_divided_bias,
            qkv_divided=qkv_divided,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_t = num_frames // self.tubelet_size
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_2D(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)
        x = rearrange(x, "(b t) c k -> b (t k) c", b=B)

        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False
    ).unsqueeze(0)


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_divided_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        init_scale=0.0,
        all_frames=16,
        tubelet_size=2,
        use_checkpoint=False,
        use_mean_pooling=True,
        qkv_divided=False,
        patch_embed_2d=False,
        st=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        if not patch_embed_2d:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                num_frames=all_frames,
                tubelet_size=self.tubelet_size,
            )
        else:
            self.tubelet_size = 1
            self.patch_embed = PatchEmbed_2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                num_frames=all_frames,
            )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.pos_embed.requires_grad = True
            # self.pos_embed = get_3d_sincos_pos_embed(embed_dim=embed_dim, grid_size=self.patch_embed.num_patches_h, t_size=self.patch_embed.num_patches_t)
            # self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=False)
            # self.pos_embed.requires_grad = False
        else:
            # sine-cosine positional embeddings is on the way
            # For videomae/mae
            if not st:
                self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            # For videomae-st
            else:
                self.pos_embed = get_3d_sincos_pos_embed(embed_dim=embed_dim, grid_size=self.patch_embed.num_patches_h, t_size=self.patch_embed.num_patches_t)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    qkv_divided_bias=qkv_divided_bias,
                    qkv_divided=qkv_divided,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()
        if self.pos_embed is not None:
            x = (
                x
                + self.pos_embed.expand(B, -1, -1)
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


def inflate_weight(weight_2d, time_dim, center=False):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        if len(weight_2d.shape) == 4:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        elif len(weight_2d.shape) == 5:
            weight_3d = weight_2d.repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
    return weight_3d


def init_(pretrain_path, state_dict, patch_embed_2d=False):
    checkpoint = torch.load(pretrain_path, map_location="cpu")
    if "model_state" in checkpoint.keys() and 'timesformer' in pretrain_path:
        checkpoint = checkpoint["model_state"]
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if patch_embed_2d and k in state_dict.keys() and state_dict[k].shape != checkpoint[k].shape:
                if len(state_dict[k].shape) <= 3:
                    print(f'Ignore: {k}')
                    continue
                print(f'Inflate: {k}, {state_dict[k].shape} => {checkpoint[k].shape}')
                time_dim = state_dict[k].shape[2]
                v = inflate_weight(checkpoint[k], time_dim)
            # strip `model.` prefix
            name = k[6:] if k.startswith("model") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

    elif "model" in checkpoint.keys():
        checkpoint = checkpoint["model"]

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if patch_embed_2d and k in state_dict.keys() and state_dict[k].shape != checkpoint[k].shape:
                if len(state_dict[k].shape) <= 3:
                    print(f'Ignore: {k}')
                    continue
                print(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict[k].shape}')
                time_dim = state_dict[k].shape[2]
                v = inflate_weight(checkpoint[k], time_dim)
            # strip `model.` prefix
            name = k[8:] if k.startswith("encoder") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

    elif "module" in checkpoint.keys():
        checkpoint = checkpoint["module"]

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if patch_embed_2d and k in state_dict.keys() and state_dict[k].shape != checkpoint[k].shape:
                if len(state_dict[k].shape) <= 3:
                    print(f'Ignore: {k}')
                    continue
                print(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict[k].shape}')
                time_dim = state_dict[k].shape[2]
                v = inflate_weight(checkpoint[k], time_dim)
            # strip `model.` prefix
            name = k[8:] if k.startswith("encoder") else k
            new_state_dict[name] = v
        checkpoint = new_state_dict

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

    else:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("visual.transformer"):
                name = k[19:].replace("resblocks", "blocks")
                if "c_fc" in name:
                    name = name.replace('c_fc', 'fc1')
                elif "c_proj" in name:
                    name = name.replace('c_proj', 'fc2')
                elif "ln_1" in name:
                    name = name.replace('ln_1', 'norm1')
                elif "ln_2" in name:
                    name = name.replace('ln_2', 'norm2')
                elif "in_proj" in name:
                    name = name.replace('in_proj_', 'qkv.')
                elif "out_proj" in name:
                    name = name.replace('out_proj', 'proj')
            elif k.startswith("visual"):
                name = k[7:]
                if name == "conv1.weight":
                    name = "patch_embed.proj.weight"
            elif k.startswith("encoder"):
                name = k[8:]
            else:
                name = k
            new_state_dict[name] = v
        checkpoint = new_state_dict
        if "patch_embed.proj.weight" in checkpoint.keys() and "patch_embed.proj.weight" in state_dict.keys() and checkpoint['patch_embed.proj.weight'].shape != state_dict['patch_embed.proj.weight'].shape:
            v = inflate_weight(checkpoint['patch_embed.proj.weight'], 2)
            checkpoint['patch_embed.proj.weight'] = v

        remove_list = []
        for k in state_dict.keys():
            if (
                ("head" in k or "patch_embed" in k)
                and k in checkpoint
                and k in state_dict
                and checkpoint[k].shape != state_dict[k].shape
            ):
                remove_list.append(k)
                del checkpoint[k]
        print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

    return checkpoint


@register_model
def unified_base(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_embed_2d=False,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        state_dict = model.state_dict()
        checkpoint = init_(pretrain_path, state_dict)
        utils.load_state_dict(model, checkpoint)

    return model


@register_model
def unified_base_st(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_embed_2d=False,
        st=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        state_dict = model.state_dict()
        checkpoint = init_(pretrain_path, state_dict)
        utils.load_state_dict(model, checkpoint)

    return model


@register_model
def unified_base_2D(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_embed_2d=True,
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        state_dict = model.state_dict()
        checkpoint = init_(pretrain_path, state_dict, patch_embed_2d=False)
        utils.load_state_dict(model, checkpoint)

    return model

if __name__ == "__main__":
    import utils
    from args import get_args_finetuning

    args = get_args_finetuning()[0]
    # print('================================')
    # model = unified_base(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/videomae_base_k400_1600e.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/videomae_base_ssv2_2400e.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/internvideo_base_hybrid_800e.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/videomaev2_base_ft_k710_pt_hybrid_giant.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/umt_base_k710_200e.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base_st(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/mvd_base_k400_400e.pth",
    #     qkv_bias = False,
    #     qkv_divided_bias = True,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base_2D(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/supervised_base_imagenet1k.bin",
    #     qkv_bias = True,
    #     qkv_divided_bias = False,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base_2D(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/supervised_base_imagenet21k.bin",
    #     qkv_bias = True,
    #     qkv_divided_bias = False,
    #     qkv_divided = False
    # )
    # print('================================')
    # model = unified_base_2D(
    #     pretrained=True,
    #     pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/mae_base_imagenet1k.bin",
    #     qkv_bias = True,
    #     qkv_divided_bias = False,
    #     qkv_divided = False
    # )
    print('================================')
    model = unified_base_2D(
        pretrained=True,
        pretrain_path="/home/syangcw/SurgVISTA/pretrain_params/clip_base_wit400m_in1k.bin",
        qkv_bias = True,
        qkv_divided_bias = False,
        qkv_divided = False
    )