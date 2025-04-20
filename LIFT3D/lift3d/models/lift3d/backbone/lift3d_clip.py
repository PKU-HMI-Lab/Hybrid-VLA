import os
import pathlib
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from timm.models.layers import trunc_normal_

from lift3d.models.lift3d.model_utils.mv_utils import PCViews
from lift3d.models.lift3d.model_utils.networks import Point_PN_scan
from lift3d.models.lift3d.model_utils.util_funcs import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AdapterSuper_noout(nn.Module):
    def __init__(self, embed_dims, reduction_dims, drop_rate_adapter=0):
        super(AdapterSuper_noout, self).__init__()

        self.embed_dims = embed_dims

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        if identity is None:
            identity = x
        return out


def bilinear_interpolation_3d_to_1d(x, pos_embed):

    x_normalized = x * (
        torch.tensor([76.0], requires_grad=False, device=x.device) / (x.max() - x.min())
    )
    x_left = torch.floor(x_normalized).long()
    x_right = torch.ceil(x_normalized).long()

    x_left = x_left.clamp(0, 76)
    x_right = x_right.clamp(0, 76)

    pos_embed_left = pos_embed[x_left]
    pos_embed_right = pos_embed[x_right]

    weight_left = (x_right.float() - x_normalized).unsqueeze(-1)
    weight_right = (x_normalized - x_left.float()).unsqueeze(-1)

    interpolated_pos_embed = (
        weight_left * pos_embed_left + weight_right * pos_embed_right
    )

    return interpolated_pos_embed.squeeze()


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
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        adapter_dim=None,
        drop_rate_adapter=None,
    ):
        super().__init__()
        self.ln_1 = norm_layer(dim)
        self.ln_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(dim, mlp_hidden_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(mlp_hidden_dim, dim)),
                ]
            )
        )
        self.attn = nn.MultiheadAttention(dim, num_heads)

        self.adapter = AdapterSuper_noout(
            embed_dims=dim,
            reduction_dims=adapter_dim,
            drop_rate_adapter=drop_rate_adapter,
        )
        self.out_transform_3d = nn.Sequential(nn.BatchNorm1d(dim), nn.GELU())
        self.out_transform_1d = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(dim), nn.GELU()) for i in range(6)]
        )
        self.out_transform_2d = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(dim), nn.GELU()) for i in range(6)]
        )

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x, args=None):
        x = x + self.attention(self.ln_1(x))

        x_ffn = self.mlp(self.ln_2(x))
        x = x + x_ffn + args.scale_factor * self.adapter(x_ffn)
        return x


class Attention1(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mid_dim=12,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or mid_dim**-0.5
        self.qkv = nn.Linear(dim, mid_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(mid_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mid_dim = mid_dim

    def forward(self, x, mask=None):
        B, N, C = x.shape[0] // 128, 128, x.shape[-1]

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.mid_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = torch.where(
                mask,
                torch.tensor(-100000.0, device=mask.device),
                torch.tensor(0.0, device=mask.device),
            )
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * N, self.mid_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Lift3dCLIP(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.patch_embed = Point_PN_scan(k_neighbors=config.patchknn)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        pc_views = PCViews()
        self.get_pos_2d = pc_views.get_pos_2d

        # load original PEs from clip vision model
        current_dir = pathlib.Path(__file__).parent.parent
        self.ckpt_dir = os.path.join(current_dir, "ckpt")
        clip_ckpt_path = os.path.join(self.ckpt_dir, config.clip_ckpt_path)
        if not os.path.exists(clip_ckpt_path):
            repo_id = "jiayueru/Lift3d"
            filename = "ViT-B-32.pt"
            cache_dir = self.ckpt_dir
            clip_ckpt_path = hf_hub_download(
                repo_id=repo_id, filename=filename, cache_dir=cache_dir
            )
        clip_base_dict = torch.load(clip_ckpt_path)
        self.pos_embed_2d = clip_base_dict.state_dict()[
            "visual.positional_embedding"
        ].unsqueeze(0)
        self.pos_embed_2d.requires_grad = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=self.trans_dim, n_head=self.num_heads, attn_mask=None
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

        self.apply(self._init_weights)

        self.width = config.trans_dim
        self.patch_size = config.patch_size
        self.img_size = config.img_size
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.trans_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn((config.img_size // self.patch_size) ** 2 + 1, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)
        self._set_requires_grad(requires_grad=False)
        self.cnt = 0
        self.feature_dim = config.encoder_dims

    def _set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def bilinear_interpolation_3d_to_2d(self, x, y, pos_embed):
        grid_x = (2.0 * x / (self.img_size - 1)) - 1
        grid_y = (2.0 * y / (self.img_size - 1)) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(2)

        pos_embed_reshaped = (
            pos_embed.permute(0, 2, 1)
            .view(
                1,
                -1,
                int(self.img_size / self.patch_size),
                int(self.img_size / self.patch_size),
            )
            .repeat(grid.shape[0], 1, 1, 1)
        )
        pos_embed_reshaped = pos_embed_reshaped.to(x.device)
        interpolated_pos_embed = F.grid_sample(
            pos_embed_reshaped,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return interpolated_pos_embed.squeeze()

    def load_model_from_ckpt_mae(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            repo_id = "jiayueru/Lift3d"
            filename = "lift3d_clip_base.pth"
            cache_dir = self.ckpt_dir
            ckpt_path = hf_hub_download(
                repo_id=repo_id, filename=filename, cache_dir=cache_dir
            )
        ckpt_state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(ckpt_state_dict, strict=False)
        if incompatible.missing_keys:
            print(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        """
        Forward pass for processing point clouds and generating class tokens.

        Args:
            pts (Tensor): Input tensor of point clouds of shape [B, N, 3] or [N, 3]

        Returns:
            Tensor: The class token, [B, 768].
        """
        tokens, pos = [], []

        if len(pts.shape) == 2:
            pts = pts.unsqueeze(0).float()
        pts = pts[:, :, :3]
        batch_size = pts.shape[0]
        pts_trans = pts.clone().transpose(1, 2).contiguous()
        center, group_input_tokens = self.patch_embed(pts_trans, pts)
        group_input_tokens = group_input_tokens.transpose(1, 2)

        pos_x, pos_y, _ = self.get_pos_2d(center)
        self.patch_pos_embed_2D = self.pos_embed_2d[:, 1:]

        interpolated_pos_embed = self.bilinear_interpolation_3d_to_2d(
            pos_x, pos_y, self.patch_pos_embed_2D
        )
        interpolated_pos_embed = interpolated_pos_embed.reshape(
            center.shape[0], -1, center.shape[1], self.trans_dim
        )
        interpolated_pos_embed = interpolated_pos_embed.mean(dim=1)

        tokens.append(group_input_tokens)
        pos.append(interpolated_pos_embed)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        tokens.insert(0, cls_tokens)
        pos.insert(0, cls_pos)

        tokens = torch.cat(tokens, dim=1)
        pos = torch.cat(pos, dim=1)

        x = (tokens + pos).permute(1, 0, 2)
        for _, block in enumerate(self.resblocks):
            x = block(x)

        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x[:, 0, :]
