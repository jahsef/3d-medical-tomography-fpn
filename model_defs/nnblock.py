import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import logging
import numpy as np


def check_tensor(name, tensor, related_tensors=None):
    if torch.isnan(tensor).any():
        print(f"[ASSERT FAIL] {name} has NaNs")
        print(f"{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}")
        if related_tensors:
            for rel_name, rel_tensor in related_tensors.items():
                print(f"{rel_name} stats: min={rel_tensor.min().item()}, max={rel_tensor.max().item()}, mean={rel_tensor.mean().item()}, std={rel_tensor.std().item()}")
        sys.stdout.flush()
        raise AssertionError(f"{name} has NaNs")

def get_norm_layer(num_channels, target_channels_per_group=4, min_groups=4):
    target_group_size = max(target_channels_per_group, num_channels // min(max(num_channels // target_channels_per_group, min_groups), num_channels))
    num_groups = None

    for g in range(target_group_size, 0, -1):
        if num_channels % g == 0:
            num_groups = g
            break

    if num_groups is None or num_groups > num_channels or num_groups < 1:
        num_groups = 1
    if num_groups == 1:
        print(f'WARNING get_norm_layer returned 1 failsafe for {num_channels} channels')
    return nn.GroupNorm(num_groups, num_channels)

def get_activation(activation_type, inplace:bool = True):
    if activation_type == 'relu':
        return nn.ReLU(inplace= inplace)
    elif activation_type == 'silu':
        return nn.SiLU(inplace = inplace)
    else:
        raise NotImplementedError('no activation support')


class BasicFCBlock(nn.Module):
    def __init__(self, in_features, out_features, p):
        
        """fc, bn, silu, dropout"""
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features,out_features, bias = False),
            nn.BatchNorm1d(out_features),
            nn.SiLU(inplace= True),
            nn.Dropout(p = p, inplace=True),
        )
        
    def forward(self,x):
        return self.block(x)


class PreActResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, target_channels_per_group=4, drop_path_p=0):
        """padding will always be dynamic to keep spatial size the same"""
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        #3:1, 5:2, 7:3

        self.features = nn.Sequential(
            get_norm_layer(in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=False),

            get_norm_layer(out_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),
        )

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                get_norm_layer(out_channels, target_channels_per_group),
            )
        else:
            self.skip = nn.Identity()

        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.features(x)) + self.skip(x)  

#essentially the downchanneling block but without the out channels flexibility and uses kenrel 3
class PreActRefinementBlock3d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, drop_path_p=0):
        """fullfat3d conv for refinement, only 1 conv"""
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        #3:1, 5:2, 7:3

        self.features = nn.Sequential(
            get_norm_layer(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, stride=1, bias=False),
        )

        self.skip = nn.Identity()
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()
        
    def forward(self, x):
        return self.drop_path(self.features(x)) + self.skip(x)  


class PreActResBottleneckBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, target_channels_per_group=4, bottleneck_ratio=4):
        """Standard 1x1x1 -> 3x3x3 -> 1x1x1 bottleneck block"""
        super().__init__()

        mid_channels = out_channels // bottleneck_ratio

        self.features = nn.Sequential(
            get_norm_layer(in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),

            get_norm_layer(mid_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, stride=stride, bias=False),

            get_norm_layer(mid_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False),
        )

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                get_norm_layer(out_channels, target_channels_per_group),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)

class PreActGroupPointBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False
            ),
            get_norm_layer(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
        )
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_layer(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)
    
class PreActDownchannel3d(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_p=0):
        """1x1 kernel preact wrapped block for downchanneling with normalized projection skip"""
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),
        )
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_layer(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity()
    
    def forward(self, x):
        return self.drop_path(self.features(x)) + self.skip(x)  

 
class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1,bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1,bias = True),
        )

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        pooled = self.pool(x)
        scale = F.sigmoid(torch.clamp(self.fc(pooled),min = 1e-3, max = 1000))
        scale = torch.clamp(scale, min=1e-3, max=1.0)  # prevent zeros



        result = x * scale

        return result


class DropPath(nn.Module):
    """
    Stochastic Depth (drop entire residual paths with given probability).
    Randomly drops entire residual branches during training to improve regularization.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
	
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class PatchEmbedding(nn.Module):
    """
    Patchify images into non-overlapping or overlapping patches and project to embedding dimension.

    Args:
        patch_size: Size of each patch (single int for square patches)
        embed_dim: Output embedding dimension
        stride: Stride for patch extraction (stride=patch_size for no overlap)
        in_channels: Number of input channels (default 3 for RGB)
    """
    def __init__(self, in_channels, patch_size, embed_dim, stride =None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = patch_size if stride == None else stride
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            bias=False
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        x = self.proj(x)
        # Flatten spatial dimensions: [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        B, C, H, W = x.shape
        x = x.flatten(2)
        # Transpose to sequence format: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block for (B, N, D) embeddings.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default 0.0)
        bias: Whether to use bias in qkv projection (default True)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        # qkv: [B, N, 3*D] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #windows has shit flash attention support, just use this to have cleaner code tbh
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Reshape: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, D]
        x = x.transpose(1, 2).reshape(B, N, D)

        # Output projection
        x = self.proj(x)


        return x