import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d

from .. import nnblock
from ..nnblock import check_tensor
from .utils import growth_fn as _growth_fn

from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F


class BackboneOnly(nn.Module):
    def __init__(self, base_features, growth_rate, dropout_p=0, drop_path_p=0):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_path_p = drop_path_p
        F_I = base_features
        G_R = growth_rate
        print(f"backbone only, F_I: {F_I}, G_R: {G_R}")
        # Local wrapper that captures F_I, G_R
        def gf(level):
            return _growth_fn(level, F_I, G_R)

        self.stem = nn.Conv3d(1, F_I, kernel_size=5, padding=2, bias=False)

        self.backbone = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, gf(1), stride=2, norm_type="gn", drop_path_p=drop_path_p),  # 1/2
            nnblock.PreActResBlock3d(gf(1), gf(1), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(1), gf(2), stride=2, norm_type="gn", drop_path_p=drop_path_p),  # 1/4
            nnblock.PreActResBlock3d(gf(2), gf(2), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(2), gf(3), stride=2, norm_type="gn", drop_path_p=drop_path_p),  # 1/8
            nnblock.PreActResBlock3d(gf(3), gf(3), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(3), gf(4), stride=2, norm_type="gn", drop_path_p=drop_path_p),  # 1/16
            nnblock.PreActResBlock3d(gf(4), gf(4), norm_type="gn", drop_path_p=drop_path_p),
        )
        print(f"feature progression: {F_I}, {[gf(i) for i in [1,2,3,4]]}")

        self.head = nn.Sequential(
            nnblock.get_norm_layer("gn", gf(4)),
            nn.SiLU(inplace=True),
            nn.Conv3d(gf(4), 1, kernel_size=1, bias=True)
        )
    
    
    def forward(self, x):
        input_x = x
        assert not torch.isnan(x).any(), 'x nan yo'
        x = self.stem(x)
        assert not torch.isnan(x).any(), f'stem nan: input min={input_x.min():.3f} max={input_x.max():.3f} mean={input_x.mean():.3f}'
        x = self.backbone(x)
        assert not torch.isnan(x).any(), 'backbone nan'
        x = self.head(x)
        assert not torch.isnan(x).any(), 'head nan'
        return x

    
        
    def print_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stem_params = sum(p.numel() for p in self.stem.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        print(
            f'Total trainable params: {trainable_params:,}\n'
            f'Stem params: {stem_params:,}\n'
            f'Backbone params: {backbone_params:,}\n'
            f'Head params: {head_params:,}'
        )




if __name__ == '__main__':
    model = BackboneOnly(base_features=5, growth_rate=2.47)
    model.print_params()