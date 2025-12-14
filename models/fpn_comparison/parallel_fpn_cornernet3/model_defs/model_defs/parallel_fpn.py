import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d



from .. import nnblock
from ..nnblock import check_tensor
    
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F
import math
from .utils import dc_growth_fn as _dc_growth_fn, power_2_rounding, growth_fn as _growth_fn, raw_growth_fn as _raw_growth_fn


class ParallelFPN(nn.Module):

    def __init__(self, base_features, growth_rate, downchanneling_factor, dropout_p=0, drop_path_p=0):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_path_p = drop_path_p
        F_I = base_features
        G_R = growth_rate
        DC = downchanneling_factor

        # Local wrappers that capture F_I, G_R, DC
        def gf(level):
            return _growth_fn(level, F_I, G_R)

        def dcf(level):
            return _dc_growth_fn(level, F_I, G_R, DC)

        def rgf(level):
            return _raw_growth_fn(level, F_I, G_R)

        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)

        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, gf(1), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(1), gf(1), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p)
        )
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(1), gf(2), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(gf(2), gf(2), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p)
        )
        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(2), gf(3), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )
        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(3), gf(4), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(gf(4), gf(5), stride=2, kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
        )

        self.interp_r_2 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(1), dcf(1), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActRefinementBlock3d(dcf(1), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer("gn", dcf(1)),
            nnblock.SEBlock3d(dcf(1), reduction=4),
            nnblock.get_norm_layer("gn", dcf(1)),
            nn.SiLU(inplace = True),
            
        )
        self.interp_r_4 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(2), dcf(2), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActRefinementBlock3d(dcf(2), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer("gn", dcf(2)),
            nnblock.SEBlock3d(dcf(2), reduction=4),
            nnblock.get_norm_layer("gn", dcf(2)),
            nn.SiLU(inplace = True),
            
        )
        self.interp_r_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(3), dcf(3), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer("gn", dcf(3)),
            nnblock.SEBlock3d(dcf(3), reduction=4),
            nnblock.get_norm_layer("gn", dcf(3)),
            nn.SiLU(inplace = True),
        )
        self.interp_r_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(4), dcf(4), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer("gn", dcf(4)),
            nnblock.SEBlock3d(dcf(4), reduction=4),
            nnblock.get_norm_layer("gn", dcf(4)),
            nn.SiLU(inplace = True),
        )
        self.interp_r_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(gf(5), dcf(5), norm_type="gn", drop_path_p=drop_path_p),
            nnblock.get_norm_layer("gn", dcf(5)),
            nnblock.SEBlock3d(dcf(5), reduction=4),
            nnblock.get_norm_layer("gn", dcf(5)),
            nn.SiLU(inplace = True),
            
        )

        self.backbone = nn.ModuleList([self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32])
        self.decoder = nn.ModuleList([self.interp_r_2, self.interp_r_4, self.interp_r_8, self.interp_r_16, self.interp_r_32])

        feature_progression = [gf(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in model: {F_I}, {feature_progression}')

        raw_feature_progression = [rgf(i) for i in [1, 2, 3, 4, 5]]
        print(f'raw feature progression in backbone: {F_I}, {raw_feature_progression}')
        print(f'feature progression in backbone: {F_I}, {feature_progression}')
        downchannel_progression = [dcf(i) for i in [1, 2, 3, 4, 5]]
        print(f'after downchanneling : {downchannel_progression}')
        total_concat_channels = sum(downchannel_progression)
        print(f'total concat features: {total_concat_channels}')
        
        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nnblock.SEBlock3d(total_concat_channels, reduction=4),
            nnblock.PreActDownchannel3d(total_concat_channels,power_2_rounding(total_concat_channels/2), norm_type="gn", drop_path_p=drop_path_p ),
            nnblock.PreActResBlock3d(power_2_rounding(total_concat_channels/2), power_2_rounding(total_concat_channels/4), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(power_2_rounding(total_concat_channels/4), power_2_rounding(total_concat_channels/8), kernel_size=3, norm_type="gn", drop_path_p=drop_path_p),
            nnblock.PreActDownchannel3d(power_2_rounding(total_concat_channels/8),1, norm_type="gn", drop_path_p=drop_path_p ),
        )

    def forward(self, x):
        assert not torch.isnan(x).any(), 'x nan yo'
        
        stem_x = self.stem(x)
        check_tensor('stem_x', stem_x)
        enc_2_x = self.enc_2(stem_x)
        enc_4_x = self.enc_4(enc_2_x)
        enc_8_x = self.enc_8(enc_4_x)
        enc_16_x = self.enc_16(enc_8_x)
        enc_32_x = self.enc_32(enc_16_x)
        check_tensor('enc_2_x', enc_2_x)
        check_tensor('enc_4_x', enc_4_x)
        check_tensor('enc_8_x', enc_8_x)
        check_tensor('enc_16_x', enc_16_x)
        check_tensor('enc_32_x', enc_32_x)
        target_size = enc_16_x.shape[2:]
        
        interp_2_x = F.interpolate(enc_2_x, size=target_size, mode='trilinear')
        
        interp_2_x = self.interp_r_2(interp_2_x)
        # interp_2_x = torch.clamp(interp_2_x, min=-1e2, max=1e2)

        interp_4_x = F.interpolate(enc_4_x, size=target_size, mode='trilinear')
        
        interp_4_x = self.interp_r_4(interp_4_x)
        # interp_4_x = torch.clamp(interp_4_x, min=-1e2, max=1e2)

        interp_8_x = F.interpolate(enc_8_x, size=target_size, mode='trilinear')
        
        interp_8_x = self.interp_r_8(interp_8_x)
        # interp_8_x = torch.clamp(interp_8_x, min=-1e2, max=1e2)

        interp_16_x = F.interpolate(enc_16_x, size=target_size, mode='trilinear')
        
        interp_16_x = self.interp_r_16(interp_16_x) #we still interp 1/16th x still because the 1/16th 
        # interp_16_x = torch.clamp(interp_16_x, min=-1e2, max=1e2)

        interp_32_x = F.interpolate(enc_32_x, size=target_size, mode='trilinear')
        
        interp_32_x = self.interp_r_32(interp_32_x)
        # interp_32_x = torch.clamp(interp_32_x, min=-1e2, max=1e2)
        
        check_tensor('interp_2_x', interp_2_x)
        check_tensor('interp_4_x', interp_4_x)
        check_tensor('interp_8_x', interp_8_x)
        check_tensor('interp_16_x', interp_16_x)
        check_tensor('interp_32_x', interp_32_x)

        fpn_concat = torch.cat([interp_2_x, interp_4_x, interp_8_x, interp_16_x, interp_32_x], dim=1)
        check_tensor('fpn_concat', fpn_concat)

        results = self.head(fpn_concat)
        
        check_tensor("results", results, related_tensors={
            "interp_2_x": interp_2_x,
            "interp_4_x": interp_4_x,
            "interp_8_x": interp_8_x,
            "interp_16_x": interp_16_x,
            "interp_32_x": interp_32_x,
        })
        
        return results


    def print_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        stem_params = sum(p.numel() for p in self.stem.parameters() if p.requires_grad)
        enc_2_params = sum(p.numel() for p in self.enc_2.parameters() if p.requires_grad)
        enc_4_params = sum(p.numel() for p in self.enc_4.parameters() if p.requires_grad)
        enc_8_params = sum(p.numel() for p in self.enc_8.parameters() if p.requires_grad)
        enc_16_params = sum(p.numel() for p in self.enc_16.parameters() if p.requires_grad)
        enc_32_params = sum(p.numel() for p in self.enc_32.parameters() if p.requires_grad)

        interp_2_params = sum(p.numel() for p in self.interp_r_2.parameters() if p.requires_grad)
        interp_4_params = sum(p.numel() for p in self.interp_r_4.parameters() if p.requires_grad)
        interp_8_params = sum(p.numel() for p in self.interp_r_8.parameters() if p.requires_grad)
        interp_16_params = sum(p.numel() for p in self.interp_r_16.parameters() if p.requires_grad)
        interp_32_params = sum(p.numel() for p in self.interp_r_32.parameters() if p.requires_grad)

        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        total_encoder = stem_params + enc_2_params + enc_4_params + enc_8_params + enc_16_params + enc_32_params
        total_interp = interp_2_params + interp_4_params + interp_8_params + interp_16_params + interp_32_params

        print(
            f'Total trainable params: {trainable_params:,}\n'
            f'Encoder params: {total_encoder:,} | stem: {stem_params:,}, enc_2: {enc_2_params:,}, enc_4: {enc_4_params:,}, enc_8: {enc_8_params:,}, enc_16: {enc_16_params:,}, enc_32: {enc_32_params:,}\n'
            f'Interpolation blocks params: {total_interp:,} | interp_2: {interp_2_params:,}, interp_4: {interp_4_params:,}, interp_8: {interp_8_params:,}, interp_16: {interp_16_params:,}, interp_32: {interp_32_params:,}\n'
            f'Head params: {head_params:,}'
        )

if __name__ == '__main__':
    model = MotorIdentifier()
    model.print_params()