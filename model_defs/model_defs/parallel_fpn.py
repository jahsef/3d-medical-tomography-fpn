import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d


if __name__ == '__main__':
    import nnblock
    from nnblock import check_tensor
else:
    from . import nnblock
    from .nnblock import check_tensor
    
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F
import math



class MotorIdentifier(nn.Module):
    
    def __init__(self, dropout_p=0, drop_path_p=0):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_path_p = drop_path_p
        F_I = 4  # base feature count
        G_R = 2.3
        DC = 2 #downchanneling factor
        def raw_growth_fn(level):
            return int(F_I * (G_R ** level))  
        
        def power_2_rounding(raw_num):
            sqrt_val = math.sqrt(raw_num)
            rounding_factor = 2 ** int(math.log2(sqrt_val))
            return int(((raw_num + rounding_factor - 1) // rounding_factor) * rounding_factor)
        
        def growth_fn(level):
            #rounds to floor power of 2 (log2(raw))
            #raw=15 → sqrt=3.87 → log2=1.95 → floor=1 → round to 2^1=2 → 16
            #raw 256 => sqrt 16 log2 = > 4 floor round to nearest 2^4 = 16
            return power_2_rounding(raw_growth_fn(level))
        
        def dc_growth_fn(level):
            return power_2_rounding(raw_growth_fn(level)/DC)
        
        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)
        
        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, growth_fn(1), stride=2, kernel_size=3, drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(growth_fn(1), growth_fn(1), kernel_size=3, drop_path_p=drop_path_p)
        )
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(1), growth_fn(2), stride=2, kernel_size=3, drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(growth_fn(2), growth_fn(2), kernel_size=3, drop_path_p=drop_path_p)
        )

        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(2), growth_fn(3), stride=2, kernel_size=3, drop_path_p=drop_path_p),
        )

        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(3), growth_fn(4), stride=2, kernel_size=3, drop_path_p=drop_path_p),
        )
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(4), growth_fn(5), stride=2, kernel_size=3, drop_path_p=drop_path_p),
        )
        self.interp_r_2 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(1), dc_growth_fn(1), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(1)),
            nnblock.SEBlock3d(dc_growth_fn(1), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(1), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(1)),
            nn.SiLU(inplace=True),
        )
        self.interp_r_4 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(2), dc_growth_fn(2), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(2)),
            nnblock.SEBlock3d(dc_growth_fn(2), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(2), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(2)),
            nn.SiLU(inplace=True),
        )

        self.interp_r_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(3), dc_growth_fn(3), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(3)),
            nnblock.SEBlock3d(dc_growth_fn(3), reduction=4),
            nnblock.get_norm_layer(dc_growth_fn(3)),
            nn.SiLU(inplace=True),
        )
        self.interp_r_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(4), dc_growth_fn(4), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(4)),
            nnblock.SEBlock3d(dc_growth_fn(4), reduction=4),
            nnblock.get_norm_layer(dc_growth_fn(4)),
            nn.SiLU(inplace=True),
        )
        self.interp_r_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(5), dc_growth_fn(5), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(dc_growth_fn(5)),
            nnblock.SEBlock3d(dc_growth_fn(5), reduction=4),
            nnblock.get_norm_layer(dc_growth_fn(5)),
            nn.SiLU(inplace=True),
        )

        self.backbone = nn.ModuleList([self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32])
        self.decoder = nn.ModuleList([self.interp_r_2, self.interp_r_4, self.interp_r_8, self.interp_r_16, self.interp_r_32])
        
        feature_progression = [growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in model: {F_I}, {feature_progression}')        


        raw_feature_progression = [raw_growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'raw feature progression in backbone: {F_I}, {raw_feature_progression}')
        feature_progression = [growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in backbone: {F_I}, {feature_progression}')
        downchannel_progression = [dc_growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'after downchanneling : {downchannel_progression}')
        total_concat_channels = sum(downchannel_progression)
        print(f'total concat features: {total_concat_channels}')
        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nn.Conv3d(total_concat_channels, power_2_rounding(total_concat_channels/2), kernel_size=1),
            nnblock.PreActResBlock3d(power_2_rounding(total_concat_channels/2), power_2_rounding(total_concat_channels/4), kernel_size=3, drop_path_p=drop_path_p),
            nnblock.PreActResBlock3d(power_2_rounding(total_concat_channels/4), power_2_rounding(total_concat_channels/8), kernel_size=3, drop_path_p=drop_path_p),
            nnblock.get_norm_layer(power_2_rounding(total_concat_channels/8)),
            nn.SiLU(inplace=True),
            nn.Conv3d(power_2_rounding(total_concat_channels/8), 1, kernel_size=1)
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


    @torch.inference_mode()
    def inference(self, tomo_tensor, batch_size, patch_size, overlap, device, tqdm_progress:bool, sigma_scale = 1/8, mode = 'gaussian'):
        # sigmoid_model = MotorIdentifierWithSigmoid(self)
        inferer = monai.inferers.inferer.SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=batch_size, overlap=overlap, 
            mode=mode, sigma_scale=sigma_scale, device=device, 
            progress=tqdm_progress, buffer_dim=0
        )
        
        with torch.amp.autocast(device_type="cuda"):
            results = inferer(inputs=tomo_tensor, network=self)
        

        del inferer
        torch.cuda.empty_cache()
        gc.collect()
        
        return torch.sigmoid(results)
    
        
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


        
class MotorIdentifierWithSigmoid(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return torch.sigmoid(self.base_model(x))
    
    # def _get_start_indices(self,length, window_size, stride):
    #     indices = []
    #     n_windows = (length - window_size) // stride + 1
    #     for i in range(n_windows):
    #         start = i * stride
    #         indices.append(start)
    #     last_start = (n_windows - 1) * stride
    #     if last_start + window_size < length:
    #         indices.append(length - window_size)
    #     return indices

if __name__ == '__main__':
    model = MotorIdentifier()
    model.print_params()