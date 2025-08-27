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


def get_optimal_groups(channels, preferred_groups=None, max_groups=8):
    """
    Determine optimal group number for grouped convolution.
    
    Args:
        channels: Number of input/output channels
        preferred_groups: Preferred number of groups (will find closest divisible)
        max_groups: Maximum number of groups to consider
    
    Returns:
        int: Optimal number of groups
    """
    if preferred_groups is None:
        # Find all divisors of channels up to max_groups
        divisors = [i for i in range(1, min(channels, max_groups) + 1) if channels % i == 0]
        # Return the largest divisor (most grouping while staying efficient)
        return max(divisors) if divisors else 1
    else:
        # Find closest divisible number to preferred_groups
        for offset in range(min(preferred_groups, channels)):
            # Try preferred_groups - offset
            if preferred_groups - offset > 0 and channels % (preferred_groups - offset) == 0:
                return preferred_groups - offset
            # Try preferred_groups + offset
            if preferred_groups + offset <= channels and channels % (preferred_groups + offset) == 0:
                return preferred_groups + offset
        # Fallback to 1 if no divisible groups found
        print(f'WARNING get_optimal_groups returned 1, depthwise basically')
        return 1

def make_gp_stack(ch, num_blocks, groups, norm_type):
    return nn.Sequential(*[
        nnblock.PreActGroupPointBlock3d(ch, ch,groups = get_optimal_groups(ch, groups), norm_type = norm_type) for _ in range(num_blocks)
    ])

def make_dsephw_stack(ch,num_blocks, norm_type):
    return nn.Sequential(*[
        nnblock.PreActDSepHWBlock3d(ch, norm_type = norm_type) for _ in range(num_blocks)
    ])

def make_asymmetric_stack(ch,num_blocks, norm_type):
    return nn.Sequential(*[
        nnblock.PreActAsymmetricBlock3d(ch, norm_type = norm_type) for _ in range(num_blocks)
    ])

class MotorIdentifier(nn.Module):
    
    def __init__(self, dropout_p=0, norm_type="gn"):
        super().__init__()
        self.dropout_p = dropout_p
        F_I = 8  # base feature count
        G_R = 2.3  # growth rate factor
        FEATURE_PROGRESSION = [20,48,100,224,496]
        DC = 2 #downchanneling factor
        print(f'F_I, G_R, DC: {F_I}, {G_R}, {DC}')
        #extra dcf only used on 1/16 and 1/32 size
        DEEP_EXTRA_DCF = 1#deep extra channel downchanneling factor
        
        GROUPS = 4
        BNR = 2#bottleneck ratio
        BB_REF = 0#backbone refinement blocks

        
        def growth_fn(level):
            # return int(F_I * (G_R ** level))  
            return FEATURE_PROGRESSION[level-1]

        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)

        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, growth_fn(1), stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActRefinementBlock3d(growth_fn(1)),
        )
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(1), growth_fn(2), stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActRefinementBlock3d(growth_fn(2)),
        )
        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(2), growth_fn(3), stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActRefinementBlock3d(growth_fn(3)),
        )
        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(3), growth_fn(4), stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActRefinementBlock3d(growth_fn(4)),
        )
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(4), growth_fn(5), stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActRefinementBlock3d(growth_fn(5)),
        )
        
        self.interp_r_2 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(1), growth_fn(1)//DC),
            nnblock.PreActRefinementBlock3d(growth_fn(1)//DC),
            nnblock.get_norm_layer("gn", growth_fn(1)//DC),
            nn.SiLU(inplace=True),
        )
        self.interp_r_4 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(2), growth_fn(2)//DC),
            nnblock.PreActRefinementBlock3d(growth_fn(2)//DC),
            nnblock.get_norm_layer("gn", growth_fn(2)//DC),
            nn.SiLU(inplace=True),
        )
        self.interp_r_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(3), growth_fn(3)//DC),
            nnblock.PreActRefinementBlock3d(growth_fn(3)//DC),
            nnblock.get_norm_layer("gn", growth_fn(3)//DC),
            nn.SiLU(inplace=True),
        )
        self.interp_r_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(4), growth_fn(4)//(DC*DEEP_EXTRA_DCF)),
            nnblock.PreActRefinementBlock3d(growth_fn(4)//(DC*DEEP_EXTRA_DCF)),
            nnblock.get_norm_layer("gn", growth_fn(4)//(DC*DEEP_EXTRA_DCF)),
            nn.SiLU(inplace=True),
        )
        self.interp_r_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(5), growth_fn(5)//(DC*DEEP_EXTRA_DCF)),
            nnblock.PreActRefinementBlock3d(growth_fn(5)//(DC*DEEP_EXTRA_DCF)),
            nnblock.get_norm_layer("gn", growth_fn(5)//(DC*DEEP_EXTRA_DCF)),
            nn.SiLU(inplace=True),
        )

        self.backbone = nn.ModuleList([self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32])
        self.decoder = nn.ModuleList([self.interp_r_2, self.interp_r_4, self.interp_r_8, self.interp_r_16, self.interp_r_32])
        
        feature_progression = [growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in model: {F_I}, {feature_progression}')

        downchannel_progression = [
            channels // DC if i < 3 else channels // (DC * DEEP_EXTRA_DCF) 
            for i, channels in enumerate(feature_progression)
        ]
        
        print(f'after downchanneling : {downchannel_progression}')
        total_concat_channels = sum(downchannel_progression)
        print(f'total concat features: {total_concat_channels}')
        
        # self.head = nn.Sequential(
        #     nn.Dropout3d(p=self.dropout_p),
        #     nn.Conv3d(total_concat_channels, total_concat_channels//2, kernel_size=1),
        #     nnblock.get_norm_layer("gn", total_concat_channels//2),
        #     nn.SiLU(inplace=True),
        #     nnblock.SEBlock3d(total_concat_channels//2, reduction=4),
        #     nnblock.PreActResBlock3d(total_concat_channels//2,total_concat_channels//4,kernel_size=3),
        #     nnblock.PreActResBlock3d(total_concat_channels//4,total_concat_channels//8,kernel_size=3),
        #     nnblock.get_norm_layer("gn", total_concat_channels // 8),
        #     nn.SiLU(inplace=True),
        #     nn.Conv3d(total_concat_channels // 8, 1, kernel_size=1)
        # )
        
        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nn.Conv3d(444, 200, kernel_size=1),
            nnblock.get_norm_layer("gn", 200),
            nn.SiLU(inplace=True),
            nnblock.SEBlock3d(200, reduction=4),
            nnblock.PreActResBlock3d(200,80,kernel_size=3),
            nnblock.PreActResBlock3d(80,24,kernel_size=3),
            nnblock.get_norm_layer("gn", 24),
            nn.SiLU(inplace=True),
            nn.Conv3d(24, 1, kernel_size=1)
        )

    def forward(self, x):
        assert not torch.isnan(x).any(), 'x nan yo'
        
        stem_x = self.stem(x)
        
        enc_2_x = self.enc_2(stem_x)
        enc_4_x = self.enc_4(enc_2_x)
        enc_8_x = self.enc_8(enc_4_x)
        enc_16_x = self.enc_16(enc_8_x)
        enc_32_x = self.enc_32(enc_16_x)

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
        
        interp_16_x = self.interp_r_16(interp_16_x)
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