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
    from ....model_defs import nnblock
    from ....model_defs.nnblock import check_tensor
    
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F
import math


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
    
    def __init__(self, dropout_p=0, drop_path_p = 0, norm_type="gn"):
        super().__init__()
        self.dropout_p = dropout_p
        self.drop_path_p = drop_path_p
        F_I = 4  # base feature count
        # FEATURE_PROGRESSION = [18,40,96,200,420]
        G_R = 2.26
        
        
        DC = 2 #downchanneling factor
        # print(f'F_I, feature_prssioon, DC: {F_I}, {}, {DC}')
        
        def raw_growth_fn(level):
            return int(F_I * (G_R ** level))  
            # return FEATURE_PROGRESSION[level-1]
            
        def growth_fn(level):
            #rounds to floor power of 2 (log2(raw))
            #raw=15 → sqrt=3.87 → log2=1.95 → floor=1 → round to 2^1=2 → 16
            #raw 256 => sqrt 16 log2 = > 4 floor round to nearest 2^4 = 16
            raw = raw_growth_fn(level)
            sqrt_val = math.sqrt(raw)
            rounding_factor = 2 ** int(math.log2(sqrt_val))
            return ((raw + rounding_factor - 1) // rounding_factor) * rounding_factor
        
        def dc_growth_fn(level):
            raw = raw_growth_fn(level) // DC
            sqrt_val = math.sqrt(raw)
            rounding_factor = 2 ** int(math.log2(sqrt_val))
            return ((raw + rounding_factor - 1) // rounding_factor) * rounding_factor

        
        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)
        
        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, growth_fn(1), stride=2, kernel_size=3, norm_type=norm_type, drop_path_p = drop_path_p),
            nnblock.PreActRefinementBlock3d(growth_fn(1), kernel_size=3, norm_type= norm_type, drop_path_p=drop_path_p)
        )
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(1), growth_fn(2), stride=2, kernel_size=3, norm_type=norm_type, drop_path_p = drop_path_p),
            nnblock.PreActRefinementBlock3d(growth_fn(2), kernel_size=3, norm_type= norm_type, drop_path_p=drop_path_p)
        )
        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(2), growth_fn(3), stride=2, kernel_size=3, norm_type=norm_type, drop_path_p = drop_path_p),
            nnblock.PreActRefinementBlock3d(growth_fn(3), kernel_size=3, norm_type= norm_type, drop_path_p=drop_path_p)
        )
        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(3), growth_fn(4), stride=2, kernel_size=3, norm_type=norm_type, drop_path_p = drop_path_p),
        )
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(growth_fn(4), growth_fn(5), stride=2, kernel_size=3, norm_type=norm_type, drop_path_p = drop_path_p),
        )
        
        self.interp_r_2 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(1), dc_growth_fn(1), drop_path_p = drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(1)),
            nnblock.SEBlock3d(dc_growth_fn(1), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(1), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(1)),
            nn.SiLU(inplace=True)
        )
        self.interp_r_4 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(2), dc_growth_fn(2), drop_path_p = drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(2)),
            nnblock.SEBlock3d(dc_growth_fn(2), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(2), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(2)),
            nn.SiLU(inplace=True)
        )
        self.interp_r_8 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(3), dc_growth_fn(3), drop_path_p = drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(3)),
            nnblock.SEBlock3d(dc_growth_fn(3), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(3), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(3)),
            nn.SiLU(inplace=True)
        )
        self.interp_r_16 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(4), dc_growth_fn(4), drop_path_p = drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(4)),
            nnblock.SEBlock3d(dc_growth_fn(4), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(4), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(4)),
            nn.SiLU(inplace=True)
        )
        self.interp_r_32 = nn.Sequential(
            nnblock.PreActDownchannel3d(growth_fn(5), dc_growth_fn(5), drop_path_p = drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(5)),
            nnblock.SEBlock3d(dc_growth_fn(5), reduction=4),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(5), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(5)),
            nn.SiLU(inplace=True)
        )
        
        #on our diagram this cascade processing block is the same level as the number in its name
        #usage: takes in current level interp'd output concat with previous level output
        
        self.cascade_2 = nn.Sequential(
            nn.Conv3d(dc_growth_fn(1), dc_growth_fn(1), kernel_size=1),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(1)),
            nn.SiLU(inplace=True)
        )
        self.cascade_4 = nn.Sequential(
            nn.Conv3d(dc_growth_fn(1) + dc_growth_fn(2), dc_growth_fn(2), kernel_size=1),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(2), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(2)),
            nn.SiLU(inplace=True)
        )
        self.cascade_8 = nn.Sequential(
            nn.Conv3d(dc_growth_fn(2) + dc_growth_fn(3), dc_growth_fn(3), kernel_size=1),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(3), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(3)),
            nn.SiLU(inplace=True)
        )
        self.cascade_16 = nn.Sequential(
            nn.Conv3d(dc_growth_fn(3) + dc_growth_fn(4), dc_growth_fn(4), kernel_size=1),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(4), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(4)),
            nn.SiLU(inplace=True)
        )
        self.cascade_32 = nn.Sequential(
            nn.Conv3d(dc_growth_fn(4) + dc_growth_fn(5), dc_growth_fn(5), kernel_size=1),
            nnblock.PreActRefinementBlock3d(dc_growth_fn(5), drop_path_p=drop_path_p),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=dc_growth_fn(5)),
            nn.SiLU(inplace=True)
        )
        
        # Organized module lists
        self.backbone = nn.ModuleList([self.stem, self.enc_2, self.enc_4, self.enc_8, self.enc_16, self.enc_32])
        self.neck = nn.ModuleList([
            self.interp_r_2, self.interp_r_4, self.interp_r_8, self.interp_r_16, self.interp_r_32,
            self.cascade_2, self.cascade_4, self.cascade_8, self.cascade_16, self.cascade_32
        ])
        
        raw_feature_progression = [raw_growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'raw feature progression in backbone: {F_I}, {raw_feature_progression}')
        
        feature_progression = [growth_fn(i) for i in [1, 2, 3, 4, 5]]
        print(f'feature progression in backbone: {F_I}, {feature_progression}')

        downchannel_progression = [dc_growth_fn(i) for i in [1, 2, 3, 4, 5]]
        
        print(f'after downchanneling : {downchannel_progression}')
        # total_concat_channels = sum(downchannel_progression)
        # print(f'total concat features: {total_concat_channels}')
        
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
        
        # cascade_32 outputs dc_growth_fn(5) channels
        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nn.Conv3d(dc_growth_fn(5), dc_growth_fn(4), kernel_size=1, bias = False),#raw inputs
            nnblock.PreActResBlock3d(dc_growth_fn(4), dc_growth_fn(3), kernel_size=3, drop_path_p = drop_path_p),
            nnblock.PreActResBlock3d(dc_growth_fn(3), dc_growth_fn(2), kernel_size=3, drop_path_p = drop_path_p),
            nnblock.PreActDownchannel3d(dc_growth_fn(2), 1, drop_path_p=drop_path_p),
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
        
        cascade_x = self.cascade_2(interp_2_x)
        cascade_x = self.cascade_4(torch.cat([cascade_x, interp_4_x], dim=1))
        cascade_x = self.cascade_8(torch.cat([cascade_x, interp_8_x], dim=1))
        cascade_x = self.cascade_16(torch.cat([cascade_x, interp_16_x], dim=1))
        cascade_x = self.cascade_32(torch.cat([cascade_x, interp_32_x], dim=1))
        
        results = self.head(cascade_x)
        
        check_tensor("results", results, related_tensors={
            "interp_2_x": interp_2_x,
            "interp_4_x": interp_4_x,
            "interp_8_x": interp_8_x,
            "interp_16_x": interp_16_x,
            "interp_32_x": interp_32_x,
            "cascade_x": cascade_x,
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

        # Backbone params (stem + encoders)
        backbone_names = ['stem', 'enc_2', 'enc_4', 'enc_8', 'enc_16', 'enc_32']
        backbone_params = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                          for module in self.backbone]
        total_backbone = sum(backbone_params)

        # Neck params (interp + cascade)
        neck_names = ['interp_2', 'interp_4', 'interp_8', 'interp_16', 'interp_32',
                      'cascade_2', 'cascade_4', 'cascade_8', 'cascade_16', 'cascade_32']
        neck_params = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                      for module in self.neck]
        total_neck = sum(neck_params)

        # Head params
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        # Format backbone details
        backbone_details = ', '.join(f'{name}: {params:,}' for name, params in zip(backbone_names, backbone_params))

        # Format neck details
        neck_details = ', '.join(f'{name}: {params:,}' for name, params in zip(neck_names, neck_params))

        print(
            f'Total trainable params: {trainable_params:,}\n'
            f'---\n'
            f'Backbone params: {total_backbone:,}\n'
            f'  {backbone_details}\n'
            f'---\n'
            f'Neck params: {total_neck:,}\n'
            f'  {neck_details}\n'
            f'---\n'
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