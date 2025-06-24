import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d
from . import nnblock
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import monai
import gc
import time
import torch.nn.functional as F
from monai.networks.nets import ResNetFeatures

class MotorIdentifier(nn.Module):
    def __init__(self, dropout_p=0, norm_type="gn"):
        super().__init__()
        
        self.dropout_p = dropout_p
        F_I = 64

        # Load pretrained ResNet - keep as complete module to preserve weights
        self.backbone = ResNetFeatures("resnet18", pretrained=True, spatial_dims=3, in_channels=1)
        
        # FPN blocks - channels: 64_2, 64_4, 128_8, 256_16, 512_32
        self.fpn_2_block = nnblock.FPNBlock3d(F_I, F_I, activation_order='post')
        self.fpn_2_conv = nn.Conv3d(F_I, 32, kernel_size=1)

        self.fpn_4_block = nnblock.FPNBlock3d(F_I, F_I, activation_order='post')
        self.fpn_4_conv = nn.Conv3d(F_I, 32, kernel_size=1)

        self.fpn_8_block = nnblock.FPNBlock3d(F_I*2, F_I*2, activation_order='post')
        self.fpn_8_conv = nn.Conv3d(F_I*2, 32, kernel_size=1)

        self.fpn_32_block = nnblock.FPNBlock3d(F_I*8, F_I*8, activation_order='post')
        self.fpn_32_conv = nn.Conv3d(F_I*8, 32, kernel_size=1)
        
        # Direct encoder feature reductions (no FPN needed at target scale)
        self.enc_16_conv = nn.Conv3d(F_I*4, 32, kernel_size=1)
        self.enc_32_conv = nn.Conv3d(F_I*8, 32, kernel_size=1)

        total_concat_channels = 32 * 6  # 192 channels total

        # Head with proper activation ordering
        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nnblock.ResBlock3d(total_concat_channels, total_concat_channels//2, kernel_size=1, norm_type=norm_type),
            nnblock.ResBlock3d(total_concat_channels//2, total_concat_channels//4, kernel_size=3, norm_type=norm_type),
            nnblock.ResBlock3d(total_concat_channels//4, total_concat_channels//8, kernel_size=3, norm_type=norm_type),
            nnblock.ResBlock3d(total_concat_channels//8, total_concat_channels//16, kernel_size=3, norm_type=norm_type),
            nnblock.ResBlock3d(total_concat_channels//16, total_concat_channels//32, kernel_size=3, norm_type=norm_type),
            nnblock.ResBlock3d(total_concat_channels//32, total_concat_channels//64, kernel_size=3, norm_type=norm_type),
            nn.Conv3d(total_concat_channels//64, 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Extract features at different scales while preserving pretrained weights
        enc_2_x = self.backbone.conv1(x)
        enc_2_x = self.backbone.bn1(enc_2_x)
        enc_2_x = self.backbone.act(enc_2_x)
        
        enc_4_x = self.backbone.maxpool(enc_2_x)
        enc_4_x = self.backbone.layer1(enc_4_x)
        
        enc_8_x = self.backbone.layer2(enc_4_x)
        enc_16_x = self.backbone.layer3(enc_8_x)
        enc_32_x = self.backbone.layer4(enc_16_x)

        # Use 1/16 scale as target size for FPN
        target_size = enc_16_x.shape[2:]

        # FPN processing for scales that need resizing
        fpn_2_x = self.fpn_2_conv(self.fpn_2_block(enc_2_x, target_size))
        fpn_4_x = self.fpn_4_conv(self.fpn_4_block(enc_4_x, target_size))
        fpn_8_x = self.fpn_8_conv(self.fpn_8_block(enc_8_x, target_size))
        fpn_32_x = self.fpn_32_conv(self.fpn_32_block(enc_32_x, target_size))
        
        # Direct reductions (no FPN needed - already at target scale or will be upsampled)
        enc_16_reduced = self.enc_16_conv(enc_16_x)
        enc_32_upsampled = F.interpolate(enc_32_x, size=target_size, mode='trilinear')
        enc_32_reduced = self.enc_32_conv(enc_32_upsampled)
        
        # Concatenate all features
        fpn_concat = torch.cat([fpn_2_x, fpn_4_x, fpn_8_x, fpn_32_x, enc_16_reduced, enc_32_reduced], dim=1)
        
        return self.head(fpn_concat)


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
        
        # Backbone breakdown
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        fpn_2_params = sum(p.numel() for p in self.fpn_2_block.parameters() if p.requires_grad) + sum(p.numel() for p in self.fpn_2_conv.parameters() if p.requires_grad)
        fpn_4_params = sum(p.numel() for p in self.fpn_4_block.parameters() if p.requires_grad) + sum(p.numel() for p in self.fpn_4_conv.parameters() if p.requires_grad)
        fpn_8_params = sum(p.numel() for p in self.fpn_8_block.parameters() if p.requires_grad) + sum(p.numel() for p in self.fpn_8_conv.parameters() if p.requires_grad)
        fpn_32_params = sum(p.numel() for p in self.fpn_32_block.parameters() if p.requires_grad) + sum(p.numel() for p in self.fpn_32_conv.parameters() if p.requires_grad)
        
        # Direct conv params
        direct_params = sum(p.numel() for p in self.enc_16_conv.parameters() if p.requires_grad) + sum(p.numel() for p in self.enc_32_conv.parameters() if p.requires_grad)
        
        # Head
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        total_fpn = fpn_2_params + fpn_4_params + fpn_8_params + fpn_32_params
        
        print(f'Total trainable params: {trainable_params:,}')
        print(f'Backbone params: {backbone_params:,}')
        print(f'FPN params: {total_fpn:,}')
        print(f'  - fpn_2: {fpn_2_params:,}')
        print(f'  - fpn_4: {fpn_4_params:,}')
        print(f'  - fpn_8: {fpn_8_params:,}')
        print(f'  - fpn_32: {fpn_32_params:,}')
        print(f'Direct conv params: {direct_params:,}')
        print(f'Head params: {head_params:,}')
        
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

