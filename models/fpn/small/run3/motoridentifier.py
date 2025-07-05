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

class MotorIdentifier(nn.Module):
    def __init__(self, dropout_p = 0, norm_type = "gn"):
        super().__init__()
        self.dropout_p = dropout_p
        F_I = 4
        feature_list_thing = [8,16,32,72,40,72]#fpn2,4,8,32,enc16,32
        #F_I = 8, [8,16,32,72,52,72]
        #features initial

        # Encoder (Downsampling path)
        self.stem = nn.Conv3d(1,F_I, kernel_size=5, stride = 1, padding = 2)
        #1/2
        self.enc_2 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, F_I*2, stride = 2, kernel_size=3, norm_type=norm_type),
        )
        #1/4
        self.enc_4 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I*2, F_I*4, stride = 2, kernel_size=3, norm_type=norm_type),
        )
        #1/8
        self.enc_8 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I*4, F_I*8, stride = 2, kernel_size=3, norm_type=norm_type),
        )
        #1/16
        self.enc_16 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I*8, F_I*16, stride = 2, kernel_size=3, norm_type=norm_type),
        )
        #1/32
        self.enc_32 = nn.Sequential(
            nnblock.PreActResBlock3d(F_I*16, F_I*32, stride = 2, kernel_size=3, norm_type=norm_type),
        )

        # FPN blocks
        self.fpn_2_block = nnblock.FPNBlock3d(F_I*2, F_I*2, activation_order='pre')
        self.fpn_2_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*2),
        nn.SiLU(),
        nn.Conv3d(F_I*2, feature_list_thing[0], kernel_size=1)
        )

        self.fpn_4_block = nnblock.FPNBlock3d(F_I*4, F_I*4, activation_order='pre')
        self.fpn_4_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*4),
        nn.SiLU(),
        nn.Conv3d(F_I*4, feature_list_thing[1], kernel_size=1)
        )

        self.fpn_8_block = nnblock.FPNBlock3d(F_I*8, F_I*8, activation_order='pre')
        self.fpn_8_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*8),
        nn.SiLU(),
        nn.Conv3d(F_I*8, feature_list_thing[2], kernel_size=1)
        )

        self.fpn_32_block = nnblock.FPNBlock3d(F_I*32, F_I*16, activation_order='pre')
        self.fpn_32_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*16),
        nn.SiLU(),
        nn.Conv3d(F_I*16, feature_list_thing[3], kernel_size=1)
        )

        # Encoder feature reductions
        self.enc_16_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*16),
        nn.SiLU(),
        nn.Conv3d(F_I*16, feature_list_thing[4], kernel_size=1)
        )

        self.enc_32_conv = nn.Sequential(
        nnblock.get_norm_layer(norm_type="gn", num_channels=F_I*32),
        nn.SiLU(),
        nn.Conv3d(F_I*32, feature_list_thing[5], kernel_size=1)
        )   

        total_concat_channels = sum(feature_list_thing)

        self.head = nn.Sequential(
            nn.Dropout3d(p=self.dropout_p),
            nnblock.PreActResBlock3d(total_concat_channels, total_concat_channels//3, kernel_size=1, norm_type="gn"),
            nnblock.PreActResBlock3d(total_concat_channels//3, total_concat_channels//6, kernel_size=1, norm_type="gn"),
            nnblock.get_norm_layer(norm_type="gn", num_channels=total_concat_channels//6),
            nn.SiLU(inplace= True),
            nn.Conv3d(total_concat_channels//6, 1, kernel_size=1)
        )
        
    # Forward pass updates needed:
    def forward(self, x):
        
        stem_x = self.stem(x)
        
        enc_2_x = self.enc_2(stem_x)
        enc_4_x = self.enc_4(enc_2_x)
        enc_8_x = self.enc_8(enc_4_x)
        enc_16_x = self.enc_16(enc_8_x)
        enc_32_x = self.enc_32(enc_16_x)

        target_size = enc_16_x.shape[2:]
        # print(f"x shape: {x.shape}")
        # print(f"stem_x shape: {stem_x.shape}")
        # print(f"enc_2_x shape: {enc_2_x.shape}")
        # print(f"enc_4_x shape: {enc_4_x.shape}")
        # print(f"enc_8_x shape: {enc_8_x.shape}")
        # print(f"enc_16_x shape: {enc_16_x.shape}")
        # print(f"enc_32_x shape: {enc_32_x.shape}")
        
        # print(f'model forward target size: {target_size}')
        # Process FPN blocks with reductions
        fpn_2_x = self.fpn_2_conv(self.fpn_2_block(enc_2_x, target_size))
        fpn_4_x = self.fpn_4_conv(self.fpn_4_block(enc_4_x, target_size))
        fpn_8_x = self.fpn_8_conv(self.fpn_8_block(enc_8_x, target_size))
        fpn_32_x = self.fpn_32_conv(self.fpn_32_block(enc_32_x, target_size))
        
        # Reduce encoder features
        enc_16_reduced = self.enc_16_conv(enc_16_x)
        enc_32_upsampled = F.interpolate(enc_32_x, size=target_size, mode='trilinear')
        enc_32_reduced = self.enc_32_conv(enc_32_upsampled)
        
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
        
        # Encoder breakdown
        stem_params = sum(p.numel() for p in self.stem.parameters() if p.requires_grad)
        enc_2_params = sum(p.numel() for p in self.enc_2.parameters() if p.requires_grad)
        enc_4_params = sum(p.numel() for p in self.enc_4.parameters() if p.requires_grad)
        enc_8_params = sum(p.numel() for p in self.enc_8.parameters() if p.requires_grad)
        enc_16_params = sum(p.numel() for p in self.enc_16.parameters() if p.requires_grad)
        enc_32_params = sum(p.numel() for p in self.enc_32.parameters() if p.requires_grad)
        
        # FPN blocks
        fpn_2_block_params = sum(p.numel() for p in self.fpn_2_block.parameters() if p.requires_grad)
        fpn_2_conv_params = sum(p.numel() for p in self.fpn_2_conv.parameters() if p.requires_grad)
        
        fpn_4_block_params = sum(p.numel() for p in self.fpn_4_block.parameters() if p.requires_grad)
        fpn_4_conv_params = sum(p.numel() for p in self.fpn_4_conv.parameters() if p.requires_grad)
        
        fpn_8_block_params = sum(p.numel() for p in self.fpn_8_block.parameters() if p.requires_grad)
        fpn_8_conv_params = sum(p.numel() for p in self.fpn_8_conv.parameters() if p.requires_grad)
        
        fpn_32_block_params = sum(p.numel() for p in self.fpn_32_block.parameters() if p.requires_grad)
        fpn_32_conv_params = sum(p.numel() for p in self.fpn_32_conv.parameters() if p.requires_grad)
        
        # Encoder feature reductions
        enc_16_conv_params = sum(p.numel() for p in self.enc_16_conv.parameters() if p.requires_grad)
        enc_32_conv_params = sum(p.numel() for p in self.enc_32_conv.parameters() if p.requires_grad)
        
        # Head
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        # Totals
        total_encoder = stem_params + enc_2_params + enc_4_params + enc_8_params + enc_16_params + enc_32_params
        total_fpn_blocks = fpn_2_block_params + fpn_4_block_params + fpn_8_block_params + fpn_32_block_params
        total_fpn_convs = fpn_2_conv_params + fpn_4_conv_params + fpn_8_conv_params + fpn_32_conv_params
        total_enc_convs = enc_16_conv_params + enc_32_conv_params
        total_fpn = total_fpn_blocks + total_fpn_convs + total_enc_convs
        
        print(f'Total trainable params: {trainable_params:,}')
        print(f'Encoder params: {total_encoder:,}')
        print(f'  - stem: {stem_params:,}')
        print(f'  - enc_2: {enc_2_params:,}')
        print(f'  - enc_4: {enc_4_params:,}')
        print(f'  - enc_8: {enc_8_params:,}')
        print(f'  - enc_16: {enc_16_params:,}')
        print(f'  - enc_32: {enc_32_params:,}')
        print(f'FPN total params: {total_fpn:,}')
        print(f'  FPN blocks: {total_fpn_blocks:,}')
        print(f'    - fpn_2_block: {fpn_2_block_params:,}')
        print(f'    - fpn_4_block: {fpn_4_block_params:,}')
        print(f'    - fpn_8_block: {fpn_8_block_params:,}')
        print(f'    - fpn_32_block: {fpn_32_block_params:,}')
        print(f'  FPN convs: {total_fpn_convs:,}')
        print(f'    - fpn_2_conv: {fpn_2_conv_params:,}')
        print(f'    - fpn_4_conv: {fpn_4_conv_params:,}')
        print(f'    - fpn_8_conv: {fpn_8_conv_params:,}')
        print(f'    - fpn_32_conv: {fpn_32_conv_params:,}')
        print(f'  Encoder convs: {total_enc_convs:,}')
        print(f'    - enc_16_conv: {enc_16_conv_params:,}')
        print(f'    - enc_32_conv: {enc_32_conv_params:,}')
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

