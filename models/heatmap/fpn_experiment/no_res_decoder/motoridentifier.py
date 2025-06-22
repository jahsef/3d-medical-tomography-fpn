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

class MotorIdentifier(nn.Module):
    
    def __init__(self, norm_type = "bn3d"):
        super().__init__()

        
        FOC = 96  # Reduced from 128

        # Encoder (Downsampling path)
        self.stem = nn.Conv3d(1,FOC//8, kernel_size=5, stride = 1, padding = 2)
        
        self.enc1 = nn.Sequential(
            nnblock.PreActResBlock3d(FOC//8, FOC//8, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//8, FOC//4, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//4, FOC//4, norm_type=norm_type),
        )
        
        self.enc2 = nn.Sequential(
            nnblock.PreActResBlock3d(FOC//4, FOC//2, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//2, FOC//2, norm_type=norm_type),
        )
        
        self.enc3 = nn.Sequential(
            nnblock.PreActResBlock3d(FOC//2, FOC, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
            nn.Dropout3d(p=0.1, inplace=True)
        )
        
        self.dec3 = nnblock.UpsamplePreActBlock3d(FOC,FOC//2,stride =2, kernel_size= 3, norm_type=norm_type)
        self.dec2 = nnblock.UpsamplePreActBlock3d(FOC//2,FOC//4,stride =2, kernel_size= 3, norm_type=norm_type)
        self.dec1 = nnblock.UpsamplePreActBlock3d(FOC//4,FOC//8,stride =2, kernel_size= 3, norm_type=norm_type)

        
        self.head = nn.Sequential(
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=FOC//8),
            nn.SiLU(inplace=True),
            nn.Conv3d(FOC//8,1, stride = 1, kernel_size=3,padding = 1, bias=False)
        )


    def forward(self, x):
        
        stem_x = self.stem(x)
        enc1_x = self.enc1(stem_x)
        enc2_x = self.enc2(enc1_x)
        enc3_x = self.enc3(enc2_x)
        #number for enc/dec is for depth
        
        dec3_x = self.dec3(enc3_x)
        dec2_x = self.dec2(dec3_x + enc2_x)
        dec1_x = self.dec1(dec2_x + enc1_x)
        
        output = self.head(dec1_x)
        
        return output


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
        enc1_params = sum(p.numel() for p in self.enc1.parameters() if p.requires_grad)
        enc2_params = sum(p.numel() for p in self.enc2.parameters() if p.requires_grad)
        enc3_params = sum(p.numel() for p in self.enc3.parameters() if p.requires_grad)
        
        # Decoder breakdown
        dec3_params = sum(p.numel() for p in self.dec3.parameters() if p.requires_grad)
        dec2_params = sum(p.numel() for p in self.dec2.parameters() if p.requires_grad)
        dec1_params = sum(p.numel() for p in self.dec1.parameters() if p.requires_grad)
        
        # Classification head
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        total_encoder = stem_params + enc1_params + enc2_params + enc3_params
        total_decoder = dec3_params + dec2_params + dec1_params
        
        print(f'Total trainable params: {trainable_params:,}')
        print(f'Encoder params: {total_encoder:,} (stem: {stem_params:,}, enc1: {enc1_params:,}, enc2: {enc2_params:,}, enc3: {enc3_params:,})')
        print(f'Decoder params: {total_decoder:,} (dec3: {dec3_params:,}, dec2: {dec2_params:,}, dec1: {dec1_params:,})')
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

