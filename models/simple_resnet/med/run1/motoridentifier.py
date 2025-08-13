import monai.inferers
import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch

from . import nnblock

import monai
import gc
import time
import torch.nn.functional as F

class MotorIdentifier(nn.Module):
    def __init__(self, dropout_p = 0, norm_type = "gn"):
        super().__init__()
        self.dropout_p = dropout_p
        F_I = 12  # Initial feature size

        # Stem
        self.stem = nn.Conv3d(1, F_I, kernel_size=5, stride=1, padding=2)

        # Encoder / Backbone
        self.backbone = nn.Sequential(
            nnblock.PreActResBlock3d(F_I, F_I * 2, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(F_I * 2, F_I * 4, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(F_I * 4, F_I * 8, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(F_I * 8, F_I * 16, stride=2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(F_I * 16, F_I * 32, stride=2, kernel_size=3, norm_type=norm_type),
        )

        # Decoder Block: 1/32 -> 1/16
        self.decoder = nn.Sequential(
            nnblock.get_norm_layer("gn", F_I * 32),
            nn.SiLU(),
            nn.Conv3d(F_I * 32, F_I * 16, kernel_size=1),  # <<< Reduce channels first
            nnblock.UpsamplePreActResBlock3d(F_I * 16, F_I * 8)  # Now only upsampling 128 → 128
        )
        
        # Final Head
        self.head = nn.Sequential(
            nn.Dropout3d(p=dropout_p),
            nnblock.PreActResBlock3d(F_I * 8, F_I * 4, kernel_size=1, norm_type=norm_type),
            nnblock.PreActResBlock3d(F_I * 4, F_I * 2, kernel_size=1, norm_type=norm_type),
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=F_I * 2),
            nn.SiLU(inplace=True),
            nn.Conv3d(F_I * 2, 1, kernel_size=1)
        )

    def forward(self, x):
        stem_x = self.stem(x)            
        enc_x = self.backbone(stem_x)    
        dec_x = self.decoder(enc_x)       
        out = self.head(dec_x)            
        return out


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

        # Encoder (Backbone) breakdown
        stem_params = sum(p.numel() for p in self.stem.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        # Decoder
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        # Head
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        # Totals
        encoder_params = stem_params + backbone_params

        print(f'Total trainable params: {trainable_params:,}')
        print(f'Encoder params: {encoder_params:,}')
        print(f'  - stem: {stem_params:,}')
        print(f'  - backbone: {backbone_params:,}')
        print(f'Decoder params: {decoder_params:,}')
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

