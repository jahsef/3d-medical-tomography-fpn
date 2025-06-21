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


class MotorIdentifier(nn.Module):
    
    def __init__(self, norm_type = "bn3d"):
        super().__init__()

        
        FOC = 96  # Reduced from 128

        # Encoder (Downsampling path)
        self.encoder = nn.Sequential(
            #c16, 64^3
            nn.Conv3d(1,FOC//8, kernel_size=5, stride = 1),
            nnblock.PreActResBlock3d(FOC//8, FOC//8, norm_type=norm_type),
            #c32, 64^3
            nnblock.PreActResBlock3d(FOC//8, FOC//4, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//4, FOC//4, norm_type=norm_type),
            #c64, 16^3
            nnblock.PreActResBlock3d(FOC//4, FOC//2, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC//2, FOC//2, norm_type=norm_type),
            #c128, 8^3
            nnblock.PreActResBlock3d(FOC//2, FOC, stride = 2, kernel_size=3, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
            nnblock.PreActResBlock3d(FOC, FOC, norm_type=norm_type),
        )
        
        #encoder increases channels, decreases spatial size
        #decoder obviously decodes
        
        # Decoder (Upsampling path)
        self.decoder = nn.Sequential(
            #64, 16^3
            nnblock.UpsamplePreActResBlock3d(FOC,FOC//2,stride =2, kernel_size= 3, norm_type=norm_type),
            
            #32, 32^3
            nnblock.UpsamplePreActResBlock3d(FOC//2,FOC//4,stride =2, kernel_size= 3, norm_type=norm_type),
            
            #16, 64^3
            nnblock.UpsamplePreActResBlock3d(FOC//4,FOC//8,stride =2, kernel_size= 3, norm_type=norm_type),
        )

        # Classification head
        
        self.classification_head = nn.Sequential(
            nnblock.get_norm_layer(norm_type=norm_type, num_channels=FOC//8),
            nn.SiLU(inplace=True),
            nn.Conv3d(FOC//8, 1, kernel_size=3,padding = 1, bias=False)
            
        )
        
        
                
        



    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode
        decoded = self.decoder(features)
        
        # Final prediction
        output = self.classification_head(decoded)
        
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
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f'trainable params: {trainable_params}')
        print(f'encoder params: {encoder_params}')
        print(f'decoder params: {decoder_params}')
        
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

