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


class MotorIdentifier(nn.Module):
    
    def __init__(self,initial_features = 12, dropout_p=0, norm_type="gn"):
        super().__init__()
        self.dropout_p = dropout_p
        
        self.stem = nn.Conv3d(1,8,kernel_size=5,padding=2,bias=False)
        
        self.backbone = nn.Sequential(
            nnblock.PreActResBlock3d(8, 16, stride = 2),#1/2
            nnblock.PreActResBlock3d(16, 32, stride = 2),#1/4
            nnblock.PreActResBlock3d(32, 64, stride = 2),#1/8
            nnblock.PreActResBlock3d(64, 64),
            nnblock.PreActResBlock3d(64, 128, stride = 2),#1/16
            nnblock.PreActResBlock3d(128, 128),
            nnblock.PreActResBlock3d(128, 256, stride = 2),#1/32
            nnblock.PreActResBlock3d(256, 256),
        )

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'trilinear', align_corners=False),
            nn.Dropout3d(p=self.dropout_p),
            nnblock.PreActResBlock3d(256,128, kernel_size=3),
            nnblock.PreActResBlock3d(128,32, kernel_size=3),
            # nnblock.PreActResBlock3d(STEM_FEATURES*4,1, kernel_size=1, target_channels_per_group=4)
            nnblock.get_norm_layer(num_channels =32, norm_type = 'gn'),
            nn.SiLU(inplace=True),
            nn.Conv3d(32,1, kernel_size=1, bias = False)
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
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        print(
            f'Total trainable params: {trainable_params:,}\n'
            f'Stem params: {stem_params:,}\n'
            f'Backbone params: {backbone_params:,}\n'
            f'Head params: {head_params:,}'
        )




if __name__ == '__main__':
    model = MotorIdentifier()
    model.print_params()