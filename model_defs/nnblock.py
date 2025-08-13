import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(norm_type, num_channels, target_channels_per_group=4, min_groups=4):
    if norm_type == "bn3d":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "gn":
        # Compute ideal group size based on target
        target_group_size = max(target_channels_per_group, num_channels // min(max(num_channels // target_channels_per_group, min_groups), num_channels))
        num_groups = None

        # Try all values from target_group_size down to 1
        for g in range(target_group_size, 0, -1):
            if num_channels % g == 0:
                num_groups = g
                break

        # Failsafe
        if num_groups is None or num_groups > num_channels or num_groups < 1:
            num_groups = 1

        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')

def get_activation(activation_type, inplace:bool = True):
    if activation_type == 'relu':
        return nn.ReLU(inplace= inplace)
    elif activation_type == 'silu':
        return nn.SiLU(inplace = inplace)
    else:
        raise NotImplementedError('no activation support')


class BasicFCBlock(nn.Module):
    def __init__(self, in_features, out_features, p):
        
        """fc, bn, silu, dropout"""
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(inplace= True),
            nn.Dropout(p = p, inplace=True),
        )
        
    def forward(self,x):
        return self.block(x)
        
        
class PreActResBlock2d(nn.Module):
    """#TODO: MAKE STRUCTURE SIMILAR TO 3D FIRST CONV SHOULD BE FOR EXPANSION"""
    #TODO: MAKE STRUCTURE SIMILAR TO 3D FIRST CONV SHOULD BE FOR EXPANSION
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=1, stride=stride, bias=False),
            
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)

class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, norm_type="gn", target_channels_per_group=4, activation = 'relu'):
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        
        self.features = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride, bias=False),
            get_norm_layer(norm_type, out_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, bias=False),
            get_norm_layer(norm_type, out_channels, target_channels_per_group),
            
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                get_norm_layer(norm_type, out_channels, target_channels_per_group),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return F.silu(self.features(x) + self.skip(x), inplace = True)   
    
class PreActResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, norm_type="gn", target_channels_per_group=4, dilation = 1):
        """padding will always be dynamic to keep spatial size the same"""
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        #3:1, 5:2, 7:3
        
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride, bias=False),
            
            get_norm_layer(norm_type, out_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, bias=False, dilation= dilation),
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                get_norm_layer(norm_type, out_channels, target_channels_per_group),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)
    
class UpsamplePreActResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, norm_type="gn", target_channels_per_group=4):
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2

        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            # Upsample spatially first, keep channels same
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride, output_padding=stride-1, bias=False),

            get_norm_layer(norm_type, in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            # Then adjust channels
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, bias=False),
        )

        # Skip connection
        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, output_padding=stride-1, bias=False),
                get_norm_layer(norm_type, out_channels, target_channels_per_group),
            )

    def forward(self, x):
        return self.features(x) + self.skip(x)

class FPNBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels,activation_order, norm_type = "gn"):
        super().__init__()
        if activation_order == 'pre':
            self.block = PreActResBlock3d(in_channels, out_channels, stride=1, norm_type=norm_type)
        elif activation_order == 'post':
            self.block = ResBlock3d(in_channels, out_channels, stride=1, norm_type=norm_type)
        # self.target_size = target_size
        
    def forward(self, x ,target_size):
        x = self.block(x)
        x = F.interpolate(x, size=target_size, mode='trilinear')
        return x
    
class DropoutPreActResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p = 0.1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=1, stride=stride, bias=False),
            
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p = p),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)
    
