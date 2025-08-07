import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import logging

def check_tensor(name, tensor, related_tensors=None):
    if torch.isnan(tensor).any():
        print(f"[ASSERT FAIL] {name} has NaNs")
        print(f"{name} stats: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}, std={tensor.std().item()}")
        if related_tensors:
            for rel_name, rel_tensor in related_tensors.items():
                print(f"{rel_name} stats: min={rel_tensor.min().item()}, max={rel_tensor.max().item()}, mean={rel_tensor.mean().item()}, std={rel_tensor.std().item()}")
        sys.stdout.flush()
        raise AssertionError(f"{name} has NaNs")

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
        if num_groups == 1:
            print(f'WARNING get_norm_layer returned 1 failsafe')
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
    
class PreActRefinementBlock3d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, norm_type="gn"):
        """fullfat3d conv for refinement"""
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        #3:1, 5:2, 7:3
        
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, 
                      padding=padding, stride=1, bias=False),
        )
        
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


class PreActResBottleneckBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_type="gn", target_channels_per_group=4, bottleneck_ratio=4):
        """Standard 1x1x1 -> 3x3x3 -> 1x1x1 bottleneck block"""
        super().__init__()
        
        mid_channels = out_channels // bottleneck_ratio
        
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            # 1x1x1 compress
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False),
            
            get_norm_layer(norm_type, mid_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            # 3x3x3 process
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, 
                      padding=1, stride=stride, bias=False),
            
            get_norm_layer(norm_type, mid_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            # 1x1x1 expand
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False),
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

class PreActGroupPointBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels,groups, norm_type = 'gn'):
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False
            ),
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),     
        )
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        
        out = self.features(x)
        skip = self.skip(x)
        return out+skip
    
class PreActDownchannel3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type = 'gn'):
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),    
        )
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        
        out = self.features(x)
        skip = self.skip(x)
        return out+skip 
    
    
class PreActDSepHWBlock3d(nn.Module):
    def __init__(self, in_channels, norm_type = 'gn'):
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,3), padding=(1,0,0), groups=in_channels, bias= False),  # depth
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(0,1,1), groups=in_channels,bias=False)  # hw   
        )
        self.skip = nn.Identity()
        
    def forward(self, x):
        out = self.features(x)
        skip = self.skip(x)
        return out+skip
    
class PreActAsymmetricBlock3d(nn.Module):
    def __init__(self, in_channels, norm_type = 'gn'):
        """133 311 331 kernel, spatial -> temporal -> spatiotemporal mixing"""
        super().__init__()
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,3), padding=(0,1,1), groups=in_channels, bias= False),  # depth
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0), groups=in_channels,bias=False),  # hw   
            get_norm_layer(norm_type, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3,3,1), padding=(1,1,0), groups=in_channels,bias=False)  # hw   
        )
        self.skip = nn.Identity()
        
    def forward(self, x):
        out = self.features(x)
        skip = self.skip(x)
        return out+skip


 
class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1,bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1,bias = True),
        )
    
    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        pooled = self.pool(x)
        scale = F.sigmoid(torch.clamp(self.fc(pooled),min = 1e-3, max = 1000))
        scale = torch.clamp(scale, min=1e-3, max=1.0)  # prevent zeros

        
        
        result = x * scale

        return result

