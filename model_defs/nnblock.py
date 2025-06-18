import torch.nn as nn

def get_norm_layer(norm_type, num_channels, target_channels_per_group=8):
    if norm_type == "bn3d":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "gn":
        # Calculate optimal number of groups for target channels per group
        num_groups = max(1, num_channels // target_channels_per_group)
        # Ensure num_channels is divisible by num_groups
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise Exception(f'Normalization type "{norm_type}" not supported')
    
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
    

class PreActResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, norm_type = "bn3d", target_channels_per_group = 8):
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
                      padding=padding, bias=False),
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
    def __init__(self, in_channels, out_channels, stride=2, kernel_size = 3, norm_type = "bn3d", target_channels_per_group = 8):
        super().__init__()
        fart = kernel_size - 1
        padding = fart//2
        
        self.features = nn.Sequential(
            get_norm_layer(norm_type, in_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride, output_padding=1, bias=False),
            
            get_norm_layer(norm_type, out_channels, target_channels_per_group),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, bias=False),
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, output_padding=1, bias=False),
                get_norm_layer(norm_type, out_channels, target_channels_per_group),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.features(x) + self.skip(x)

 
    
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
    
