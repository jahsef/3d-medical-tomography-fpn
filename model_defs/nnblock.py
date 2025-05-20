import torch.nn as nn



class PreActResBlock2d(nn.Module):
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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(in_channels, in_channels, kernel_size=3, 
                      padding=1, stride=stride, bias=False),
            
            nn.BatchNorm3d(in_channels),
            nn.SiLU(inplace=True),
            #dropout here
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
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
    
class BottleneckPreActResBlock2d(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4
    def __init__(self, in_channels, out_channels,downsample=None, groups=1,
                 base_width=64, dilation=1):
        super().__init__()

        width = int(out_channels * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.features = nn.Sequential([
            nn.BatchNorm2d(),
            nn.Conv2d()
        ])
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out