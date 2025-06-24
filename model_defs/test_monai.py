from monai.networks.nets import ResNetFeatures
import torch



# Load weights non-strictly
pretrained = ResNetFeatures("resnet18", pretrained=True, spatial_dims=3, in_channels=1)
print(pretrained)

#   (conv1): Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
#   (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (act): ReLU(inplace=True)
#   (maxpool): MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  #layer 1,2,3,4