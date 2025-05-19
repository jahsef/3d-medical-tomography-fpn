import torch.nn as nn
import torch


# target output size of 5x7x9
m = nn.AdaptiveAvgPool3d((5, 7, 9))
input = torch.randn(1, 64, 8, 9, 10) #b,c,d,h,w
output = m(input)
print(output.shape)

# target output size of 7x7x7 (cube)
m = nn.AdaptiveAvgPool3d(7)
input = torch.randn(1, 64, 10, 9, 8)
output = m(input)
print(output.shape)
# target output size of 7x9x8
m = nn.AdaptiveAvgPool3d((7, None, None))
input = torch.randn(1, 64, 10, 9, 8)
output = m(input)
print(output.shape)