import torch.nn as nn
import torch.nn.functional as F

class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 1, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)