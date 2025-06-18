import torch.nn as nn


# Test with trivial model to verify training pipeline
class TrivialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, padding=1)
        self.final = nn.Conv3d(1, 1, 1)
    
    def forward(self, x):
        return self.final(self.conv(x))

# Train for a few epochs - should show meaningful loss decrease