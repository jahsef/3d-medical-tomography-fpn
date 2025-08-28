import torch
import torch.nn as nn


class TrivialModel(nn.Module):
    def __init__(self, initial_features=12, dropout_p=0, norm_type="gn"):
        super().__init__()
        # Actually trivial model - just 1x1 convs
        self.stem = nn.Conv3d(1, 8, kernel_size=1, stride=1)
        self.backbone = nn.AvgPool3d(kernel_size=16, stride=16)  # Downsample to 1/16
        self.head = nn.Conv3d(8, 1, kernel_size=1)
        
    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.backbone(x)
        x = self.head(x)
        return x
    
    def print_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total trainable params: {trainable_params:,}')