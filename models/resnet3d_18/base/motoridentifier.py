import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch

from . import nnblock

class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors:int):
        self.max_motors = max_motors
        super().__init__()

        self.features = nn.Sequential(
            #stem
            nn.Conv3d(in_channels=1, out_channels= 16, kernel_size= 3, stride = 1, padding = 1),
            #blocks
            nnblock.PreActResBlock3d(in_channels=16, out_channels=32, stride = 2),
            nnblock.PreActResBlock3d(in_channels=32, out_channels=64, stride = 2),
            nnblock.PreActResBlock3d(in_channels=64, out_channels=128, stride = 2),
            
        )
        
        
        #we should just get average of 512 feature maps?
        #these are basic blocks not preact so no need to apply activations after
        self.intermediate = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size = (1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128,256),
            nn.SiLU()
        )
        
        self.regression_head = nn.Sequential(
            #outputs a 3d point in space (use mse or something similar)
            nn.Linear(256, max_motors * 3)
        )
        
        self.classification_head = nn.Sequential(
            #outputs conf logits? bce
            nn.Linear(256,max_motors * 1)
            #use sigmoid function later on in forward
        )
        
        #in model.predict we can handle the (-1,-1,-1) output
    
    def forward(self,x):
        """

        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, w)

        Returns:
            torch.Tensor: Output tensor of shape (b, max_motors, 4)
                        where the last dimension is [x, y, z, conf]
        """
        
        x = self.features(x)
        x = self.intermediate(x)
        # x = x.view(x.size(0), -1)
        points = self.regression_head(x)
        #points sahpe (b,max_motors,3), conf (b,max_motors)
        points = points.view(-1, self.max_motors, 3)
        conf_logits = self.classification_head(x)
        
        # Combine along feature dimension
        outputs = torch.cat([points, conf_logits.unsqueeze(-1)], dim=-1)

        return outputs

    def predict(self,x):
        with torch.no_grad():
            outputs = self.forward(x)
            outputs[:, :, 3] = torch.nn.functional.sigmoid(outputs[:, :, 3])
            return outputs
  
if __name__ == '__main__':
    model = r3d_18(R3D_18_Weights)
    # # conv1_weight = model.stem[0].weight.data  # shape (64, 3, 3, 7, 7)
    print(model.layer4)#should output 512,512,3,3,3

    # print(model)
    # for name, module in model.named_children():
    #     print(name)
