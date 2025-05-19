import torch.nn as nn
from torchvision.models.video.resnet import r3d_18
from torchvision.models.video.resnet import R3D_18_Weights
import torch


class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors = 20):
        self.max_motors = max_motors
        super().__init__()
        r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Get pretrained weights for first conv
        conv1_weight = r3d.stem[0].weight.data  # shape (64, 3, 3, 7, 7)
        # Average over channel dim to get grayscale kernel
        new_weight = conv1_weight.mean(dim=1, keepdim=True)  # shape (64, 1, 3, 7, 7)
        # Replace conv1 weights with new grayscale weights
        r3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7),stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        r3d.stem[0].weight.data = new_weight
        
        self.features = nn.Sequential(
            #avg tomo stats: 415 depth, 956 h/w
            #we are going to do avg down about 1/3 per dimension
            #actually 1/20 for prototyping
            nn.AdaptiveAvgPool3d(output_size=(20, 47, 47)),
            #adaptive avg pool preserves b,c dim just pool the d,h,w
            r3d.stem,
            r3d.layer1,
            r3d.layer2,
            r3d.layer3,
            r3d.layer4,#outputs 512 feature maps
        )
        
        self.globalpool = nn.AdaptiveAvgPool3d(output_size = (1, 1, 1))
        #we should just get average of 512 feature maps?
        #these are basic blocks not preact so no need to apply activations after
        
        
        self.regression_head = nn.Sequential(
            #outputs a 3d point in space (use mse or something similar)
            nn.Linear(512,max_motors * 3)
        )
        
        self.classification_head = nn.Sequential(
            #outputs conf logits? bce
            nn.Linear(512,max_motors * 1)
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
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
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
