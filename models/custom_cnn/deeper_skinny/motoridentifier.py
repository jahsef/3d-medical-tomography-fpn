import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch

from . import nnblock
from itertools import product

class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors:int):
        self.max_motors = max_motors
        super().__init__()

        
        features_out_channels = 128
        self.features = nn.Sequential(
            #stem
            nn.Conv3d(in_channels=1, out_channels= 16, kernel_size= 3, stride = 1, padding = 1),
            #blocks
            nnblock.PreActResBlock3d(in_channels=16, out_channels=16),
            nnblock.PreActResBlock3d(in_channels=16, out_channels=16),
            nnblock.PreActResBlock3d(in_channels=16, out_channels=32, stride = 2),
            nnblock.PreActResBlock3d(in_channels=32, out_channels=32),
            nnblock.PreActResBlock3d(in_channels=32, out_channels=64, stride = 2),
            nnblock.PreActResBlock3d(in_channels=64, out_channels=64),
            nnblock.PreActResBlock3d(in_channels=64, out_channels=features_out_channels, stride = 2),
        )
        
        
        #we should just get average of 512 feature maps?
        #these are basic blocks not preact so no need to apply activations after
        self.intermediate = nn.Sequential(
            nn.BatchNorm3d(features_out_channels),
            nn.SiLU(inplace= True),
            nn.AdaptiveAvgPool3d(output_size = (1, 1, 1)),
            nn.Flatten(),
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(features_out_channels,features_out_channels*2),
            nn.SiLU(),
            #outputs a 3d point in space (use mse or something similar)
            nn.Linear(features_out_channels*2, max_motors * 3)
        )
        
        self.classification_head = nn.Sequential(
            #outputs conf logits? bce
            nn.Linear(features_out_channels,max_motors * 1)
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

    @torch.inference_mode()
    #just disabled until i know eveyrtihng works lol
    # @torch.amp.autocast(device_type='cuda')
    def inference(self,x:torch.Tensor, num_patches_per_batch:int, patch_size:int, stride:int, conf_threshold:float):
        """_summary_
        Args:
            x (torch.Tensor): shape (b,c,d,h,w)
            
            batching not working yet

        Returns:
            _type_: _description_
        """
        #THIS IS FOR TRAINING SCRIPT LATER
        # 1. create a new dir for full tomograms in .pt format (only from my validation stuff since it takes 10000000 gb of data)
        # 2. i have paths to my tomogram patch directories right, so at start of runtime i would reconstruct the full tomogram dirs from my list (i dont want to manually do this)
        # 3. pass in the list of paths to validation
        
    
        
        # 1. create all subpatches 
        # 2. remove padded rows since its nice for forward passes but not here
        # 3. compute global prediction coords
        # 4. concat along dimension 1
        # 5. apply nms / voxel downsampling / whatever
        # 6. profit
        
        D,H,W = x.shape[2:]
        start_d = self._get_start_indices(D, patch_size, stride)
        start_h = self._get_start_indices(H, patch_size, stride)
        start_w = self._get_start_indices(W, patch_size, stride)
        
        #loop through all patches creating batches on the fly
    
        for d, h, w in product(start_d, start_h, start_w):
            #add batching later
            patch = x[d:d+patch_size, h:h+patch_size, w:w+patch_size]
            output = self.forward(patch)
            #output (b, max_motors, 4)
            output[:, :, 3] = torch.sigmoid(output[:, :,  3])#conf scaling
            #apply conf threshold here
            conf_mask = output[:, :, 3] > conf_threshold
            output = output[conf_mask]
            if output.numel() == 0:
                continue
            #now we only have relevant stuff
            #global coords in this case is d,h,w
            #no need to pass stuff in
            #now we need to just add global coords to each dimension lol
            output
            
        #log metrics like how many points were pruned
            
            
            
            

            


    def _get_start_indices(self,length, window_size, stride):
        indices = []
        n_windows = (length - window_size) // stride + 1
        for i in range(n_windows):
            start = i * stride
            indices.append(start)
        last_start = (n_windows - 1) * stride
        if last_start + window_size < length:
            indices.append(length - window_size)
        return indices


if __name__ == '__main__':
    # model = r3d_18(R3D_18_Weights)
    # # conv1_weight = model.stem[0].weight.data  # shape (64, 3, 3, 7, 7)
    print(model.layer4)#should output 512,512,3,3,3

    # print(model)
    # for name, module in model.named_children():
    #     print(name)
