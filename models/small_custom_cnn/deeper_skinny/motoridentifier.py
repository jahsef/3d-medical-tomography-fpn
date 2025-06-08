import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d
from . import nnblock
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio




class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors:int):
        self.max_motors = max_motors
        super().__init__()

        
        features_out_channels = 256
        self.features = nn.Sequential(
            #stem
            nn.Conv3d(in_channels=1, out_channels= 16, kernel_size= 3, stride = 1, padding = 1, bias = False),
            #blocks
            # nnblock.PreActResBlock3d(in_channels=16, out_channels=16),#32
            # nnblock.PreActResBlock3d(in_channels=32, out_channels=32),
            nnblock.PreActResBlock3d(in_channels=16, out_channels=32, stride = 2),#32
            # nnblock.PreActResBlock3d(in_channels=64, out_channels=64),#16
            nnblock.PreActResBlock3d(in_channels=32, out_channels=64, stride = 2),#16
            # nnblock.PreActResBlock3d(in_channels=96, out_channels=96),#16
            nnblock.PreActResBlock3d(in_channels=64, out_channels=128, stride = 2),#8
            # nnblock.PreActResBlock3d(in_channels=192, out_channels=192),#8
            nnblock.PreActResBlock3d(in_channels=128, out_channels=features_out_channels, stride = 2),#4
            # nnblock.PreActResBlock3d(in_channels=features_out_channels, out_channels=features_out_channels),#4

        )
        
        
        #we should just get average of 512 feature maps?
        #these are basic blocks not preact so no need to apply activations after
        feature_map_size = 4**3
        linear_channels = features_out_channels * feature_map_size
        i_lin_channels = 384
        self.intermediate = nn.Sequential(
            nn.BatchNorm3d(features_out_channels),
            nn.SiLU(inplace= True),
            # DropBlock3d(p = 0.1, block_size= 1,inplace=True ),
            nn.Dropout3d(p = 0.05, inplace=True),
            # nn.AdaptiveAvgPool3d(output_size = (1, 1, 1)),
            nn.Flatten(),

        )
        

        self.regression_head = nn.Sequential(
            nnblock.BasicFCBlock(in_features=linear_channels, out_features= i_lin_channels*2, p = 0.1),
            nnblock.BasicFCBlock(in_features=i_lin_channels*2, out_features= i_lin_channels, p = 0.1),
            # nnblock.BasicFCBlock(in_features=i_lin_channels, out_features= i_lin_channels, p = 0.1),
            #outputs a 3d point in space (use mse or something similar)
            nn.Linear(i_lin_channels, max_motors * 3)
        )
        
        self.classification_head = nn.Sequential(
            # nnblock.BasicFCBlock(in_features=linear_channels, out_features= i_lin_channels, p = 0.1),
            nnblock.BasicFCBlock(in_features=linear_channels, out_features= i_lin_channels, p = 0.1),
            nn.Linear(i_lin_channels,max_motors * 1)
        )

    def forward(self,x):
        """

        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, w)

        Returns:
            torch.Tensor: Output tensor of shape (b, max_motors, 4)
                        where the last dimension is [x, y, z, conf]
        """
        
        x = self.features(x)
        # print('here1'*10)
        x = self.intermediate(x)
        # print('here2'*10)
        # x = x.view(x.size(0), -1)
        points = self.regression_head(x)
        # print('here3'*10)
        #points sahpe (b,max_motors,3), conf (b,max_motors)
        points = points.view(-1, self.max_motors, 3)
        # print('here4'*10)
        # points = points.view(-1, self.max_motors, 3).sigmoid() * patch_size  # Constrained to [0,64]
        conf_logits = self.classification_head(x)
        # print('here5'*10)
        
        # Combine along feature dimension
        outputs = torch.cat([points, conf_logits.unsqueeze(-1)], dim=-1)

        return outputs

    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda')
    def inference(self, tomo_tensor, patch_size, overlap, conf_threshold):
        # Convert tensor to TorchIO subject
        subject = tio.Subject({
            #image expects c,d,h,w
            'image': tio.ScalarImage(tensor=tomo_tensor),
        })
        
        # Create sampler with the subject
        sampler = GridSampler(subject, patch_size, overlap)
        # aggregator = GridAggregator(sampler)
        outputs = []
        for patch in sampler:
            #inferencing expects b,c,d,h,w
            # loc = patch['location']
            #for torch we can use the tio.CONSTANTS or just key accessing

            patch_data = patch['image'][tio.DATA].unsqueeze(0)
            # print(patch_data.shape)
            origin = torch.tensor(patch[tio.LOCATION][:3]).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            
            output = self._patch_inference(patch_data, conf_threshold)
            output[:, :3] += origin
            outputs.append(output)

        
        return torch.cat(outputs, dim = 0)
    
    def _patch_inference(self,patch:torch.Tensor,conf_threshold:float):
        """_summary_
        Args:
            patch (torch.Tensor): shape (b,c,d,h,w)

        Returns:
            _type_: _description_
        """
        if patch.ndim == 4:
            patch = patch.unsqueeze(0)
        output = self.forward(patch)
        #output (b, max_motors, 4)
        output[:, :, 3] = torch.sigmoid(output[:, :,  3])#conf scaling
        mask = output[..., 3] > conf_threshold
        output = output[mask]
        return output
        #run point cloud voxel downsample or nms like algorithm
        #return the
        #log metrics like how many points were pruned maybe?
        #if validation then log it/ return it somehow idk
        
        
            
            
            

            


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
