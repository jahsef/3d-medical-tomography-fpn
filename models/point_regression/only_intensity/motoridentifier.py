import torch.nn as nn
# from torchvision.models.video.resnet import r3d_18
# from torchvision.models.video.resnet import R3D_18_Weights
import torch
from torchvision.ops import DropBlock3d
from . import nnblock
from itertools import product
from torchio import GridSampler, GridAggregator
import torchio as tio
from torch.utils.data.dataloader import DataLoader



class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors:int):
        self.max_motors = max_motors
        super().__init__()

        
        features_out_channels = 256
        self.features = nn.Sequential(
            #stem
            nn.Conv3d(in_channels=1, out_channels= 16, kernel_size= 3, stride = 1, padding = 1, bias = False),
            # nnblock.PreActResBlock3d(in_channels=16, out_channels=16),#16
            nnblock.PreActResBlock3d(in_channels=16, out_channels=32, stride = 2),#32
            # nnblock.PreActResBlock3d(in_channels=32, out_channels=32),#16
            nnblock.PreActResBlock3d(in_channels=32, out_channels=64, stride = 2),#16
            # nnblock.PreActResBlock3d(in_channels=64, out_channels=64),#16
            nnblock.PreActResBlock3d(in_channels=64, out_channels=128, stride = 2),#8
            # nnblock.PreActResBlock3d(in_channels=128, out_channels=128),#8
            nnblock.PreActResBlock3d(in_channels=128, out_channels=features_out_channels, stride = 2),#4
            # nnblock.PreActResBlock3d(in_channels=features_out_channels, out_channels=features_out_channels),#4
            
        )
        
        
        #we should just get average of 512 feature maps?
        #these are basic blocks not preact so no need to apply activations after
        
        feature_map_size = 4**3
        linear_channels = features_out_channels * feature_map_size
        i_lin_channels = 256
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
    def inference(self, tomo_tensor, batch_size, patch_size, overlap, conf_threshold):
        subject = tio.Subject(image=tio.ScalarImage(tensor=tomo_tensor))
        dataset = tio.SubjectsDataset([subject])
        sampler = GridSampler(dataset[0], patch_size, overlap)
        loader = tio.SubjectsLoader(sampler, batch_size=batch_size)
        
        outputs = []
        for batch in loader:
            images = batch['image'][tio.DATA].to('cuda', non_blocking=True)
            locations = torch.stack([loc[:3] for loc in batch['location']]).to('cuda')

            with torch.amp.autocast(device_type='cuda'):
                output = self._patch_inference(images, conf_threshold)
                # print(output.shape)#batch,
                # print(locations.shape)
                output[:, :3] += locations
                outputs.append(output)
        concat_outputs = torch.cat(outputs, dim=0)
        conf_mask = concat_outputs[:,3] > conf_threshold
        return concat_outputs[conf_mask]


    
    def _patch_inference(self,patch:torch.Tensor,conf_threshold:float):
        """_summary_
        Args:
            patch (torch.Tensor): shape (b,c,d,h,w)

        Returns:
            _type_: _description_ (b,4)
        """
        if patch.ndim == 4:
            patch = patch.unsqueeze(0)
        output = self.forward(patch)
        #output (b, max_motors, 4)
        output[:, :, 3] = torch.sigmoid(output[:, :,  3])#conf scaling
        # mask = output[..., 3] > conf_threshold
        # output = output[mask]
        
        # Create mask based only on confidence scores
        # conf_mask = output[:, :, 3] > conf_threshold
        
        # Set confidence to 0 for low-confidence detections, keep bbox coords
        # output[:, :, 3] = torch.where(conf_mask, output[:, :, 3], torch.zeros_like(output[:, :, 3]))
        
        return output.view(output.shape[0], output.shape[2])
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
