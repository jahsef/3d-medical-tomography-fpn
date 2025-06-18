import monai.inferers
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
import monai



class MotorIdentifier(nn.Module):
    
    def __init__(self,max_motors:int):
        self.max_motors = max_motors
        super().__init__()

        
        FOC = 128  # Reduced from 128

        # Encoder (Downsampling path)
        self.encoder = nn.Sequential(
            #c16, 32^3
            nnblock.PreActResBlock3d(1, FOC//4, stride = 2, kernel_size=5),
            #c32, 16^3
            nnblock.PreActResBlock3d(FOC//4, FOC//2, stride = 2, kernel_size=3),
            #c64, 8^3
            nnblock.PreActResBlock3d(FOC//2, FOC, stride = 2, kernel_size=3),
            nnblock.PreActResBlock3d(FOC, FOC, stride = 1, kernel_size=3),
        )
        
        #encoder increases channels, decreases spatial size
        #decoder obviously decodes
        
        # Decoder (Upsampling path)
        self.decoder = nn.Sequential(
            #c32, 16^3
            # nnblock.UpsamplePreActResBlock3d(FOC,FOC,stride =1, kernel_size= 3),
            nnblock.UpsamplePreActResBlock3d(FOC,FOC//2,stride =2, kernel_size= 3),
            
            #c16, 32^3
            nnblock.UpsamplePreActResBlock3d(FOC//2,FOC//4,stride =2, kernel_size= 3),
            
            #c8, 64^3
            nnblock.UpsamplePreActResBlock3d(FOC//4,FOC//8,stride =2, kernel_size= 3),
        )

        # Classification head
        self.classification_head = nn.Conv3d(FOC//8, 1, kernel_size=1, bias=False)



    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode
        decoded = self.decoder(features)
        
        # Final prediction
        output = self.classification_head(decoded)
        
        return output

    @torch.inference_mode()
    def inference(self, tomo_tensor, batch_size, patch_size, overlap, device, tqdm_progress:bool):
        
        # monai.inferers.inferer.SlidingWindowInfererAdapt()
        #SlidingWindowInfererAdapt extends SlidingWindowInferer to automatically switch to buffered and then to CPU stitching, when OOM on GPU.
        sigmoid_model = MotorIdentifierWithSigmoid(self)
        inferer = monai.inferers.inferer.SlidingWindowInferer(roi_size=patch_size, sw_batch_size=batch_size, overlap=overlap, 
                                                    mode = 'gaussian', sigma_scale= 1/4, device = device, progress = tqdm_progress, buffer_dim=0 )#buffer_steps num iterations before sending patches to gpu?
        with torch.amp.autocast(device_type="cuda"):
            results = inferer(inputs = tomo_tensor, network = sigmoid_model)
        return results

class MotorIdentifierWithSigmoid(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        return torch.sigmoid(self.base_model(x))
    
    # def _get_start_indices(self,length, window_size, stride):
    #     indices = []
    #     n_windows = (length - window_size) // stride + 1
    #     for i in range(n_windows):
    #         start = i * stride
    #         indices.append(start)
    #     last_start = (n_windows - 1) * stride
    #     if last_start + window_size < length:
    #         indices.append(length - window_size)
    #     return indices


if __name__ == '__main__':
    # model = r3d_18(R3D_18_Weights)
    # # conv1_weight = model.stem[0].weight.data  # shape (64, 3, 3, 7, 7)
    print(model.layer4)#should output 512,512,3,3,3

    # print(model)
    # for name, module in model.named_children():
    #     print(name)
