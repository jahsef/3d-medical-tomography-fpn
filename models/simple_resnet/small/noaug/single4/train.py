import monai.transforms
import torch.nn.backends
import torchvision
# import torchvision.transforms as transforms
# import torchvision.transforms.v2 as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torchvision.transforms
from torchvision.ops import sigmoid_focal_loss
from trainer import Trainer
import time
import os
import pandas as pd
from balancedrandomnsampler import RandomNSampler

from pathlib import Path
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motoridentifier import MotorIdentifier
from model_defs.trivialmodel import TrivialModel
from sklearn.model_selection import train_test_split

import utils
from patchtomodataset import PatchTomoDataset

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

# import torchio as tio
import monai
from monai import transforms


def write_tomos(list_val_paths):
    """
    writes tomos if they dont exist, used for validation
    """
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    transform = t.Compose([
    t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
    ])
    
    dst = Path.cwd() / 'normalized_val_fulltomo'
    for patches_path in list_val_paths:
        path:Path

        tomo_id = patches_path.name
        
        print(tomo_id)
        
        tomo_pt_path = dst / Path(str(tomo_id) + '.pt')
        
        if tomo_pt_path.exists():
            continue
        
        print(f'Writing full tomogram: {patches_path.name}')
        #find original images path
        images_path = Path.cwd() / 'original_data/train' / patches_path.name
        
        files = [
            f for f in images_path.rglob('*')
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        
        files = natsorted(files, key=lambda x: x.name)
        
        imgs = [iio.imread(file, mode="L") for file in files]
        
        tomo_array = np.stack(imgs)
        
        # Convert to tensor and normalize
        tomo_tensor = torch.as_tensor(tomo_array)
        tomo_tensor = transform(tomo_tensor)
        
        torch.save(tomo_tensor, tomo_pt_path)

import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * torch.abs(targets-pt)**self.gamma * bce_loss
        return focal_loss.mean()
    
class BCETopKLoss(nn.Module):
    def __init__(self, k=20):
        super().__init__()
        self.k = k
    
    def forward(self, inputs, targets):
        # print(f'input shape : {inputs.shape}')
        # print(f'targets shape : {targets.shape}')
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce_reshaped = bce_loss.view(bce_loss.size(0), -1)  # [B, H*W*D]
        # print(bce_loss.shape)
        # print(bce_reshaped.shape)
        topk_values = torch.topk(bce_reshaped, self.k, dim=1)[0]  # [B, k]
        return topk_values.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Weight based on target values - higher targets get more weight
        weights = 1.0 + (targets * self.pos_weight)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = bce_loss * weights
        return weighted_loss.mean()    


if __name__ == "__main__":
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    train_transform = monai.transforms.Compose([
        #seems to be a sweetspot at least for single sample, dont use for larger datasets prob
        # transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=0.02),
        # transforms.RandShiftIntensityd(keys='patch', offsets=0.02, safe=True, prob=1),
        # transforms.RandAdjustContrastd(keys="patch", gamma=(0.98, 1.02), prob=1),
        # transforms.RandScaleIntensityd(keys="patch", factors=0.02, prob=1),

        
        #these are probably sufficient
        # transforms.RandRotate90d(keys=["patch", "label"], prob=0.5, spatial_axes=[1,2]),
        # transforms.RandFlipd(keys=['patch', 'label'], prob=0.5, spatial_axis=[0,1,2]),
        # transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=[168,304,304], mode='reflect'),
        # transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=[160,288,288], random_center=True),
        
        # transforms.RandRotated(keys=['patch', 'label'], range_x=0.33, range_y=0.33, range_z=0.33, prob=0.25, mode=['trilinear', 'nearest']),
        # transforms.RandZoomd(keys=['patch', 'label'], min_zoom = 0.9, max_zoom = 1.1, prob = 0.25, mode = ['trilinear', 'nearest']),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # np.random.seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    model = MotorIdentifier(dropout_p= 0.15, norm_type="gn")
    model.print_params()
    # time.sleep(1000)


    # print('loading state dict into model\n'*20)
    # model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\simple_resnet/overfit_test/single_medspatial/best.pt'))


    save_dir = './models/simple_resnet/small/noaug/single4/'
    
    os.makedirs(save_dir, exist_ok= True)#just so we dont accidentally overwrite stuff
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]

    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size= 0.95, test_size= 0.05, random_state= 42)
    # train_id_list = train_id_list[:len(train_id_list)//30]
    train_id_list = ['tomo_d7475d']
    
    # val_id_list = val_id_list[:len(val_id_list)//10]
    val_id_list =    []
    # val_id_list = ['tomo_d7475d']
    
    epochs = 25
    
    lr = 1e-3
    batch_size = 1
    batches_per_step = 1 #for gradient accumulation (every n batches we step)
    steps_per_epoch = 10
    
    angstrom_blob_sigma = 200 #this is in real coord space not downsampled
    #gaussian edge weighting basically
    weight_sigma_scale = 1.5 #larger is prolly fine for downscaled heatmaps
    downsampling_factor = 16



    # Collect parameter groups
    backbone_params = list(model.stem.parameters()) + list(model.backbone.parameters())
    decoder_params = list(model.decoder.parameters())
    head_params = list(model.head.parameters())

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr},      # Lower LR for backbone
        {'params': decoder_params, 'lr': lr},            # Normal LR for decoder
        {'params': head_params, 'lr': lr}                # Normal LR for head
    ], weight_decay=1e-4)


    #we only load optimizer state when basically everything we are doing is the same
    #optimizer state has a bunch of running avgs
    
    # print('Loading state dict into optimizer')
    # optimizer_state = torch.load(
    #     r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\simple_resnet/overfit_test/single_medspatial/best_optimizer.pt', 
    #     map_location=device
    # )
    
    # optimizer.load_state_dict(optimizer_state)
    # # Force move optimizer state to device after loading
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if torch.is_tensor(v):
    #             state[k] = v.to(device)
    

    # conf_loss_fn = nn.BCEWithLogitsLoss()
    
    # conf_loss_fn = BCETopKLoss(k = 1000)
    
    #1000 pos weight might not be that crazy
    #since 160x288x288 patches, gauss std dev of 12.5 with 1 motor
    #2.5 std devs of the blob covers about 30k voxels
    #so with 13.3m pixels total 1k weight isnt super crazy
    
    conf_loss_fn = WeightedBCELoss(pos_weight=1000)
    
    # conf_loss_fn = ContinuousFocalLoss(alpha = 1, gamma = 2)
    
    
    train_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = train_transform,
        tomo_id_list= train_id_list
    )
    
    val_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = None,
        tomo_id_list= val_id_list
    )
    
    pin_memory = True
    num_workers =5
    val_workers = 1 
    persistent_workers = True
    prefetch_factor = 1
    
    sampler = RandomNSampler(train_dataset, n = batch_size*batches_per_step*steps_per_epoch)
    # g = torch.Generator()
    # g.manual_seed(42)
    
    train_loader = DataLoader(train_dataset,shuffle = False, sampler = sampler, batch_size = batch_size, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=val_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)


    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*batches_per_step*steps_per_epoch*epochs}')
    
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)#% of steps warmup, 5% is about 2 epochs
    
                
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps= warmup_steps, total_steps= total_steps)


    # Train and validate the model
    trainer = Trainer(
        model=model,
        batches_per_step = batches_per_step,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler = scheduler,
        # regression_loss_fn=regression_loss_fn,
        conf_loss_fn = conf_loss_fn,
        # regression_loss_weight = 1.0,
        # conf_loss_weight= 2.0,
        device=device,
        save_dir = save_dir,
        topk_values=[10, 50, 300]
        )
    
    trainer.train(
        epochs=epochs,
        save_period=1
    )
    