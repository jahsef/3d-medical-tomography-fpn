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
from balancedrandomnsampler import BalancedRandomNSampler

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
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()



    
if __name__ == "__main__":
    
    train_transform = None
    # intensity
    # rotation
    # scale
    # spatial
    
    train_transform = monai.transforms.Compose([
        #mild intensity
        transforms.RandGaussianNoised(keys = 'patch' ,dtype = torch.float16, prob = 0.5, std = 0.01),
        transforms.RandShiftIntensityd(keys = 'patch', offsets = 0.1,safe = True, prob = 0.50, ),
        
        #mild spatial/rotational
        transforms.RandRotate90d(keys=["patch", "label"], prob=0.5),
        # transforms.SpatialPadd(keys = ['patch', 'label'], spatial_size= [80,80,80], mode = 'reflect'),
        # transforms.RandSpatialCropd(keys = ['patch', 'label'], roi_size = [64,64,64], random_center=True), 
        
        #slightly more aggressive ones below
        # transforms.RandRotated(keys=["patch", "label"], range_x=0.45, range_y=0.45, range_z=0.45, prob=0.5),
        # transforms.RandZoomd(keys=["patch", "label"], min_zoom=0.8, max_zoom=1.2, prob=0.5),
        # transforms.RandGaussianSmoothd(keys="patch", sigma_x=(0.25, 0.35), sigma_y=(0.25, 0.35), sigma_z=(0.25, 0.35), prob=0.5),
        # transforms.RandAdjustContrastd(keys="patch", gamma=(0.8, 1.25), prob=0.5)
        
    ])
    train_transform = None
    
    
    #TODO visualization/logging
    #log f1 beta weighted stuff with precision + recall too
    #plot lr vs epoch
    #log some basic predictions + slices + ground_truth for a few key examples
    #maybe we can run predictions on a few tomos at the end of training then show ground truth vs prediction?

    #apply max, min, and average pooling to get some good visualizations
    #over depth dimension
    #of base image and convolutions??

    #plot predicted on a certain slice, also plot ground truth on the same slice or another one if needed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = MotorIdentifier()
    # model.print_params()
    # time.sleep(1000)
    
    # model = TrivialModel()
    
    print('loading state dict into model\n'*20)
    model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\curriculum\run4\best.pt'))
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Prefreeze Trainable params: {trainable_params}")
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Postfreeze Trainable params: {trainable_params}")
    
    save_dir = './models/heatmap/curriculum/run5/'
    #curriculum //60, //10, //2 , 1?
    os.makedirs(save_dir, exist_ok= True)   
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
    # tomo_id_list = tomo_id_list[:len(tomo_id_list)]
    
    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size= 0.95, test_size= 0.05, random_state= 42)
    train_id_list = train_id_list[:len(train_id_list)//10]
    
    # train_id_list = ['tomo_d7475d']
    # print(f'train tomograms for debugging: {train_id_list}')
    # val_id_list = val_id_list[:len(val_id_list)//10]
    val_id_list = []
    # print(f'validation tomograms for debugging: {val_id_list}')
    
    # train tomograms for debugging: ['tomo_bdc097', 'tomo_d7475d', 'tomo_51a47f', 'tomo_2c607f', 'tomo_975287', 'tomo_51a77e', 'tomo_3e7407', 'tomo_412d88', 'tomo_91beab', 'tomo_cc65a9', 'tomo_1f0e78', 'tomo_e71210', 'tomo_00e463', 'tomo_f36495', 'tomo_6943e6', 'tomo_711fad', 'tomo_aff073', 'tomo_fe050c', 'tomo_24795a', 'tomo_c46d3c', 'tomo_be4a3a', 'tomo_0d4c9e', 'tomo_821255', 'tomo_47ac94', 'tomo_ac4f0d', 'tomo_12f896', 'tomo_675583', 'tomo_20a9ed', 'tomo_b2b342', 'tomo_28f9c1', 'tomo_94c173', 'tomo_935f8a', 'tomo_746d88', 'tomo_8e4919', 'tomo_da79d8', 'tomo_40b215', 'tomo_c36b4b', 'tomo_1af88d', 'tomo_a2a928', 'tomo_13973d', 'tomo_c4db00', 'tomo_568537', 'tomo_101279', 'tomo_512f98', 'tomo_7fbc49', 'tomo_0333fa', 'tomo_f2fa4a', 'tomo_a37a5c', 'tomo_ec607b', 'tomo_a8bf76', 'tomo_dfc627', 'tomo_7a9b64', 'tomo_8b6795', 'tomo_23a8e8', 'tomo_651ecd', 'tomo_67565e', 'tomo_e9fa5f', 'tomo_2bb588', 'tomo_3a0914', 'tomo_10c564', 'tomo_8e30f5']
    # validation tomograms for debugging: ['tomo_fbb49b', 'tomo_56b9a3', 'tomo_e72e60']
    
    epochs =50
    lr = 2e-5
    batch_size = 16
    batches_per_step = 1 #for gradient accumulation (every n batches we step)
    steps_per_epoch = 128
    std_dev = 8

    train_dataset = PatchTomoDataset(
        sigma=std_dev,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = train_transform,
        tomo_id_list= train_id_list
    )
    
    val_dataset = PatchTomoDataset(
        sigma=std_dev,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        transform = None,
        tomo_id_list= val_id_list
    )
    
    pin_memory = True
    num_workers = 3
    val_workers = 1
    persistent_workers = True
    prefetch_factor = 1
    
    sampler = BalancedRandomNSampler(train_dataset, n = batch_size*batches_per_step*steps_per_epoch, balance_ratio= 0.1, class_labels= train_dataset.index_df['has_motor'])
    
    train_loader = DataLoader(train_dataset,sampler = sampler, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=val_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)
    
    #regression loss with all false labels results in nan
    #setting regression loss to 0 in that case
    # regression_loss_fn = torch.nn.SmoothL1Loss(beta = 1)#lower beta = more robust to outliers
    
    # pos_weight = torch.tensor([10]).to(device)
    # conf_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # conf_loss_fn = torch.nn.MSELoss()
    
    conf_loss_fn = FocalLoss(alpha = 20, gamma = 3)#alpha is pos weight, gamma is focusing param
    #lower for early learning, higher to focus on hard examples when fine tuning
    
    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*batches_per_step*steps_per_epoch*epochs}')
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    warmup_steps = int(0.1 * total_steps)#% of steps warmup, 5% is about 2 epochs
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay= 1e-4)
    
    print('Loading state dict into optimizer')
    optimizer_state = torch.load(
        r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\curriculum\run4\best_optimizer.pt', 
        map_location=device
    )
    optimizer.load_state_dict(optimizer_state)

    # Force move optimizer state to device after loading
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                
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
        save_dir = save_dir
        )
    
    trainer.train(
        epochs=epochs,
        save_period=1
    )
    