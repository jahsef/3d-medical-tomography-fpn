import torch.nn.backends
import torchvision
# import torchvision.transforms as transforms
import torchvision.transforms.v2 as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.ops import sigmoid_focal_loss
from trainer import Trainer
import time
import os
import pandas as pd


from pathlib import Path
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motoridentifier import MotorIdentifier
from sklearn.model_selection import train_test_split

import utils
from patchtomodataset import PatchTomoDataset

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

import torchio as tio
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ChainedScheduler


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
    max_motors = 1#max motors must be the same as our labels
    batch_size = 96
    model = MotorIdentifier(max_motors=max_motors)

    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    # tomo_dir_list = tomo_dir_list[:len(tomo_dir_list)//10]
    # train_set, val_set = train_test_split(tomo_dir_list, train_size= 0.9, test_size= 0.1, random_state= 42)
    
    patch_training = True
    # num_patches = 96
    train_dataset = PatchTomoDataset(tomo_dir_list,patches_per_batch = 96 * 512, transform= train_transform)
    val_dataset = PatchTomoDataset(tomo_dir_list[:2], transform= None)
    
    pin_memory = True
    num_workers = 3
    persistent_workers = True
    prefetch_factor = 1
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    regression_loss_fn = torch.nn.SmoothL1Loss(beta = 1)#lower beta = more robust to outliers
    pos_weight = torch.tensor([5e-5]).to(device)
    conf_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # conf_loss_fn = FocalLoss(alpha = 0.01, gamma = 2)#alpha is pos weight, gamma is focusing param

    epochs = 50
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)#% of steps warmup, 5% is about 2 epochs
    
    batches_per_step = 3 #for gradient accumulation (every n batches we step)
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-3, weight_decay= 1e-3)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps= warmup_steps, total_steps= total_steps)
    
    save_dir = './models/small_custom_cnn/deeper_skinny/'

    os.makedirs(save_dir, exist_ok= True)


    # Train and validate the model
    trainer = Trainer(
        model=model,
        batches_per_step = batches_per_step,
        patch_training=patch_training,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler = scheduler,
        regression_loss_fn=regression_loss_fn,
        conf_loss_fn = conf_loss_fn,
        regression_loss_weight = 1.0,
        conf_loss_weight= 2.0,
        device=device,
        save_dir = save_dir
        )
    
    trainer.train(
        epochs=epochs,
        save_period=0
    )
    