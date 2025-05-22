import torch.nn.backends
import torchvision
# import torchvision.transforms as transforms
import torchvision.transforms.v2 as t
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


if __name__ == "__main__":
    #WE NEED TO USE TORCH VIDEO TRANSFORMS
    #transforms usually work for 2d stuff with chw
    #we need vide ofor c,d,h,w
    #also dont work on batch dimension
    # train_transform = t.Compose([
    #     # t.ToDtype(torch.float16, scale=True),
    #     t.Normalize((0.479915,), (0.224932,))
    # ])
    
    # val_transform = t.Compose([
    #     # t.ToDtype(torch.float16, scale=True),
    #     t.Normalize((0.479915,), (0.224932,))
    # ])

    # device = torch.device('cpu')

    #TODO visualization/logging
    #log f1 beta weighted stuff with precision + recall too
    #plot lr vs epoch
    #log some basic predictions + slices + ground_truth for a few key examples
    #maybe we can run predictions on a few tomos at the end of training then show ground truth vs prediction?

    #apply max, min, and average pooling to get some good visualizations
    #over depth dimension
    #of base image and convolutions??

    #plot predicted on a certain slice, also plot ground truth on the same slice or another one if needed
    
    
    epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_motors = 5#max motors must be the same as our labels
        
    batch_size = 6
    model = MotorIdentifier(max_motors=max_motors)

    master_tomo_path = Path.cwd() / 'normalized_pt_data/train'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    train_set, val_set = train_test_split(tomo_dir_list, train_size= 0.8, test_size= 0.2, random_state= 42)
    patch_training = True
    num_patches = 16
    train_dataset = PatchTomoDataset(train_set, num_patches= num_patches, mmap = False, transform= None)
    val_dataset = PatchTomoDataset(val_set, num_patches= num_patches, mmap = False, transform= None)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory= False   , num_workers=2, persistent_workers= True, prefetch_factor= 1)
    
    #we need to load full tomos for validation
    #create tomos at runtime if they are missing
    #dir full_tomos (need to apply transforms too)
    
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory=False, num_workers=2, persistent_workers= True, prefetch_factor= 1)
    
    #loader yields tuple of tensor, label(tomo_id, shape, coords, mask)
    # time.sleep(1000)
    
    regression_loss_fn = torch.nn.MSELoss()
    
    conf_loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')#default mean
    #we can weight the positives more for better recall
    #actually we can do progressive class weighting
    #as epochs go on, pos weight increases!
    #criterion_conf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
    #label smoothing too?? idk
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay= 1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= epochs, eta_min = 1e-6)

    save_dir = './models/resnet3d_18/base/'
    

    os.makedirs(save_dir, exist_ok= True)
    
    # Train and validate the model
    trainer = Trainer(
        model=model,
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
    
