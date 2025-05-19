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
from tomodataset import TomoDataset


if __name__ == "__main__":

    train_transform = t.Compose([
        t.ToDtype(torch.float16, scale=True),
        t.Normalize((0.479915,), (0.224932,))
    ])
    
    val_transform = t.Compose([
        t.ToDtype(torch.float16, scale=True),
        t.Normalize((0.479915,), (0.224932,))
    ])

    batch_size = 128
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    
    #TODO:model architecture (resnet3d-18?) and custom head
    #we need to use adaptive avg pooling to downscale to a predefined size
    #add confidence to the head and add a threshold to assign no point (-1,-1,-1)
    #use mse and bce loss, we can weight these differently too
    #bce more since recall is weighted more
    #distance > 1000 angstroms considered FP, < TP
    
    #break it down into the backbone, then the 2 heads
    #use resnet3d-18 pretrained backbone?
    #model head plan:
    #regression head with mse, confidence head with threshold for bce?
    max_motors = 20
    
    model = MotorIdentifier(max_motors=max_motors)
    # print(model)
    #TODO: custom dataloader
    #we should split into train and val using scikit learn
    #get file paths, dont use torch.load yet
    #then we pass those split lists into our data loaders
    #once we get that we just use loaders as normal
    # okay i have a labels csv and a bunch of .pt files where the csv contains the names of the .pt files
    # I should just create 2 lists of .pt paths and the corresponding csv file rows
    # i can just have a if .pt name == csv row name then add them here or something like that
    # then i can call the scikit learn split
    
    
    tomo_list = utils.create_file_list(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train')
    csv = pd.read_csv(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')
    tomo_csvrow_fart = utils.map_csvs_to_pt(csv, tomo_list, max_motors= max_motors)
    # print(len(tomo_csvrow_fart))
    
    train_set, val_set = train_test_split(tomo_csvrow_fart, train_size= 0.8, test_size= 0.2, random_state= 42)
    train_dataset = TomoDataset(train_set, train_transform)
    val_dataset = TomoDataset(val_set, val_transform)
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory= False   , num_workers=6, persistent_workers= True, prefetch_factor= 4)
    val_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True, pin_memory=False, num_workers=6, persistent_workers= True, prefetch_factor= 4)
    #loader yields tuple of tensor, label(tomo_id, shape, coords, mask)
    
    train_iter = iter(train_loader)
    
    # tomo, label = next(train_iter)
    
    # print(tomo.shape)
    # print(label)
    # print(len(train_set), len(val_set))
    # print(tomo_csvrow_fart[2])
    
    # time.sleep(1000)
    
    regression_loss_fn = torch.nn.MSELoss()
    
    conf_loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')#default mean

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay= 1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= epochs, eta_min = 1e-6)

    save_dir = './models/resnet3d_18/base/'
    

    os.makedirs(save_dir, exist_ok= True)
    
    # Train and validate the model
    trainer = Trainer(
        model=model,
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
        save_period=0,
    )
    
