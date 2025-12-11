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
from model_defs.simple_unet import MotorIdentifier
from model_defs.parallel_fpn import MotorIdentifier as FPNModel
from sklearn.model_selection import train_test_split

import utils
from patchtomodataset import PatchTomoDataset

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

# import torchio as tio
import monai
from monai import transforms


import torch.nn.functional as F



class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0, reduction = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Weight based on target values - higher targets get more weight
        weights = 1.0 + (targets * self.pos_weight)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = bce_loss * weights
        
        return weighted_loss.mean() if self.reduction == 'mean' else weighted_loss

class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.weighted_bce = WeightedBCELoss(pos_weight=pos_weight, reduction='none')
    def forward(self, inputs, targets):
        bce_loss = self.weighted_bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = torch.abs(targets-pt)**self.gamma * bce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    
    CONFIG = {
        # Model
        'dropout_p': 0.00,
        'drop_path_p': 0.00,
        'norm_type': 'gn',
        
        # Training
        'epochs': 100,
        'lr': 5e-4,
        'batch_size': 1,
        'batches_per_step': 5,
        # 'steps_per_epoch': 128,
        
        # Data
        'angstrom_blob_sigma': 200,
        'weight_sigma_scale': 1.5,
        'downsampling_factor': 16,
        'train_size': 0.25,  
        'random_state': 42,
        
        # Loss
        'loss_function': 'vanilla_bce',  # 'vanilla_bce', 'weighted_bce', 'focal'
        'pos_weight': 3,
        'gamma': 1.2,  # For focal loss
        
        
        # Optimizer
        'weight_decay': 1e-4,
        'backbone_lr_factor': 1.0,
        'warmup_ratio': 0.1,
        #how much data u want lol
        'use_subset': False,
        'subset_fraction': 1,
        'empty_validation': True,
        
        # DataLoader
        'num_workers': 4,
        'val_workers': 2,
        'pin_memory': False,
        'persistent_workers': True,
        'prefetch_factor': 1,
        
        # Paths
        'save_dir': './models/fpn/vanilla_bce/test',
        'exist_ok':True,
        
        # Other
        'seed': 42,
        'topk_values': [10, 50, 300],
        'save_period': 1,
        
        # Feature toggles
        'enable_augmentation': False,
        'enable_deterministic': False,  # Disabled due to CUDA upsample_trilinear3d_backward_out_cuda non-deterministic implementation
        'load_pretrained': False,
        # Model loading (only used if enabled)
        'pretrained': {
            'model_path': r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\simple_resnet/med/noaug/full2/best.pt',
            'optimizer_path': r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\simple_resnet/med/noaug/full2/best_optimizer.pt',
            'load_optimizer': True,
        },
        'debug_mode': False,

        
        
        # Augmentation settings (only used if enabled)
        'augmentation': {
            'gaussian_std': 0.02,
            'shift_offsets': 0.02,
            'contrast_gamma': (0.98, 1.02),
            'scale_factors': 0.02,
            'rotate90_prob': 0.5,
            'flip_prob': 0.5,
            'spatial_pad_size': [168, 304, 304],
            'spatial_crop_size': [160, 288, 288],
            'rotation_range': 0.33,
            'rotation_prob': 0.25,
            'zoom_range': (0.9, 1.1),
            'zoom_prob': 0.25,
        }
        

        

    }
    
    save_dir = CONFIG['save_dir']
    os.makedirs(save_dir, exist_ok=CONFIG['exist_ok'])
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # Create augmentation transforms based on config
    aug_transforms = []
    if CONFIG['enable_augmentation']:
        aug = CONFIG['augmentation']
        aug_transforms.extend([
            # transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=aug['gaussian_std']),
            # transforms.RandShiftIntensityd(keys='patch', offsets=aug['shift_offsets'], safe=True, prob=1),
            # transforms.RandAdjustContrastd(keys="patch", gamma=aug['contrast_gamma'], prob=1),
            # transforms.RandScaleIntensityd(keys="patch", factors=aug['scale_factors'], prob=1),
            transforms.RandRotate90d(keys=["patch", "label"], prob=aug['rotate90_prob'], spatial_axes=[1,2]),
            transforms.RandFlipd(keys=['patch', 'label'], prob=aug['flip_prob'], spatial_axis=[0,1,2]),
            transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=aug['spatial_pad_size'], mode='reflect'),
            transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=aug['spatial_crop_size'], random_center=True),
            # transforms.RandRotated(keys=['patch', 'label'], range_x=aug['rotation_range'], range_y=aug['rotation_range'], range_z=aug['rotation_range'], prob=aug['rotation_prob'], mode=['trilinear', 'nearest']),
            # transforms.RandZoomd(keys=['patch', 'label'], min_zoom=aug['zoom_range'][0], max_zoom=aug['zoom_range'][1], prob=aug['zoom_prob'], mode=['trilinear', 'nearest']),
        ])
    
    train_transform = monai.transforms.Compose(aug_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Set deterministic behavior if enabled
    if CONFIG['enable_deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    model = FPNModel(dropout_p=CONFIG['dropout_p'], norm_type=CONFIG['norm_type'], drop_path_p= CONFIG['drop_path_p'])
    print(f'MODEL TYPE: {type(model)}')
    model.print_params()
    
    

    # Load pretrained model if enabled
    if CONFIG['load_pretrained']:
        print('Loading state dict into model\n' * 20)
        model.load_state_dict(torch.load(CONFIG['pretrained']['model_path']))


    
    master_tomo_path = Path.cwd() / 'data/processed/patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]

    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=CONFIG['train_size'], test_size=1-CONFIG['train_size'], random_state=CONFIG['random_state'])
    



    epochs = CONFIG['epochs']
    lr = CONFIG['lr']
    # Optimizer - handle parameter groups if model has backbone/stem
    if hasattr(model, 'backbone') and hasattr(model, 'stem'):
        params = [
            {'params': list(model.stem.parameters()) + list(model.backbone.parameters()), 
             'lr': lr * CONFIG['backbone_lr_factor']},
            {'params': list(model.head.parameters()), 'lr': lr},
            
        ]
        if hasattr(model, 'decoder'):
            params.append({'params': list(model.decoder.parameters()), 'lr': lr})
            #TODO: SHOULD FIX THIS CONDITIONAL LOGIC LATER TO BE LESS STUPID LOL
        optimizer = torch.optim.AdamW(params, weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])

    # Load optimizer state if pretrained model is loaded
    if CONFIG['load_pretrained'] and CONFIG['pretrained']['load_optimizer']:
        print('Loading state dict into optimizer')
        optimizer_state = torch.load(CONFIG['pretrained']['optimizer_path'], map_location=device)
        optimizer.load_state_dict(optimizer_state)
        # Force move optimizer state to device after loading
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    # Configure loss function based on config
    if CONFIG['loss_function'] == 'vanilla_bce':
        conf_loss_fn = nn.BCEWithLogitsLoss()
        print(f"Using vanilla BCE loss")
    elif CONFIG['loss_function'] == 'weighted_bce':
        conf_loss_fn = WeightedBCELoss(pos_weight=CONFIG['pos_weight'])
        print(f"Using weighted BCE loss with pos_weight={CONFIG['pos_weight']}")
    elif CONFIG['loss_function'] == 'focal':
        conf_loss_fn = FocalLoss(pos_weight=CONFIG['pos_weight'], gamma=CONFIG['gamma'])
        print(f"Using focal loss with pos_weight={CONFIG['pos_weight']}, gamma={CONFIG['gamma']}")
    else:
        raise ValueError(f"Unknown loss function: {CONFIG['loss_function']}. Choose from: 'vanilla_bce', 'weighted_bce', 'focal'")
    

    
    # Must use CPU for dataset processing when num_workers > 0
    # MONAI MetaTensors on CUDA cannot be pickled across multiprocessing queues
    # IF USING MONAI TRANSFORMS, MUST USE CPU PROCESSING
    angstrom_blob_sigma = CONFIG['angstrom_blob_sigma']
    weight_sigma_scale = CONFIG['weight_sigma_scale']
    downsampling_factor = CONFIG['downsampling_factor']
    # Apply dataset filtering based on config
    if CONFIG['use_subset']:
        train_id_list = train_id_list[:int(len(train_id_list) * CONFIG['subset_fraction'])]
        val_id_list = val_id_list[:int(len(val_id_list) * CONFIG['subset_fraction'])]
    if CONFIG['empty_validation']:
        val_id_list = []
    train_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        transform = train_transform,
        tomo_id_list= train_id_list,
        processing_device='cuda'
    )

    val_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        transform = None,
        tomo_id_list= val_id_list,
        processing_device='cuda'
    )

    

    batch_size = CONFIG['batch_size']
    batches_per_step = CONFIG['batches_per_step']


    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*len(train_dataset)*epochs}')
    
    total_steps = epochs *len(train_dataset)
    warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)
    print(f'WARMUP STEPS: {warmup_steps}')
    
    # sampler = RandomNSampler(train_dataset, n=batch_size*len(train_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        # sampler=sampler, 
        batch_size=batch_size, 
        pin_memory=CONFIG['pin_memory'], 
        num_workers=CONFIG['num_workers'], 
        persistent_workers=CONFIG['persistent_workers'], 
        prefetch_factor=CONFIG['prefetch_factor']
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=CONFIG['pin_memory'], 
        num_workers=CONFIG['val_workers'], 
        persistent_workers=CONFIG['persistent_workers'], 
        prefetch_factor=CONFIG['prefetch_factor']
    )
                
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # Train and validate the model
    trainer = Trainer(
        model=model,
        batches_per_step=batches_per_step,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        conf_loss_fn=conf_loss_fn,
        device=device,
        run_dir=save_dir,
        topk_values=CONFIG['topk_values']
    )
    
    trainer.train(
        epochs=epochs,
        save_period=CONFIG['save_period']
    )
    