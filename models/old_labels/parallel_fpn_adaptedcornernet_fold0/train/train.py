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
from pathlib import Path
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motor_detector import MotorDetector
from sklearn.model_selection import train_test_split

import utils
from patchtomodataset import PatchTomoDataset
from augments import get_augmentation_preset
from loss import WeightedBCELoss, BCEFocalLoss, AdaptedCornerNetLoss, FuzzyCornerNetLoss

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

# import torchio as tio
import monai
from monai import transforms
import math

if __name__ == "__main__":
    
    CONFIG = {
        # Model
        'dropout_p': 0.00,
        'drop_path_p': 0.0,
        
        # Training
        'epochs': 69,
        'lr': 2e-4,
        'batch_size': 1,
        'batches_per_step': 5,

        # Model
        'model_name': 'parallel_fpn',  # 'simple_unet', 'parallel_fpn', 'cascade_fpn'
        'model_size': '4m',  # depends on model_name
         
        # Data
        'train_folds': [0],
        'val_folds' :[],
        'dataset_path':Path('./data/processed/old_labels/'),
        'labels_path' :Path('./data/original_data/train_labels.csv'),
        
        # Loss
        'loss_function': 'adaptedcornernet',  # 'bce', 'mse', 'bcefocal', 'weightedbce', 'adaptedcornernet', 'fuzzycornernet'
        'pos_weight':10,#for weighted bce
        'gamma': 2,  # For focal loss
        
        'alpha': 2.0,  # cornernet focal power for hard examples
        'beta': 4.0,   # cornernet background suppression near peaks
        'pos_threshold':0.65, # cornernet adapted for true continuous targets, threshold to swap to pos loss
        
        # Optimizer
        'weight_decay': 1e-4,
        'warmup_ratio': 0.1,
        #how much data u want lol
        
        
        # DataLoader
        'num_workers': 0,
        'val_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': None,
        
        # Paths
        'save_dir': './models/old_labels/parallel_fpn_adaptedcornernet_fold0',  
        'exist_ok':False,
        
        # Other
        'seed': 42,
        'topk_percent_values': [0.1, 1, 10],
        'save_period': 1,
        
        # Feature toggles
        'augmentation_preset': None,  # 'none', 'light', 'medium', 'high'
        'enable_deterministic': False,  # Disabled due to CUDA upsample_trilinear3d_backward_out_cuda non-deterministic implementation
        'load_pretrained': False,
        # Model loading (only used if enabled)
        'pretrained': {
            'model_path': './models/fpn_comparison/new_fpn2/weights/best.pt',
            'optimizer_path': './models/fpn_comparison/new_fpn2/weights/best_optimizer.pt',
            'load_optimizer': True,
        },
        'debug_mode': False,
        

        

    }
    
    save_dir = CONFIG['save_dir']
    if CONFIG['exist_ok'] and os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
        
    os.makedirs(save_dir, exist_ok=CONFIG['exist_ok'])
    
    
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        import math
        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.5 * (1. + math.cos(math.pi * progress)), 0.01)  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # Create augmentation transforms based on preset
    aug_transforms = get_augmentation_preset(CONFIG['augmentation_preset'])
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

    # Load pretrained model if enabled, otherwise create new
    if CONFIG['load_pretrained']:
        print('Loading from checkpoint...')
        detector, optimizer_state = MotorDetector.load_checkpoint(
            path=CONFIG['pretrained']['model_path'],
            dropout_p=CONFIG['dropout_p'],
            drop_path_p=CONFIG['drop_path_p']
        )
    else:
        detector = MotorDetector(
            model_name=CONFIG['model_name'],
            model_size=CONFIG['model_size'],
            dropout_p=CONFIG['dropout_p'],
            drop_path_p=CONFIG['drop_path_p']
        )
        optimizer_state = None

    print(f'MODEL: {CONFIG["model_name"]} ({CONFIG["model_size"]})')
    detector.print_params()


    



    epochs = CONFIG['epochs']
    lr = CONFIG['lr']

    optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=CONFIG['weight_decay'])

    # Load optimizer state if pretrained model is loaded
    if optimizer_state is not None and CONFIG['pretrained']['load_optimizer']:
        print('Loading optimizer state from checkpoint')
        optimizer.load_state_dict(optimizer_state)
        # Force move optimizer state to device after loading
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    # Configure loss function based on config
    if CONFIG['loss_function'] == 'bce':
        conf_loss_fn = nn.BCEWithLogitsLoss()
        print(f"Using vanilla BCE loss")
    elif CONFIG['loss_function'] == 'mse':
        conf_loss_fn = nn.MSELoss()
    elif CONFIG['loss_function'] == 'bcefocal':
        conf_loss_fn = BCEFocalLoss(gamma=CONFIG['gamma'])
        print(f"Using bce focal loss with gamma={CONFIG['gamma']}")
    elif CONFIG['loss_function'] == 'weightedbce':
        conf_loss_fn = WeightedBCELoss(pos_weight = CONFIG['pos_weight'])
        print(f"Using weighted bce loss with pos_weight={CONFIG['pos_weight']}")
    # elif CONFIG['loss_function'] == 'maefocal':
    #     conf_loss_fn = MAEFocalLoss(gamma=CONFIG['gamma'])
    #     print(f"Using mae focal loss with gamma={CONFIG['gamma']}")
    # elif CONFIG['loss_function'] == 'msefocal':
    #     conf_loss_fn = MAEFocalLoss(gamma=CONFIG['gamma'])
    #     print(f"Using mae focal loss with gamma={CONFIG['gamma']}")
    elif CONFIG['loss_function'] == 'adaptedcornernet':
        conf_loss_fn = AdaptedCornerNetLoss(pos_threshold=CONFIG['pos_threshold'], alpha=CONFIG['alpha'], beta=CONFIG['beta'])
        print(f"Using AdaptedCornerNet loss with alpha={CONFIG['alpha']}, beta={CONFIG['beta']}")
    elif CONFIG['loss_function'] == 'fuzzycornernet':
        conf_loss_fn = FuzzyCornerNetLoss(pos_threshold=CONFIG['pos_threshold'], alpha=CONFIG['alpha'], beta=CONFIG['beta'])
        print(f"Using FuzzyCornerNet loss with alpha={CONFIG['alpha']}, beta={CONFIG['beta']}")
    else:
        raise ValueError(f"Unknown loss function: {CONFIG['loss_function']}. Choose from: 'bce', 'mse', 'bcefocal', 'weightedbce', 'adaptedcornernet', 'fuzzycornernet'")
    



    train_dataset = PatchTomoDataset(
        transform = train_transform,
        folds = CONFIG['train_folds'],
        dataset_path=CONFIG['dataset_path'],
        labels_path = CONFIG['labels_path']
    )

    val_dataset = PatchTomoDataset(
        transform = None,
        folds = CONFIG['val_folds'],
        dataset_path=CONFIG['dataset_path'],
        labels_path = CONFIG['labels_path']
    )

    

    batch_size = CONFIG['batch_size']
    batches_per_step = CONFIG['batches_per_step']

    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*len(train_dataset)*epochs}')
    
    total_steps = epochs * math.ceil(len(train_dataset) / batch_size / batches_per_step)
    warmup_steps = int(CONFIG['warmup_ratio'] * total_steps)
    print(f'WARMUP STEPS: {warmup_steps}')
    

    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
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
        detector=detector,
        batches_per_step=batches_per_step,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        conf_loss_fn=conf_loss_fn,
        device=device,
        run_dir=save_dir,
        topk_percent_values=CONFIG['topk_percent_values']
    )
    
    trainer.train(
        epochs=epochs,
        save_period=CONFIG['save_period']
    )
    