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

from natsort import natsorted
import imageio.v3 as iio
import numpy as np

# import torchio as tio
import monai
from monai import transforms
import math

import torch.nn.functional as F


# class PeakBCELoss(nn.Module):
#     def __init__(self, epsilon:float, lambda_param:float, reduction = 'mean'):
#         """PeakBCELoss for continuous regression targets [0,1]. can focus on peaks or be used without peak weighting.

#         Args:
#             epsilon (float) : if target is all 0s, then this is the total weight. so its roughly epsilon vs peak weight (about 15-40 for 10x18x18 sigma about 1.3 (read patchtomodataset.py for more info on how gaussian is computed))
#             lambda_param (float): [1,inf), 1 places no emphasis on peaks, anything above 1 weights peaks higher
            
#             reduction (str, optional): _description_. Defaults to 'mean'.
#         """
#         super().__init__()
#         self.epsilon = epsilon
#         self.lambda_param = lambda_param
#         self.reduction = reduction
    
#     def forward(self, inputs, targets):
#         extra_peak_weighting = targets ** self.lambda_param
#         background_suppression_weighting = self.epsilon + targets #small base weight for background regardless
#         weights = background_suppression_weighting + extra_peak_weighting
#         #normalizing epsilon by patch.numel() makes it patch size invariant
#         #so total background weight is now epsilon * total_peak_weight
#         #for example when eps = 0.1, total_background_weight = 0.1 * total_peak_weight
        
#         #centernet style: y^lambda + (1 - y)^ beta (explicitly tunable background weighting but adds weird nonlinear hyperparam)
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         weighted_loss =  bce_loss * weights
#     #     print(f"Weight stats: min={weights.min():.6f}, max={weights.max():.6f}, "
#     #   f"mean={weights.mean():.6f}, peak_sum={total_peak_weight:.2f}")
#         return weighted_loss.mean() if self.reduction == 'mean' else weighted_loss

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

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pred_prob = F.sigmoid(inputs)
        focal_loss = torch.abs(targets-pred_prob)**self.gamma * bce_loss
        return focal_loss.mean()
    
# class MAEFocalLoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super().__init__()
#         self.gamma = gamma
#     def forward(self, inputs, targets):
#         mae_loss = F.l1_loss(inputs, targets, reduction='none')
#         pred_prob = F.sigmoid(inputs)
#         focal_loss = torch.abs(targets-pred_prob)**self.gamma * mae_loss
#         return focal_loss.mean()

# class MSEFocalLoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super().__init__()
#         self.gamma = gamma
#     def forward(self, inputs, targets):
#         mse_loss = F.mse_loss(inputs, targets, reduction='none')
#         pred_prob = F.sigmoid(inputs)
#         focal_loss = torch.abs(targets-pred_prob)**self.gamma * mse_loss
#         return focal_loss.mean()

class CornerNetFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        """CornerNet-style focal loss for heatmap regression.
        
        Args:
            alpha: focal power for hard examples (default 2)
            beta: gaussian penalty reduction power (default 4)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted logits [B, C, H, W]
            targets: ground truth heatmaps with gaussian bumps [B, C, H, W]
        """
        pred = torch.sigmoid(inputs)
        #need dynamic epsilon based on numerical precision, otherwise can become 1.0 or 0.0 which in log(1-p) or log(p) results in inf which causes explosions
        if pred.dtype == torch.float16:
            eps = 1e-3
        elif pred.dtype == torch.bfloat16:
            eps = 1e-2 
        else:  # float32 or float64
            eps = 1e-6
        pred = torch.clamp(pred, min=eps, max=1 - eps)
        
        # Positive loss: -(1-p)^α * log(p) 
        # Applied where y = 1 (peak centers)
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred)
        
        
        
        # Negative loss: -(1-y)^β * p^α * log(1-p)
        # Applied where y < 1 (background and gaussian falloff)
        # (1-y)^β reduces penalty near peaks (gaussian bumps)
        neg_loss = -((1 - targets) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred)
        #gaussian falloff weighting ((1 - targets) ** self.beta)
        #high confidence weighting (pred ** self.alpha) (if high confidence then the loss matters A LOT MORE)
        #torch.log(1 - pred) (essentially does the same thing)
        
        #proposed
        
        # pos_loss = -(torch.abs(1 - pred) ** self.alpha) * torch.log(pred)
        # neg_loss = -((1 - targets) ** self.beta) * (torch.abs(targets - pred) ** self.alpha) * torch.log(1-pred)
        #gaussian falloff weighting: ((1 - targets) ** self.beta)
        #weighting based on abs error((targets - pred) ** self.alpha) 
        #1-pred is needed to penalize 
        
        
        # Use positive loss where targets are 1 (or very close to 1)
        # Use negative loss elsewhere
        loss = torch.where(targets >= 0.90, pos_loss, neg_loss)
        #modified to be 0.90 since our peaks arent always super close to 1 due to our weighting
        
        return loss.mean()


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
        # 'steps_per_epoch': 128,
        
        # Data
        'angstrom_blob_sigma': 200,
        'downsampling_factor': 16,
        'train_size': 0.25,  
        'random_state': 42, 
        
        # Loss
        'loss_function': 'cornernet',  # 'bce', 'mse', 'focal', 'cornernet'
        'pos_weight':10,#for weighted bce
        'gamma': 2,  # For focal loss
        'alpha': 2.0,  # CenterNet focal power for hard examples
        'beta': 4.0,   # CenterNet background suppression near peaks
        
        
        # Optimizer
        'weight_decay': 1e-4,
        'warmup_ratio': 0.1,
        #how much data u want lol
        'empty_validation': True,
        
        # DataLoader
        'num_workers': 4,
        'val_workers': 1,
        'pin_memory': False,
        'persistent_workers': True,
        'prefetch_factor': 1,

        # Paths
        'save_dir': './models/fpn_comparison/parallel_fpn_cornernet5',  
        'exist_ok':False,
        
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
            'model_path': './models/fpn_comparison/new_fpn2/weights/best.pt',
            'optimizer_path': './models/fpn_comparison/new_fpn2/weights/best_optimizer.pt',
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
            if current_step <= warmup_steps:
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


    
    master_tomo_path = Path.cwd() / 'data/processed/patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]

    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=CONFIG['train_size'], test_size=1-CONFIG['train_size'], random_state=CONFIG['random_state'])
    



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
    elif CONFIG['loss_function'] == 'cornernet':
        conf_loss_fn = CornerNetFocalLoss(alpha=CONFIG['alpha'], beta=CONFIG['beta'])
        print(f"Using Cornernet focal loss with alpha={CONFIG['alpha']}, beta={CONFIG['beta']}")
    else:
        raise ValueError(f"Unknown loss function: {CONFIG['loss_function']}. Choose from: 'vanilla_bce', 'weighted_bce', 'focal', 'centernet'")
    

    
    # Must use CPU for dataset processing when num_workers > 0
    # MONAI MetaTensors on CUDA cannot be pickled across multiprocessing queues
    # IF USING MONAI TRANSFORMS, MUST USE CPU PROCESSING
    angstrom_blob_sigma = CONFIG['angstrom_blob_sigma']
    downsampling_factor = CONFIG['downsampling_factor']
    
    if CONFIG['empty_validation']:
        val_id_list = []
    train_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        downsampling_factor= downsampling_factor,
        transform = train_transform,
        tomo_id_list= train_id_list,
        processing_device='cuda'
    )

    val_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        downsampling_factor= downsampling_factor,
        transform = None,
        tomo_id_list= val_id_list,
        processing_device='cuda'
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
        topk_values=CONFIG['topk_values']
    )
    
    trainer.train(
        epochs=epochs,
        save_period=CONFIG['save_period']
    )
    