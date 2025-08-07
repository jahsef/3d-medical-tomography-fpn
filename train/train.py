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
    
    def forward(self, inputs, targets, reduction = 'mean'):
        # Weight based on target values - higher targets get more weight

        
        weights = 1.0 + (targets * self.pos_weight)
        inputs = torch.clamp(inputs, min=-15.0, max=15.0)
        assert not torch.isnan(inputs).any(), 'inputs nan yo'
        assert not torch.isnan(targets).any(), 'targets nan yo'
        assert not torch.isinf(inputs).any(), 'inputs inf yo'
        assert not torch.isinf(targets).any(), 'targets inf yo'
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = torch.clamp(bce_loss, min=1e-6, max=10.0) * weights
        loss = weighted_loss.mean() if reduction == 'mean' else weighted_loss 
        
        assert not torch.isnan(loss).any(), 'loss nan yo'
        assert not torch.isinf(loss).any(), 'loss inf yo'
        
        return loss

class FocalWrappedWeightedBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.gamma = gamma
        self.weighted_bce = WeightedBCELoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        
        eps = 1e-6
        inputs = torch.clamp(inputs, min=-15.0, max=15.0)
        pt = torch.sigmoid(inputs)
        focal_weight = torch.clamp(torch.abs(targets - pt), min=eps) ** self.gamma
        base_loss = self.weighted_bce(inputs, targets, reduction='none')  # modify WeightedBCELoss to allow reduction override
        product = focal_weight * base_loss    
        loss = (product).mean()
        assert not torch.isnan(loss).any(), 'loss function nan yo'
        
        return loss


if __name__ == "__main__":
    import math
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + math.cos(math.pi * progress)) + 0.01  # Min LR = 1% of initial LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_polynomial_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return (current_step / float(warmup_steps)) ** 0.5
            
            denom = max(1, total_steps - warmup_steps)
            progress = (current_step - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)  # Clamp to [0, 1]

            decay_power = progress ** 0.8
            decay_power = min(max(decay_power, 0.0), 1.0)  # Extra safety

            cosine_decay = 0.5 * (1. + math.cos(math.pi * decay_power))
            return max(cosine_decay + min_lr_ratio, 0.0)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
    def get_linear_warmup(optimizer, warmup_steps):
        """Linear warmup then constant LR"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    train_transform = monai.transforms.Compose([
        #seems to be a sweetspot at least for single sample, dont use for larger datasets prob
        # transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=0.02),
        # transforms.RandShiftIntensityd(keys='patch', offsets=0.02, safe=True, prob=1),
        # transforms.RandAdjustContrastd(keys="patch", gamma=(0.96, 1.04), prob=0.6),
        # transforms.RandScaleIntensityd(keys="patch", factors=0.03, prob=0.6),

        
        #these are probably sufficient
        # transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=[176,312,312], mode='reflect'),
        # transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=[160,288,288], random_center=True),
        
        transforms.RandRotate90d(keys=["patch", "label"], prob=0.5, spatial_axes=[1,2]),
        transforms.RandFlipd(keys=['patch', 'label'], prob=0.5, spatial_axis=[0,1,2]),

        #range x does not affect depth dimension
        # transforms.RandRotated(keys=['patch', 'label'], range_x=0.25, range_y=0.1, range_z=0.1, prob=0.07, mode=['trilinear', 'trilinear']),
        # transforms.RandZoomd(keys=['patch', 'label'], min_zoom = 0.925, max_zoom = 1.075, prob = 0.07, mode = ['trilinear', 'trilinear']),
    ])
    # train_transform = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # np.random.seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)#trilinear conflict

    model = MotorIdentifier(dropout_p= 0.25, norm_type="gn")
    model.print_params()
    # time.sleep(1000)
    
    print('loading state dict into model\n'*20)
    model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp/models/relabel/full/lite_augs/24m/best.pt'))
    
    save_dir = './models/relabel/full/lite_augs/24m2/'
    patch_index_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_relabel_index.csv')
    dataset_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\relabel_data')
    labels_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\relabel.csv')
    os.makedirs(save_dir, exist_ok= True)#just so we dont accidentally overwrite stuff
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]

    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size= 0.85, test_size= 0.15, random_state= 42)
    
    train_id_list = train_id_list[:len(train_id_list)]
    # train_id_list = ['tomo_d7475d']
    
    # val_id_list = val_id_list[:len(val_id_list)//10]
    val_id_list =    []
    # val_id_list = ['tomo_d7475d']
    
    epochs = 10
    lr = 1e-3  
    batch_size = 1
    batches_per_step = 2 #for gradient accumulation (every n batches we step)
    steps_per_epoch = 512
    
    angstrom_blob_sigma = 200 #this is in real coord space not downsampled
    #gaussian edge weighting basically
    weight_sigma_scale = 1.5 #larger is prolly fine for downscaled heatmaps
    downsampling_factor = 16
    print(f'TOTAL EXPECTED PATCHES TRAINED: {batch_size*batches_per_step*steps_per_epoch*epochs}')

    backbone_params = list(model.stem.parameters()) + \
                    list(model.enc_2.parameters()) + \
                    list(model.enc_4.parameters()) + \
                    list(model.enc_8.parameters()) + \
                    list(model.enc_16.parameters()) + \
                    list(model.enc_32.parameters())

    interp_params = list(model.interp_r_2.parameters()) + \
                    list(model.interp_r_4.parameters()) + \
                    list(model.interp_r_8.parameters()) + \
                    list(model.interp_r_16.parameters()) + \
                    list(model.interp_r_32.parameters())

    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr / 10},
        {'params': interp_params,  'lr': lr},
        {'params': head_params,    'lr': lr}
    ], weight_decay=5e-4)
    
    

    for name, param in model.named_parameters():
        if param.requires_grad and not any(param is p for g in optimizer.param_groups for p in g['params']):
            print(f"{name} is not in optimizer!")




    #we only load optimizer state when basically everything we are doing is the same
    #optimizer state has a bunch of running avgs
    
    print('Loading state dict into optimizer')
    optimizer_state = torch.load(
        r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models/relabel/full/lite_augs/24m/best_optimizer.pt', 
        map_location=device
    )
    
    optimizer.load_state_dict(optimizer_state)
    # Force move optimizer state to device after loading
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    
    # conf_loss_fn = FocalWrappedWeightedBCELoss(gamma = 1.0, pos_weight=50)
    conf_loss_fn = WeightedBCELoss(pos_weight=420)
    
    train_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        patch_index_path=patch_index_path,
        dataset_path=dataset_path,
        labels_path=labels_path,
        transform = train_transform,
        tomo_id_list= train_id_list
    )
    
    val_dataset = PatchTomoDataset(
        angstrom_blob_sigma=angstrom_blob_sigma,
        sigma_scale=weight_sigma_scale,
        downsampling_factor= downsampling_factor,
        patch_index_path=patch_index_path,
        dataset_path=dataset_path,
        labels_path=labels_path,
        transform = None,
        tomo_id_list= val_id_list
    )
    
    steps_per_epoch = min(steps_per_epoch, len(train_dataset) // (batches_per_step*batch_size))
    if steps_per_epoch == len(train_dataset) // (batches_per_step*batch_size):
        print(f'WARNING WARNING WARNING steps per epoch truncated to len of training set')
        
        
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)#% of steps warmup, 5% is about 2 epochs
    
    print(f'WARMUP STEPS: {warmup_steps}')
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps= warmup_steps, total_steps=total_steps)
    
    pin_memory = True
    num_workers =6
    val_workers = 1
    persistent_workers = True
    prefetch_factor = 1
    val_batch_mult = 1
    
    sampler = RandomNSampler(train_dataset, n = batch_size*batches_per_step*steps_per_epoch)
    # g = torch.Generator()
    # g.manual_seed(42)
    
    train_loader = DataLoader(train_dataset,shuffle = False, sampler = sampler, batch_size = batch_size, pin_memory =pin_memory, num_workers=num_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size = batch_size*val_batch_mult, shuffle = False, pin_memory =pin_memory, num_workers=val_workers, persistent_workers= persistent_workers, prefetch_factor= prefetch_factor)



                
    

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
        topk_values=[5, 100, 1000]
        )
    
    trainer.train(
        epochs=epochs,
        save_period=1
    )
    