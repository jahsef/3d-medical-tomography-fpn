#added model_defs to path

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
import sys

# Add transforms support
from monai import transforms
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs.motoridentifier import MotorIdentifier

patch_dir = Path.cwd() / 'patch_pt_data' / 'tomo_d7475d'
device = torch.device('cuda')
model = MotorIdentifier().to(device)
model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn/small/noaug/30subset/best.pt'))
model.eval()

DOWNSAMPLING_FACTOR = 16
GAUSSIAN_SIGMA = 12.5  # sigma in realpixel space units

# Define transform pipeline (matching your training augmentations)
test_transforms = Compose([
    # transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=0.4, std=0.2),
    # transforms.RandShiftIntensityd(keys='patch', offsets=0.2, safe=True, prob=0.4),
    # transforms.RandAdjustContrastd(keys="patch", gamma=(0.8, 1.2), prob=0.4),
    # transforms.RandScaleIntensityd(keys="patch", factors=0.2, prob=0.4),
    # transforms.RandCoarseDropoutd(
    #     keys="patch", 
    #     holes=32, 
    #     spatial_size=(10, 20, 20), 
    #     prob=0.4
    # ),
])
# test_transforms = None

def create_gaussian_blob(shape, center, sigmas):
    """Create a 3D gaussian blob"""
    d, h, w = shape
    z, y, x = np.ogrid[:d, :h, :w]
    
    # Create gaussian
    gaussian = np.exp(-((z - center[0])**2 / (2 * sigmas[0]**2) +
                       (y - center[1])**2 / (2 * sigmas[1]**2) +
                       (x - center[2])**2 / (2 * sigmas[2]**2)))
    
    return gaussian

for patch_file in patch_dir.glob('*.pt'):
    print(f"\nProcessing: {patch_file.name}")
    
    patch_dict = torch.load(patch_file)

    transformed_dict = test_transforms(patch_dict)
    patch = transformed_dict['patch'].to(device)
    
    if patch.dim() == 3:
        patch = patch.unsqueeze(0).unsqueeze(0) 
    if patch.dim() == 4:
        patch = patch.unsqueeze(0)
    print(f"Input shape: {patch.shape}")
    
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            results = torch.sigmoid_(model.forward(patch))
    
    label = patch_dict['labels']
    gt_coords_full = label[0, :3].numpy()  # Full resolution coordinates
    gt_coords_down = gt_coords_full // DOWNSAMPLING_FACTOR  # Downsampled coordinates
    
    results = results.reshape(results.shape[2:]).cpu().numpy()
    print(f"Results shape: {results.shape}")
    
    # Create theoretical perfect gaussian
    d_i, h_i, w_i = results.shape
    min_dim = min([d_i, h_i, w_i])
    blob_sigma_d = GAUSSIAN_SIGMA * (d_i / min_dim) / DOWNSAMPLING_FACTOR
    blob_sigma_h = GAUSSIAN_SIGMA * (h_i / min_dim) / DOWNSAMPLING_FACTOR  
    blob_sigma_w = GAUSSIAN_SIGMA * (w_i / min_dim) / DOWNSAMPLING_FACTOR
    
    theoretical_heatmap = create_gaussian_blob(results.shape, gt_coords_down, 
                                             [blob_sigma_d, blob_sigma_h, blob_sigma_w])
    
    # Get slices
    gt_slice = results[gt_coords_down[0], ...]  # GT depth slice
    
    argmax = np.asarray(np.unravel_index(np.argmax(results), shape=results.shape))
    global_max_slice = results[argmax[0], ...]  # Global max depth slice
    
    original_slice = patch[0,0, gt_coords_full[0], ...].cpu().numpy() 
    
    theoretical_slice = theoretical_heatmap[gt_coords_down[0], ...]  # Theoretical perfect slice
    
    # Calculate distances (in downsampled space)
    distance = np.sqrt(np.sum((gt_coords_down - argmax)**2))
    print(f'GT coords (full res): {gt_coords_full}')
    print(f'GT coords (downsampled): {gt_coords_down}')
    print(f'Global max coords: {argmax}')
    print(f'Distance: {distance:.4f}')
    print(f'Blob sigmas (d,h,w): {blob_sigma_d:.2f}, {blob_sigma_h:.2f}, {blob_sigma_w:.2f}')
    
    # Find slice maxima
    gt_slice_max = np.unravel_index(np.argmax(gt_slice), shape=gt_slice.shape)
    global_slice_max = np.unravel_index(np.argmax(global_max_slice), shape=global_max_slice.shape)
    
    # Create 4-panel plot
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # GT slice
    im1 = axes[0].imshow(gt_slice, cmap='plasma', vmin = 0, vmax = 1)
    axes[0].set_title(f'GT Slice (d={gt_coords_down[0]})')
    gt_circle = Circle((gt_coords_down[2], gt_coords_down[1]), radius=2, 
                      fill=False, color='green', linewidth=2)
    pred_circle = Circle((gt_slice_max[1], gt_slice_max[0]), radius=2, 
                        fill=False, color='red', linewidth=2)
    axes[0].add_patch(gt_circle)
    axes[0].add_patch(pred_circle)
    plt.colorbar(im1, ax=axes[0])
    
    # Global max slice
    im2 = axes[1].imshow(global_max_slice, cmap='plasma', vmin = 0, vmax = 1)
    axes[1].set_title(f'Global Max Slice (d={argmax[0]})')
    global_circle = Circle((global_slice_max[1], global_slice_max[0]), radius=2, 
                          fill=False, color='red', linewidth=2)
    axes[1].add_patch(global_circle)
    plt.colorbar(im2, ax=axes[1])
    
    # Original slice
    im3 = axes[2].imshow(original_slice, cmap='gray')
    axes[2].set_title(f'Original Slice (d={gt_coords_full[0]})')
    orig_gt_circle = Circle((gt_coords_full[2], gt_coords_full[1]), radius=3, 
                           fill=False, color='green', linewidth=2)
    axes[2].add_patch(orig_gt_circle)
    plt.colorbar(im3, ax=axes[2])
    
    # Theoretical perfect slice
    im4 = axes[3].imshow(theoretical_slice, cmap='plasma', vmin = 0, vmax = 1)
    axes[3].set_title(f'Theoretical Perfect (d={gt_coords_down[0]})')
    theo_circle = Circle((gt_coords_down[2], gt_coords_down[1]), radius=2, 
                        fill=False, color='green', linewidth=2)
    axes[3].add_patch(theo_circle)
    plt.colorbar(im4, ax=axes[3])
    
    plt.suptitle(f'{patch_file.name} - Realspace dist: {distance*DOWNSAMPLING_FACTOR:.4f}', fontsize=16)
    plt.tight_layout()
    plt.show()