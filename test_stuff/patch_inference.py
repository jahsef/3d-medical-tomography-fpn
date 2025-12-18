#added model_defs to path

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from monai.transforms import Compose

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))

# from models.fpn_comparison.parallel_fpn_cornernet20.model_defs.motor_detector import MotorDetector
from model_defs.motor_detector import MotorDetector
from train.utils import get_tomo_folds

tomo_folds = get_tomo_folds()

# Config
VIS_FOLDS = [0]

master_tomo_path = Path.cwd() / 'data/processed/old_data_300sigma'
tomo_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir() and tomo_folds[dir.name] in VIS_FOLDS]

patch_files = []
for tomo_dir in tomo_list:
    patch_files.extend(tomo_dir.glob('*.pt'))

print(f'Found {len(patch_files)} patches')

print(f'Total tomos in folds {VIS_FOLDS}: {len(tomo_list)}')



checkpoint = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\old_data_300sigma/parallel_fpn_cornernet_fold0/weights/best.pt'
device = torch.device('cpu')
model, _ = MotorDetector.load_checkpoint(checkpoint)
model = model.to(device)
model.eval()
# print("HERE")
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

class InferenceViewer:
    def __init__(self, patch_files, model, device):
        self.patch_files = list(patch_files)
        self.model = model
        self.device = device
        self.patch_idx = 0
        self.slice_idx = 0

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self._load_current_patch()
        self._update_display()
        plt.show()

    def _load_current_patch(self):
        patch_dict = torch.load(self.patch_files[self.patch_idx], weights_only=False)
        self.patch_data = patch_dict['patch'].numpy()

        # Squeeze channel dim if present (old format is 4D)
        if self.patch_data.ndim == 4:
            self.patch_data = self.patch_data.squeeze(0)

        # Handle old format without gaussian key
        if 'gaussian' in patch_dict:
            self.gt_gaussian = patch_dict['gaussian'].numpy()
            if self.gt_gaussian.ndim == 4:
                self.gt_gaussian = self.gt_gaussian.squeeze(0)
            self.has_gt = True
        else:
            # Old format - create zeros array from squeezed patch shape
            ds_shape = tuple(s // DOWNSAMPLING_FACTOR for s in self.patch_data.shape)
            self.gt_gaussian = np.zeros(ds_shape, dtype=np.float32)
            self.has_gt = False

        self.patch_type = patch_dict.get('patch_type', 'unknown')

        # Run inference - handle different patch formats
        patch_tensor = patch_dict['patch'].float()
        if patch_tensor.ndim == 3:  # (D, H, W) - new format
            patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
        elif patch_tensor.ndim == 4:  # (1, D, H, W) - old format with channel
            patch_tensor = patch_tensor.unsqueeze(0)
        patch_tensor = patch_tensor.to(self.device)
        with torch.no_grad():
            self.pred = torch.sigmoid(self.model.forward(patch_tensor)).squeeze().cpu().numpy()

        # Compute metrics (only meaningful if GT exists)
        if self.has_gt:
            self.mae = np.mean(np.abs(self.pred - self.gt_gaussian))
            intersection = np.sum(self.pred * self.gt_gaussian)
            self.dice = 2 * intersection / (np.sum(self.pred) + np.sum(self.gt_gaussian) + 1e-8)
        else:
            self.mae = None
            self.dice = None
        self.max_conf = self.pred.max()

        # Check if max locations match (only meaningful if GT has a motor)
        self.gt_max_idx = [int(i) for i in np.unravel_index(np.argmax(self.gt_gaussian), self.gt_gaussian.shape)]
        self.pred_max_idx = [int(i) for i in np.unravel_index(np.argmax(self.pred), self.pred.shape)]
        self.has_motor = self.has_gt and self.gt_gaussian.max() > 0.1
        self.max_match = self.gt_max_idx == self.pred_max_idx if self.has_motor else None

        # Set initial slice to GT peak depth or center
        self.slice_idx = self.gt_max_idx[0] if self.has_motor else self.gt_gaussian.shape[0] // 2

    def _update_display(self):
        for ax in self.axes:
            ax.clear()

        # Map downsampled slice to real space
        real_slice = self.slice_idx * DOWNSAMPLING_FACTOR + DOWNSAMPLING_FACTOR // 2
        real_slice = min(real_slice, self.patch_data.shape[0] - 1)

        # Left: patch
        self.axes[0].imshow(self.patch_data[real_slice], cmap='gray')
        self.axes[0].set_title(f'Patch (z={real_slice})')
        self.axes[0].axis('off')

        # Middle: inference
        self.axes[1].imshow(self.pred[self.slice_idx], cmap='hot', vmin=0, vmax=1)
        self.axes[1].set_title(f'Pred (max={self.max_conf:.3f})')
        self.axes[1].axis('off')

        # Right: GT gaussian
        self.axes[2].imshow(self.gt_gaussian[self.slice_idx], cmap='hot', vmin=0, vmax=1)
        self.axes[2].set_title(f'GT')
        self.axes[2].axis('off')
        
        # Suptitle with metrics
        if not self.has_gt:
            match_str = f"NO GT (pred={self.pred_max_idx})"
            metrics_str = f"max={self.max_conf:.3f}"
        elif self.max_match is None:
            match_str = f"NO MOTOR (pred={self.pred_max_idx})"
            metrics_str = f"MAE={self.mae:.4f} | Dice={self.dice:.3f}"
        elif self.max_match:
            match_str = "MATCH"
            metrics_str = f"MAE={self.mae:.4f} | Dice={self.dice:.3f}"
        else:
            match_str = f"MISS (pred={self.pred_max_idx} gt={self.gt_max_idx})"
            metrics_str = f"MAE={self.mae:.4f} | Dice={self.dice:.3f}"
        fname = self.patch_files[self.patch_idx].name
        self.fig.suptitle(
            f'{fname} | {self.patch_type} | {metrics_str} | {match_str}\n'
            f'Patch {self.patch_idx+1}/{len(self.patch_files)} | Slice {self.slice_idx+1}/{self.gt_gaussian.shape[0]}'
        )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'left':
            self.patch_idx = (self.patch_idx - 1) % len(self.patch_files)
            self._load_current_patch()
        elif event.key == 'right':
            self.patch_idx = (self.patch_idx + 1) % len(self.patch_files)
            self._load_current_patch()
        elif event.key == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.gt_gaussian.shape[0] - 1)
        elif event.key == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        elif event.key == 'q':
            plt.close(self.fig)
            return

        self._update_display()



InferenceViewer(patch_files, model, device)