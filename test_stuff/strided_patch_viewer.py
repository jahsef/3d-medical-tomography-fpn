"""Debug viewer for strided patch inference on full tomogram."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from natsort import natsorted
import imageio.v3 as iio

sys.path.append(str(Path(__file__).parent.parent))
from model_defs.motor_detector import MotorDetector

# Config - edit these
TOMO_PATH = Path(r'data/original_data/train/tomo_00e047')
MODEL_PATH = Path(r'models/old_data/parallel_fpn_cornernet_fold0_2/weights/best.pt')
PATCH_SIZE = np.array([160, 288, 288])
OVERLAP = 0.25
DOWNSAMPLING_FACTOR = 16
DEVICE = torch.device('cuda')

# Normalization constants (same as convert_pt.py and inferencing.py)
NORM_MEAN = 0.479915
NORM_STD = 0.224932


def load_tomogram(tomo_path: Path) -> np.ndarray:
    """Load and normalize tomogram (EXACTLY like convert_pt.py)."""
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [f for f in tomo_path.rglob('*') if f.is_file() and f.suffix.lower() in exts]
    files = natsorted(files, key=lambda x: x.name)
    imgs = [iio.imread(f, mode='L') for f in files]
    # Match convert_pt.py EXACTLY: float16 first, then operations
    tomo = np.stack(imgs).astype(np.float16) / 255.0
    tomo = (tomo - NORM_MEAN) / NORM_STD
    return tomo


def generate_patch_origins(tomo_shape: tuple, patch_size: np.ndarray, overlap: float) -> list:
    """Generate strided patch origins like inference pipeline."""
    stride = (patch_size * (1 - overlap)).astype(int)
    origins = []

    for z in range(0, tomo_shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, tomo_shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, tomo_shape[2] - patch_size[2] + 1, stride[2]):
                origins.append(np.array([z, y, x]))

    # Add edge cases if needed
    # Handle last positions if stride doesn't land exactly at end
    max_z = tomo_shape[0] - patch_size[0]
    max_y = tomo_shape[1] - patch_size[1]
    max_x = tomo_shape[2] - patch_size[2]

    return origins


class StridedPatchViewer:
    def __init__(self, tomo: np.ndarray, model, patch_origins: list):
        self.tomo = tomo
        self.model = model
        self.patch_origins = patch_origins
        self.patch_idx = 0
        self.slice_idx = 0

        self.current_patch = None
        self.current_pred = None

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self._load_current_patch()
        self._update_display()
        plt.show()

    def _load_current_patch(self):
        origin = self.patch_origins[self.patch_idx]
        end = origin + PATCH_SIZE

        self.current_patch = self.tomo[origin[0]:end[0], origin[1]:end[1], origin[2]:end[2]]
        self.current_origin = origin

        # Run inference
        patch_tensor = torch.from_numpy(self.current_patch).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            self.current_pred = torch.sigmoid(self.model.forward(patch_tensor)).squeeze().cpu().numpy()

        self.slice_idx = self.current_pred.shape[0] // 2

    def _update_display(self):
        for ax in self.axes:
            ax.clear()

        # Map downsampled slice to real space
        real_slice = self.slice_idx * DOWNSAMPLING_FACTOR + DOWNSAMPLING_FACTOR // 2
        real_slice = min(real_slice, self.current_patch.shape[0] - 1)

        # Left: patch
        self.axes[0].imshow(self.current_patch[real_slice], cmap='gray')
        self.axes[0].set_title(f'Patch (z={real_slice})')
        self.axes[0].axis('off')

        # Right: prediction
        self.axes[1].imshow(self.current_pred[self.slice_idx], cmap='hot', vmin=0, vmax=1)
        self.axes[1].set_title(f'Pred (max={self.current_pred.max():.3f})')
        self.axes[1].axis('off')

        origin = self.current_origin
        self.fig.suptitle(
            f'Origin: ({origin[0]}, {origin[1]}, {origin[2]}) | '
            f'Patch {self.patch_idx+1}/{len(self.patch_origins)} | '
            f'DS Slice {self.slice_idx+1}/{self.current_pred.shape[0]}'
        )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'left':
            self.patch_idx = (self.patch_idx - 1) % len(self.patch_origins)
            self._load_current_patch()
        elif event.key == 'right':
            self.patch_idx = (self.patch_idx + 1) % len(self.patch_origins)
            self._load_current_patch()
        elif event.key == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.current_pred.shape[0] - 1)
        elif event.key == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        elif event.key == 'q':
            plt.close(self.fig)
            return

        self._update_display()


if __name__ == '__main__':
    print(f'Loading model from {MODEL_PATH}')
    model, _ = MotorDetector.load_checkpoint(str(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    print(f'Loading tomogram from {TOMO_PATH}')
    tomo = load_tomogram(TOMO_PATH)
    print(f'Tomogram shape: {tomo.shape}')

    print(f'Generating patch origins with overlap={OVERLAP}')
    origins = generate_patch_origins(tomo.shape, PATCH_SIZE, OVERLAP)
    print(f'Generated {len(origins)} patches')

    StridedPatchViewer(tomo, model, origins)
