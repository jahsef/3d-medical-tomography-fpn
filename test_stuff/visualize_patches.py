from pathlib import Path
import torch
import matplotlib.pyplot as plt
from natsort import natsorted

DOWNSAMPLING_FACTOR = 16


def get_initial_slice(gaussian: torch.Tensor) -> int:
    """Return first peak depth if gaussian has label, else center (in downsampled space)."""
    depth_profile = gaussian.sum(dim=(1, 2))  # sum over H, W
    if depth_profile.max() > 0.1:
        return int(depth_profile.argmax())
    return gaussian.shape[0] // 2


def load_patch(patch_path: Path) -> dict:
    """Load patch data on demand."""
    tomo_id = patch_path.parent.name
    parts = patch_path.stem.split('_')
    coords = (int(parts[1]), int(parts[2]), int(parts[3]))

    data = torch.load(patch_path, weights_only=False)
    return {
        'tomo_id': tomo_id,
        'coords': coords,
        'patch': data['patch'],
        'gaussian': data['gaussian'].squeeze(),
        'patch_type': data['patch_type'],
    }


class PatchViewer:
    def __init__(self, patches_dir: Path):
        self.patch_files = list(patches_dir.rglob('*.pt'))
        self.patch_files = natsorted(self.patch_files, key=lambda p: (p.parent.name, p.name))

        self.patch_idx = 0
        self.slice_idx = 0
        self.current_patch = None

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self._load_current_patch()
        self._update_display()
        plt.show()

    def _load_current_patch(self):
        """Load current patch on demand and set initial slice."""
        self.current_patch = load_patch(self.patch_files[self.patch_idx])
        self.slice_idx = get_initial_slice(self.current_patch['gaussian'])

    def _update_display(self):
        info = self.current_patch
        patch = info['patch']
        gaussian = info['gaussian']
        patch_origin = info['coords']

        # Convert downsampled slice to real space (center of downsampled voxel)
        real_slice_idx = self.slice_idx * DOWNSAMPLING_FACTOR + DOWNSAMPLING_FACTOR // 2
        real_slice_idx = min(real_slice_idx, patch.shape[0] - 1)

        self.axes[0].clear()
        self.axes[0].imshow(patch[real_slice_idx], cmap='gray')
        self.axes[0].set_title('Patch')
        self.axes[0].axis('off')

        self.axes[1].clear()
        self.axes[1].imshow(gaussian[self.slice_idx], cmap='hot', vmin=0, vmax=1)
        self.axes[1].set_title('Label (Gaussian)')
        self.axes[1].axis('off')

        # Find gaussian peaks (conf == 1) and convert to real space
        peak_indices = (gaussian == 1).nonzero()
        if len(peak_indices) > 0:
            peak_coords_str = ", ".join([
                f"({patch_origin[0] + int(p[0]) * DOWNSAMPLING_FACTOR}, "
                f"{patch_origin[1] + int(p[1]) * DOWNSAMPLING_FACTOR}, "
                f"{patch_origin[2] + int(p[2]) * DOWNSAMPLING_FACTOR})"
                for p in peak_indices
            ])
            motor_info = f"Motors: {peak_coords_str}"
        else:
            motor_info = "No motor"

        # Real depth = patch_origin_z + ds_slice * factor
        real_depth = patch_origin[0] + self.slice_idx * DOWNSAMPLING_FACTOR

        title = f"{info['tomo_id']} | origin=({patch_origin[0]}, {patch_origin[1]}, {patch_origin[2]}) | {info['patch_type']}"
        subtitle = f"Patch {self.patch_idx + 1}/{len(self.patch_files)} | DS Slice {self.slice_idx + 1}/{gaussian.shape[0]} | Real Depth: {real_depth}"
        motor_line = motor_info
        self.fig.suptitle(f"{title}\n{subtitle}\n{motor_line}")

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        gaussian = self.current_patch['gaussian']

        if event.key == 'left':
            self.patch_idx = (self.patch_idx - 1) % len(self.patch_files)
            self._load_current_patch()
        elif event.key == 'right':
            self.patch_idx = (self.patch_idx + 1) % len(self.patch_files)
            self._load_current_patch()
        elif event.key == 'up':
            self.slice_idx = min(self.slice_idx + 1, gaussian.shape[0] - 1)
        elif event.key == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        elif event.key == 'q':
            plt.close(self.fig)
            return

        self._update_display()

    
if __name__ == '__main__':
    patches_dir = Path('data/processed/old_data')
    PatchViewer(patches_dir)
