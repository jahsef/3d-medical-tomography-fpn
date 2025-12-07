import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import imageio.v3 as iio
from natsort import natsorted
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs._OLD_FPN import MotorIdentifier

# Configuration (same as visualize_inference.py)
device = torch.device('cuda')
model_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn/focal/test/weights\epoch25.pt'
labels_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv'
original_data_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train')
master_tomo_path = Path.cwd() / 'data\processed\patch_pt_data'

batch_size = 6
patch_size = (160, 288, 288)
overlap = 0.5
sigma_scale = 1/8
downsampling_factor = 16

norm_type = "gn"

# Dataset split (same as visualize_inference.py)
tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.25, random_state=42)
train_id_list = train_id_list[::5]  # Same striding

print(f'Tomograms to visualize: {len(train_id_list)}')


def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized


def load_tomogram(src: Path):
    """Load full tomogram and return both normalized (for inference) and raw (for display)."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    print(f'Loading tomogram: {src.name}')

    files = [f for f in src.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)

    imgs = []
    for file in files:
        img = iio.imread(file, mode="L")
        imgs.append(img)

    raw_array = np.stack(imgs)  # Keep raw for display
    normalized = normalize_tomogram(raw_array)

    return torch.as_tensor(normalized), raw_array


def get_gt_motors(tomo_id, labels_path):
    """Get all ground truth motor locations for a tomogram."""
    df = pd.read_csv(labels_path)
    tomo_rows = df[df['tomo_id'] == tomo_id]

    motors = []
    for _, row in tomo_rows.iterrows():
        if row['Motor axis 0'] != -1:  # Valid motor
            motors.append([
                row['Motor axis 0'],
                row['Motor axis 1'],
                row['Motor axis 2']
            ])
    return motors


class InteractiveTomoViewer:
    """Interactive viewer for tomogram heatmaps with depth slider and keyboard navigation."""

    def __init__(self, tomo_id, heatmap, gt_motors, raw_tomo, downsampling_factor):
        self.tomo_id = tomo_id
        self.heatmap = heatmap  # (D, H, W) in downsampled space
        self.gt_motors = gt_motors  # List of [z, y, x] in original space
        self.raw_tomo = raw_tomo  # (D, H, W) in original space
        self.ds = downsampling_factor

        self.depth_idx = 0
        self.max_depth_ds = heatmap.shape[0] - 1
        self.max_depth_real = raw_tomo.shape[0] - 1

        # Convert GT motors to downsampled space
        self.gt_motors_ds = [[m[0] // self.ds, m[1] // self.ds, m[2] // self.ds] for m in gt_motors]

        self.setup_figure()

    def setup_figure(self):
        """Create figure with subplots and slider."""
        self.fig = plt.figure(figsize=(18, 8))

        # Create axes: left for heatmap, right for original, bottom for slider
        self.ax_heatmap = self.fig.add_axes([0.05, 0.25, 0.4, 0.65])
        self.ax_original = self.fig.add_axes([0.50, 0.25, 0.4, 0.65])
        self.ax_slider = self.fig.add_axes([0.15, 0.08, 0.7, 0.03])
        self.ax_info = self.fig.add_axes([0.05, 0.92, 0.9, 0.06])
        self.ax_info.axis('off')

        # Create slider
        self.slider = Slider(
            ax=self.ax_slider,
            label='Depth (DS)',
            valmin=0,
            valmax=self.max_depth_ds,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.on_slider)

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Initial display
        self.update_display(0)

        # Set window title
        self.fig.canvas.manager.set_window_title(f'Interactive Viewer: {self.tomo_id}')

    def update_display(self, depth_idx):
        """Update both plots for the given depth index."""
        self.depth_idx = depth_idx
        real_depth = min(depth_idx * self.ds + self.ds // 2, self.max_depth_real)

        # Clear axes
        self.ax_heatmap.clear()
        self.ax_original.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')

        # --- Heatmap plot ---
        slice_heatmap = self.heatmap[depth_idx]
        im1 = self.ax_heatmap.imshow(slice_heatmap, cmap='viridis', vmin=0, vmax=1)

        # Add GT motor circles (if within ±1 depth slice)
        for i, motor_ds in enumerate(self.gt_motors_ds):
            if abs(motor_ds[0] - depth_idx) <= 1:
                circle = plt.Circle(
                    (motor_ds[2], motor_ds[1]),  # (x, y) in image coords
                    radius=2,
                    color='orange' if motor_ds[0] == depth_idx else 'yellow',
                    fill=False,
                    linewidth=2,
                    label=f'GT Motor {i}' if motor_ds[0] == depth_idx else None
                )
                self.ax_heatmap.add_patch(circle)

        # Find and mark prediction peak in this slice
        pred_y, pred_x = np.unravel_index(np.argmax(slice_heatmap), slice_heatmap.shape)
        self.ax_heatmap.plot(pred_x, pred_y, 'r+', markersize=10, markeredgewidth=2, label='Slice Max')

        self.ax_heatmap.set_title(f'Heatmap (DS depth {depth_idx})\nMax: {slice_heatmap.max():.4f} | Mean: {slice_heatmap.mean():.4f}')
        self.ax_heatmap.legend(loc='upper right', fontsize=8)

        # --- Original slice plot ---
        original_slice = self.raw_tomo[real_depth]
        self.ax_original.imshow(original_slice, cmap='gray')

        # Add GT motor circles in real space (if within ±ds depth)
        for i, motor in enumerate(self.gt_motors):
            if abs(motor[0] - real_depth) <= self.ds:
                circle = plt.Circle(
                    (motor[2], motor[1]),  # (x, y) in image coords
                    radius=24,
                    color='orange' if abs(motor[0] - real_depth) < self.ds // 2 else 'yellow',
                    fill=False,
                    linewidth=2
                )
                self.ax_original.add_patch(circle)

        self.ax_original.set_title(f'Original (real depth {real_depth})')

        # --- Info panel ---
        info_text = f'Tomogram: {self.tomo_id} | Total Motors: {len(self.gt_motors)} | '
        info_text += f'Depth: {depth_idx} (DS) = {real_depth} (real) | '
        info_text += f'Use Left/Right arrows or slider to navigate'
        self.ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                         transform=self.ax_info.transAxes)

        # Motor coordinates
        if self.gt_motors_ds:
            motor_str = ' | '.join([f'M{i}:({m[2]},{m[1]},{m[0]})' for i, m in enumerate(self.gt_motors_ds)])
            self.ax_info.text(0.5, 0.0, f'GT Motors (x,y,z DS): {motor_str}', ha='center', va='center',
                             fontsize=8, transform=self.ax_info.transAxes)

        self.fig.canvas.draw_idle()

    def on_slider(self, val):
        """Handle slider change."""
        self.update_display(int(val))

    def on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            new_depth = min(self.depth_idx + 1, self.max_depth_ds)
            self.slider.set_val(new_depth)
        elif event.key == 'left':
            new_depth = max(self.depth_idx - 1, 0)
            self.slider.set_val(new_depth)

    def show(self):
        """Display the interactive viewer."""
        plt.show()


def run_interactive_visualization():
    """Main function to run interactive tomogram visualization."""
    # Load model
    model = MotorIdentifier(norm_type=norm_type).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Processing {len(train_id_list)} tomograms")
    print("Close the window to move to next tomogram\n")

    for tomo_id in train_id_list:
        print(f"\n{'='*50}")
        print(f"Processing {tomo_id}...")

        # Get GT motors for this tomogram
        gt_motors = get_gt_motors(tomo_id, labels_path)
        print(f"Found {len(gt_motors)} ground truth motors")

        # Load tomogram
        tomo_path = original_data_path / tomo_id
        tomo_normalized, raw_tomo = load_tomogram(tomo_path)

        print(f"Tomogram shape: {raw_tomo.shape}")

        # Prepare for inference
        tomo_batch = tomo_normalized.reshape(1, 1, *tomo_normalized.shape).to(device)

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            results = model.inference(
                tomo_batch,
                batch_size=batch_size,
                patch_size=patch_size,
                overlap=overlap,
                device=device,
                tqdm_progress=True,
                sigma_scale=sigma_scale
            )
        heatmap = results.view(results.shape[2:]).cpu().numpy()
        print(f"Heatmap shape: {heatmap.shape}")

        # Create interactive viewer
        viewer = InteractiveTomoViewer(
            tomo_id=tomo_id,
            heatmap=heatmap,
            gt_motors=gt_motors,
            raw_tomo=raw_tomo,
            downsampling_factor=downsampling_factor
        )
        viewer.show()

    print("\nVisualization complete!")


if __name__ == "__main__":
    run_interactive_visualization()
