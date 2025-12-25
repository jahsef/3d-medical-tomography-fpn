import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v3 as iio
from natsort import natsorted
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))

from model_defs.motor_detector import MotorDetector
from train.utils import get_tomo_folds, load_ground_truth

# Configuration
device = torch.device('cpu')
model_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\architecture_ablation\backbone_4m_combined_a2b6_fold0\weights\best.pt'
labels_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv'
original_data_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train')

batch_size = 6
patch_size = (160, 288, 288)
overlap = 0.5
sigma_scale = 1/8
downsampling_factor = 16
dtype = torch.float32  # Use float32 for CPU compatibility

# Fold-based filtering
VIS_FOLDS = [0]
tomo_folds = get_tomo_folds()
vis_id_list = [tomo_id for tomo_id, fold in tomo_folds.items() if fold in VIS_FOLDS]
vis_id_list = vis_id_list[::5]
# vis_id_list = ['tomo_00e047']

print(f'possible samples to inference: {len(vis_id_list)}')

def normalize_tomogram(tomo_array):
    """Normalize tomogram: scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float32) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized



def load_tomogram(src: Path):
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    """Process a single tomogram with memory-efficient approach."""
    
    print(f'Loading tomogram: {src.name}')
    
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    
    files = natsorted(files, key=lambda x: x.name)
    
    imgs = []
    for file in files:
        img = iio.imread(file, mode="L")
        imgs.append(img)
    
    tomo_array = np.stack(imgs)
    tomo_array = normalize_tomogram(tomo_array)
    
    return torch.as_tensor(tomo_array)

def load_original_slice(tomo_path: Path, depth_idx: int):
    """Load a specific slice from the original tomogram data."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    
    files = [
        f for f in tomo_path.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    
    files = natsorted(files, key=lambda x: x.name)
    
    slice_img = iio.imread(files[depth_idx], mode="L")
    return slice_img

def get_single_motor_tomograms(ground_truth):
    """Get dict of tomo_id -> motor_coords for tomos with exactly 1 motor."""
    return {tomo_id: motors[0] for tomo_id, motors in ground_truth.items() if len(motors) == 1}

def visualize_tomogram_results(tomo_id, heatmap, ground_truth_coords, original_data_path):
    """Visualize heatmap with ground truth and prediction circles, including original slice."""
    # Original coordinates for the actual slice
    gt_depth = int(ground_truth_coords[0])
    gt_y = int(ground_truth_coords[1])
    gt_x = int(ground_truth_coords[2])
    
    # Scale ground truth coordinates to match heatmap resolution
    gt_depth_ds = int(ground_truth_coords[0] // downsampling_factor)
    gt_y_ds = int(ground_truth_coords[1] // downsampling_factor)
    gt_x_ds = int(ground_truth_coords[2] // downsampling_factor)
    
    # Get the slice at ground truth depth
    slice_heatmap = heatmap[gt_depth_ds, ...]
    
    # Find argmax (prediction) coordinates in this slice
    pred_y_ds, pred_x_ds = np.unravel_index(np.argmax(slice_heatmap), slice_heatmap.shape)
    
    # Convert prediction coordinates to real space
    pred_depth_real = gt_depth  # Same depth as GT for this slice
    pred_y_real = pred_y_ds * downsampling_factor + downsampling_factor // 2
    pred_x_real = pred_x_ds * downsampling_factor + downsampling_factor // 2
    
    # Find the slice with maximum confidence in entire heatmap
    heatmap_max = np.max(heatmap)
    max_depth_idx_ds = np.unravel_index(np.argmax(heatmap), heatmap.shape)[0]
    max_slice_heatmap = heatmap[max_depth_idx_ds, ...]
    max_slice_y_ds, max_slice_x_ds = np.unravel_index(np.argmax(max_slice_heatmap), max_slice_heatmap.shape)
    
    # Convert max confidence coordinates to real space
    max_depth_real = max_depth_idx_ds * downsampling_factor + downsampling_factor // 2
    max_y_real = max_slice_y_ds * downsampling_factor + downsampling_factor // 2
    max_x_real = max_slice_x_ds * downsampling_factor + downsampling_factor // 2
    
    # Calculate confidence values
    gt_conf_val = slice_heatmap[gt_y_ds, gt_x_ds]
    slice_max_conf_val = np.max(slice_heatmap)
    avg_conf = np.average(slice_heatmap)
    
    # Load original slice
    tomo_path = original_data_path / tomo_id
    original_slice = load_original_slice(tomo_path, gt_depth)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 9))
    
    # First plot: Ground truth depth slice (heatmap)
    im1 = ax1.imshow(slice_heatmap, cmap='viridis')
    
    # Add circles for first plot
    circle_gt = plt.Circle((gt_x_ds, gt_y_ds), radius=24//downsampling_factor, 
                          color='orange', fill=False, linewidth=2, label='Ground Truth')
    circle_pred = plt.Circle((pred_x_ds, pred_y_ds), radius=24//downsampling_factor, 
                           color='orangered', fill=False, linewidth=2, label='Prediction (ArgMax)')
    
    ax1.add_patch(circle_gt)
    ax1.add_patch(circle_pred)
    
    # Title with both downsampled and real coordinates
    ax1.set_title(f'Heatmap - GT Depth Slice\n'
                  f'Downsampled: GT({gt_x_ds}, {gt_y_ds}, {gt_depth_ds}) | Pred({pred_x_ds}, {pred_y_ds}, {gt_depth_ds})\n'
                  f'Real Space: GT({gt_x}, {gt_y}, {gt_depth}) | Pred({pred_x_real}, {pred_y_real}, {pred_depth_real})\n'
                  f'Confidence: GT={gt_conf_val:.4f} | Pred={slice_max_conf_val:.4f} | Avg={avg_conf:.4f}')
    ax1.legend()
    plt.colorbar(im1, ax=ax1,shrink = 0.66)
    
    # Second plot: Max confidence slice (heatmap)
    im2 = ax2.imshow(max_slice_heatmap, cmap='viridis')
    
    # Add circle for max confidence location
    circle_max = plt.Circle((max_slice_x_ds, max_slice_y_ds), radius=24//downsampling_factor, 
                           color='red', fill=False, linewidth=2, label='Max Confidence')
    ax2.add_patch(circle_max)
    
    ax2.set_title(f'Max Confidence Slice\n'
                  f'Downsampled: ({max_slice_x_ds}, {max_slice_y_ds}, {max_depth_idx_ds})\n'
                  f'Real Space: ({max_x_real}, {max_y_real}, {max_depth_real})\n'
                  f'Max Confidence: {heatmap_max:.4f}')
    ax2.legend()
    plt.colorbar(im2, ax=ax2 ,shrink = 0.66)
    
    # Third plot: Original slice
    ax3.imshow(original_slice, cmap='gray')
    
    # Add ground truth circle on original slice
    circle_orig_gt = plt.Circle((gt_x, gt_y), radius=24, color='orange', 
                               fill=False, linewidth=2, label='Ground Truth')
    ax3.add_patch(circle_orig_gt)
    
    ax3.set_title(f'Original Slice\n'
                  f'Ground Truth: ({gt_x}, {gt_y}, {gt_depth})\n'
                  f'Tomogram: {tomo_id}')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def run_automated_visualization():
    """Main function to run automated tomogram visualization."""
    # Load model
    detector, _ = MotorDetector.load_checkpoint(model_path)
    detector = detector.to(device).to(dtype)
    detector.eval()

    # Load ground truth (coords in downsampled space)
    ground_truth = load_ground_truth(labels_path, vis_id_list, downsampling_factor)

    # Get valid tomograms with single motor
    single_motor_tomos = get_single_motor_tomograms(ground_truth)
    print(f"Found {len(single_motor_tomos)} tomograms with exactly 1 motor")
    print(f"Using downsampling factor: {downsampling_factor}")

    for tomo_id, motor_coords_ds in single_motor_tomos.items():
        # Convert back to real space for visualization
        motor_coords = [c * downsampling_factor for c in motor_coords_ds]

        print(f"\nProcessing {tomo_id}...")
        print(f"Original motor coordinates: {motor_coords}")
        print(f"Downsampled motor coordinates: {motor_coords_ds}")

        # Load tomogram
        tomo_path = original_data_path / tomo_id
        tomo = load_tomogram(tomo_path)

        if tomo is None:
            print(f"Failed to load {tomo_id}, skipping...")
            continue

        print(f"Tomogram shape: {tomo.shape}")

        # Prepare for inference
        original_shape = tomo.shape
        tomo_batch = tomo.reshape(1, 1, *original_shape).to(device).to(dtype)

        # Run inference
        print("Running inference...")
        results = detector.inference(tomo_batch, batch_size=batch_size, patch_size=patch_size, overlap=overlap, device=device, tqdm_progress=True, sigma_scale=sigma_scale, dtype=dtype)
        heatmap = results.view(results.shape[2:]).cpu().numpy()

        print(f"Heatmap shape: {heatmap.shape}")

        # Visualize (pass real-space coords)
        visualize_tomogram_results(tomo_id, heatmap, motor_coords, original_data_path)

    print("Visualization complete!")

if __name__ == "__main__":
    run_automated_visualization()