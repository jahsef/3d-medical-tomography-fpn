import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import imageio.v3 as iio
from natsort import natsorted
import sys

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs._OLD_FPN import MotorIdentifier

# Configuration
device = torch.device('cuda')
model_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn/focal/test/weights\epoch25.pt'
labels_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv'
original_data_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train')
master_tomo_path = Path.cwd() / 'data\processed\patch_pt_data'

batch_size = 6
patch_size = (160,288,288)
overlap = 0.5#when overfitted looks like overlap matters a lot more lol
sigma_scale = 1/8
downsampling_factor = 16

norm_type = "gn"
# Load dataset split
tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.25, random_state=42)
#curriculum4
#1/30:2
#1/15:3
#2/15:4(overwritten by 4/15)
#4/15:4

#after successful curriculum we can move to skip connection ae better enc:dec ratio 4:1 and gaussian weighting for labels (we can keep edge motors)

train_id_list = train_id_list[: :5]
# train_id_list = val_id_list
# train_id_list = ['tomo_d7475d']
# train_id_list = ['tomo_00e047']

print(f'possible samples to inference: {len(train_id_list)}')

def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
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

def get_single_motor_tomograms():
    """Get list of tomogram IDs that have exactly 1 motor."""
    
    # Load labels and filter for single motors
    df = pd.read_csv(labels_path)
    single_motor_mask = df['Number of motors'] == 1
    valid_motors = df[single_motor_mask]
    bool_mask = valid_motors['tomo_id'].isin(train_id_list)
    valid_motors = valid_motors[bool_mask]
    
    return valid_motors

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
    model = MotorIdentifier(norm_type= norm_type).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get valid tomograms
    valid_motors = get_single_motor_tomograms()
    print(f"Found {len(valid_motors)} tomograms with exactly 1 motor")
    print(f"Using downsampling factor: {downsampling_factor}")
    
    for idx, row in valid_motors.iterrows():
        tomo_id = row['tomo_id']
        motor_coords = [row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']]
        
        print(f"\nProcessing {tomo_id}...")
        print(f"Original motor coordinates: {motor_coords}")
        print(f"Downsampled motor coordinates: {[c//downsampling_factor for c in motor_coords]}")
        
        # Load tomogram (now with downsampling applied)
        tomo_path = original_data_path / tomo_id
        tomo = load_tomogram(tomo_path)
        
        if tomo is None:
            print(f"Failed to load {tomo_id}, skipping...")
            continue
        
        print(f"Downsampled tomogram shape: {tomo.shape}")
        
        # Prepare for inference
        original_shape = tomo.shape
        tomo_batch = tomo.reshape(1, 1, *original_shape).to(device)
        
        # Run inference
        print("Running inference...")
        results = model.inference(tomo_batch, batch_size=batch_size, patch_size=patch_size, overlap=overlap, device=device, tqdm_progress=True, sigma_scale=sigma_scale)
        heatmap = results.view(results.shape[2:]).cpu().numpy()
        
        print(f"Heatmap shape: {heatmap.shape}")
        
        # Visualize
        visualize_tomogram_results(tomo_id, heatmap, motor_coords, original_data_path)
    
    print("Visualization complete!")

if __name__ == "__main__":
    run_automated_visualization()