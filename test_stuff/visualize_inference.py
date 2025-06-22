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
from model_defs.motoridentifier import MotorIdentifier

# Configuration
device = torch.device('cuda')
model_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\fart\run1\best.pt'
labels_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv'
original_data_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
master_tomo_path = Path.cwd() / 'patch_pt_data'

batch_size = 8
patch_size = 96
overlap = 0.25#when overfitted looks like overlap matters a lot more lol
sigma_scale = 1/8

norm_type = "gn"
# Load dataset split
tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.95, test_size=0.05, random_state=42)
#curriculum4
#1/30:2
#1/15:3
#2/15:4(overwritten by 4/15)
#4/15:4

#after successful curriculum we can move to skip connection ae better enc:dec ratio 4:1 and gaussian weighting for labels (we can keep edge motors)

train_id_list = train_id_list[len(train_id_list)*0:len(train_id_list)//10 :]
# train_id_list = val_id_list
# train_id_list = ['tomo_d7475d']

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
    
    if not files:
        print(f"No image files found in {src}")
        return None
    
    imgs = []
    for file in files:
        img = iio.imread(file, mode="L")
        imgs.append(img)
    
    tomo_array = np.stack(imgs)
    tomo_array = normalize_tomogram(tomo_array)
    return torch.as_tensor(tomo_array)

def get_single_motor_tomograms():
    """Get list of tomogram IDs that have exactly 1 motor."""
    
    # Load labels and filter for single motors
    df = pd.read_csv(labels_path)
    single_motor_mask = df['Number of motors'] == 1
    valid_motors = df[single_motor_mask]
    bool_mask = valid_motors['tomo_id'].isin(train_id_list)
    valid_motors = valid_motors[bool_mask]
    
    return valid_motors

def visualize_tomogram_results(tomo_id, heatmap, ground_truth_coords):
    """Visualize heatmap with ground truth and prediction circles."""
    gt_depth = int(ground_truth_coords[0])  # Motor axis 0 is depth
    gt_y = int(ground_truth_coords[1])      # Motor axis 1 is y
    gt_x = int(ground_truth_coords[2])      # Motor axis 2 is x
    
    # Get the slice at ground truth depth
    slice_heatmap = heatmap[gt_depth, ...]
    
    # Find argmax (prediction) coordinates in this slice
    pred_y, pred_x = np.unravel_index(np.argmax(slice_heatmap), slice_heatmap.shape)
    
    # Find the slice with maximum confidence in entire heatmap
    heatmap_max = np.max(heatmap)
    max_depth_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)[0]
    max_slice_heatmap = heatmap[max_depth_idx, ...]
    max_slice_y, max_slice_x = np.unravel_index(np.argmax(max_slice_heatmap), max_slice_heatmap.shape)
    
    # Calculate confidence values
    gt_conf_val = slice_heatmap[gt_y, gt_x]
    slice_max_conf_val = np.max(slice_heatmap)
    avg_conf = np.average(slice_heatmap)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
    
    # First plot: Ground truth depth slice
    im1 = ax1.imshow(slice_heatmap, cmap='viridis')
    
    # Add circles for first plot
    circle_gt = plt.Circle((gt_x, gt_y), radius=24, color='orange', fill=False, linewidth=1, label='Ground Truth')
    circle_pred = plt.Circle((pred_x, pred_y), radius=24, color='orangered', fill=False, linewidth=1, label='Prediction (ArgMax)')
    
    ax1.add_patch(circle_gt)
    ax1.add_patch(circle_pred)
    
    ax1.set_title(f'Tomogram: {tomo_id}, GT Depth: {gt_depth}\n GT: ({gt_x}, {gt_y}) | Pred: ({pred_x}, {pred_y})\n GT/Pred conf: {gt_conf_val:.4f}, {slice_max_conf_val:.4f}\n Avg conf: {avg_conf:.4f}')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # Second plot: Max confidence slice
    im2 = ax2.imshow(max_slice_heatmap, cmap='viridis')
    
    # Add circle for max confidence location
    circle_max = plt.Circle((max_slice_x, max_slice_y), radius=24, color='red', fill=False, linewidth=1, label='Max Confidence')
    ax2.add_patch(circle_max)
    
    ax2.set_title(f'Max Conf Slice, Depth: {max_depth_idx}\n Max Conf Location: ({max_slice_x}, {max_slice_y})\n Heatmap max conf: {heatmap_max:.4f}')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def run_automated_visualization():
    """Main function to run automated tomogram visualization."""
    # Load model

    model = MotorIdentifier(norm_type).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get valid tomograms
    valid_motors = get_single_motor_tomograms()
    print(f"Found {len(valid_motors)} tomograms with exactly 1 motor")
    
    for idx, row in valid_motors.iterrows():
        tomo_id = row['tomo_id']
        motor_coords = [row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']]
        
        print(f"\nProcessing {tomo_id}...")
        
        # Load tomogram
        tomo_path = original_data_path / tomo_id
        tomo = load_tomogram(tomo_path)
        
        if tomo is None:
            print(f"Failed to load {tomo_id}, skipping...")
            continue
        
        # Prepare for inference
        original_shape = tomo.shape
        tomo_batch = tomo.reshape(1, 1, *original_shape).to(device)
        
        # Run inference
        print("Running inference...")
        results = model.inference(tomo_batch, batch_size=batch_size, patch_size=patch_size, overlap=overlap, device=device, tqdm_progress=True, sigma_scale= sigma_scale)
        heatmap = results.view(results.shape[2:]).cpu().numpy()
        
        # Visualize
        visualize_tomogram_results(tomo_id, heatmap, motor_coords)
    
    print("Visualization complete!")

if __name__ == "__main__":
    run_automated_visualization()