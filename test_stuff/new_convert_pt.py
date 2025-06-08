from pathlib import Path
import torch
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count
import torchvision.transforms.v2 as t
import pandas as pd
from collections import defaultdict
import gc
from itertools import product

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    # Convert to float16 and scale to [0,1] (assuming input is uint8 0-255)
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    
    # Apply z-score normalization
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    
    return tomo_normalized

def load_labels(csv_path, global_max_motors=None):
    """Load motor coordinates from CSV file."""
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        tomo_id = row['tomo_id']
        if row['Number of motors'] > 0:
            coords = (row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'])
            if tomo_id not in labels:
                labels[tomo_id] = []
            labels[tomo_id].append(coords)
    
    if global_max_motors is not None:
        for tomo_id in labels:
            labels[tomo_id] = labels[tomo_id][:global_max_motors]
    return labels

def get_start_indices(length, window_size, stride):
    """Generate starting indices for patch extraction with proper boundary handling."""
    if length < window_size:
        return []
    
    indices = []
    start = 0
    while start + window_size <= length:
        indices.append(start)
        start += stride
    
    # Ensure we get the last possible patch
    if indices and indices[-1] + window_size < length:
        indices.append(length - window_size)
    elif not indices:  # If no patches fit, try one at the end
        indices.append(max(0, length - window_size))
    
    return indices

def process_single_tomogram(args):
    """Process a single tomogram with memory-efficient approach."""
    (src, dst, patch_size, stride, labels, patch_max_motors, 
     positive_keep_fraction, global_negative_keep_fraction, min_negative_samples) = args
    
    print(f'Processing tomogram: {src.name}')
    
    tomo_id = src.name
    motors = labels.get(tomo_id, [])
    
    # Load images
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    files = natsorted(files, key=lambda x: x.name)
    
    if not files:
        print(f"No image files found in {src}")
        return
    
    # Load all images into memory
    imgs = []
    for file in files:
        img = iio.imread(file, mode="L")
        imgs.append(img)
    
    # Stack into array and keep as numpy for now
    tomo_array = np.stack(imgs)
    
    # Apply normalization at tomogram level
    tomo_array = normalize_tomogram(tomo_array)
    
    D, H, W = tomo_array.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    print(f"Tomogram shape: {D}x{H}x{W}, Patch size: {pd}x{ph}x{pw}")
    
    # Create output directory
    tomo_dir = dst / tomo_id
    tomo_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert motors to numpy array and validate bounds
    valid_motors = []
    if len(motors) > 0:
        for motor in motors:
            d, h, w = motor
            if (0 <= d < D and 0 <= h < H and 0 <= w < W):
                valid_motors.append(motor)
            else:
                print(f"Warning: Motor at {motor} is outside tomogram bounds {(D,H,W)}")
        
        motors_rounded = np.round(valid_motors).astype(np.int32) if valid_motors else np.empty((0, 3), dtype=np.int32)
    else:
        motors_rounded = np.empty((0, 3), dtype=np.int32)
    
    print(f"Valid motors: {len(motors_rounded)}")
    
    # Generate all possible patch positions as numpy arrays
    start_d = np.array(get_start_indices(D, pd, sd))
    start_h = np.array(get_start_indices(H, ph, sh))
    start_w = np.array(get_start_indices(W, pw, sw))
    
    total_patches = len(start_d) * len(start_h) * len(start_w)
    print(f"Grid positions: {len(start_d)} x {len(start_h)} x {len(start_w)} = {total_patches}")
    
    # First pass: collect all patch coordinates and compute labels
    all_patch_coords = []
    
    for d, h, w in product(start_d, start_h, start_w):
        # Check if patch would have valid shape
        if d + pd <= D and h + ph <= H and w + pw <= W:
            all_patch_coords.append((d, h, w))
    
    print(f"Found {len(all_patch_coords)} valid patch positions, computing labels...")
    
    # Vectorized motor containment check for all patches
    positive_patches_data = []
    negative_patches_data = []
    
    if len(motors_rounded) > 0 and len(all_patch_coords) > 0:
        # Convert to numpy for vectorization
        patch_coords_array = np.array(all_patch_coords)  # Shape: (n_patches, 3)
        patch_starts = patch_coords_array
        patch_ends = patch_starts + np.array(patch_size)
        motors_array = motors_rounded  # Shape: (n_motors, 3)
        
        # Vectorized containment check
        motors_in_patches = (
            (patch_starts[:, np.newaxis, :] <= motors_array[np.newaxis, :, :]) &
            (motors_array[np.newaxis, :, :] < patch_ends[:, np.newaxis, :])
        ).all(axis=2)  # Shape: (n_patches, n_motors)
        
        patches_with_motors = motors_in_patches.any(axis=1)
        
        for patch_idx, (d, h, w) in enumerate(all_patch_coords):
            if patches_with_motors[patch_idx]:
                # Find motors in this patch
                motor_mask = motors_in_patches[patch_idx]
                contained_motors = motors_array[motor_mask]
                
                # Compute local coordinates
                local_motors = []
                patch_start = np.array([d, h, w])
                for motor in contained_motors:
                    local_coords = motor - patch_start
                    local_motors.append((*local_coords, 1))
                
                positive_patches_data.append(((d, h, w), local_motors))
            else:
                negative_patches_data.append(((d, h, w), []))
    else:
        # No motors, all patches are negative
        negative_patches_data = [((d, h, w), []) for d, h, w in all_patch_coords]
    
    print(f"Found {len(positive_patches_data)} positive and {len(negative_patches_data)} negative patches")
    
    # Sample patches
    sampled_positive = positive_patches_data
    if positive_keep_fraction < 1.0 and len(positive_patches_data) > 0:
        n_keep = int(len(positive_patches_data) * positive_keep_fraction)
        n_keep = min(n_keep, len(positive_patches_data))
        if n_keep > 0 and n_keep < len(positive_patches_data):
            indices = np.random.choice(len(positive_patches_data), n_keep, replace=False)
            sampled_positive = [positive_patches_data[i] for i in indices]
    
    target_negatives = max(
        int(len(negative_patches_data) * global_negative_keep_fraction),
        min_negative_samples
    )
    target_negatives = min(target_negatives, len(negative_patches_data))
    
    sampled_negative = negative_patches_data
    if len(negative_patches_data) > target_negatives and target_negatives > 0:
        np.random.shuffle(negative_patches_data)
        sampled_negative = negative_patches_data[:target_negatives]
    
    all_selected = sampled_positive + sampled_negative
    print(f"Processing and saving {len(sampled_positive)} positive and {len(sampled_negative)} negative patches")
    
    # Process and save patches one by one
    positive_count = 0
    negative_count = 0
    saved_patches = 0
    
    for (d, h, w), local_motors in all_selected:
        # Extract patch immediately and make independent copy
        patch_data = tomo_array[d:d+pd, h:h+ph, w:w+pw].copy()
        
        # Create label data
        if len(local_motors) > 0:
            positive_count += 1
            label_data = np.array(local_motors, dtype=np.int32)
            
            # Handle max motors constraint
            if patch_max_motors is not None and len(local_motors) > patch_max_motors:
                label_data = label_data[:patch_max_motors]
            elif patch_max_motors is not None:
                pad_needed = patch_max_motors - len(local_motors)
                if pad_needed > 0:
                    padding = np.zeros((pad_needed, 4), dtype=np.int32)
                    label_data = np.vstack([label_data, padding])
        else:
            negative_count += 1
            if patch_max_motors is not None:
                label_data = np.zeros((patch_max_motors, 4), dtype=np.int32)
            else:
                label_data = np.empty((0, 4), dtype=np.int32)
        
        # Convert to tensors - patch_data is already independent and normalized
        patch_tensor = torch.from_numpy(patch_data).unsqueeze(0)  # Add channel dimension
        
        label_tensor = torch.from_numpy(label_data)
        coords_tensor = torch.from_numpy(np.array([d, h, w], dtype=np.int32))
        
        # Create save dictionary
        save_dict = {
            'patch': patch_tensor,
            'labels': label_tensor,
            'global_coords': coords_tensor,
        }
        
        # Save immediately
        patch_name = f'patch_{d}_{h}_{w}'
        torch.save(save_dict, tomo_dir / f'{patch_name}.pt')
        
        # Clean up immediately
        del patch_data, patch_tensor, label_tensor, coords_tensor, save_dict
        saved_patches += 1
        
        # Periodic garbage collection
        if saved_patches % 100 == 0:
            gc.collect()
    
    # Clear tomogram from memory
    del tomo_array, imgs
    gc.collect()
    
    
    
    actual_positive_ratio = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
    print(f'Finished {tomo_id}: {saved_patches} patches saved, {actual_positive_ratio:.1%} positive ({positive_count} pos, {negative_count} neg)')

def write_whole_directory(
    src, 
    dst, 
    patch_size, 
    stride, 
    csv_path, 
    positive_keep_fraction=0.95,
    global_negative_keep_fraction=0.05,
    min_negative_samples=200,
    global_max_motors=None,
    patch_max_motors=None,
    max_processes=None
):
    """Main function to process all tomograms."""
    labels = load_labels(csv_path, global_max_motors=global_max_motors)
    dst.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded labels for {len(labels)} tomograms")
    for tomo_id, motors in labels.items():
        print(f"  {tomo_id}: {len(motors)} motors")
    
    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    print(f"Found {len(tomo_dirs)} tomogram directories")
    
    args_list = [
        (tomo_dir, dst, patch_size, stride, labels, patch_max_motors, 
         positive_keep_fraction, global_negative_keep_fraction, min_negative_samples)
        for tomo_dir in tomo_dirs
    ]

    max_processes = max_processes or min(cpu_count(), 4)  # Reduced default
    print(f"Using {max_processes} processes")
    
    if max_processes == 1:
        # Serial processing for debugging
        for args in args_list:
            process_single_tomogram(args)
    else:
        with Pool(processes=max_processes) as pool:
            pool.map(process_single_tomogram, args_list)

def analyze_dataset_balance(data_dir: Path):
    """Analyze the balance of positive/negative patches in the dataset."""
    stats = defaultdict(lambda: {'positive': 0, 'negative': 0})
    
    for tomo_dir in data_dir.iterdir():
        print(f'checking: {tomo_dir}')
        if not tomo_dir.is_dir():
            continue
            
        tomo_id = tomo_dir.name
        
        # Look for .pt files directly in the tomo directory
        for pt_file in tomo_dir.glob('*.pt'):
            try:
                data = torch.load(pt_file, map_location='cpu')
                label_data = data['labels']
                has_motor = False
                
                if label_data.numel() > 0:
                    # Check if any row has non-zero coordinates or confidence
                    if label_data.dim() == 2:
                        has_motor = torch.any(label_data != 0).item()
                    else:
                        has_motor = torch.any(label_data != 0).item()
                
                if has_motor:
                    stats[tomo_id]['positive'] += 1
                else:
                    stats[tomo_id]['negative'] += 1
            except Exception as e:
                print(f"Error loading {pt_file}: {e}")
    
    # Print statistics
    total_positive = sum(s['positive'] for s in stats.values())
    total_negative = sum(s['negative'] for s in stats.values())
    total_patches = total_positive + total_negative
    
    print(f"\nDataset Statistics:")
    print(f"Total patches: {total_patches}")
    if total_patches > 0:
        print(f"Positive patches: {total_positive} ({total_positive/total_patches:.1%})")
        print(f"Negative patches: {total_negative} ({total_negative/total_patches:.1%})")
    
    # print(f"\nPer-tomogram breakdown:")
    # for tomo_id, counts in sorted(stats.items()):
    #     total = counts['positive'] + counts['negative']
    #     pos_ratio = counts['positive'] / total if total > 0 else 0
    #     print(f"{tomo_id}: {total} patches, {pos_ratio:.1%} positive ({counts['positive']} pos, {counts['negative']} neg)")

def prune_empty_dirs(master_tomo_dir: Path):
    """Remove empty directories."""
    print('Removing empty directories...')
    removed_count = 0
    tomo_dirs = [x for x in master_tomo_dir.iterdir() if x.is_dir()]
    for dir in tomo_dirs:
        # Check if directory has any .pt files
        pt_files = list(dir.glob('*.pt'))
        
        if len(pt_files) == 0:
            dir.rmdir()
            removed_count += 1
    print(f'Removed {removed_count} empty directories')

if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\patch_pt_data')
    csv_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')
    
    # Parameters
    p = 64
    s = p // 4  # 16 pixel stride
    patch_size = (p, p, p)
    stride = (s, s, s)
    workers = 4  # Start with 1 for debugging

    # Sampling parameters
    positive_keep_fraction = 1.0  # Keep all positive patches
    global_negative_keep_fraction = 0.003  # 0.2% of all negative patches globally
    min_negative_samples = 200  # Minimum negative samples per tomogram
    
    global_max_motors = 20
    patch_max_motors = 1
    
    print("Starting patch extraction...")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Global negative keep fraction: {global_negative_keep_fraction}")
    print(f"Minimum negative samples per tomogram: {min_negative_samples}")
    
    # Run the extraction
    write_whole_directory(
        src=src_root,
        dst=dst_root,
        patch_size=patch_size,
        stride=stride,
        csv_path=csv_path,
        positive_keep_fraction=positive_keep_fraction,
        global_negative_keep_fraction=global_negative_keep_fraction,
        min_negative_samples=min_negative_samples,
        global_max_motors=global_max_motors,
        patch_max_motors=patch_max_motors,
        max_processes=workers
    )
    
    # Clean up empty directories
    prune_empty_dirs(dst_root)
    
    # Analyze the final dataset balance
    analyze_dataset_balance(dst_root)