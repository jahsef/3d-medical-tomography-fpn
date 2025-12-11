from pathlib import Path
import torch
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count
import pandas as pd
import gc
from itertools import product

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized

def load_labels(csv_path):
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
    
    if indices and indices[-1] + window_size < length:
        indices.append(length - window_size)
    elif not indices:
        indices.append(max(0, length - window_size))
    
    return indices

def process_single_tomogram(**kwargs):
    """Process a single tomogram."""
    src = kwargs['src']
    dst = kwargs['dst']
    patch_size = kwargs['patch_size']
    stride = kwargs['stride']
    labels = kwargs['labels']
    positive_keep_fraction = kwargs['positive_keep_fraction']
    global_negative_keep_fraction = kwargs['global_negative_keep_fraction']
    min_negative_samples = kwargs['min_negative_samples']

    print(f'Processing tomogram: {src.name}')

    tomo_id = src.name
    motors = labels.get(tomo_id, [])

    # Load images
    files = [f for f in src.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)

    if not files:
        print(f"No image files found in {src}")
        return

    # Load and normalize tomogram
    imgs = [iio.imread(file, mode="L") for file in files]
    tomo_array = normalize_tomogram(np.stack(imgs))

    D, H, W = tomo_array.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    print(f"Tomogram shape: {D}x{H}x{W}, Patch size: {pd}x{ph}x{pw}")

    # Create output directory
    tomo_dir = dst / tomo_id
    tomo_dir.mkdir(parents=True, exist_ok=True)

    # Validate motor bounds
    valid_motors = []
    for motor in motors:
        d, h, w = motor
        if (0 <= d < D and 0 <= h < H and 0 <= w < W):
            valid_motors.append(motor)
        else:
            print(f"Warning: Motor at {motor} is outside tomogram bounds {(D,H,W)}")

    motors_rounded = np.round(valid_motors).astype(np.int32) if valid_motors else np.empty((0, 3), dtype=np.int32)
    print(f"Valid motors: {len(motors_rounded)}")

    # Generate all patch positions
    start_d = np.array(get_start_indices(D, pd, sd))
    start_h = np.array(get_start_indices(H, ph, sh))
    start_w = np.array(get_start_indices(W, pw, sw))

    all_patch_coords = [(d, h, w) for d, h, w in product(start_d, start_h, start_w)
                        if d + pd <= D and h + ph <= H and w + pw <= W]

    print(f"Found {len(all_patch_coords)} valid patch positions")

    # Vectorized motor containment check
    positive_patches_data = []
    negative_patches_data = []

    if len(motors_rounded) > 0 and len(all_patch_coords) > 0:
        patch_coords_array = np.array(all_patch_coords)
        patch_starts = patch_coords_array
        patch_ends = patch_starts + np.array(patch_size)
        motors_array = motors_rounded

        # Vectorized containment check
        motors_in_patches = (
            (patch_starts[:, np.newaxis, :] <= motors_array[np.newaxis, :, :]) &
            (motors_array[np.newaxis, :, :] < patch_ends[:, np.newaxis, :])
        ).all(axis=2)

        patches_with_motors = motors_in_patches.any(axis=1)

        for patch_idx, (d, h, w) in enumerate(all_patch_coords):
            if patches_with_motors[patch_idx]:
                motor_mask = motors_in_patches[patch_idx]
                contained_motors = motors_array[motor_mask]

                # Keep patch - compute local coordinates
                local_motors = []
                patch_start = np.array([d, h, w])
                for motor in contained_motors:
                    local_coords = motor - patch_start
                    local_motors.append((*local_coords, 1))

                positive_patches_data.append(((d, h, w), local_motors))
            else:
                negative_patches_data.append(((d, h, w), []))
    else:
        negative_patches_data = [((d, h, w), []) for d, h, w in all_patch_coords]

    print(f"Found {len(positive_patches_data)} positive, {len(negative_patches_data)} negative patches")

    # Sample patches
    sampled_positive = positive_patches_data
    if positive_keep_fraction < 1.0 and len(positive_patches_data) > 0:
        n_keep = int(len(positive_patches_data) * positive_keep_fraction)
        if 0 < n_keep < len(positive_patches_data):
            indices = np.random.choice(len(positive_patches_data), n_keep, replace=False)
            sampled_positive = [positive_patches_data[i] for i in indices]

    target_negatives = max(int(len(negative_patches_data) * global_negative_keep_fraction), min_negative_samples)
    target_negatives = min(target_negatives, len(negative_patches_data))

    sampled_negative = negative_patches_data
    if len(negative_patches_data) > target_negatives > 0:
        np.random.shuffle(negative_patches_data)
        sampled_negative = negative_patches_data[:target_negatives]

    all_selected = sampled_positive + sampled_negative
    print(f"Saving {len(sampled_positive)} positive and {len(sampled_negative)} negative patches")

    # Process and save patches
    positive_count = negative_count = 0

    for (d, h, w), local_motors in all_selected:
        patch_data = tomo_array[d:d+pd, h:h+ph, w:w+pw].copy()

        # Create label data
        if len(local_motors) > 0:
            positive_count += 1
            label_data = np.array(local_motors, dtype=np.int32)
        else:
            negative_count += 1
            label_data = np.zeros((1, 4), dtype=np.int32)
        
        # Save patch
        save_dict = {
            'patch': torch.from_numpy(patch_data).unsqueeze(0),
            'labels': torch.from_numpy(label_data),
            'global_coords': torch.from_numpy(np.array([d, h, w], dtype=np.int32)),
        }
        
        patch_name = f'patch_{d}_{h}_{w}'
        torch.save(save_dict, tomo_dir / f'{patch_name}.pt')
        
        if (positive_count + negative_count) % 100 == 0:
            gc.collect()
    
    del tomo_array, imgs
    gc.collect()
    
    actual_positive_ratio = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
    print(f'Finished {tomo_id}: {positive_count + negative_count} patches saved, {actual_positive_ratio:.1%} positive')

def _process_wrapper(kwargs):
    """Wrapper to unpack kwargs for multiprocessing Pool.map"""
    return process_single_tomogram(**kwargs)

def write_whole_directory(
    src, dst, patch_size, stride, csv_path,
    positive_keep_fraction: float,
    global_negative_keep_fraction: float,
    min_negative_samples: int,
    max_processes=None
):
    """Main function to process all tomograms."""
    labels = load_labels(csv_path)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loaded labels for {len(labels)} tomograms")

    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    print(f"Found {len(tomo_dirs)} tomogram directories")

    kwargs_list = [
        {
            'src': tomo_dir,
            'dst': dst,
            'patch_size': patch_size,
            'stride': stride,
            'labels': labels,
            'positive_keep_fraction': positive_keep_fraction,
            'global_negative_keep_fraction': global_negative_keep_fraction,
            'min_negative_samples': min_negative_samples,
        }
        for tomo_dir in tomo_dirs
    ]

    max_processes = max_processes or min(cpu_count(), 4)
    print(f"Using {max_processes} processes")

    if max_processes == 1:
        for kwargs in kwargs_list:
            process_single_tomogram(**kwargs)
    else:
        with Pool(processes=max_processes) as pool:
            pool.map(_process_wrapper, kwargs_list)

if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\processed\new_patch_pt_data')
    dst_root.mkdir(parents=False, exist_ok=True)
    csv_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv')
    
    # Parameters
    patch_size = (160, 288, 288)
    stride = tuple([p//8 for p in patch_size])
    
    # Sampling parameters
    positive_keep_fraction = 0.25
    global_negative_keep_fraction = 0.001
    min_negative_samples = 1
    
    print("Starting patch extraction with fractional motor centering filter...")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    # print(f"Center fraction: {center_fraction}")
    
    write_whole_directory(
        src=src_root,
        dst=dst_root,
        patch_size=patch_size,
        stride=stride,
        csv_path=csv_path,
        positive_keep_fraction=positive_keep_fraction,
        global_negative_keep_fraction=global_negative_keep_fraction,
        min_negative_samples=min_negative_samples,
        max_processes=3 #max processes should be kept small because tomograms use hella memory
    )