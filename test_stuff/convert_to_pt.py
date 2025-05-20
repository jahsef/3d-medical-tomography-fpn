from pathlib import Path
import torch
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count
import torchvision.transforms.v2 as t
import pandas as pd

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

transform = t.Compose([
    t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
])

def load_labels(csv_path, global_max_motors=None):
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

def write_tomo_patches(args):
    src, dst, patch_size, stride, labels, patch_max_motors, empty_keep_fraction = args
    print(f'Processing tomogram: {src.name}')
    
    tomo_id = src.name
    motors = labels.get(tomo_id, [])
    
    # Load and process images
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    files = natsorted(files, key=lambda x: x.name)
    imgs = [iio.imread(file, mode="L") for file in files]
    tomo_array = np.stack(imgs)
    
    # Convert to tensor and normalize
    tomo_tensor = torch.as_tensor(tomo_array)
    tomo_tensor = transform(tomo_tensor)

    D, H, W = tomo_tensor.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Create output directory
    patch_dir = dst / tomo_id
    patch_dir.mkdir(parents=True, exist_ok=True)

    def get_start_indices(length, window_size, stride):
        indices = []
        n_windows = (length - window_size) // stride + 1
        for i in range(n_windows):
            start = i * stride
            indices.append(start)
        last_start = (n_windows - 1) * stride
        if last_start + window_size < length:
            indices.append(length - window_size)
        return indices

    start_d = get_start_indices(D, pd, sd)
    start_h = get_start_indices(H, ph, sh)
    start_w = get_start_indices(W, pw, sw)

    # Convert motors to numpy array
    if len(motors) > 0:
        motors_array = np.array(motors, dtype=np.float32)
        motors_rounded = np.round(motors_array).astype(np.int32)
    else:
        motors_rounded = np.empty((0, 3), dtype=np.int32)

    from itertools import product

    for d, h, w in product(start_d, start_h, start_w):
        # Extract patch
        patch = tomo_tensor[d:d+pd, h:h+ph, w:w+pw].clone().contiguous()
        if patch.shape != (pd, ph, pw):
            continue

        # Vectorized motor filtering
        mask = (
            (motors_rounded[:, 0] >= d) &
            (motors_rounded[:, 0] < d + pd) &
            (motors_rounded[:, 1] >= h) &
            (motors_rounded[:, 1] < h + ph) &
            (motors_rounded[:, 2] >= w) &
            (motors_rounded[:, 2] < w + pw)
        )

        local_coords = motors_rounded[mask]
        if len(local_coords) > 0:
            local_coords = local_coords - [d, h, w]
            in_patch = np.hstack([local_coords, np.ones((len(local_coords), 1))])  # (x,y,z,conf=1.0)
        else:
            in_patch = []

        # Decide whether to save empty patches
        has_motor = len(in_patch) > 0
        if has_motor:
            save_patch = True
        else:
            save_patch = torch.rand(1).item() < empty_keep_fraction  # keep only 0.15% of empty patches

        if not save_patch:
            continue

        # Enforce fixed shape
        if patch_max_motors is not None:
            in_patch_tensor = torch.tensor(in_patch, dtype=torch.float32)
            current_count = in_patch_tensor.shape[0]

            if current_count > patch_max_motors:
                in_patch_tensor = in_patch_tensor[:patch_max_motors]
            else:
                pad_needed = patch_max_motors - current_count
                padding = torch.zeros((pad_needed, 4), dtype=torch.float32)
                in_patch_tensor = torch.cat([in_patch_tensor, padding], dim=0)

            metadata = {
                'global_coords': torch.tensor((d, h, w), dtype=torch.int32),
                'xyzconf': in_patch_tensor
            }
        else:
            metadata = {
                'global_coords': torch.tensor((d, h, w), dtype=torch.int32),
                'xyzconf': torch.tensor(in_patch, dtype=torch.float32)
            }

        # Save patch
        patch_path = patch_dir / f'patch_{d}_{h}_{w}.pt'
        torch.save({
            'data': patch,
            'metadata': metadata
        }, patch_path)

    print(f'Finished processing: {tomo_id}')

def write_whole_directory(
    src, 
    dst, 
    patch_size, 
    stride, 
    csv_path, 
    empty_keep_fraction,
    global_max_motors=None,
    patch_max_motors=None,
    max_processes=None
):
    labels = load_labels(csv_path, global_max_motors=global_max_motors)
    dst.mkdir(parents=True, exist_ok=True)
    
    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    args_list = [
        (tomo_dir, dst, patch_size, stride, labels, patch_max_motors, empty_keep_fraction)
        for tomo_dir in tomo_dirs
    ]

    max_processes = max_processes or min(cpu_count(), 16)
    with Pool(processes=max_processes) as pool:
        pool.map(write_tomo_patches, args_list)

if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train')
    csv_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')
    p = 64
    s = int(1/2*p)
    patch_size = (p, p, p)
    #stride = 15/16 * p
    stride = (s, s, s)
    workers = 6
    empty_keep_fraction = 0.0005
    global_max_motors = 20
    patch_max_motors = 5
    
    write_whole_directory(
        src=src_root,
        dst=dst_root,
        empty_keep_fraction= empty_keep_fraction,
        patch_size=patch_size,
        stride=stride,
        csv_path=csv_path,
        global_max_motors=global_max_motors,
        patch_max_motors=patch_max_motors,
        max_processes=workers
    )