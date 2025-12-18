from pathlib import Path
from multiprocessing import Process, Queue
from typing import Optional, Iterator

import numpy as np
import pandas as pd
import imageio.v3 as iio
from natsort import natsorted
import gc
import torch
import torch.nn.functional as F
import logging
import time
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train.utils import _log



IMAGE_EXTS: set[str] = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def downsampled_to_real_idx(downsampled_idx: np.ndarray, downsampling_factor: int) -> np.ndarray:
    """Convert downsampled index to real-space with random [0, factor) offset."""
    return downsampled_idx * downsampling_factor + np.random.randint(0, downsampling_factor, size=3)


def probabilistic_round(value: float) -> int:
    """Round with probability based on fractional part.
    E.g., 1.3 -> 1 (70% prob) or 2 (30% prob), expected value = 1.3
    """
    floor = int(value)
    frac = value - floor
    if np.random.random() < frac:
        return floor + 1
    return floor


def random_sample_indices(availability_mask: np.ndarray, num_samples: int) -> Iterator[np.ndarray]:
    """Yield up to num_samples random indices from availability_mask."""
    valid_indices = np.argwhere(availability_mask)
    if valid_indices.shape[0] == 0:
        return

    n_to_sample = min(num_samples, valid_indices.shape[0])
    chosen = np.random.choice(valid_indices.shape[0], size=n_to_sample, replace=False)
    for idx in chosen:
        yield valid_indices[idx]


def get_motors_in_patch(
    motor_heatmap: np.ndarray,
    patch_origin: np.ndarray,
    patch_size: np.ndarray,
    downsampling_factor: int
) -> np.ndarray:
    """
    Return downsampled motor indices (N, 3) that fall within the patch region.
    Indices are relative to patch origin (local coords).
    """
    ds_origin = patch_origin // downsampling_factor
    ds_end = (patch_origin + patch_size) // downsampling_factor
    _log(f'get_motors_in_patch: patch_origin={patch_origin}, ds_origin={ds_origin}, ds_end={ds_end}, heatmap_shape={motor_heatmap.shape}', 'DEBUG', True)

    motor_mask = motor_heatmap[ds_origin[0]:ds_end[0], ds_origin[1]:ds_end[1], ds_origin[2]:ds_end[2]]
    return np.argwhere(motor_mask)


def normalize_tomogram(tomo_array: np.ndarray) -> np.ndarray:
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized

def load_tomogram(src_dir: Path) -> Optional[np.ndarray]:
    """Load and normalize a single tomogram from image directory."""
    files = [f for f in src_dir.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)
    if not files:
        return None
    imgs = [iio.imread(file, mode="L") for file in files]
    return normalize_tomogram(np.stack(imgs))


def _parse_motors_new_format(tomo_rows: pd.DataFrame) -> list[tuple]:
    """Parse motors from new relabeled format (normalized coords in 'coordinates' column)."""
    motors = []
    for _, row in tomo_rows.iterrows():
        coords_str = row['coordinates']
        if coords_str == '[]':
            continue
        z_shape, y_shape, x_shape = row['z_shape'], row['y_shape'], row['x_shape']
        coords_list = eval(coords_str)
        for norm_z, norm_y, norm_x in coords_list:
            z = int(norm_z * z_shape)
            y = int(norm_y * y_shape)
            x = int(norm_x * x_shape)
            motors.append((z, y, x))
    return motors


def _parse_motors_old_format(tomo_rows: pd.DataFrame) -> list[tuple]:
    """Parse motors from old format (one row per motor, direct coords in 'Motor axis' columns)."""
    motors = []
    for _, row in tomo_rows.iterrows():
        z, y, x = row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']
        if z == -1:  # no motor marker
            continue
        motors.append((int(z), int(y), int(x)))
    return motors


def producer(
    input_queue: Queue,
    output_queue: Queue,
    df: pd.DataFrame
) -> None:
    """Load tomograms from input queue and put on output queue."""
    is_new_format = 'coordinates' in df.columns

    while True:
        tomo_dir: Optional[Path] = input_queue.get()
        if tomo_dir is None:  # Poison pill
            break

        tomo_id = tomo_dir.name
        tomo_rows = df[df['tomo_id'] == tomo_id]

        if is_new_format:
            motors = _parse_motors_new_format(tomo_rows)
        else:
            motors = _parse_motors_old_format(tomo_rows)

        # Get fold and voxel_spacing from first row
        first_row = tomo_rows.iloc[0]
        fold = first_row['fold'] if 'fold' in df.columns else -1
        voxel_spacing = first_row['voxel_spacing'] if 'voxel_spacing' in df.columns else first_row['Voxel spacing']

        print(f'Loading tomogram: {tomo_id}')
        load_start = time.time()
        tomo_array = load_tomogram(tomo_dir)
        load_time = time.time() - load_start

        if tomo_array is not None:
            _log(f'{tomo_id} load time: {load_time:.2f}s', 'DEBUG', True)
            output_queue.put({
                'tomo_id': tomo_id,
                'tomo_array': tomo_array,
                'motors': motors,
                'voxel_spacing': voxel_spacing,
                'fold': fold,
            })
        else:
            print(f"No images found in {tomo_dir}")


def save_patches_placeholder(
    tomo_dict: dict,
    dst: Path,
    data_split_dict:dict[str,int],
    downsampling_factor: int,
    angstrom_sigma:float,
    patch_size:tuple,
) -> None:
    """Placeholder for actual patch saving logic."""
    tomo_dir = dst / tomo_dict['tomo_id']
    Path.mkdir(tomo_dir)
    tomo_array = tomo_dict['tomo_array']
    assert tomo_array.ndim == 3, "expected tomo array to be d,h,w"
    tomo_shape = np.array(tomo_array.shape)
    ds_shape = tomo_shape // downsampling_factor
    patch_size_arr = np.array(patch_size)

    # Valid bounds mask: ds origins where patch fits even with max random offset (factor-1)
    max_valid_ds = (tomo_shape - patch_size_arr - downsampling_factor + 1) // downsampling_factor
    valid_bounds_mask = np.zeros(tuple(ds_shape), dtype=np.bool_)
    valid_bounds_mask[:max_valid_ds[0]+1, :max_valid_ds[1]+1, :max_valid_ds[2]+1] = True

    availability_mask = np.zeros((5, *ds_shape), dtype=np.bool_)
    # 0 multi motor region
    # 1 single motor region
    # 2 hard negative region
    # 3 random negative region
    # 4 actual motor locations

    for motor_idx in tomo_dict['motors']:
        downsampled_motor_idx = [min(i // downsampling_factor, ds_shape[dim] - 1) for dim, i in enumerate(motor_idx)]
        _log(f'marking motor: real={motor_idx}, downsampled={downsampled_motor_idx}', 'DEBUG', True)
        availability_mask[4, *downsampled_motor_idx] = 1
    _log(f'total motors marked in slice 4: {availability_mask[4].sum()}', 'DEBUG', True)

    tomo_id = tomo_dict['tomo_id']
    total_start = time.time()

    t0 = time.time()
    num_motors = len(tomo_dict['motors'])
    multi_motor_samples = data_split_dict['single_motor_samples'] * (num_motors // 2)
    successful_multi_motors = _multi_motor_save(multi_motor_samples=multi_motor_samples,
                                                    tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                    availability_mask=availability_mask,
                                                    valid_bounds_mask=valid_bounds_mask,
                                                    downsampling_factor=downsampling_factor,
                                                    angstrom_sigma=angstrom_sigma,
                                                    patch_size=patch_size_arr)
    multi_time = time.time() - t0

    t0 = time.time()
    successful_single_motors = _single_motor_save(single_motor_samples=data_split_dict['single_motor_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                angstrom_sigma=angstrom_sigma,
                                                patch_size=patch_size_arr)
    single_time = time.time() - t0

    t0 = time.time()
    successful_hard_negatives = _hard_negative_save(hard_negative_samples=data_split_dict['hard_negative_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                patch_size=patch_size_arr)
    hard_neg_time = time.time() - t0

    t0 = time.time()
    successful_random_negatives = _random_negative_save(random_negative_samples=data_split_dict['random_negative_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                patch_size=patch_size_arr)
    random_neg_time = time.time() - t0

    total_time = time.time() - total_start
    total_patches = successful_multi_motors + successful_single_motors + successful_hard_negatives + successful_random_negatives

    _log(f'{tomo_id} saves: multi={successful_multi_motors} ({multi_time:.2f}s), '
         f'single={successful_single_motors} ({single_time:.2f}s), '
         f'hard_neg={successful_hard_negatives} ({hard_neg_time:.2f}s), '
         f'random_neg={successful_random_negatives} ({random_neg_time:.2f}s)', 'INFO', True)
    _log(f'{tomo_id} total: {total_patches} patches in {total_time:.2f}s', 'INFO', True)
    
    

def generate_gaussian_label(downsampled_local_motor_coords, downsampled_patch_shape, angstrom_sigma, voxel_spacing, downsampling_factor):
    # sigma in downsampled space = angstrom_sigma / voxel_spacing / downsampling_factor
    ds_pixel_sigma = angstrom_sigma / voxel_spacing / downsampling_factor

    label_d, label_h, label_w = downsampled_local_motor_coords
    grid_d = torch.arange(downsampled_patch_shape[0])[:, None, None]
    grid_h = torch.arange(downsampled_patch_shape[1])[None, :, None]
    grid_w = torch.arange(downsampled_patch_shape[2])[None, None, :]
    dist_sq = (grid_d-label_d)**2 + (grid_h-label_h)**2 + (grid_w-label_w)**2
    gaussian_blob = torch.exp(-dist_sq/(2*ds_pixel_sigma**2)).to(torch.float16)
    return gaussian_blob

#ALL OF THESE FUNCTIONS RETURN INDICES IF APPLICABLE
#then in the save patches orchestrator function we generate gaussians there and save
#probably return motor_idx, type
#also pass reference of shape same as heatmap indicating which voxels are taken and which are available. so i guess 0 could be available nothigns there 1 is unavailable. 
# multi,single and hrad negative will populate that and the random negative function samples from that

def _multi_motor_save(
    multi_motor_samples: int,
    tomo_dict: dict,
    tomo_dir: Path,
    availability_mask: np.ndarray,
    valid_bounds_mask: np.ndarray,
    downsampling_factor: int,
    angstrom_sigma: float,
    patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    voxel_spacing = tomo_dict['voxel_spacing']
    downsampled_patch_shape = patch_size // downsampling_factor

    ### FIND VALID RANGES FOR MULTI MOTORS (all in downsampled space)
    for i, motor_idx_tuple_i in enumerate(motors):
        ds_motor_i = np.asarray(motor_idx_tuple_i) // downsampling_factor

        for j, motor_idx_tuple_j in enumerate(motors):
            if j <= i:
                continue #only generate valid combinations (triu indices)

            ds_motor_j = np.asarray(motor_idx_tuple_j) // downsampling_factor
            if np.any(np.abs(ds_motor_i - ds_motor_j) >= downsampled_patch_shape):
                continue  # too far apart to fit in one patch

            # Valid origins D: patch [D, D+ds_patch) contains both motors
            # D > ds_max - ds_patch  AND  D <= ds_min
            ds_max = np.maximum(ds_motor_i, ds_motor_j)
            ds_min = np.minimum(ds_motor_i, ds_motor_j)
            lo = ds_max - downsampled_patch_shape + 1  # +1 for strict inequality
            hi = ds_min + 1  # +1 for exclusive slicing
            lo = np.maximum(lo, 0)
            availability_mask[0, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1

    # Erode locally for sampling - keeps availability_mask[0] as pre-erosion for exclusion later
    eroded_multi = _apply_erosion(availability_mask[0], radius=1) & valid_bounds_mask

    saved_patches = 0
    for downsampled_idx in random_sample_indices(eroded_multi, multi_motor_samples):
        patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor)
        patch_end = patch_origin + patch_size
        patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
        assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {tuple(patch_size)}"

        motor_indices = get_motors_in_patch(availability_mask[4], patch_origin, patch_size, downsampling_factor)
        assert motor_indices.shape[0] >= 2, f"Expected >=2 motors in multi motor patch, got {motor_indices.shape[0]}"

        max_gaussian = torch.zeros(tuple(downsampled_patch_shape))
        for motor_coord in motor_indices:
            assert np.all(motor_coord >= 0) and np.all(motor_coord < downsampled_patch_shape), \
                f"Motor coord {motor_coord} out of bounds for shape {downsampled_patch_shape}"
            single_gaussian = generate_gaussian_label(
                downsampled_local_motor_coords=motor_coord,
                downsampled_patch_shape=downsampled_patch_shape,
                angstrom_sigma=angstrom_sigma,
                voxel_spacing=voxel_spacing,
                downsampling_factor=downsampling_factor
            )
            max_gaussian = torch.maximum(max_gaussian, single_gaussian)

        assert max_gaussian.shape == tuple(downsampled_patch_shape), \
            f"Gaussian shape {max_gaussian.shape} != expected {tuple(downsampled_patch_shape)}"
        torch.save(
            obj={
                'patch': torch.from_numpy(patch.copy()),
                'gaussian': max_gaussian,
                'patch_type': "multi_motor"
            },
            f=tomo_dir / f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
        )
        saved_patches += 1

    return saved_patches
        


def _single_motor_save(
    single_motor_samples: int,
    tomo_dict: dict,
    tomo_dir: Path,
    availability_mask: np.ndarray,
    valid_bounds_mask: np.ndarray,
    downsampling_factor: int,
    angstrom_sigma: float,
    patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    voxel_spacing = tomo_dict['voxel_spacing']
    downsampled_patch_size = patch_size // downsampling_factor
    ds_shape = tuple(np.array(tomo_array.shape) // downsampling_factor)

    # First pass: mark all single-motor regions in availability_mask[1] for exclusion later
    for motor_idx_tuple in motors:
        ds_motor = np.array(motor_idx_tuple) // downsampling_factor
        ds_start = ds_motor - downsampled_patch_size + 1
        ds_end = ds_motor + 1
        ds_start = np.maximum(ds_start, 0)
        availability_mask[1, ds_start[0]:ds_end[0], ds_start[1]:ds_end[1], ds_start[2]:ds_end[2]] = 1

    # AND not with multi-motor regions so single doesn't overlap with multi
    availability_mask[1] = availability_mask[1] & ~availability_mask[0]

    # Second pass: sample per-motor for diversity
    saved_patches = 0
    for motor_idx_tuple in motors:
        ds_motor = np.array(motor_idx_tuple) // downsampling_factor
        ds_start = ds_motor - downsampled_patch_size + 1
        ds_end = ds_motor + 1
        ds_start = np.maximum(ds_start, 0)

        # Create temp mask for just this motor's valid region
        motor_mask = np.zeros(ds_shape, dtype=np.bool_)
        motor_mask[ds_start[0]:ds_end[0], ds_start[1]:ds_end[1], ds_start[2]:ds_end[2]] = 1
        motor_mask = motor_mask & ~availability_mask[0] & valid_bounds_mask  # exclude multi-motor, apply bounds
        motor_mask = _apply_erosion(motor_mask, radius=1)

        for downsampled_idx in random_sample_indices(motor_mask, single_motor_samples):
            patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor)
            patch_end = patch_origin + patch_size
            patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
            assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {tuple(patch_size)}"

            motor_indices = get_motors_in_patch(availability_mask[4], patch_origin, patch_size, downsampling_factor)
            assert motor_indices.shape[0] == 1, f"Expected 1 motor in single motor patch, got {motor_indices.shape[0]}"

            motor_coord = motor_indices[0]
            assert np.all(motor_coord >= 0) and np.all(motor_coord < downsampled_patch_size), \
                f"Motor coord {motor_coord} out of bounds for shape {downsampled_patch_size}"

            gaussian = generate_gaussian_label(
                downsampled_local_motor_coords=motor_coord,
                downsampled_patch_shape=downsampled_patch_size,
                angstrom_sigma=angstrom_sigma,
                voxel_spacing=voxel_spacing,
                downsampling_factor=downsampling_factor
            )
            assert gaussian.shape == tuple(downsampled_patch_size), \
                f"Gaussian shape {gaussian.shape} != expected {tuple(downsampled_patch_size)}"

            torch.save(
                obj={
                    'patch': torch.from_numpy(patch.copy()),
                    'gaussian': gaussian,
                    'patch_type': "single_motor"
                },
                f=tomo_dir / f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
            )
            saved_patches += 1

    return saved_patches


def _apply_box_conv(mask: np.ndarray, kernel_size: tuple) -> np.ndarray:
    """Apply box kernel conv to boolean mask, returns dilated boolean mask."""
    kernel_size = tuple(kernel_size)
    padding = tuple((k - 1) // 2 for k in kernel_size)
    conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
    conv.weight.data.fill_(1.0)

    tensor = torch.from_numpy(mask.astype(np.float32)).reshape(1, 1, *mask.shape)
    result = conv(tensor) > 0
    return result.squeeze().numpy()


def _apply_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode mask by radius voxels - shrinks mask from edges.

    Used to ensure motors aren't at patch boundaries where gaussians would be truncated.
    Opposite of dilation: requires ALL neighbors in kernel to be True.
    Out-of-bounds treated as True so edge voxels can survive erosion.
    """
    kernel_size = 2 * radius + 1
    kernel_volume = kernel_size ** 3
    conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
    conv.weight.data.fill_(1.0)

    tensor = torch.from_numpy(mask.astype(np.float32)).reshape(1, 1, *mask.shape)
    padded = F.pad(tensor, (radius,) * 6, mode='constant', value=1.0)
    result = conv(padded) == kernel_volume  # ALL neighbors must be True
    return result.squeeze().numpy()


def _hard_negative_save(hard_negative_samples: float,
                        tomo_dict: dict,
                        tomo_dir: Path,
                        availability_mask: np.ndarray,
                        valid_bounds_mask: np.ndarray,
                        downsampling_factor: int,
                        patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    downsampled_patch_size = patch_size // downsampling_factor
    ds_shape = np.array(tomo_array.shape) // downsampling_factor

    # +1 margin avoids ambiguous negatives directly on motor region edges
    close_radius = (0.1 * downsampled_patch_size).astype(int) + 1
    close_kernel_size = 2 * close_radius + 1
    far_radius = (0.3 * downsampled_patch_size).astype(int) + 1
    far_kernel_size = 2 * far_radius + 1

    # availability_mask[0] and [1] are pre-erosion, so this covers all motor-containing origins
    positive = availability_mask[0] | availability_mask[1]

    #close: extend exclusion zone so theres no ambiguity
    close = _apply_box_conv(positive, close_kernel_size)
    #far: set the edge of the hard negative zone
    far = _apply_box_conv(positive, far_kernel_size)

    #hard negative = within far boundary but outside close exclusion, AND within valid bounds
    availability_mask[2] = ~close & far & valid_bounds_mask
    _log(f'hard_neg: close_radius={close_radius}, far_radius={far_radius}, '
         f'positive_sum={positive.sum()}, close_sum={close.sum()}, far_sum={far.sum()}, '
         f'hard_neg_sum={availability_mask[2].sum()}', 'DEBUG', True)

    # Per-motor sampling from existing hard_neg mask
    saved_patches = 0
    for motor_idx_tuple in motors:
        num_samples = probabilistic_round(hard_negative_samples)
        if num_samples == 0:
            continue

        ds_motor = np.array(motor_idx_tuple) // downsampling_factor

        # Extract ~2x patch region around motor from existing hard_neg mask
        ds_start = ds_motor - downsampled_patch_size
        ds_end = ds_motor + downsampled_patch_size
        ds_start = np.maximum(ds_start, 0)
        ds_end = np.minimum(ds_end, ds_shape)

        motor_hard_neg = np.zeros(tuple(ds_shape), dtype=np.bool_)
        motor_hard_neg[ds_start[0]:ds_end[0], ds_start[1]:ds_end[1], ds_start[2]:ds_end[2]] = \
            availability_mask[2, ds_start[0]:ds_end[0], ds_start[1]:ds_end[1], ds_start[2]:ds_end[2]]

        for downsampled_idx in random_sample_indices(motor_hard_neg, num_samples):
            patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor)
            patch_end = patch_origin + patch_size
            patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
            assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {tuple(patch_size)}"

            motor_indices = get_motors_in_patch(availability_mask[4], patch_origin, patch_size, downsampling_factor)
            assert motor_indices.shape[0] == 0, \
                f"Expected 0 motor in hard negative patch, got {motor_indices.shape[0]} at local coords {motor_indices}"

            gaussian = torch.zeros(tuple(downsampled_patch_size), dtype=torch.float16)
            torch.save(
                obj={
                    'patch': torch.from_numpy(patch.copy()),
                    'gaussian': gaussian,
                    'patch_type': "hard_negative"
                },
                f=tomo_dir / f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
            )
            saved_patches += 1
    return saved_patches


def _random_negative_save(random_negative_samples: float,
                        tomo_dict: dict,
                        tomo_dir: Path,
                        availability_mask: np.ndarray,
                        valid_bounds_mask: np.ndarray,
                        downsampling_factor: int,
                        patch_size: np.ndarray) -> int:
    tomo_array = tomo_dict['tomo_array']
    downsampled_patch_size = patch_size // downsampling_factor

    # Compute far zone independently to exclude regions near motors
    # +1 margin avoids ambiguous negatives directly on motor region edges
    far_radius = (0.3 * downsampled_patch_size).astype(int) + 1
    far_kernel_size = 2 * far_radius + 1
    positive = availability_mask[0] | availability_mask[1]
    far = _apply_box_conv(positive, far_kernel_size)

    # Random negatives = outside far zone (truly far from any motors)
    availability_mask[3] = ~far & valid_bounds_mask
    _log(f'random_neg: mask_sum={availability_mask[3].sum()}', 'DEBUG', True)

    num_samples = probabilistic_round(random_negative_samples)
    saved_patches = 0
    for downsampled_idx in random_sample_indices(availability_mask[3], num_samples):
        patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor)
        patch_end = patch_origin + patch_size
        _log(f'random_neg patch: ds_idx={downsampled_idx}, origin={patch_origin}', 'DEBUG', True)
        patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
        assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {tuple(patch_size)}"

        motor_indices = get_motors_in_patch(availability_mask[4], patch_origin, patch_size, downsampling_factor)
        assert motor_indices.shape[0] == 0, \
            f"Expected 0 motor in random negative patch, got {motor_indices.shape[0]} at local coords {motor_indices}"
        gaussian = torch.zeros(tuple(downsampled_patch_size), dtype=torch.float16)
        torch.save(
            obj={
                'patch': torch.from_numpy(patch.copy()),
                'gaussian': gaussian,
                'patch_type': "random_negative"
            },
            f=tomo_dir / f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
            )
        saved_patches += 1
    return saved_patches
    

def write_whole_directory(
    src: Path,
    dst: Path,
    csv_path: Path,
    num_workers: int,
    queue_size: int,
    data_split_dict:dict,
    downsampling_factor:int,
    angstrom_sigma:float,
    patch_size:tuple
) -> None:
    """
    Docstring for write_whole_directory

    :param src: src dir for tomograms (jpgs). structure files/tomo_N/0.jpg
    :param dst: root dir for saving tomos in format dst/tomo_N/patch_N.pt
    :param csv_path: the csv path duh
    :param num_workers: yuh
    :param queue_size: yuh

    """
    #Main function - producer processes load tomograms, main process consumes."""
    df = pd.read_csv(csv_path)
    dst.mkdir(parents=True, exist_ok=False)

    print(f"Loaded {len(df)} rows from CSV")

    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    num_tomos = len(tomo_dirs)
    print(f"Found {num_tomos} tomogram directories")

    input_queue = Queue()
    output_queue = Queue(maxsize=queue_size)

    # Fill input queue with work
    for tomo_dir in tomo_dirs:
        input_queue.put(tomo_dir)
    # Add poison pills for each worker
    for _ in range(num_workers):
        input_queue.put(None)

    # Start producer workers
    producers = []
    for _ in range(num_workers):
        p = Process(target=producer, args=(input_queue, output_queue, df))
        p.start()
        producers.append(p)

    # Consume from output queue in main process
    processed = 0
    while processed < num_tomos:
        alive_count = sum(1 for p in producers if p.is_alive())
        if output_queue.empty() and alive_count == 0:
            break
        try:
            item = output_queue.get(timeout=1.0)
        except:
            continue
                
        tomo_dict = item
        D, H, W = tomo_dict['tomo_array'].shape
        print(f"Processing: {tomo_dict['tomo_id']} (fold={tomo_dict['fold']}, motors={len(tomo_dict['motors'])}, shape={D}x{H}x{W}, voxel_spacing={tomo_dict['voxel_spacing']})")

        save_patches_placeholder(tomo_dict, dst, data_split_dict=data_split_dict, downsampling_factor=downsampling_factor, angstrom_sigma=angstrom_sigma, patch_size=patch_size)

        del tomo_dict
        gc.collect()
        processed += 1

    # Wait for producers to finish
    for p in producers:
        p.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #SUSPICIOUS MOTOR, GIVEN LABELS SINGLE SAMPLE SHOULD BE EASILY OBTAINED
#     Processing: tomo_0da370 (fold=-1, motors=1, shape=300x928x928, voxel_spacing=13.1)
# INFO: tomo_0da370 saves: multi=0 (0.00s), single=0 (0.00s), hard_neg=1 (0.12s), random_neg=1 (0.03s)
# INFO: tomo_0da370 total: 2 patches in 0.15s\
    
    src_root = Path(r'data/original_data/train')
    dst_root = Path(r'data/processed/old_data_300sigma')
    csv_path = Path(r'data/original_data/train_labels.csv')
    data_split_dict = {
        'single_motor_samples': 7,  # per motor (also used for multi_motor: single * num_motors // 2)
        'hard_negative_samples': 0.5,  # per motor, float supported (0.5 = 50% chance of 1)
        'random_negative_samples': 1  # per tomogram, float supported
    }
    
    downsampling_factor = 16
    angstrom_sigma = 300
    patch_size = (160, 288, 288)
    
    write_whole_directory(src=src_root, dst=dst_root, csv_path=csv_path, num_workers=4, queue_size=2, data_split_dict=data_split_dict, downsampling_factor=downsampling_factor, angstrom_sigma=angstrom_sigma, patch_size=patch_size)
