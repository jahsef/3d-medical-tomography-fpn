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
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from train.utils import _log



IMAGE_EXTS: set[str] = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def downsampled_to_real_idx(
    downsampled_idx: np.ndarray,
    downsampling_factor: int,
    tomo_shape: np.ndarray,
    patch_size: np.ndarray
) -> np.ndarray:
    """Convert downsampled index to real-space with random [0, factor) offset, clamped to valid bounds."""
    base = downsampled_idx * downsampling_factor
    offset = np.random.randint(0, downsampling_factor, size=3)

    # Clamp so patch fits: origin + patch_size <= tomo_shape
    max_origin = tomo_shape - patch_size
    return np.minimum(base + offset, max_origin)


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


class AvailabilityHeatmapViewer:
    """Interactive viewer for availability heatmap channels with depth navigation.

    Row 1: Patch origin locations (where patches can START)
    Row 2: Patch coverage space (what space patches would COVER) - dilated by patch_size
    Note: Coverage is NOT the true availability mask, just a visualization aid.
    """

    def __init__(
        self,
        tomo_array: np.ndarray,
        availability_mask: np.ndarray,
        gaussian_heatmap: torch.Tensor,
        tomo_id: str,
        downsampling_factor: int,
        ds_patch_size: np.ndarray
    ):
        self.tomo_array = tomo_array
        self.availability_mask = availability_mask
        self.gaussian_heatmap = gaussian_heatmap.numpy()
        self.tomo_id = tomo_id
        self.ds = downsampling_factor
        self.ds_patch_size = ds_patch_size

        self.depth_idx = 0
        self.max_depth_ds = availability_mask.shape[1] - 1
        self.max_depth_real = tomo_array.shape[0] - 1

        # Precompute coverage masks (dilate origins by patch_size toward bottom-right)
        # This shows what space patches would cover, not where they can start
        self.coverage_mask = self._compute_coverage_masks()

        self.channel_names = [
            'Multi Motor',
            'Single Motor',
            'Hard Negative',
            'Random Negative',
            'Gaussian Heatmap'
        ]

        self.fig, self.axes = plt.subplots(2, 5, figsize=(20, 8))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._update_display()
        plt.show()

    def _compute_coverage_masks(self) -> np.ndarray:
        """Dilate availability masks by patch_size toward bottom-right.

        Top-left kernel: ones in [0:patch_size], zero padding.
        Each True at position P spreads to [P, P+patch_size).
        """
        d, h, w = self.ds_patch_size
        kernel_size = (2*d - 1, 2*h - 1, 2*w - 1)

        # Ones in top-left quadrant (indices < patch_size)
        kernel = np.zeros(kernel_size, dtype=np.float32)
        kernel[:d, :h, :w] = 1.0

        pad = (d - 1, h - 1, w - 1)
        conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=pad, bias=False)
        conv.weight.data = torch.from_numpy(kernel).reshape(1, 1, *kernel_size)

        coverage = np.zeros_like(self.availability_mask)
        for i in range(4):
            tensor = torch.from_numpy(self.availability_mask[i].astype(np.float32)).reshape(1, 1, *self.availability_mask[i].shape)
            coverage[i] = (conv(tensor) > 0).squeeze().numpy()

        return coverage

    def _update_display(self):
        real_depth = min(self.depth_idx * self.ds + self.ds // 2, self.max_depth_real)

        for row in self.axes:
            for ax in row:
                ax.clear()

        # Row 1: Tomogram + 4 origin masks
        self.axes[0, 0].imshow(self.tomo_array[real_depth], cmap='gray')
        self.axes[0, 0].set_title(f'Tomogram (z={real_depth})')
        self.axes[0, 0].axis('off')

        for i in range(4):
            self.axes[0, i + 1].imshow(self.availability_mask[i, self.depth_idx], cmap='hot')
            self.axes[0, i + 1].set_title(f'{self.channel_names[i]} (origins)')
            self.axes[0, i + 1].axis('off')

        # Row 2: Gaussian heatmap + 4 coverage masks
        self.axes[1, 0].imshow(self.gaussian_heatmap[self.depth_idx], cmap='viridis', vmin=0, vmax=1)
        self.axes[1, 0].set_title('Gaussian Heatmap')
        self.axes[1, 0].axis('off')

        for i in range(4):
            self.axes[1, i + 1].imshow(self.coverage_mask[i, self.depth_idx], cmap='hot')
            self.axes[1, i + 1].set_title(f'{self.channel_names[i]} (coverage*)')
            self.axes[1, i + 1].axis('off')

        self.fig.suptitle(
            f'{self.tomo_id} | DS Depth {self.depth_idx}/{self.max_depth_ds} | '
            f'Up/Down: navigate | Q: quit\n'
            f'*Coverage = origins dilated by patch_size (visualization only)'
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'up':
            self.depth_idx = min(self.depth_idx + 1, self.max_depth_ds)
        elif event.key == 'down':
            self.depth_idx = max(self.depth_idx - 1, 0)
        elif event.key == 'q':
            plt.close(self.fig)
            return
        self._update_display()


def visualize_availability_heatmap(
    tomo_array: np.ndarray,
    availability_mask: np.ndarray,
    gaussian_heatmap: torch.Tensor,
    tomo_id: str,
    downsampling_factor: int,
    ds_patch_size: np.ndarray
) -> None:
    """Launch interactive visualization of availability mask channels."""
    AvailabilityHeatmapViewer(tomo_array, availability_mask, gaussian_heatmap, tomo_id, downsampling_factor, ds_patch_size)


def save_patches_placeholder(
    tomo_dict: dict,
    dst: Path,
    data_split_dict:dict[str,int],
    downsampling_factor: int,
    angstrom_sigma:float,
    patch_size:tuple,
    visualize: bool,
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
    # With clamping in downsampled_to_real_idx, we can sample from edge ds voxels
    max_valid_ds = (tomo_shape - patch_size_arr) // downsampling_factor
    valid_bounds_mask = np.zeros(tuple(ds_shape), dtype=np.bool_)
    valid_bounds_mask[:max_valid_ds[0]+1, :max_valid_ds[1]+1, :max_valid_ds[2]+1] = True

    availability_mask = np.zeros((4, *ds_shape), dtype=np.bool_)
    # 0 multi motor region
    # 1 single motor region
    # 2 hard negative region
    # 3 random negative region

    # Precompute full gaussian heatmap (float coords, max pooling across motors)
    gaussian_heatmap = torch.zeros(tuple(ds_shape), dtype=torch.float32)
    for motor_idx in tomo_dict['motors']:
        ds_motor_float = np.array(motor_idx) / downsampling_factor
        single_gaussian = generate_gaussian_label(
            downsampled_local_motor_coords=ds_motor_float,
            downsampled_patch_shape=tuple(ds_shape),
            angstrom_sigma=angstrom_sigma,
            voxel_spacing=tomo_dict['voxel_spacing'],
            downsampling_factor=downsampling_factor
        )
        gaussian_heatmap = torch.maximum(gaussian_heatmap, single_gaussian.float())

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
                                                    gaussian_heatmap=gaussian_heatmap,
                                                    patch_size=patch_size_arr)
    multi_time = time.time() - t0

    t0 = time.time()
    successful_single_motors = _single_motor_save(single_motor_samples=data_split_dict['single_motor_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                gaussian_heatmap=gaussian_heatmap,
                                                patch_size=patch_size_arr)
    single_time = time.time() - t0

    t0 = time.time()
    successful_hard_negatives = _hard_negative_save(hard_negative_samples=data_split_dict['hard_negative_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                gaussian_heatmap=gaussian_heatmap,
                                                patch_size=patch_size_arr)
    hard_neg_time = time.time() - t0

    t0 = time.time()
    successful_random_negatives = _random_negative_save(random_negative_samples=data_split_dict['random_negative_samples'],
                                                tomo_dict=tomo_dict, tomo_dir=tomo_dir,
                                                availability_mask=availability_mask,
                                                valid_bounds_mask=valid_bounds_mask,
                                                downsampling_factor=downsampling_factor,
                                                gaussian_heatmap=gaussian_heatmap,
                                                patch_size=patch_size_arr)
    random_neg_time = time.time() - t0

    total_time = time.time() - total_start
    total_patches = successful_multi_motors + successful_single_motors + successful_hard_negatives + successful_random_negatives

    _log(f'{tomo_id} saves: multi={successful_multi_motors} ({multi_time:.2f}s), '
         f'single={successful_single_motors} ({single_time:.2f}s), '
         f'hard_neg={successful_hard_negatives} ({hard_neg_time:.2f}s), '
         f'random_neg={successful_random_negatives} ({random_neg_time:.2f}s)', 'INFO', True)
    _log(f'{tomo_id} total: {total_patches} patches in {total_time:.2f}s', 'INFO', True)

    if visualize:
        ds_patch_size = patch_size_arr // downsampling_factor
        visualize_availability_heatmap(tomo_array, availability_mask, gaussian_heatmap, tomo_id, downsampling_factor, ds_patch_size)


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


def _save_patch(
    tomo_array: np.ndarray,
    gaussian_heatmap: torch.Tensor,
    motors: list[tuple],
    patch_origin: np.ndarray,
    patch_size: np.ndarray,
    downsampling_factor: int,
    patch_type: str,
    tomo_dir: Path
) -> None:
    """Extract patch and gaussian crop, validate motor count, and save."""
    # Extract patch from tomo_array
    patch_end = patch_origin + patch_size
    patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
    assert patch.shape == tuple(patch_size), f"Patch shape {patch.shape} != expected {tuple(patch_size)}"

    # Extract gaussian crop from heatmap
    ds_origin = patch_origin // downsampling_factor
    ds_end = patch_end // downsampling_factor
    gaussian = gaussian_heatmap[ds_origin[0]:ds_end[0], ds_origin[1]:ds_end[1], ds_origin[2]:ds_end[2]]

    # Count motors by checking if realspace coords fall within patch bounds
    motors_in_patch = [
        m for m in motors
        if all(patch_origin[i] <= m[i] < patch_end[i] for i in range(3))
    ]
    num_motors = len(motors_in_patch)

    _log(f'_save_patch: type={patch_type}, origin={patch_origin}, end={patch_end}, '
         f'ds_origin={ds_origin}, num_motors={num_motors}, motors_in_patch={motors_in_patch}', 'DEBUG', True)

    # Validate motor count based on patch_type
    if patch_type == "multi_motor":
        assert num_motors >= 2, f"Expected >=2 motors in multi_motor patch, got {num_motors}"
    elif patch_type == "single_motor":
        assert num_motors == 1, f"Expected 1 motor in single_motor patch, got {num_motors}, motors_in_patch={motors_in_patch}, all_motors={motors}"
    elif patch_type in ("hard_negative", "random_negative"):
        assert num_motors == 0, f"Expected 0 motors in {patch_type} patch, got {num_motors}, motors_in_patch={motors_in_patch}"

    torch.save(
        obj={
            'patch': torch.from_numpy(patch.copy()),
            'gaussian': gaussian.clone().to(torch.float16),
            'patch_type': patch_type
        },
        f=tomo_dir / f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
    )


def _multi_motor_save(
    multi_motor_samples: int,
    tomo_dict: dict,
    tomo_dir: Path,
    availability_mask: np.ndarray,
    valid_bounds_mask: np.ndarray,
    downsampling_factor: int,
    gaussian_heatmap: torch.Tensor,
    patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    tomo_shape = np.array(tomo_array.shape)
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
    # Radius=2 because: (1) gaussian uses float ds coords which can be up to 0.99 voxels past int coords,
    # (2) matches old sampling strategy (~0.8*patch_size radius from center constraint)
    eroded_multi = _apply_erosion(availability_mask[0], radius=2) & valid_bounds_mask
    
    saved_patches = 0
    for downsampled_idx in random_sample_indices(eroded_multi, multi_motor_samples):
        patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor, tomo_shape, patch_size)

        _save_patch(tomo_array, gaussian_heatmap, motors, patch_origin, patch_size, downsampling_factor, "multi_motor", tomo_dir)
        saved_patches += 1

    return saved_patches
        


def _single_motor_save(
    single_motor_samples: int,
    tomo_dict: dict,
    tomo_dir: Path,
    availability_mask: np.ndarray,
    valid_bounds_mask: np.ndarray,
    downsampling_factor: int,
    gaussian_heatmap: torch.Tensor,
    patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    tomo_shape = np.array(tomo_array.shape)
    downsampled_patch_size = patch_size // downsampling_factor

    _log(f'_single_motor_save: motors={motors}, ds_patch_size={downsampled_patch_size}', 'DEBUG', True)

    # First pass: mark all single-motor regions in availability_mask[1] for exclusion later
    for motor_idx_tuple in motors:
        ds_motor = np.array(motor_idx_tuple) // downsampling_factor
        ds_start = ds_motor - downsampled_patch_size + 1
        ds_end = ds_motor + 1
        ds_start = np.maximum(ds_start, 0)
        _log(f'_single_motor_save: motor={motor_idx_tuple}, ds_motor={ds_motor}, marking ds_start={ds_start}, ds_end={ds_end}', 'DEBUG', True)
        availability_mask[1, ds_start[0]:ds_end[0], ds_start[1]:ds_end[1], ds_start[2]:ds_end[2]] = 1

    _log(f'_single_motor_save: before exclusion single_mask_sum={availability_mask[1].sum()}, multi_mask_sum={availability_mask[0].sum()}', 'DEBUG', True)

    # Dilate multi-motor mask by 1 to account for random offset uncertainty
    dilated_multi = _apply_dilation(availability_mask[0], radius=1)

    # Erosion on local copy only - don't mutate availability_mask[1] (needed for hard_neg exclusion)
    # Radius=2 because: (1) gaussian uses float ds coords which can be up to 0.99 voxels past int coords,
    # (2) matches old sampling strategy (~0.8*patch_size radius from center constraint)
    local_motor_mask = availability_mask[1].copy()
    local_motor_mask = _apply_erosion(local_motor_mask, radius=2)
    #apply erosion then & not dilated multi
    local_motor_mask = local_motor_mask & ~dilated_multi & valid_bounds_mask

    _log(f'_single_motor_save: after erosion+exclusion local_mask_sum={local_motor_mask.sum()}', 'DEBUG', True)

    #global one is unmutated so that hard negative example has clean stuff to work with
    #availability_mask[1]

    # Second pass: sample per-motor for diversity
    saved_patches = 0
    for motor_idx_tuple in motors:
        for downsampled_idx in random_sample_indices(local_motor_mask, single_motor_samples):
            patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor, tomo_shape, patch_size)
            _log(f'_single_motor_save: ds_idx={downsampled_idx}, patch_origin={patch_origin}', 'DEBUG', True)

            _save_patch(tomo_array, gaussian_heatmap, motors, patch_origin, patch_size, downsampling_factor, "single_motor", tomo_dir)
            saved_patches += 1

    return saved_patches



def _apply_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode mask by radius voxels - shrinks mask from edges.

    Used to ensure motors aren't at patch boundaries where gaussians would be truncated.
    Opposite of dilation: requires ALL neighbors in kernel to be True.
    Out-of-bounds treated as False (padding=0) so volume boundary edges also erode.
    This ensures consistent training: all motors are radius voxels from patch edge,
    no exceptions for tomogram boundaries.
    """
    kernel_size = 2 * radius + 1
    kernel_volume = kernel_size ** 3
    conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
    conv.weight.data.fill_(1.0)

    tensor = torch.from_numpy(mask.astype(np.float32)).reshape(1, 1, *mask.shape)
    padded = F.pad(tensor, (radius,) * 6, mode='constant', value=0.0)
    result = conv(padded) == kernel_volume  # ALL neighbors must be True
    return result.squeeze().numpy()


def _apply_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate mask by radius voxels - expands mask from edges.

    Used to account for random offset uncertainty when excluding multi-motor regions.
    Opposite of erosion: ANY neighbor in kernel being True makes output True.
    """
    kernel_size = 2 * radius + 1
    conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=radius, bias=False)
    conv.weight.data.fill_(1.0)

    tensor = torch.from_numpy(mask.astype(np.float32)).reshape(1, 1, *mask.shape)
    result = conv(tensor) > 0  # ANY neighbor True makes output True
    return result.squeeze().numpy()


def _asymmetric_box_conv(
    mask: np.ndarray,
    multiplier: float,
    ds_patch_size: np.ndarray,
    is_close: bool
) -> np.ndarray:
    """Apply asymmetric box kernel that extends toward positive direction.

    Used for hard negative sampling where patch origins are "behind" motors.
    The kernel extends from middle (or middle-1 for close) to end in each dim.
    """
    radius = (multiplier * ds_patch_size).astype(int)
    if is_close:
        radius = radius + 1

    kernel_size = tuple(2 * radius + 1)
    middle_idx = radius  # center of kernel

    kernel = np.zeros(kernel_size, dtype=np.float32)
    if is_close:
        # middle_idx - 1 to end (inclusive)
        kernel[middle_idx[0]-1:, middle_idx[1]-1:, middle_idx[2]-1:] = 1.0
    else:
        # middle_idx to end (inclusive)
        kernel[middle_idx[0]:, middle_idx[1]:, middle_idx[2]:] = 1.0

    padding = tuple((k - 1) // 2 for k in kernel_size)
    conv = torch.nn.Conv3d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
    conv.weight.data = torch.from_numpy(kernel).reshape(1, 1, *kernel_size)

    tensor = torch.from_numpy(mask.astype(np.float32)).reshape(1, 1, *mask.shape)
    result = conv(tensor) > 0
    return result.squeeze().numpy()


def _close_conv(mask: np.ndarray, multiplier: float, ds_patch_size: np.ndarray) -> np.ndarray:
    return _asymmetric_box_conv(mask, multiplier, ds_patch_size, is_close=True)


def _far_conv(mask: np.ndarray, multiplier: float, ds_patch_size: np.ndarray) -> np.ndarray:
    return _asymmetric_box_conv(mask, multiplier, ds_patch_size, is_close=False)


def _hard_negative_save(hard_negative_samples: float,
                        tomo_dict: dict,
                        tomo_dir: Path,
                        availability_mask: np.ndarray,
                        valid_bounds_mask: np.ndarray,
                        downsampling_factor: int,
                        gaussian_heatmap: torch.Tensor,
                        patch_size: np.ndarray) -> int:

    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    tomo_shape = np.array(tomo_array.shape)
    downsampled_patch_size = patch_size // downsampling_factor
    ds_shape = tomo_shape // downsampling_factor

    # availability_mask[0] and [1] are pre-erosion, so this covers all motor-containing origins
    positive = availability_mask[0] | availability_mask[1]

    # Asymmetric kernels: extend toward positive direction where motors are relative to origins
    close = _close_conv(positive, 0.1, downsampled_patch_size)
    far = _far_conv(positive, 0.3, downsampled_patch_size)

    #hard negative = within far boundary but outside close exclusion, AND within valid bounds
    availability_mask[2] = ~close & far & valid_bounds_mask
    _log(f'hard_neg: positive_sum={positive.sum()}, close_sum={close.sum()}, far_sum={far.sum()}, '
         f'hard_neg_sum={availability_mask[2].sum()}', 'DEBUG', True)
    
    # Per-motor sampling from existing hard_neg mask
    saved_patches = 0
    for motor_idx_tuple in motors:
        num_samples = probabilistic_round(hard_negative_samples)
        if num_samples == 0:
            continue

        hard_neg = availability_mask[2]
        for downsampled_idx in random_sample_indices(hard_neg, num_samples):
            patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor, tomo_shape, patch_size)

            _save_patch(tomo_array, gaussian_heatmap, motors, patch_origin, patch_size, downsampling_factor, "hard_negative", tomo_dir)
            saved_patches += 1
    return saved_patches


def _random_negative_save(random_negative_samples: float,
                        tomo_dict: dict,
                        tomo_dir: Path,
                        availability_mask: np.ndarray,
                        valid_bounds_mask: np.ndarray,
                        downsampling_factor: int,
                        gaussian_heatmap: torch.Tensor,
                        patch_size: np.ndarray) -> int:
    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    tomo_shape = np.array(tomo_array.shape)
    downsampled_patch_size = patch_size // downsampling_factor

    # Can't just do ~multi & ~single & ~hard_neg because hard_neg has a deadzone
    # (the "close" exclusion gap between positive regions and the hard_neg ring).
    # Random negatives could land in that deadzone if we didn't compute far manually.
    positive = availability_mask[0] | availability_mask[1]
    far = _far_conv(positive, 0.3, downsampled_patch_size)
    
    # Random negatives = outside far zone (truly far from any motors)
    availability_mask[3] = ~far & valid_bounds_mask
    _log(f'random_neg: mask_sum={availability_mask[3].sum()}', 'DEBUG', True)

    num_samples = probabilistic_round(random_negative_samples)
    saved_patches = 0
    for downsampled_idx in random_sample_indices(availability_mask[3], num_samples):
        patch_origin = downsampled_to_real_idx(downsampled_idx, downsampling_factor, tomo_shape, patch_size)
        _log(f'random_neg patch: ds_idx={downsampled_idx}, origin={patch_origin}', 'DEBUG', True)

        _save_patch(tomo_array, gaussian_heatmap, motors, patch_origin, patch_size, downsampling_factor, "random_negative", tomo_dir)
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
    patch_size:tuple,
    visualize: bool,
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

        save_patches_placeholder(tomo_dict, dst, data_split_dict=data_split_dict, downsampling_factor=downsampling_factor, angstrom_sigma=angstrom_sigma, patch_size=patch_size, visualize=visualize)
        
        del tomo_dict
        gc.collect()
        processed += 1

    # Wait for producers to finish
    for p in producers:
        p.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    #SUSPICIOUS MOTOR, GIVEN LABELS SINGLE SAMPLE SHOULD BE EASILY OBTAINED
#     Processing: tomo_0da370 (fold=-1, motors=1, shape=300x928x928, voxel_spacing=13.1)
# INFO: tomo_0da370 saves: multi=0 (0.00s), single=0 (0.00s), hard_neg=1 (0.12s), random_neg=1 (0.03s)
# INFO: tomo_0da370 total: 2 patches in 0.15s\

    src_root = Path(r'data/original_data/train')
    dst_root = Path(r'data/processed/old_labels')
    csv_path = Path(r'data/original_data/train_labels.csv')
    data_split_dict = {
        'single_motor_samples': 15,  # per motor (also used for multi_motor: single * num_motors // 2)
        'hard_negative_samples': 1,  # per motor, float supported (0.5 = 50% chance of 1)
        'random_negative_samples': 1  # per tomogram, float supported
    }
    
    downsampling_factor = 16
    angstrom_sigma = 250
    patch_size = (160, 288, 288)
    # Note on motor detection thresholds: Using float downsampled coords means motors near
    # voxel edges have <1.0 gaussian peak. E.g. with 250 angstrom sigma and 16 voxel spacing,
    # a motor 0.2 ds voxels from edge has ~0.75 confidence (ds_pixel 10.8 gives pixel 11 ~0.75 conf).

    write_whole_directory(src=src_root, dst=dst_root, csv_path=csv_path, num_workers=3, queue_size=2, data_split_dict=data_split_dict, downsampling_factor=downsampling_factor, angstrom_sigma=angstrom_sigma, patch_size=patch_size, visualize=False)
