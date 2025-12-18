"""Sanity check: verify patch gaussian peaks match GT motor locations."""
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))
from train.utils import load_ground_truth

DOWNSAMPLING_FACTOR = 16
PATCH_SIZE = np.array([160, 288, 288])


def load_patches_by_tomo(dst_root: Path) -> dict:
    """Load all patch files grouped by tomo_id."""
    patches = defaultdict(list)
    for patch_file in dst_root.rglob('*.pt'):
        tomo_id = patch_file.parent.name
        patches[tomo_id].append(patch_file)
    return dict(patches)


def get_patch_peaks(patch_file: Path) -> list:
    """Extract real-space peak coords from patch gaussian."""
    parts = patch_file.stem.split('_')
    patch_origin = np.array([int(parts[1]), int(parts[2]), int(parts[3])])

    data = torch.load(patch_file, weights_only=False)
    gaussian = data['gaussian'].squeeze()

    peak_indices = (gaussian == 1).nonzero()
    if len(peak_indices) == 0:
        return []

    peaks = []
    for p in peak_indices:
        # Center of downsampled voxel in real space
        real_coord = patch_origin + np.array([int(p[0]), int(p[1]), int(p[2])]) * DOWNSAMPLING_FACTOR
        peaks.append(real_coord)
    return peaks


def find_nearest_gt(peak: np.ndarray, gt_motors: list) -> tuple:
    """Find nearest GT motor to peak, return (nearest_coord, MAE)."""
    if not gt_motors:
        return None, float('inf')

    distances = [np.mean(np.abs(peak - np.array(gt))) for gt in gt_motors]
    min_idx = np.argmin(distances)
    return gt_motors[min_idx], distances[min_idx]


def check_patches(dst_root: Path, csv_path: Path, tomo_stride: int):
    patches_by_tomo = load_patches_by_tomo(dst_root)
    tomo_ids = list(patches_by_tomo.keys())[::tomo_stride]

    # Load GT in real space (downsampling_factor=1)
    gt = load_ground_truth(csv_path, tomo_ids, downsampling_factor=1)

    correct = 0
    incorrect = 0
    incorrect_by_tomo = defaultdict(list)

    # Max distance threshold - peak should be within 1 downsampled voxel of GT
    max_dist = DOWNSAMPLING_FACTOR

    for tomo_id in tomo_ids:
        gt_motors = gt.get(tomo_id, [])
        patch_files = patches_by_tomo[tomo_id]

        for patch_file in patch_files:
            peaks = get_patch_peaks(patch_file)

            for peak in peaks:
                nearest_gt, dist = find_nearest_gt(peak, gt_motors)

                if dist <= max_dist:
                    correct += 1
                else:
                    incorrect += 1
                    incorrect_by_tomo[tomo_id].append({
                        'patch': patch_file.name,
                        'peak': peak.tolist(),
                        'nearest_gt': nearest_gt,
                        'distance': dist
                    })

    total = correct + incorrect
    if total == 0:
        print("No patches with gaussian peaks found")
        return

    print(f"\n=== Patch-GT Sanity Check ===")
    print(f"Tomos checked: {len(tomo_ids)} (stride={tomo_stride})")
    print(f"Total peaks: {total}")
    print(f"Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/total:.1f}%)")
    print(f"Max distance threshold: {max_dist:.1f} (patch diagonal)")

    if incorrect_by_tomo:
        print(f"\n=== Incorrect by Tomo ===")
        for tomo_id, errors in sorted(incorrect_by_tomo.items()):
            print(f"\n{tomo_id}: {len(errors)} incorrect")
            for e in errors[:5]:  # Show first 5
                print(f"  {e['patch']}: peak={e['peak']}, nearest_gt={e['nearest_gt']}, MAE ={e['distance']:.1f}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")


if __name__ == '__main__':
    dst_root = Path(r'data/processed/old_data')
    csv_path = Path(r'data/original_data/train_labels.csv')
    tomo_stride = 10

    check_patches(dst_root, csv_path, tomo_stride)
