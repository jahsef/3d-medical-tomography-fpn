"""Compare pre-saved patch vs fresh extraction from tomogram."""
import torch
import numpy as np
from pathlib import Path
import imageio.v3 as iio
from natsort import natsorted

# Pick a specific patch to compare
PATCH_PATH = Path(r'data/processed/old_data')
TOMO_PATH = Path(r'data/original_data/train')

# Normalization (same as convert_pt.py)
def normalize_convert_pt(tomo_array):
    """Exact normalize from convert_pt.py"""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized

def load_raw_tomogram(tomo_dir):
    """Load tomogram WITHOUT normalization"""
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [f for f in tomo_dir.rglob('*') if f.is_file() and f.suffix.lower() in exts]
    files = natsorted(files, key=lambda x: x.name)
    imgs = [iio.imread(f, mode='L') for f in files]
    return np.stack(imgs)

def main():
    # Find first tomo with patches
    for tomo_dir in PATCH_PATH.iterdir():
        if not tomo_dir.is_dir():
            continue

        patch_files = list(tomo_dir.glob('*.pt'))
        if not patch_files:
            continue

        tomo_id = tomo_dir.name
        print(f'Analyzing {tomo_id}')

        # Load first patch
        patch_file = patch_files[0]
        patch_dict = torch.load(patch_file, weights_only=False)
        saved_patch = patch_dict['patch'].numpy()

        # Parse origin from filename: patch_Z_Y_X.pt
        parts = patch_file.stem.split('_')
        origin = [int(parts[1]), int(parts[2]), int(parts[3])]
        print(f'Patch origin: {origin}')
        print(f'Saved patch dtype: {saved_patch.dtype}, shape: {saved_patch.shape}')

        # Load raw tomogram and extract same region
        raw_tomo = load_raw_tomogram(TOMO_PATH / tomo_id)
        print(f'Raw tomo dtype: {raw_tomo.dtype}, shape: {raw_tomo.shape}')

        # Normalize using convert_pt method
        normalized_tomo = normalize_convert_pt(raw_tomo)
        print(f'Normalized tomo dtype: {normalized_tomo.dtype}')

        # Extract same region
        d, h, w = origin
        patch_size = saved_patch.shape
        extracted_patch = normalized_tomo[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
        print(f'Extracted patch dtype: {extracted_patch.dtype}, shape: {extracted_patch.shape}')

        # Compare
        saved_f32 = saved_patch.astype(np.float32)
        extracted_f32 = extracted_patch.astype(np.float32)

        diff = np.abs(saved_f32 - extracted_f32)
        print(f'\n=== COMPARISON ===')
        print(f'Max absolute difference: {diff.max():.10f}')
        print(f'Mean absolute difference: {diff.mean():.10f}')
        print(f'Saved range: [{saved_f32.min():.6f}, {saved_f32.max():.6f}]')
        print(f'Extracted range: [{extracted_f32.min():.6f}, {extracted_f32.max():.6f}]')

        # Check a few sample values
        print(f'\nSample values at [0,0,0]:')
        print(f'  Saved: {saved_f32[0,0,0]:.10f}')
        print(f'  Extracted: {extracted_f32[0,0,0]:.10f}')

        print(f'\nSample values at center:')
        c = [s//2 for s in patch_size]
        print(f'  Saved: {saved_f32[c[0],c[1],c[2]]:.10f}')
        print(f'  Extracted: {extracted_f32[c[0],c[1],c[2]]:.10f}')

        # Check if they're essentially identical
        if diff.max() < 1e-5:
            print('\n✓ Patches are essentially IDENTICAL')
        else:
            print('\n✗ Patches DIFFER significantly!')

            # Find where they differ most
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f'Max diff at index {max_idx}:')
            print(f'  Saved: {saved_f32[max_idx]:.10f}')
            print(f'  Extracted: {extracted_f32[max_idx]:.10f}')

        break  # Just check one

if __name__ == '__main__':
    main()
