"""Test offset sensitivity with visualization."""
import torch
import numpy as np
from pathlib import Path
import imageio.v3 as iio
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))
from model_defs.motor_detector import MotorDetector

# Config
TOMO_PATH = Path(r'data/original_data/train')
PATCH_DATA_PATH = Path(r'data/processed/old_data')
CSV_PATH = Path(r'data/original_data/train_labels.csv')
MODEL_PATH = Path(r'models/old_data/parallel_fpn_cornernet_fold0/weights/best.pt')
DEVICE = torch.device('cuda')
PATCH_SIZE = (160, 288, 288)
DS_FACTOR = 16

# Which tomo to use
TOMO_ID = 'tomo_00e047'

# Offsets to test per axis
OFFSETS_1D = [0, 1, 2, 4, 8, 12, 14, 15, 16, 17, 32]

# Which axis to visualize: 1=Y, 2=X (no Z offset for visualization)
VIS_AXIS = 2


def normalize_tomogram(tomo_array):
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized


def load_tomogram(tomo_dir):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [f for f in tomo_dir.rglob('*') if f.is_file() and f.suffix.lower() in exts]
    files = natsorted(files, key=lambda x: x.name)
    imgs = [iio.imread(f, mode='L') for f in files]
    return normalize_tomogram(np.stack(imgs))


def run_inference(model, patch_np):
    patch_tensor = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model.forward(patch_tensor)).squeeze().cpu().numpy()
    return pred


def get_motor_from_csv(csv_path, tomo_id):
    """Get first motor coord from CSV."""
    df = pd.read_csv(csv_path)
    rows = df[df['tomo_id'] == tomo_id]
    for _, row in rows.iterrows():
        z, y, x = row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']
        if z != -1:
            return np.array([int(z), int(y), int(x)])
    return None


def find_motor_patch_origin(patch_data_path, tomo_id):
    """Find a patch with motor for this tomo, return origin from filename."""
    tomo_dir = patch_data_path / tomo_id
    if not tomo_dir.exists():
        return None
    for patch_file in tomo_dir.glob('*.pt'):
        patch_dict = torch.load(patch_file, weights_only=False)
        # New format
        if 'gaussian' in patch_dict and patch_dict['gaussian'].max() > 0.5:
            parts = patch_file.stem.split('_')
            origin = np.array([int(parts[1]), int(parts[2]), int(parts[3])])
            return origin, patch_file
        # Old format
        if 'labels' in patch_dict:
            labels = patch_dict['labels'].numpy()
            if labels.shape[0] > 0 and labels[0, 3] == 1:
                parts = patch_file.stem.split('_')
                origin = np.array([int(parts[1]), int(parts[2]), int(parts[3])])
                return origin, patch_file
    return None


def main():
    print('Loading model...')
    model, _ = MotorDetector.load_checkpoint(str(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    # Get motor from CSV
    motor = get_motor_from_csv(CSV_PATH, TOMO_ID)
    if motor is None:
        print(f'No motor found in CSV for {TOMO_ID}')
        return
    print(f'Motor: {motor}')

    # Find a training patch and get its origin from filename
    result = find_motor_patch_origin(PATCH_DATA_PATH, TOMO_ID)
    if result is None:
        print(f'No motor patch found for {TOMO_ID}')
        return
    base_origin, patch_file = result
    print(f'Patch: {patch_file}')
    print(f'Base origin: {base_origin}')
    print(f'Base origin mod 16: {base_origin % DS_FACTOR}')

    # Load tomogram
    print(f'Loading tomogram {TOMO_ID}...')
    tomo = load_tomogram(TOMO_PATH / TOMO_ID)
    tomo_shape = np.array(tomo.shape)

    axis_names = ['Z', 'Y', 'X']

    # Print table for all axes
    for axis in range(3):
        print(f'\n=== Offsets along {axis_names[axis]} axis ===')
        print(f'{"Offset":<8} {"Max Pred":>10} {"Argmax":<20} {"Motor DS pos"}')
        print('-' * 60)

        for off in OFFSETS_1D:
            offset = np.array([0, 0, 0])
            offset[axis] = off
            origin = base_origin + offset

            if np.any(origin < 0) or np.any(origin + PATCH_SIZE > tomo_shape):
                print(f'{off:<8} {"OOB":>10}')
                continue

            d, h, w = origin
            patch = tomo[d:d+PATCH_SIZE[0], h:h+PATCH_SIZE[1], w:w+PATCH_SIZE[2]]
            pred = run_inference(model, patch)
            motor_in_patch_ds = (motor - origin) // DS_FACTOR
            argmax = np.unravel_index(np.argmax(pred), pred.shape)

            print(f'{off:<8} {pred.max():>10.6f} {str(argmax):<20} {motor_in_patch_ds}')

    # Visualize selected axis (Y or X only)
    assert VIS_AXIS in [1, 2], "VIS_AXIS must be 1 (Y) or 2 (X)"
    print(f'\n=== Visualizing {axis_names[VIS_AXIS]} axis ===')

    valid_offsets = []
    for off in OFFSETS_1D:
        offset = np.array([0, 0, 0])
        offset[VIS_AXIS] = off
        origin = base_origin + offset
        if not (np.any(origin < 0) or np.any(origin + PATCH_SIZE > tomo_shape)):
            valid_offsets.append(off)

    # Get depth slice from offset=0
    d, h, w = base_origin
    patch_0 = tomo[d:d+PATCH_SIZE[0], h:h+PATCH_SIZE[1], w:w+PATCH_SIZE[2]]
    pred_0 = run_inference(model, patch_0)
    argmax_0 = np.unravel_index(np.argmax(pred_0), pred_0.shape)
    pred_slice = argmax_0[0]
    real_slice = pred_slice * DS_FACTOR + DS_FACTOR // 2

    n_offsets = len(valid_offsets)
    fig, axes = plt.subplots(2, n_offsets, figsize=(3 * n_offsets, 6), squeeze=False)

    for idx, off in enumerate(valid_offsets):
        offset = np.array([0, 0, 0])
        offset[VIS_AXIS] = off
        origin = base_origin + offset

        d, h, w = origin
        patch = tomo[d:d+PATCH_SIZE[0], h:h+PATCH_SIZE[1], w:w+PATCH_SIZE[2]]
        pred = run_inference(model, patch)

        # Patch - same depth slice for all
        axes[0, idx].imshow(patch[real_slice], cmap='gray')
        axes[0, idx].set_title(f'off={off}')
        axes[0, idx].axis('off')

        # Prediction - same depth slice for all
        axes[1, idx].imshow(pred[pred_slice], cmap='hot', vmin=0, vmax=1)
        axes[1, idx].set_title(f'max={pred.max():.3f}')
        axes[1, idx].axis('off')

    plt.suptitle(f'{TOMO_ID} - {axis_names[VIS_AXIS]} axis offsets')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
