import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import imageio.v3 as iio

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

# MUST be a top-level function for multiprocessing/threading
def process_file(file_path):
    try:
        img = iio.imread(str(file_path), mode='L')  # Grayscale
        mean = img.mean()
        std = img.std()
        return mean, std, img.size
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0.0, 0.0, 0


def compute_dataset_stats_from_master(master_dir, extensions=IMAGE_EXTS, num_workers=8):
    """
    Compute global mean and std across all grayscale image files using threading.
    
    Args:
        master_dir (str or Path): Root directory containing subdirs.
        extensions (set): Image file extensions to include.
        num_workers (int): Number of worker threads (I/O bound).

    Returns:
        tuple: (global_mean, global_std)
    """
    master_dir = Path(master_dir)
    if not master_dir.is_dir():
        raise ValueError(f"{master_dir} is not a valid directory")

    print(f"Computing dataset stats from: {master_dir}")

    # Collect all image paths
    files = []
    for tomo_dir in master_dir.iterdir():
        if tomo_dir.is_dir():
            files.extend([f for f in tomo_dir.rglob('*') if f.is_file() and f.suffix.lower() in extensions])
    
    if not files:
        raise ValueError("No valid image files found in the provided directory")

    total_mean = 0.0
    total_var = 0.0
    total_pixels = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for mean, std, n in executor.map(process_file, files):
            if n == 0:
                continue
            total_mean += mean * n
            # Var(X) = E[X^2] - (E[X])^2
            total_var += (std ** 2 + mean ** 2) * n
            total_pixels += n

    global_mean = total_mean / total_pixels
    global_var = (total_var / total_pixels) - (global_mean ** 2)
    global_std = np.sqrt(global_var)

    print(f"Global Mean: {global_mean:.4f}")
    print(f"Global Std:  {global_std:.4f}")

    return global_mean, global_std


if __name__ == '__main__':
    mean, std = compute_dataset_stats_from_master(
        r'C:\Users\kevin\Documents\GitHub\kaggle BYU bacteria motor\original_data\train',
        num_workers=16
    )
    print(f"Final Mean: {mean:.4f}, Final Std: {std:.4f}")