import time
import numpy as np
from pathlib import Path

class ThroughputTracker:
    def __init__(self, name: str = None, update_interval=5):
        self.running_mb = 0
        self.updates = 0
        self.last_update = time.perf_counter()
        self.update_interval = update_interval
        self.name = name

    def update(self, mb):
        self.running_mb += mb
        self.updates += 1
        current_time = time.perf_counter()
        
        if current_time - self.last_update >= self.update_interval:
            time_elapsed = current_time - self.last_update
            mb_s = self.running_mb / time_elapsed
            iters_s = self.updates / time_elapsed
            if self.name:
                print(f'thruput tracker: {self.name}')
            print(f'iterations/s: {iters_s:.2f}')
            print(f'mb/s: {mb_s:.2f}')
            print('-' * 30)
            self.running_mb = 0
            self.updates = 0
            self.last_update = current_time

def load_patch_data(tomo_dir, patch_id):
    """Load patches, labels, and global_coords for a given patch_id"""
    patches_path = tomo_dir / 'patches' / f'{patch_id}.npy'
    labels_path = tomo_dir / 'labels' / f'{patch_id}.npy'
    coords_path = tomo_dir / 'global_coords' / f'{patch_id}.npy'
    
    patches = np.load(patches_path)
    labels = np.load(labels_path)
    coords = np.load(coords_path)
    
    return patches, labels, coords

def calculate_mb_size(*arrays):
    """Calculate total size in MB for given numpy arrays"""
    total_bytes = sum(arr.nbytes for arr in arrays)
    return total_bytes / (1024 * 1024)

def main():
    master_dir = Path.cwd() / 'patch_np_data'
    assert master_dir.exists()
    
    # Get all tomo directories
    tomo_dirs = [d for d in master_dir.iterdir() if d.is_dir()]
    
    # Collect all patch files from all tomos
    all_patches = []
    for tomo_dir in tomo_dirs:
        patches_dir = tomo_dir / 'patches'
        if patches_dir.exists():
            patch_files = list(patches_dir.glob('*.npy'))
            for patch_file in patch_files:
                all_patches.append((tomo_dir, patch_file.stem))
    
    tracker = ThroughputTracker('numpy_patch_loader', update_interval=1)
    
    while True:
        for tomo_dir, patch_id in all_patches:
            patches, labels, coords = load_patch_data(tomo_dir, patch_id)
            total_mb = calculate_mb_size(patches, labels, coords)
            tracker.update(total_mb)

if __name__ == '__main__':
    main()