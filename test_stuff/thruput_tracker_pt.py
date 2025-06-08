import time
import torch
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
    """Load pt file for a given patch_id"""
    patch_path = tomo_dir / f'{patch_id}.pt'
    return torch.load(patch_path)

def main():
    master_dir = Path.cwd() / 'patch_pt_data'
    assert master_dir.exists()
    
    # Get all tomo directories
    tomo_dirs = [d for d in master_dir.iterdir() if d.is_dir()]
    
    # Collect all patch files from all tomos
    all_patches = []
    for tomo_dir in tomo_dirs:
        patch_files = list(tomo_dir.glob('*.pt'))
        for patch_file in patch_files:
            all_patches.append((tomo_dir, patch_file.stem))
    
    tracker = ThroughputTracker('pt_patch_loader', update_interval=1)
    
    while True:
        for tomo_dir, patch_id in all_patches:
            data_dict = load_patch_data(tomo_dir, patch_id)
            patches = data_dict['patch']
            coords = data_dict['global_coords']
            labels = data_dict['labels']
            total_mb = (patches.numel() * patches.element_size() + coords.numel() * coords.element_size() + labels.numel() * labels.element_size()) / (1024 * 1024)
            tracker.update(total_mb)

if __name__ == '__main__':
    main()