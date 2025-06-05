import time
import random
from collections import defaultdict
import numpy as np
import dask.array as da
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import pandas as pd

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
            print('-'*30)
            self.running_mb = 0
            self.updates = 0
            self.last_update = current_time

def load_labels(csv_path, max_motors_per_patch=5):
    """Load labels from CSV and organize by tomo_id"""
    df = pd.read_csv(csv_path)
    labels_dict = {}
    
    for tomo_id, group in df.groupby('tomo_id'):
        # Filter out rows with no motors
        motor_rows = group[group['Number of motors'] > 0]
        
        if len(motor_rows) > 0:
            coords = motor_rows[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
            # Add confidence of 1.0 for all motors (xyzconf format)
            conf = np.ones((len(coords), 1))
            xyzconf = np.hstack([coords, conf]).astype(np.float32)
            
            # Limit to max motors
            if len(xyzconf) > max_motors_per_patch:
                xyzconf = xyzconf[:max_motors_per_patch]
                
            labels_dict[tomo_id] = xyzconf
        else:
            labels_dict[tomo_id] = np.empty((0, 4), dtype=np.float32)
    
    return labels_dict

class OnDemandBalancedPatchDataset(Dataset):
    def __init__(self, zarr_files, labels_dict, patch_size=(64, 64, 64), 
                 max_motors_per_patch=5, positive_ratio=0.5, 
                 samples_per_epoch=10000, stride_factor=0.5, seed=None):
        """
        Memory-efficient dataset that generates patch coordinates on-demand
        """
        self.zarr_files = zarr_files
        self.labels_dict = labels_dict
        self.patch_size = patch_size
        self.max_motors_per_patch = max_motors_per_patch
        self.positive_ratio = positive_ratio
        self.samples_per_epoch = samples_per_epoch
        self.stride_factor = stride_factor
        self.base_seed = seed or 42
        self.epoch = 0
        
        # Pre-open zarr arrays as dask arrays (minimal memory overhead)
        self.dask_arrays = {}
        self.tomo_metadata = {}
        
        for zarr_path in zarr_files:
            tomo_id = zarr_path.stem.replace('.zarr', '')
            z = zarr.open(str(zarr_path), mode='r')
            dask_array = da.from_zarr(str(zarr_path))
            
            self.dask_arrays[tomo_id] = dask_array
            
            # Store only essential metadata for patch generation
            shape = z.shape
            stride = [int(s * self.stride_factor) for s in self.patch_size]
            max_starts = [max(0, shape[i] - self.patch_size[i]) for i in range(3)]
            
            self.tomo_metadata[tomo_id] = {
                'shape': shape,
                'stride': stride,
                'max_starts': max_starts,
                'total_patches': self._count_total_patches(shape, stride)
            }
        
        # Calculate tomo sampling weights
        self.tomo_ids = list(self.tomo_metadata.keys())
        self._setup_sampling_weights()
        
        # Calculate sample splits
        self.n_positive = int(self.samples_per_epoch * self.positive_ratio)
        self.n_negative = self.samples_per_epoch - self.n_positive
        
        print(f"Dataset initialized: {len(self.tomo_ids)} tomos, "
              f"{self.n_positive} positive + {self.n_negative} negative samples per epoch")
    
    def _setup_sampling_weights(self):
        """Setup sampling weights for positive and negative patches"""
        # For positive sampling: weight by motor count
        # For negative sampling: weight by total patches
        self.positive_tomo_weights = np.array([
            len(self.labels_dict.get(tomo_id, [])) 
            for tomo_id in self.tomo_ids
        ], dtype=np.float32)
        
        self.negative_tomo_weights = np.array([
            self.tomo_metadata[tomo_id]['total_patches'] 
            for tomo_id in self.tomo_ids
        ], dtype=np.float32)
        
        # Normalize weights (handle case where no motors exist)
        if self.positive_tomo_weights.sum() > 0:
            self.positive_tomo_weights = self.positive_tomo_weights / self.positive_tomo_weights.sum()
        else:
            self.positive_tomo_weights = self.negative_tomo_weights / self.negative_tomo_weights.sum()
            
        self.negative_tomo_weights = self.negative_tomo_weights / self.negative_tomo_weights.sum()
    
    def _count_total_patches(self, shape, stride):
        """Count total possible patches for a given shape and stride"""
        patches_per_dim = [
            (shape[i] - self.patch_size[i]) // stride[i] + 1
            for i in range(3)
        ]
        return np.prod(patches_per_dim)
    
    def _generate_patch_coords(self, tomo_id, rng, target_positive=False):
        """Generate patch coordinates, either around a motor (positive) or random (negative)"""
        if target_positive:
            return self._generate_positive_patch_coords(tomo_id, rng)
        else:
            return self._generate_negative_patch_coords(tomo_id, rng)
    
    def _generate_positive_patch_coords(self, tomo_id, rng):
        """Generate a patch coordinates around a motor"""
        motors = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
        
        if len(motors) == 0:
            # Fallback to random sampling if no motors
            return self._generate_random_patch_coords(tomo_id, rng)
        
        # Pick a random motor
        motor_idx = rng.randint(0, len(motors))
        motor_pos = motors[motor_idx][:3].astype(int)
        
        # Generate patch around this motor with some randomness
        metadata = self.tomo_metadata[tomo_id]
        max_starts = metadata['max_starts']
        
        # Calculate valid start range that ensures motor is inside patch
        start = []
        for i in range(3):
            # Motor must be inside patch: start[i] <= motor_pos[i] < start[i] + patch_size[i]
            min_start = max(0, motor_pos[i] - self.patch_size[i] + 1)
            max_start = min(max_starts[i], motor_pos[i])
            
            if min_start <= max_start:
                start.append(rng.randint(min_start, max_start + 1))
            else:
                # Motor is too close to edge, use boundary
                start.append(max(0, min(max_starts[i], motor_pos[i] - self.patch_size[i] // 2)))
        
        end = [start[i] + self.patch_size[i] for i in range(3)]
        return start, end
    
    def _generate_negative_patch_coords(self, tomo_id, rng):
        """Generate a patch that doesn't contain motors"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate random patch coordinates
            start, end = self._generate_random_patch_coords(tomo_id, rng)
            
            # Check if this patch contains motors
            if not self._patch_contains_motor(tomo_id, start, end):
                return start, end
        
        # Fallback: return last generated patch (might contain motors but rare)
        return start, end
    
    def _patch_contains_motor(self, tomo_id, start, end):
        """Check if a patch contains any motor coordinates"""
        global_labels = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
        
        if len(global_labels) == 0:
            return False
            
        start_np = np.array(start)
        end_np = np.array(end)
        
        for label in global_labels:
            xyz = label[:3]
            if np.all(xyz >= start_np) and np.all(xyz < end_np):
                return True
        return False
    
    def _generate_random_patch_coords(self, tomo_id, rng):
        """Generate random patch coordinates for a given tomo"""
        metadata = self.tomo_metadata[tomo_id]
        
        # Generate random start coordinates within valid ranges
        start = [
            rng.randint(0, metadata['max_starts'][i] + 1)
            for i in range(3)
        ]
        
        end = [start[i] + self.patch_size[i] for i in range(3)]
        return start, end
    
    def set_epoch(self, epoch):
        """Update epoch for different sampling each epoch"""
        self.epoch = epoch
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Use deterministic RNG to decide class for this index
        rng = np.random.RandomState(self.base_seed + self.epoch * 100000 + idx)
        target_positive = rng.random() < self.positive_ratio
        
        # Select tomo based on target class
        if target_positive:
            tomo_id = rng.choice(self.tomo_ids, p=self.positive_tomo_weights)
        else:
            tomo_id = rng.choice(self.tomo_ids, p=self.negative_tomo_weights)
        
        # Generate patch coordinates
        start, end = self._generate_patch_coords(tomo_id, rng, target_positive)
        
        # Extract patch using dask
        dask_array = self.dask_arrays[tomo_id]
        patch = dask_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        patch_data = patch.compute().astype(np.float16)
        
        # Add channel dimension
        if patch_data.ndim == 3:
            patch_data = patch_data[None, ...]
        
        # Get local labels
        local_labels = self._get_local_labels(tomo_id, start, end)
        
        return {
            'image': torch.from_numpy(patch_data),
            'labels': torch.from_numpy(local_labels),
            'num_motors': int(np.sum(local_labels[:, 3] > 0)),
            'tomo_id': tomo_id,
            'patch_coords': start,
            'is_positive': int(np.sum(local_labels[:, 3] > 0) > 0)
        }
    
    def _get_local_labels(self, tomo_id, start, end):
        """Extract labels within patch and convert to local coordinates"""
        global_labels = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
        
        local_labels = []
        start_np = np.array(start)
        end_np = np.array(end)
        
        for label in global_labels:
            xyz = label[:3]
            if np.all(xyz >= start_np) and np.all(xyz < end_np):
                # Convert to local coordinates
                local_xyz = xyz - start_np
                local_labels.append([local_xyz[0], local_xyz[1], local_xyz[2], label[3]])
        
        # Pad to fixed size
        padded_labels = np.zeros((self.max_motors_per_patch, 4), dtype=np.float32)
        if local_labels:
            local_labels = np.array(local_labels, dtype=np.float32)
            n_labels = min(len(local_labels), self.max_motors_per_patch)
            padded_labels[:n_labels] = local_labels[:n_labels]
        
        return padded_labels
    
    def get_class_distribution(self):
        """Get current class distribution statistics"""
        return {
            'target_positive_ratio': self.positive_ratio,
            'samples_per_epoch': self.samples_per_epoch,
            'positive_samples': self.n_positive,
            'negative_samples': self.n_negative,
            'total_tomos': len(self.tomo_ids)
        }

# Usage example
if __name__ == '__main__':
    # Load labels
    csv_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')
    max_motors_per_patch = 1
    labels_dict = load_labels(csv_path, max_motors_per_patch=max_motors_per_patch)
    
    # Create data items
    zarr_files = list(Path("normalized_zarr_data/train").glob("*.zarr"))
    
    # Create memory-efficient dataset
    batch_size = 64
    batches_per_epoch = 128
    dataset = OnDemandBalancedPatchDataset(
        zarr_files=zarr_files,
        labels_dict=labels_dict,
        patch_size=(64, 64, 64),
        max_motors_per_patch=max_motors_per_patch,
        positive_ratio=0.5,
        samples_per_epoch=batch_size * batches_per_epoch,
        stride_factor=0.5,
        seed=42
    )
    
    # Print class distribution
    print("Class distribution:", dataset.get_class_distribution())
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=2,
        shuffle=False,  # Deterministic but varied sampling
        pin_memory=True,
        persistent_workers=True
    )
    
    # Test throughput and class balance
    tracker = ThroughputTracker(name="on_demand_balanced_patches")
    positive_count = 0
    total_count = 0
    epochs = 50
    
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            patches = batch["image"]
            labels = batch["labels"]
            num_motors = batch["num_motors"]
            is_positive = batch["is_positive"]
            
            positive_count += is_positive.sum().item()
            total_count += len(is_positive)
            
            num_bytes = patches.numel() * patches.element_size()
            tracker.update(num_bytes / (1024 ** 2))
            
            if batch_idx == 0:
                print(f"Epoch {epoch} - First batch positive samples: {is_positive.sum().item()}/{len(is_positive)}")
    
    print(f"\nOverall class balance: {positive_count}/{total_count} = {positive_count/total_count:.2%} positive")