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
from scipy.spatial import cKDTree
import pickle
import hashlib
from functools import lru_cache

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

class PatchCoordinateCache:
    """Cache for precomputed patch coordinates with disk persistence"""
    
    def __init__(self, cache_dir="./patch_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        
    def _get_cache_key(self, tomo_id, patch_size, stride_factor, max_motors_per_patch):
        """Generate unique cache key for configuration"""
        key_str = f"{tomo_id}_{patch_size}_{stride_factor}_{max_motors_per_patch}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get_cached_coordinates(self, tomo_id, patch_size, stride_factor, max_motors_per_patch):
        """Retrieve cached coordinates"""
        cache_key = self._get_cache_key(tomo_id, patch_size, stride_factor, max_motors_per_patch)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache[cache_key] = data
                    return data
            except Exception:
                # Cache corrupted, remove it
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def save_coordinates(self, tomo_id, patch_size, stride_factor, max_motors_per_patch, data):
        """Save coordinates to cache"""
        cache_key = self._get_cache_key(tomo_id, patch_size, stride_factor, max_motors_per_patch)
        
        # Save to memory
        self.memory_cache[cache_key] = data
        
        # Save to disk
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")

class SpatialIndex:
    """Spatial index for fast motor lookup using KD-tree"""
    
    def __init__(self, motors):
        self.motors = motors
        if len(motors) > 0:
            self.kdtree = cKDTree(motors[:, :3])
        else:
            self.kdtree = None
    
    def query_box(self, box_min, box_max):
        """Find all motors within a box defined by min/max coordinates"""
        if self.kdtree is None or len(self.motors) == 0:
            return []
        
        # Find points within the bounding box
        indices = []
        for i, motor in enumerate(self.motors):
            pos = motor[:3]
            if np.all(pos >= box_min) and np.all(pos < box_max):
                indices.append(i)
        
        return indices
    
    def has_motors_in_box(self, box_min, box_max):
        """Fast check if any motors exist in the box"""
        return len(self.query_box(box_min, box_max)) > 0

class OnDemandBalancedPatchDataset(Dataset):
    def __init__(self, zarr_files, labels_dict, patch_size=(64, 64, 64), 
                 max_motors_per_patch=5, positive_ratio=0.5, 
                 samples_per_epoch=10000, stride_factor=0.5, seed=None,
                 cache_dir="./patch_cache", precompute_patches=True,
                 negative_cache_size=1000):
        """
        Memory-efficient dataset with patch coordinate caching and spatial indexing
        
        Args:
            cache_dir: Directory to store patch coordinate cache
            precompute_patches: Whether to precompute and cache patch coordinates
            negative_cache_size: Number of negative patches to cache per tomo
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
        self.negative_cache_size = negative_cache_size
        
        # Initialize caching
        self.coord_cache = PatchCoordinateCache(cache_dir)
        
        # Pre-open zarr arrays as dask arrays (minimal memory overhead)
        self.dask_arrays = {}
        self.tomo_metadata = {}
        self.spatial_indices = {}
        
        print("Initializing dataset and building spatial indices...")
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
            
            # Build spatial index for motors
            motors = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
            self.spatial_indices[tomo_id] = SpatialIndex(motors)
        
        # Calculate tomo sampling weights
        self.tomo_ids = list(self.tomo_metadata.keys())
        self._setup_sampling_weights()
        
        # Precompute patch coordinates if requested
        if precompute_patches:
            self._precompute_patch_coordinates()
        
        # Calculate sample splits
        self.n_positive = int(self.samples_per_epoch * self.positive_ratio)
        self.n_negative = self.samples_per_epoch - self.n_positive
        
        print(f"Dataset initialized: {len(self.tomo_ids)} tomos, "
              f"{self.n_positive} positive + {self.n_negative} negative samples per epoch")
    
    def _precompute_patch_coordinates(self):
        """Precompute and cache patch coordinates for all tomos"""
        print("Precomputing patch coordinates...")
        
        for tomo_id in self.tomo_ids:
            # Check if already cached
            cached_data = self.coord_cache.get_cached_coordinates(
                tomo_id, self.patch_size, self.stride_factor, self.max_motors_per_patch
            )
            
            if cached_data is not None:
                print(f"Using cached coordinates for {tomo_id}")
                continue
            
            print(f"Computing coordinates for {tomo_id}...")
            
            # Generate positive patches (around motors)
            motors = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
            positive_patches = []
            
            if len(motors) > 0:
                for motor in motors:
                    motor_pos = motor[:3].astype(int)
                    # Generate multiple patches around each motor
                    for _ in range(10):  # Generate 10 variations per motor
                        start, end = self._generate_positive_patch_coords_around_motor(
                            tomo_id, motor_pos, np.random.RandomState(42)
                        )
                        positive_patches.append((start, end))
            
            # Generate negative patches (without motors)
            negative_patches = []
            rng = np.random.RandomState(42)
            attempts = 0
            max_attempts = self.negative_cache_size * 10
            
            while len(negative_patches) < self.negative_cache_size and attempts < max_attempts:
                start, end = self._generate_random_patch_coords(tomo_id, rng)
                if not self._patch_contains_motor_fast(tomo_id, start, end):
                    negative_patches.append((start, end))
                attempts += 1
            
            # Cache the results
            cache_data = {
                'positive_patches': positive_patches,
                'negative_patches': negative_patches
            }
            
            self.coord_cache.save_coordinates(
                tomo_id, self.patch_size, self.stride_factor, 
                self.max_motors_per_patch, cache_data
            )
            
            print(f"Cached {len(positive_patches)} positive and {len(negative_patches)} negative patches for {tomo_id}")
    
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
    
    @lru_cache(maxsize=1000)
    def _get_cached_patch_coords(self, tomo_id, idx, target_positive, epoch):
        """Get patch coordinates with LRU caching"""
        return self._generate_patch_coords_uncached(tomo_id, idx, target_positive, epoch)
    
    def _generate_patch_coords_uncached(self, tomo_id, idx, target_positive, epoch):
        """Generate patch coordinates without caching"""
        rng = np.random.RandomState(self.base_seed + epoch * 100000 + idx)
        
        # Try to use precomputed coordinates first
        cached_data = self.coord_cache.get_cached_coordinates(
            tomo_id, self.patch_size, self.stride_factor, self.max_motors_per_patch
        )
        
        if cached_data is not None:
            if target_positive and cached_data['positive_patches']:
                patch_idx = rng.randint(0, len(cached_data['positive_patches']))
                return cached_data['positive_patches'][patch_idx]
            elif not target_positive and cached_data['negative_patches']:
                patch_idx = rng.randint(0, len(cached_data['negative_patches']))
                return cached_data['negative_patches'][patch_idx]
        
        # Fallback to dynamic generation
        if target_positive:
            return self._generate_positive_patch_coords(tomo_id, rng)
        else:
            return self._generate_negative_patch_coords(tomo_id, rng)
    
    def _generate_patch_coords(self, tomo_id, idx, target_positive, epoch):
        """Generate patch coordinates with caching"""
        return self._get_cached_patch_coords(tomo_id, idx, target_positive, epoch)
    
    def _generate_positive_patch_coords_around_motor(self, tomo_id, motor_pos, rng):
        """Generate patch coordinates around a specific motor position"""
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
    
    def _generate_positive_patch_coords(self, tomo_id, rng):
        """Generate a patch coordinates around a motor"""
        motors = self.labels_dict.get(tomo_id, np.empty((0, 4), dtype=np.float32))
        
        if len(motors) == 0:
            # Fallback to random sampling if no motors
            return self._generate_random_patch_coords(tomo_id, rng)
        
        # Pick a random motor
        motor_idx = rng.randint(0, len(motors))
        motor_pos = motors[motor_idx][:3].astype(int)
        
        return self._generate_positive_patch_coords_around_motor(tomo_id, motor_pos, rng)
    
    def _generate_negative_patch_coords(self, tomo_id, rng):
        """Generate a patch that doesn't contain motors"""
        max_attempts = 20  # Reduced from 50 for better performance
        
        for attempt in range(max_attempts):
            # Generate random patch coordinates
            start, end = self._generate_random_patch_coords(tomo_id, rng)
            
            # Check if this patch contains motors using fast spatial index
            if not self._patch_contains_motor_fast(tomo_id, start, end):
                return start, end
        
        # Fallback: return last generated patch (might contain motors but rare)
        return start, end
    
    def _patch_contains_motor_fast(self, tomo_id, start, end):
        """Fast check if a patch contains any motor coordinates using spatial index"""
        spatial_index = self.spatial_indices[tomo_id]
        return spatial_index.has_motors_in_box(np.array(start), np.array(end))
    
    def _patch_contains_motor(self, tomo_id, start, end):
        """Legacy method - kept for compatibility"""
        return self._patch_contains_motor_fast(tomo_id, start, end)
    
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
        # Clear LRU cache when epoch changes
        self._get_cached_patch_coords.cache_clear()
    
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
        
        # Generate patch coordinates using cache
        start, end = self._generate_patch_coords(tomo_id, idx, target_positive, self.epoch)
        
        # Extract patch using dask - use optimized slicing
        dask_array = self.dask_arrays[tomo_id]
        patch = dask_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # Compute with optimized memory usage
        patch_data = patch.compute().astype(np.float16)  # Use float16 to save memory
        
        # Add channel dimension
        if patch_data.ndim == 3:
            patch_data = patch_data[None, ...]
        
        # Get local labels using fast spatial index
        local_labels = self._get_local_labels_fast(tomo_id, start, end)
        
        return {
            'image': torch.from_numpy(patch_data),
            'labels': torch.from_numpy(local_labels),
            'num_motors': int(np.sum(local_labels[:, 3] > 0)),
            'valid_mask': local_labels[:,3] > 0,
            'tomo_id': tomo_id,
            'patch_coords': start,
            'is_positive': int(np.sum(local_labels[:, 3] > 0) > 0)
        }
    
    def _get_local_labels_fast(self, tomo_id, start, end):
        """Extract labels within patch using spatial index"""
        spatial_index = self.spatial_indices[tomo_id]
        start_np = np.array(start)
        end_np = np.array(end)
        
        # Use spatial index to find motors in patch
        motor_indices = spatial_index.query_box(start_np, end_np)
        
        local_labels = []
        for idx in motor_indices:
            motor = spatial_index.motors[idx]
            xyz = motor[:3]
            # Convert to local coordinates
            local_xyz = xyz - start_np
            local_labels.append([local_xyz[0], local_xyz[1], local_xyz[2], motor[3]])
        
        # Pad to fixed size
        padded_labels = np.zeros((self.max_motors_per_patch, 4), dtype=np.float32)
        if local_labels:
            local_labels = np.array(local_labels, dtype=np.float32)
            n_labels = min(len(local_labels), self.max_motors_per_patch)
            padded_labels[:n_labels] = local_labels[:n_labels]
        
        return padded_labels
    
    def _get_local_labels(self, tomo_id, start, end):
        """Legacy method - kept for compatibility"""
        return self._get_local_labels_fast(tomo_id, start, end)
    
    def get_class_distribution(self):
        """Get current class distribution statistics"""
        return {
            'target_positive_ratio': self.positive_ratio,
            'samples_per_epoch': self.samples_per_epoch,
            'positive_samples': self.n_positive,
            'negative_samples': self.n_negative,
            'total_tomos': len(self.tomo_ids)
        }
    
    def get_cache_stats(self):
        """Get caching statistics"""
        cache_info = self._get_cached_patch_coords.cache_info()
        return {
            'lru_cache_hits': cache_info.hits,
            'lru_cache_misses': cache_info.misses,
            'lru_cache_size': cache_info.currsize,
            'lru_cache_maxsize': cache_info.maxsize,
            'disk_cache_dir': str(self.coord_cache.cache_dir),
            'memory_cache_size': len(self.coord_cache.memory_cache)
        }