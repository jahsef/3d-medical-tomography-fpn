import torch
from pathlib import Path
import numpy as np
import random

class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, tomo_dir_list: list[Path], patches_per_batch: int = 16, transform=None):
        """
        Args:
            tomo_dir_list (list[Path]): list of tomogram directories containing patch files
            patches_per_batch (int): number of patches to sample per batch
            transform: optional transforms to apply to patches
        """
        super().__init__()
        self.tomo_dir_list = tomo_dir_list
        self.patches_per_batch = patches_per_batch
        self.transform = transform
        self._build_patch_index()
        self._determine_patch_stats()
        self.resample_batch()  # Initialize first batch
        
    def _build_patch_index(self):
        """Build an index of all individual patches across all tomograms"""
        self.patch_index = []
        
        for tomo_dir in self.tomo_dir_list:
            patch_paths = [x for x in tomo_dir.iterdir() if x.is_file()]
            for patch_path in patch_paths:
                self.patch_index.append(patch_path)
        
        print(f"Found {len(self.patch_index)} total patches across {len(self.tomo_dir_list)} tomograms")
    
    def _determine_patch_stats(self):
        """Determine patch statistics from the first patch"""
        if not self.patch_index:
            raise Exception("No patches found in the provided directories")
            
        # Load first patch to get dimensions
        test_patch_path = self.patch_index[0]
        test_dict = torch.load(test_patch_path)
        test_patch = test_dict['patch']
        test_xyzconf = test_dict['labels']
        test_globalcoords = test_dict['global_coords']
        
        # Determine channels
        if test_patch.ndim == 3:
            self.channels = 1
        elif test_patch.ndim == 4:
            self.channels = test_patch.shape[0]
        else:
            raise Exception(f'test_patch ndim should be 3 or 4, got: {test_patch.ndim}')
        
        # Get patch size (assuming cubic patches)
        if test_patch.ndim == 3:
            self.patch_size = test_patch.shape[0]  # d, h, w
            expected_elements = self.patch_size ** 3
        else:
            self.patch_size = test_patch.shape[1]  # c, d, h, w
            expected_elements = self.channels * (self.patch_size ** 3)
            
        actual_elements = test_patch.numel()
        assert actual_elements == expected_elements, \
            f'Expected {expected_elements} elements, got {actual_elements}. Shape: {test_patch.shape}'
        
        # Validate labels
        assert test_xyzconf.ndim == 2, f'xyzconf should be 2D, got: {test_xyzconf.ndim}'
        assert test_xyzconf.shape[1] == 4, f'xyzconf should have 4 columns, got: {test_xyzconf.shape[1]}'
        
        # Validate global coordinates
        assert test_globalcoords.shape == (3,) or test_globalcoords.numel() == 3, \
            f'global_coords wrong shape: {test_globalcoords.shape}'
        
        self.max_motors = test_xyzconf.shape[0]
        self.patch_dtype = test_patch.dtype
        
        print(f"Dataset stats:")
        print(f"  Channels: {self.channels}")
        print(f"  Patch size: {self.patch_size}^3")
        print(f"  Max motors per patch: {self.max_motors}")
        print(f"  Patch dtype: {self.patch_dtype}")
        print(f"  Patches per batch: {self.patches_per_batch}")

    def resample_batch(self):
        """Resample a new batch of random patches"""
        if len(self.patch_index) < self.patches_per_batch:
            # If we have fewer patches than requested, sample with replacement
            self.current_batch_indices = random.choices(range(len(self.patch_index)), k=self.patches_per_batch)
        else:
            # Sample without replacement
            self.current_batch_indices = random.sample(range(len(self.patch_index)), self.patches_per_batch)



    def __len__(self):
        return self.patches_per_batch

    def __getitem__(self, idx):
        """Returns a single patch and its corresponding metadata from the current batch"""
        if idx >= self.patches_per_batch:
            raise IndexError(f"Index {idx} out of range for batch size {self.patches_per_batch}")
        
        # Get the actual patch index from the current batch
        actual_idx = self.current_batch_indices[idx]
        patch_path = self.patch_index[actual_idx]
        
        # Load the patch file
        file_data = torch.load(patch_path)
        
        patch = file_data['patch']
        sparse_labels = file_data['labels'].to(torch.float32)
        global_coords = file_data['global_coords']
        
        # Ensure patch has channel dimension
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)  # Add channel dimension: (D,H,W) -> (1,D,H,W)


        
        
        if self.transform:
            #something is wrong with my dense label stuff 99% sure
            # spatial_dims = patch.shape[1:] # (D, H, W)
            
            dense_labels = torch.zeros(size=patch.shape, dtype=torch.float32)
            
            valid_motor = sparse_labels[sparse_labels[:,3] > 0]
            
            if valid_motor.numel() > 0:
                x,y,z = valid_motor[0,:3].to(torch.int32)#ignoring motors > 1 since im not gonna deal with that in the future anyways
                dense_labels[0, x,y,z] = 1.0
            
            monai_dict = {
                
                'image' : patch, #vector <1,2,3> -> [1,3], [c,d,h,w] -> 1 x 64 x 64 x 64
                'label' : dense_labels #1x64x64x64
                #coords x,y,z,  1x3
            }
            
            monai_dict = self.transform(monai_dict)
            patch = monai_dict['image']
            dense_labels = monai_dict['label']
            
            coords = torch.nonzero(dense_labels).to(torch.float32)#returns bool mask
            # print(coords.shape)
            if coords.numel() > 0:
                # print(f'original labels: {sparse_labels[0,:3]}')
                # print(f'transformed coords: {coords[0, 1:]}')
                # assert torch.allclose(sparse_labels[0,:3], coords[0, 1:])
                sparse_labels[0, :3] = coords[0, 1:]#since dense mask is c,d,h,w we need to slice out the channel dim
            #1x3
            
            #global coords should remain the same i think?
        
        #should handle valid mask in trainer script later on
        #then we can modify this to support dict based transforms
        # print(f'patch tomo dataset labels shape: {labels.shape}')
        return patch, sparse_labels, global_coords


import time

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
                print(f'Throughput tracker: {self.name}')
            print(f'Iterations/s: {iters_s:.2f}')
            print(f'MB/s: {mb_s:.2f}')
            print('-' * 30)
            self.running_mb = 0
            self.updates = 0
            self.last_update = current_time


if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    
    # Create dataset with patches_per_batch parameter
    dataset = PatchTomoDataset(
        tomo_dir_list=tomo_dir_list, 
        patches_per_batch=128*128,  # Control how many patches per batch
        transform=None
    )
    
    # DataLoader will handle batching normally
    dataloader = DataLoader(
        dataset, 
        batch_size=128,  # This will batch the patches_per_batch samples
        shuffle=True, 
        pin_memory=True,
        num_workers=2, 
        persistent_workers=True, 
        prefetch_factor=2
    )
    
    main_throughput_tracker = ThroughputTracker('patch_loading')
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Total available patches: {len(dataset.patch_index)}")
    print(f"Starting throughput test...")
    
    while True:
        for batch_idx, (patches, labels, global_coords, valid_mask) in enumerate(dataloader):
            # patches shape: [batch_size, channels, depth, height, width]
            # labels shape: [batch_size, max_motors, 4]
            # global_coords shape: [batch_size, 3]
            # valid_mask shape: [batch_size, max_motors]
            
            num_bytes = patches.numel() * patches.element_size()
            main_mb = num_bytes / (1024 * 1024)
            main_throughput_tracker.update(main_mb)
            
            # if batch_idx == 0:
            #     print(f"First batch shapes:")
            #     print(f"  Patches: {patches.shape}")
            #     print(f"  Labels: {labels.shape}")
            #     print(f"  Global coords: {global_coords.shape}")
            #     print(f"  Valid mask: {valid_mask.shape}")
        
        # Resample for next epoch
        dataset.resample_batch()
        print("Resampled patches for next epoch")