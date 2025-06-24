import torch
from pathlib import Path
import numpy as np
import random
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, blob_sigma:float,sigma_scale:float,downsampling_factor:int,patch_index_path: Path, transform=None, tomo_id_list: list[str] = None):
        """
        Args:
            patch_index_path (Path): self explanatory
            patches_per_batch (int): number of patches to sample per batch
            transform: optional transforms to apply to patches
            blob_sigma: gaussian std dev for labels (idk is good i think)
            tomo_id_list: list of tomograms(SHOULD BE TOMO IDS ONLY) we want to be in our patch based dataset 
        """
        super().__init__()
        self.dataset_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\patch_pt_data')
        self.index_df = pd.read_csv(patch_index_path)
        
        if tomo_id_list is not None:
            bool_mask = self.index_df["tomo_id"].isin(tomo_id_list)
            self.index_df = self.index_df[bool_mask]
    
            
        
        class_labels = np.array(self.index_df['has_motor'])
        positive_indices = np.where(class_labels == 1)[0]  # Assuming 1 = positive class
        negative_indices = np.where(class_labels == 0)[0]  # Assuming 0 = negative class
        
        print(f"Dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        
        self.transform = transform
        self.blob_sigma = blob_sigma
        self.sigma_scale = sigma_scale
        self.downsampling_factor = downsampling_factor
        
        # self._resample_batch()
        
        
        #define gaussian blobs?

        #blob_sigma std of 2 is fine for my usecase maybe 1.5-3 would work idk
        
        
    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        """Returns a single patch and its corresponding metadata from the current batch"""
        #we want to just index a list or something similar that we get from the random indices
        
        row = self.index_df.iloc[idx]
        patch_path = self.dataset_path / row['tomo_id'] / row['patch_id'] #patch_id already has .pt
        assert type(patch_path) is pathlib.WindowsPath, type(patch_path)
        
        patch_dict = torch.load(patch_path)
        
        patch = patch_dict['patch']
        sparse_label = patch_dict['labels']
        global_coords = patch_dict['global_coords']
        
        if patch.ndim == 3:
            # print('unsqueezing')
            patch = patch.unsqueeze(0)
        
        # Generate dense label at FULL resolution first
        
        d_i, h_i, w_i = patch.shape[1:]
        min_dim = min([d_i, h_i, w_i])
        blob_sigma_d = self.blob_sigma * (min_dim / d_i)
        blob_sigma_h = self.blob_sigma * (min_dim / h_i) 
        blob_sigma_w = self.blob_sigma * (min_dim / w_i)

        if sparse_label[0,3] == 1:  # if valid motor
            label_d, label_h, label_w = sparse_label[0, :3]  # Keep original coordinates
            
            d, h, w = torch.arange(d_i), torch.arange(h_i), torch.arange(w_i)
            grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
            
            # Use anisotropic gaussian for blob
            pre_weighted_label = torch.exp(-((grid_d-label_d)**2/(2*blob_sigma_d**2) + 
                                            (grid_h-label_h)**2/(2*blob_sigma_h**2) + 
                                            (grid_w-label_w)**2/(2*blob_sigma_w**2))).to(torch.float16).unsqueeze(0)
            
            # Center and weight calculations at full res
            center_d, center_h, center_w = [(x-1)/2 for x in [d_i, h_i, w_i]]
            
            # Anisotropic weighting gaussian too
            weight_sigma_d = self.sigma_scale * d_i
            weight_sigma_h = self.sigma_scale * h_i
            weight_sigma_w = self.sigma_scale * w_i
            
            gaussian_weight = torch.exp(-((grid_d-center_d)**2/(2*weight_sigma_d**2) + 
                                        (grid_h-center_h)**2/(2*weight_sigma_h**2) + 
                                        (grid_w-center_w)**2/(2*weight_sigma_w**2))).to(torch.float16).unsqueeze(0)
            
            dense_label = pre_weighted_label * gaussian_weight
        else:
            dense_label = torch.zeros(size=(1, d_i, h_i, w_i), dtype=torch.float16)

        
        if self.transform:
            dict = {'patch': patch, 'label': dense_label}
            dict = self.transform(dict)
            patch = dict['patch']
            dense_label = dict['label']

        target_size = (d_i//self.downsampling_factor, h_i//self.downsampling_factor, w_i//self.downsampling_factor)
        downsampled_label = F.interpolate(dense_label.unsqueeze(0), size=target_size, mode='trilinear').squeeze(0)
        
        # Visualization at the end - only for non-empty labels
        if sparse_label[0,3] == 1 and False:  # Change True to False to disable plotting
            print(f"Label coords: {label_d}, {label_h}, {label_w}")
            print(f"Patch dims: {d_i}, {h_i}, {w_i}")
            print(f'downsample dims: {downsampled_label.shape}')
            print(f"Dense label max: {dense_label.max()}")
            print(f'Downsampled label max: {downsampled_label.max()}')
            
            print(f'max unweighted label value: {torch.max(pre_weighted_label)}')
            max_idx = torch.argmax(pre_weighted_label)
            
            c,d,h,w = torch.unravel_index(max_idx, dense_label.shape)#0,n,n,n?
            print(f'max_idx: {d,h,w}')

            motor_z = int(sparse_label[0, 0]) 
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 5, 1)
            plt.imshow(patch[0, motor_z].cpu(), cmap='gray')
            plt.title(f'Raw patch at depth {motor_z}')
            plt.plot(sparse_label[0, 2], sparse_label[0, 1], 'ro', markersize=8)
            
            # Find the global min/max across all your data
            global_min = 0
            global_max = max(dense_label.max(), pre_weighted_label.max(), gaussian_weight.max(), downsampled_label.max())

            plt.subplot(1, 5, 2)
            plt.imshow(dense_label[0, motor_z].cpu(), cmap='viridis', vmin=global_min, vmax=global_max)
            plt.title(f'Full Res Heatmap at depth {motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 5, 3)
            plt.imshow(pre_weighted_label[0, motor_z, ...].numpy(), cmap='viridis', vmin=global_min, vmax=global_max)
            plt.title(f'Unweighted Heatmap at depth {motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 5, 4)
            plt.imshow(gaussian_weight[0, motor_z, ...].numpy(), cmap='viridis', vmin=global_min, vmax=global_max)
            plt.title('Gaussian Weighting')
            plt.colorbar(shrink=0.25)
            
            # Downsampled heatmap in the 5th subplot
            downsampled_z = int(motor_z // self.downsampling_factor)
            plt.subplot(1, 5, 5)
            plt.imshow(downsampled_label[0, downsampled_z].cpu(), cmap='viridis', vmin=global_min, vmax=global_max)
            plt.title(f'Downsampled Heatmap at depth {downsampled_z}')
            plt.colorbar(shrink=0.25)
            
            plt.tight_layout()
            plt.show()
        
        # if idx == self.patches_per_batch-1:
        #     self._resample_batch()
        
        # print(f'loading from: {patch_path}')
        
        # print(f'dataset patch shape: {patch.shape}')
        # assert patch.shape == (1,128,240,240), f'{patch.shape},  {patch_path}'
        # print(f'dataset label shape: {downsampled_label.shape}')
        # print(downsampled_label.shape)
        # print('-----')
        return patch, downsampled_label, global_coords



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
        blob_sigma=69,
        sigma_scale=0.7,
        downsampling_factor=16,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        tomo_id_list=['tomo_d7475d'],
        transform = None,
    )
    
    # DataLoader will handle batching normally
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # This will batch the patches_per_batch samples
        shuffle=True, 
        pin_memory=True,
        num_workers=1, 
        persistent_workers=True, 
        prefetch_factor=1
    )
    
    main_throughput_tracker = ThroughputTracker('patch_loading')
    
    print(f"Dataset length: {len(dataset)}")
    # print(f"Total available patches: {len(dataset.patch_index)}")
    print(f"Starting throughput test...")
    
    while True:
        for batch_idx, (patches, labels, global_coords) in enumerate(dataloader):
            # patches shape: [batch_size, channels, depth, height, width]
            # labels shape: [batch_size, max_motors, 4]
            # global_coords shape: [batch_size, 3]
            # valid_mask shape: [batch_size, max_motors]
            
            positive_pixels = (labels > 0).sum().item()
            total_pixels = labels.numel()
            
            # print(f"Batch {batch_idx}: {positive_pixels}/{total_pixels} positive pixels ({positive_pixels/total_pixels*100:.2f}%)")
            # print(patches.shape)
            # print(labels.shape)
            
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
        # dataset.resample_batch()
        # print("Resampled patches for next epoch")