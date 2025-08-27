import torch
from pathlib import Path
import numpy as np
import random
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch.nn.functional as F




def downsample_with_max_weighting(dense_label, downsampling_factor):
    """Vectorized downsample by weighting around max positions"""
    
    # Simple maxpool to find peak locations
    max_pooled = F.max_pool3d(dense_label.float().unsqueeze(0), 
                             kernel_size=downsampling_factor, 
                             stride=downsampling_factor, 
                             return_indices=True)
    
    max_values, max_indices = max_pooled
    
    # Create a smoothed version using avg pool for non-zero regions
    avg_pooled = F.avg_pool3d(dense_label.float().unsqueeze(0), 
                             kernel_size=downsampling_factor, 
                             stride=downsampling_factor)
    
    # Blend: use max where there are peaks, avg elsewhere
    # You can adjust the blend ratio here
    blend_ratio = 0.7  # 70% max, 30% avg
    result = blend_ratio * max_values + (1 - blend_ratio) * avg_pooled
    
    return result.squeeze(0).half()


class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, angstrom_blob_sigma:float, sigma_scale:float, downsampling_factor:int, 
                 patch_index_path: Path = Path.cwd() / '_patch_index.csv',
                 dataset_path: Path = Path.cwd() / 'data/processed/patch_pt_data',
                 labels_path: Path = Path.cwd() / 'data/original_data/train_labels.csv',
                 transform=None, tomo_id_list: list[str] = None):
        """
        Args:
            patch_index_path (Path): Path to patch index CSV
            dataset_path (Path): Path to processed patch data directory
            labels_path (Path): Path to train labels CSV
            transform: optional transforms to apply to patches
            tomo_id_list: list of tomograms(SHOULD BE TOMO IDS ONLY) we want to be in our patch based dataset 
        """
        super().__init__()
        
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.labels_csv = pd.read_csv(self.labels_path)
        
        self.index_df = pd.read_csv(patch_index_path)
        self.angstrom_blob_sigma = angstrom_blob_sigma
        
        if tomo_id_list is not None:
            bool_mask = self.index_df["tomo_id"].isin(tomo_id_list)
            self.index_df = self.index_df[bool_mask]
    
            
        
        class_labels = np.array(self.index_df['has_motor'])
        positive_indices = np.where(class_labels == 1)[0]  # Assuming 1 = positive class
        negative_indices = np.where(class_labels == 0)[0]  # Assuming 0 = negative class
        
        print(f"Dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        
        self.transform = transform
        self.angstrom_blob_sigma = angstrom_blob_sigma
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

        bool_mask = self.labels_csv['tomo_id'] == row['tomo_id']
        voxel_spacing = self.labels_csv['Voxel spacing'][bool_mask].iloc[0]

        d_i, h_i, w_i = patch.shape[1:]

        if sparse_label[0,3] == 1:  # if valid motor
            label_d, label_h, label_w = sparse_label[0, :3]  # Keep original coordinates
            
            d, h, w = torch.arange(d_i), torch.arange(h_i), torch.arange(w_i)
            grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
            
            # Convert angstrom blob sigma to pixel space
            blob_sigma_pixels = self.angstrom_blob_sigma / voxel_spacing
            
            # Use spherical gaussian for blob
            pre_weighted_label = torch.exp(-((grid_d-label_d)**2 + 
                                            (grid_h-label_h)**2 + 
                                            (grid_w-label_w)**2)/(2*blob_sigma_pixels**2)).to(torch.float16).unsqueeze(0)
            
            # Center and weight calculations at full res
            center_d, center_h, center_w = [(x-1)/2 for x in [d_i, h_i, w_i]]
            
            # Spherical weighting gaussian too
            weight_sigma_pixels = self.sigma_scale * min([d_i, h_i, w_i])
            
            gaussian_weight = torch.exp(-((grid_d-center_d)**2 + 
                                        (grid_h-center_h)**2 + 
                                        (grid_w-center_w)**2)/(2*weight_sigma_pixels**2)).to(torch.float16).unsqueeze(0)
            
            dense_label = pre_weighted_label * gaussian_weight
        else:
            dense_label = torch.zeros(size=(1, d_i, h_i, w_i), dtype=torch.float16)

        if self.transform:
            dict = {'patch': patch, 'label': dense_label}
            dict = self.transform(dict)
            patch = dict['patch']   
            dense_label = dict['label']

        # target_size = (d_i//self.downsampling_factor, h_i//self.downsampling_factor, w_i//self.downsampling_factor)
        # downsampled_label = F.interpolate(dense_label.unsqueeze(0), size=target_size, mode='area').squeeze(0)

        downsampled_label = downsample_with_max_weighting(dense_label, self.downsampling_factor)

        
        
        # downsampled_label = F.avg_pool3d(dense_label.float().unsqueeze(0), 
        #                                 kernel_size=self.downsampling_factor, 
        #                                 stride=self.downsampling_factor).squeeze(0).half()
        
        # Visualization at the end - only for non-empty labels
        if sparse_label[0,3] == 1 and False:  # Change True to False to disable plotting
            # Get all variables at the top
            tomo_origin = patch_path.parent.name
            original_label_coords = (label_d, label_h, label_w)
            global_coords = patch_dict['global_coords']
            motor_weight_value = gaussian_weight[0, int(label_d), int(label_h), int(label_w)]
            
            dense_label_max = dense_label.max()
            downsampled_label_max = downsampled_label.max()
            
            # Find actual coordinates after transforms
            actual_max_idx = torch.argmax(dense_label)
            _, actual_motor_d, actual_motor_h, actual_motor_w = torch.unravel_index(actual_max_idx, dense_label.shape)
            actual_motor_z = actual_motor_d.item()
            
            # Find actual peak in downsampled version
            downsampled_max_idx = torch.argmax(downsampled_label)
            _, actual_peak_d, actual_peak_h, actual_peak_w = torch.unravel_index(downsampled_max_idx, downsampled_label.shape)
            
            downsample_slice_max = downsampled_label[0, actual_peak_d, ...].max()
            color_scale_max = max(dense_label_max, pre_weighted_label.max(), gaussian_weight.max(), downsampled_label_max)
            
            # Print info
            print(f'Tomo: {tomo_origin} | local Motor coords:({actual_motor_d}, {actual_motor_h}, {actual_motor_w}) | global motor coords: {actual_motor_d+global_coords[0]}, {actual_motor_h+global_coords[1]}, {actual_motor_w+global_coords[2]}')
            print(f"Dense max: {dense_label_max:.3f} | Downsampled max: {downsampled_label_max:.3f} | Slice max: {downsample_slice_max:.3f}")
            print(f"Downsampled peak at slice {actual_peak_d}, full-res peak at slice {actual_motor_z}")

            # Plotting - use same depth for all full-res plots
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 5, 1)
            plt.imshow(patch[0, actual_motor_z].cpu(), cmap='gray')
            plt.title(f'Raw patch at depth {actual_motor_z}')
            plt.plot(actual_motor_w, actual_motor_h, 'ro', markersize=8)
            
            plt.subplot(1, 5, 2)
            plt.imshow(dense_label[0, actual_motor_z].cpu(), cmap='viridis', vmin=0, vmax=color_scale_max)
            plt.title(f'Full Res Heatmap at depth {actual_motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 5, 3)
            plt.imshow(pre_weighted_label[0, actual_motor_z, ...].numpy(), cmap='viridis', vmin=0, vmax=color_scale_max)
            plt.title(f'Unweighted Heatmap at depth {actual_motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 5, 4)
            plt.imshow(gaussian_weight[0, actual_motor_z, ...].numpy(), cmap='viridis', vmin=0, vmax=color_scale_max)
            plt.title(f'Gaussian Weighting at depth {actual_motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 5, 5)
            plt.imshow(downsampled_label[0, actual_peak_d].cpu(), cmap='viridis', vmin=0, vmax=color_scale_max)
            plt.title(f'Downsampled Heatmap at depth {actual_peak_d}')
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
    import monai
    import monai.transforms as transforms
    
    train_transform = transforms.Compose([
        # Stronger intensity augmentations
        transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=0.5),
        transforms.RandShiftIntensityd(keys='patch', offsets=0.5, safe=True, prob=1),
        transforms.RandAdjustContrastd(keys="patch", gamma=(0.5, 1.5), prob=1),
        transforms.RandScaleIntensityd(keys="patch", factors=0.1, prob=1),
        
        transforms.RandCoarseDropoutd(#basically just noise at these settings
            keys="patch", 
            holes=8, 
            spatial_size=(10, 20, 20), 
            prob = 1
        ),
        
        # Spatial augmentations (keeping these efficient)
        # transforms.RandRotate90d(keys=["patch", "label"], prob=0.5, spatial_axes=[1,2]),
        # transforms.RandFlipd(keys=['patch', 'label'], prob=0.5, spatial_axis=[0,1,2]),
        
        # transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=[168,304,304], mode='reflect'),
        # transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=[160,288,288], random_center=True),

        # transforms.RandRotated(keys=['patch', 'label'], range_x=0.33, range_y=0.33, range_z=0.33, prob=0.15, mode=['trilinear', 'nearest']),
        # transforms.RandZoomd(keys=['patch', 'label'], min_zoom = 0.9, max_zoom = 1.1, prob = 0.15, mode = ['trilinear', 'nearest']),
        


    ])
    # train_transform = None
    
    master_tomo_path = Path.cwd() / 'patch_pt_data'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    
    # Create dataset with patches_per_batch parameter
    dataset = PatchTomoDataset(
        angstrom_blob_sigma=200,
        sigma_scale=1.5,
        downsampling_factor=16,
        patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        tomo_id_list=['tomo_d7475d'],
        transform = train_transform,
    )
    
    # DataLoader will handle batching normally
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # This will batch the patches_per_batch samples
        shuffle=False, 
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