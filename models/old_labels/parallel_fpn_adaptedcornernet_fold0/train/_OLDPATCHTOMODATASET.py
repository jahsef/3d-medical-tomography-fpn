import torch
from pathlib import Path
import numpy as np
import random
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging
import time
from model_defs.nnblock import check_tensor




def generate_gaussian_label(grid_d, grid_h, grid_w, motor_coords, blob_sigma_pixels, center_coords, target_device='cpu'):
    """
    Generate gaussian blob and apply weighting on target device with no intermediate transfers.
    Assumes input grids are already on target_device.
    
    Returns:
        torch.Tensor: Weighted gaussian label on target_device
    """
    label_d, label_h, label_w = motor_coords
    
    # Gaussian blob computation on target device
    gaussian_blob = torch.exp(-((grid_d-label_d)**2 + 
                               (grid_h-label_h)**2 + 
                               (grid_w-label_w)**2)/(2*blob_sigma_pixels**2)).to(torch.float16).unsqueeze(0)

    
    # Apply weighting and return result on target device
    return gaussian_blob 


def downsample(dense_label, downsampling_factor, target_device=None):
    """
    Vectorized downsample using max pooling on target device with no transfers.
    
    Args:
        target_device: If None, uses dense_label's current device
    
    Returns:
        torch.Tensor: Downsampled tensor on same device as input
    """
    if target_device is None:
        target_device = dense_label.device
    
    # Ensure tensor is on target device
    if dense_label.device != target_device:
        dense_label = dense_label.to(target_device)
    
    # Max pooling on target device
    max_pooled = F.max_pool3d(dense_label.unsqueeze(0), 
                             kernel_size=downsampling_factor, 
                             stride=downsampling_factor, 
                             return_indices=False)
    
    return max_pooled.squeeze(0)


class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, angstrom_blob_sigma:float, downsampling_factor:int, 
                 patch_index_path: Path = Path.cwd() / '_patch_index.csv',
                 dataset_path: Path = Path.cwd() / 'data/processed/patch_pt_data',
                 labels_path: Path = Path.cwd() / 'data/original_data/train_labels.csv',
                 transform=None, tomo_id_list: list[str] = None, debug_visualization: bool = False,
                 verbose_profiling: bool = False, processing_device: str = None):
        """
        Args:
            angstrom_blob_sigma (float) : gaussian blob sigma in angstroms (will be divided using GT angstrom per voxel data to get voxel space sigma)
            patch_index_path (Path): Path to patch index CSV
            dataset_path (Path): Path to processed patch data directory
            labels_path (Path): Path to train labels CSV
            transform: optional transforms to apply to patches
            tomo_id_list: list of tomograms(SHOULD BE TOMO IDS ONLY) we want to be in our patch based dataset
            debug_visualization: Enable matplotlib visualization of patches and labels
            verbose_profiling: Enable detailed debug logging (default: False, only timing logs)
            processing_device: Device for processing ('cuda', 'cpu', or None for auto). If None, uses 'cuda' if available, else 'cpu'
        """
        super().__init__()
        logging.debug("[DATASET_INIT] Starting PatchTomoDataset initialization")
        
        init_start = time.perf_counter()
        
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.debug_visualization = debug_visualization
        self.verbose_profiling = verbose_profiling
        
        # Determine processing device
        if processing_device is None:
            if torch.cuda.is_available():
                self.processing_device = torch.device('cuda')
            else:
                self.processing_device = torch.device('cpu')
        else:
            self.processing_device = torch.device(processing_device)
        
        logging.debug(f"[DATASET_INIT] Processing device set to: {self.processing_device}")
        if self.verbose_profiling:
            logging.debug(f"[DEVICE_DEBUG] CUDA available: {torch.cuda.is_available()}, Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
        
        logging.debug(f"[DATASET_INIT] Loading labels CSV from: {self.labels_path}")
        csv_start = time.perf_counter()
        self.labels_csv = pd.read_csv(self.labels_path)
        csv_elapsed = (time.perf_counter() - csv_start) * 1000
        logging.debug(f"[PERF_TIMING] Labels CSV loaded in {csv_elapsed:.1f}ms")
        
        logging.debug(f"[DATASET_INIT] Loading patch index CSV from: {patch_index_path}")
        index_start = time.perf_counter()
        self.index_df = pd.read_csv(patch_index_path)
        index_elapsed = (time.perf_counter() - index_start) * 1000
        logging.debug(f"[PERF_TIMING] Patch index loaded in {index_elapsed:.1f}ms ({len(self.index_df)} total patches)")
        self.angstrom_blob_sigma = angstrom_blob_sigma
        
        if tomo_id_list is not None:
            logging.debug(f"[DATASET_INIT] Filtering dataset to {len(tomo_id_list)} specified tomograms")
            filter_start = time.perf_counter()
            bool_mask = self.index_df["tomo_id"].isin(tomo_id_list)
            original_count = len(self.index_df)
            self.index_df = self.index_df[bool_mask]
            filter_elapsed = (time.perf_counter() - filter_start) * 1000
            logging.debug(f"[PERF_TIMING] Dataset filtering took {filter_elapsed:.1f}ms ({original_count} -> {len(self.index_df)} patches)")
        
        class_labels = np.array(self.index_df['has_motor'])
        positive_indices = np.where(class_labels == 1)[0]
        negative_indices = np.where(class_labels == 0)[0]
        
        logging.debug(f"[DATASET_INIT] Final dataset statistics: {len(positive_indices)} positive, {len(negative_indices)} negative patches")
        print(f"Dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        self.transform = transform
        self.angstrom_blob_sigma = angstrom_blob_sigma
        self.downsampling_factor = downsampling_factor
        
        init_elapsed = (time.perf_counter() - init_start) * 1000
        logging.debug(f"[PERF_TIMING] Total dataset initialization took {init_elapsed:.1f}ms")
        
        # self._resample_batch()
        
        
        #define gaussian blobs?

        #blob_sigma std of 2 is fine for my usecase maybe 1.5-3 would work idk
        
        
    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        """
        Returns a single patch and its corresponding metadata.
        
        IMPORTANT: Returned tensors (patch, downsampled_label) will be on self.processing_device,
        not necessarily CPU. The DataLoader will handle final device placement if needed.
        
        Returns:
            patch: Tensor on self.processing_device
            downsampled_label: Tensor on self.processing_device  
            global_coords: Tensor (unchanged, stays on original device)
        """
        #we want to just index a list or something similar that we get from the random indices
        
        getitem_start = time.perf_counter()
        logging.debug(f"[GETITEM_START] Loading idx {idx}")
        
        row = self.index_df.iloc[idx]
        patch_path = self.dataset_path / row['tomo_id'] / row['patch_id'] #patch_id already has .pt
        assert type(patch_path) is pathlib.WindowsPath, type(patch_path)
        
        load_start = time.perf_counter()
        patch_dict = torch.load(patch_path)
        load_elapsed = (time.perf_counter() - load_start) * 1000
        logging.debug(f"[PERF_TIMING] Patch loading took {load_elapsed:.1f}ms for {patch_path.name}")
        
        patch = patch_dict['patch']
        sparse_label = patch_dict['labels']
        global_coords = patch_dict['global_coords']
        
        # Move patch to processing device early
        if patch.device != self.processing_device:
            patch = patch.to(self.processing_device)
            if self.verbose_profiling:
                logging.debug(f"[DEVICE_TRANSFER] Moved patch from {patch.device} to {self.processing_device}")
        
        if self.verbose_profiling:
            logging.debug(f"[DATA_EXTRACTION] Patch shape: {patch.shape}, Has motor: {sparse_label[0,3] == 1}")
        
        if patch.ndim == 3:
            # print('unsqueezing')
            patch = patch.unsqueeze(0)
            if self.verbose_profiling:
                logging.debug(f"[DATA_TRANSFORM] Unsqueezed patch to shape: {patch.shape}")

        voxel_start = time.perf_counter()
        bool_mask = self.labels_csv['tomo_id'] == row['tomo_id']
        voxel_spacing = self.labels_csv['Voxel spacing'][bool_mask].iloc[0]
        voxel_elapsed = (time.perf_counter() - voxel_start) * 1000
        if self.verbose_profiling:
            logging.debug(f"[VOXEL_LOOKUP] Voxel spacing lookup took {voxel_elapsed:.1f}ms, spacing: {voxel_spacing}")

        d_i, h_i, w_i = patch.shape[1:]
        if self.verbose_profiling:
            logging.debug(f"[PATCH_DIMS] Patch dimensions: {d_i}x{h_i}x{w_i}")

        if sparse_label[0,3] == 1:  # if valid motor
            label_gen_start = time.perf_counter()
            logging.debug(f"[LABEL_GENERATION] Processing positive sample with motor")
            
            label_d, label_h, label_w = sparse_label[0, :3]  # Keep original coordinates
            if self.verbose_profiling:
                logging.debug(f"[MOTOR_COORDS] Motor location: ({label_d}, {label_h}, {label_w})")
            
            grid_start = time.perf_counter()
            # Create grids directly on processing device to avoid transfers
            d = torch.arange(d_i, device=self.processing_device)
            h = torch.arange(h_i, device=self.processing_device) 
            w = torch.arange(w_i, device=self.processing_device)
            grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
            
            grid_elapsed = (time.perf_counter() - grid_start) * 1000
            if self.verbose_profiling:
                logging.debug(f"[PERF_TIMING] Grid generation on {self.processing_device} took {grid_elapsed:.1f}ms")
            
            # Convert angstrom blob sigma to pixel space
            blob_sigma_pixels = self.angstrom_blob_sigma / voxel_spacing
            if self.verbose_profiling:
                logging.debug(f"[BLOB_PARAMS] Blob sigma: {self.angstrom_blob_sigma}Å -> {blob_sigma_pixels:.2f} pixels")
            
            # Center and weight calculations at full res
            center_d, center_h, center_w = [(x-1)/2 for x in [d_i, h_i, w_i]]
            if self.verbose_profiling:
                logging.debug(f"[WEIGHT_PARAMS] Patch center: ({center_d:.1f}, {center_h:.1f}, {center_w:.1f})")
            

            # Combined gaussian blob generation and weighting in single GPU operation
            gaussian_start = time.perf_counter()
            dense_label = generate_gaussian_label(grid_d, grid_h, grid_w,
                                                 (label_d, label_h, label_w),
                                                 blob_sigma_pixels,
                                                 (center_d, center_h, center_w),
                                                 target_device=self.processing_device)
            check_tensor("dense_label after Gaussian generation", dense_label)
            gaussian_elapsed = (time.perf_counter() - gaussian_start) * 1000
            logging.debug(f"[PERF_TIMING] Gaussian label generation took {gaussian_elapsed:.1f}ms")
            
            label_gen_elapsed = (time.perf_counter() - label_gen_start) * 1000
            logging.debug(f"[PERF_TIMING] Total label generation took {label_gen_elapsed:.1f}ms")
        else:
            logging.debug(f"[LABEL_GENERATION] Processing negative sample (no motor)")
            dense_label = torch.zeros(size=(1, d_i, h_i, w_i), dtype=torch.float16, device=self.processing_device)
        check_tensor("dense_label zeros for negative sample", dense_label)

        if self.transform:
            transform_start = time.perf_counter()
            logging.debug(f"[TRANSFORM] Applying MONAI transforms (requires CPU)")
            
            # Move tensors to CPU for MONAI transforms
            transfer_to_cpu_start = time.perf_counter()
            patch_cpu = patch.cpu() if patch.device != torch.device('cpu') else patch
            dense_label_cpu = dense_label.cpu() if dense_label.device != torch.device('cpu') else dense_label
            transfer_to_cpu_elapsed = (time.perf_counter() - transfer_to_cpu_start) * 1000
            
            if self.verbose_profiling:
                logging.debug(f"[PERF_TIMING] GPU→CPU transfer for transforms took {transfer_to_cpu_elapsed:.1f}ms")
            
            # Apply MONAI transforms on CPU
            transform_compute_start = time.perf_counter()
            dict = {'patch': patch_cpu, 'label': dense_label_cpu}
            dict = self.transform(dict)
            patch_transformed = dict['patch']
            dense_label_transformed = dict['label']
            transform_compute_elapsed = (time.perf_counter() - transform_compute_start) * 1000
            
            # Move back to processing device
            transfer_to_device_start = time.perf_counter()
            patch = patch_transformed.to(self.processing_device) if self.processing_device != torch.device('cpu') else patch_transformed
            dense_label = dense_label_transformed.to(self.processing_device) if self.processing_device != torch.device('cpu') else dense_label_transformed
            transfer_to_device_elapsed = (time.perf_counter() - transfer_to_device_start) * 1000
            
            transform_elapsed = (time.perf_counter() - transform_start) * 1000
            logging.debug(f"[PERF_TIMING] Total transforms took {transform_elapsed:.1f}ms (compute: {transform_compute_elapsed:.1f}ms, transfers: {transfer_to_cpu_elapsed + transfer_to_device_elapsed:.1f}ms)")
            
            if self.verbose_profiling:
                logging.debug(f"[PERF_TIMING] CPU→GPU transfer after transforms took {transfer_to_device_elapsed:.1f}ms")
                logging.debug(f"[TRANSFORM_RESULT] Post-transform patch shape: {patch.shape}, label shape: {dense_label.shape}")

        # target_size = (d_i//self.downsampling_factor, h_i//self.downsampling_factor, w_i//self.downsampling_factor)
        # downsampled_label = F.interpolate(dense_label.unsqueeze(0), size=target_size, mode='area').squeeze(0)
        
        downsample_start = time.perf_counter()
        logging.debug(f"[DOWNSAMPLING] Downsampling by factor {self.downsampling_factor}, device: {self.processing_device}")
        downsampled_label = downsample(dense_label, self.downsampling_factor, target_device=self.processing_device)
        check_tensor("downsampled_label after downsample", downsampled_label)
        downsample_elapsed = (time.perf_counter() - downsample_start) * 1000
        logging.debug(f"[PERF_TIMING] Downsampling took {downsample_elapsed:.1f}ms")
        if self.verbose_profiling:
            logging.debug(f"[DOWNSAMPLE_RESULT] Final label shape: {downsampled_label.shape}")

        
        
        # downsampled_label = F.avg_pool3d(dense_label.float().unsqueeze(0), 
        #                                 kernel_size=self.downsampling_factor, 
        #                                 stride=self.downsampling_factor).squeeze(0).half()
        
        # Visualization at the end - only for non-empty labels
        if sparse_label[0,3] == 1 and self.debug_visualization:
            # Get all variables at the top
            tomo_origin = patch_path.parent.name
            original_label_coords = (label_d, label_h, label_w)
            global_coords = patch_dict['global_coords']
            
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
            color_scale_max = max(dense_label_max, downsampled_label_max)
            
            # Print info
            print(f'Tomo: {tomo_origin} | local Motor coords:({actual_motor_d}, {actual_motor_h}, {actual_motor_w}) | global motor coords: {actual_motor_d+global_coords[0]}, {actual_motor_h+global_coords[1]}, {actual_motor_w+global_coords[2]}')
            print(f"Dense max: {dense_label_max:.3f} | Downsampled max: {downsampled_label_max:.3f} | Slice max: {downsample_slice_max:.3f}")
            print(f"Downsampled peak at slice {actual_peak_d}, full-res peak at slice {actual_motor_z}")

            # Plotting - simplified to 3 subplots
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(patch[0, actual_motor_z].cpu() if patch.device != torch.device('cpu') else patch[0, actual_motor_z], cmap='gray')
            plt.title(f'Raw patch at depth {actual_motor_z}')
            plt.plot(actual_motor_w, actual_motor_h, 'ro', markersize=8)
            
            plt.subplot(1, 3, 2)
            plt.imshow(dense_label[0, actual_motor_z].cpu() if dense_label.device != torch.device('cpu') else dense_label[0, actual_motor_z], cmap='viridis', vmin=0, vmax=color_scale_max)
            plt.title(f'Full Res Heatmap at depth {actual_motor_z}')
            plt.colorbar(shrink=0.25)

            plt.subplot(1, 3, 3)
            plt.imshow(downsampled_label[0, actual_peak_d].cpu() if downsampled_label.device != torch.device('cpu') else downsampled_label[0, actual_peak_d], cmap='viridis', vmin=0, vmax=color_scale_max)
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
        getitem_elapsed = (time.perf_counter() - getitem_start) * 1000
        logging.debug(f"[PERF_TIMING] Total __getitem__ took {getitem_elapsed:.1f}ms for idx {idx}\n")
        
        check_tensor("final patch before return", patch)
        check_tensor("final downsampled_label before return", downsampled_label)
        
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
    # Configure debug logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    
    from pathlib import Path
    from torch.utils.data import DataLoader
    import monai
    import monai.transforms as transforms
    
    train_transform = transforms.Compose([
        # Stronger intensity augmentations
        # transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=0.5),
        # transforms.RandShiftIntensityd(keys='patch', offsets=0.5, safe=True, prob=1),
        # transforms.RandAdjustContrastd(keys="patch", gamma=(0.5, 1.5), prob=1),
        # transforms.RandScaleIntensityd(keys="patch", factors=0.1, prob=1),
        
        # transforms.RandCoarseDropoutd(#basically just noise at these settings
        #     keys="patch", 
        #     holes=8, 
        #     spatial_size=(10, 20, 20), 
        #     prob = 1
        # ),
        
        # Spatial augmentations (keeping these efficient)
        # transforms.RandRotate90d(keys=["patch", "label"], prob=0.5, spatial_axes=[1,2]),
        # transforms.RandFlipd(keys=['patch', 'label'], prob=0.5, spatial_axis=[0,1,2]),
        
        # transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=[168,304,304], mode='reflect'),
        # transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=[160,288,288], random_center=True),

        # transforms.RandRotated(keys=['patch', 'label'], range_x=0.33, range_y=0.33, range_z=0.33, prob=0.15, mode=['trilinear', 'nearest']),
        # transforms.RandZoomd(keys=['patch', 'label'], min_zoom = 0.9, max_zoom = 1.1, prob = 0.15, mode = ['trilinear', 'nearest']),
        


    ])
    train_transform = None
    
    master_tomo_path = Path.cwd() / 'data/processed/patch_pt_data'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    
    # Create dataset with patches_per_batch parameter
    dataset = PatchTomoDataset(
        angstrom_blob_sigma=200,
        downsampling_factor=16,
        # patch_index_path=Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\_patch_index.csv'),
        # tomo_id_list=['tomo_d7475d'],
        transform = train_transform,
        processing_device=torch.device('cpu'),
        debug_visualization=True,
        verbose_profiling=True
    )
    
    # DataLoader will handle batching normally
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # This will batch the patches_per_batch samples
        shuffle=False, 
        pin_memory=False,
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