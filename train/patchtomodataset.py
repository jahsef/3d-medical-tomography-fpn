import torch
from pathlib import Path

class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, list_tuple_path_labels: list[tuple],  num_patches:int, patch_size:int, mmap:bool, transform = None):
        """_summary_
        Args:
            list_tuple_path_labels (list[tuple]): list of tuple of path, labels
            
            transform (type) doesnt work with labels yet lol
        """
        
        super().__init__()
        self.list_tuple_path_labels = list_tuple_path_labels
        self.transform = transform
        self.num_patches = num_patches
        self.mmap = mmap
        self.patch_size = patch_size
        

    def __len__(self):
        return len(self.list_tuple_path_labels)

    def __getitem__(self, idx):
        """Returns a batch of random patches and corresponding label metadata from a tensor file."""
        path, labels = self.list_tuple_path_labels[idx]
        
        # Validate ID match
        assert path.stem == labels['tomo_id'], \
            f"ID mismatch: tomo={path.stem}, label={labels['tomo_id']}"

        # Load tensor with memory mapping
        tensor = torch.load(path, mmap=self.mmap)

        # Add channel dimension if missing
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # shape becomes [1, D, H, W]

        # Validate shape match
        assert tensor.shape[1:] == labels['shape'], \
            f"Shape mismatch in {path.stem}: got {tensor.shape[1:]}, expected {labels['shape']}"

        # Get tensor dimensions
        d, h, w = tensor.shape[1:]

        # Generate random patch origins
        rand_d = torch.randint(low=0, high=d - self.patch_size, size=(self.num_patches, 1))
        rand_h = torch.randint(low=0, high=h - self.patch_size, size=(self.num_patches, 1))
        rand_w = torch.randint(low=0, high=w - self.patch_size, size=(self.num_patches, 1))
        rand_dhw = torch.cat([rand_d, rand_h, rand_w], dim=1)

        # Extract patches using slice-based indexing
        patches = []
        for i in range(self.num_patches):
            d_start = rand_dhw[i, 0]
            h_start = rand_dhw[i, 1]
            w_start = rand_dhw[i, 2]
            patch = tensor[:, d_start:d_start + self.patch_size,
                        h_start:h_start + self.patch_size,
                        w_start:w_start + self.patch_size]
            patches.append(patch)

        # Stack patches into a single tensor
        patches = torch.stack(patches, dim=0)  # Shape: (num_patches, C, patch_size, patch_size, patch_size)
        del tensor  # Free memory early

        # Apply transform if provided
        if self.transform:
            patches = self.transform(patches)

        # Compute patch labels (unchanged from original)
        patch_origins = rand_dhw  # (num_patches, 3)
        patch_ends = patch_origins + self.patch_size  # (num_patches, 3)
        patch_origins = patch_origins.unsqueeze(1)  # (num_patches, 1, 3)
        patch_ends = patch_ends.unsqueeze(1)        # (num_patches, 1, 3)

        xyzconf = labels['xyzconf']
        label_coords = xyzconf[:, :3].unsqueeze(0)  # (1, max_motors, 3)

        # Check if each label is inside each patch
        is_inside = (label_coords >= patch_origins) & (label_coords < patch_ends)
        is_inside = is_inside.all(dim=-1)  # (num_patches, max_motors)

        # Expand mask and apply to labels
        mask_expanded = is_inside.unsqueeze(-1)  # (num_patches, max_motors, 1)
        
        final_labels_per_patch = torch.where(
            mask_expanded,
            xyzconf.unsqueeze(0).expand(self.num_patches, -1, -1),
            torch.zeros_like(xyzconf.unsqueeze(0).expand(self.num_patches, -1, -1))
        )

        return patches, rand_dhw, final_labels_per_patch
        
import time


class ThroughputTracker:
    def __init__(self, update_interval=16):
        self.running_mb = 0
        self.last_update = time.perf_counter()
        self.update_interval = update_interval

    def update(self, mb):
        self.running_mb += mb
        current_time = time.perf_counter()
        
        if current_time - self.last_update >= self.update_interval:
            mb_s = self.running_mb / (current_time - self.last_update)
            print(f'mb/s: {mb_s:.2f}')
            self.running_mb = 0
            self.last_update = current_time
            
if __name__ == '__main__':
    from pathlib import Path
    import utils
    import pandas as pd
    from torch.utils.data import DataLoader
    import time
    max_motors = 20
    tomo_list = utils.create_file_list(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train')
    csv = pd.read_csv(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train_labels.csv')
    #we load labels with some metadata for sanity check below
    #may not need those sanity checks but whatever
    #we only send relevant data to training loop
    tomo_csvrow_fart = utils.map_csvs_to_pt(csv, tomo_list, max_motors= max_motors)
    
    dataset = PatchTomoDataset(tomo_csvrow_fart, num_patches=1, patch_size=128, mmap=True, transform= None)
    
    # patches, patch_origin_metadata, labels = dataset.__getitem__(idx = 0)
    train_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory= False   , num_workers=16, persistent_workers= True, prefetch_factor= 3)
    
    thruput_tracker = ThroughputTracker()
    for patches, patch_origin_metadata, labels in train_loader._get_iterator():
        
        # print(patches.shape)
        # print(patch_origin_metadata.shape)
        # print(labels.shape)
        # bytes = patches.numel() * patches.dtype.itemsize
        # mb = bytes / 1024 / 1024
        # thruput_tracker.update(mb)
        print(labels.shape)
        # sum = torch.sum(labels)
        # if sum != 0:
        #     print(f'found correct label')
        #     print(labels)
        #     print(labels.shape)
        # print(f'array dtype: {patches.dtype}')

