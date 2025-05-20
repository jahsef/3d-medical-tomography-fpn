import torch
from pathlib import Path
import numpy as np

class PatchTomoDataset(torch.utils.data.Dataset):
    def __init__(self, tomo_dir_list: list[Path],  num_patches:int, mmap:bool, transform = None):
        """_summary_
        Args:
            list_tuple_path_labels (list[tuple]): list of tuple of path, labels
            
            num_patches (int): this must be lower than total number of tomos
            
            transform (type) WE NEED TO USE TORCH VIDEO TRANSFORMS
        """
        
        super().__init__()
        self.tomo_dir_list = tomo_dir_list
        self.num_patches = num_patches
        self.mmap = mmap
        self.transform = transform
        self._determine_patch_stats()
        
    
    def _determine_patch_stats(self):
        #save channels and patch_size
        #open first tomo dir open a random patch
        tomo_path = self.tomo_dir_list[0]
        tomo_path:Path
        test_patch:torch.Tensor
        for patch_path in tomo_path.iterdir():
            test_patch = torch.load(patch_path)
            break
        
        #c,d,h,w
        self.channels = test_patch.shape[0]
        assert torch.allclose(test_patch.shape), 'd,h,w not all close, currently only supports d,h,w = patchsize not seperate'
        
        self.patch_size = test_patch.shape[1]
        

    def __len__(self):
        return len(self.tomo_dir_list)

    def __getitem__(self, idx):
        """Returns a batch of random patches and corresponding label metadata from a tensor file."""
        #currently random sample, stratified/tomograph aware
        rand_indices = np.random.choice(self.__len__, size = (self.num_patches,))
        #(p,c,d,h,w)
        cdhw = [self.channels, self.patch_size, self.patch_size, self.patch_size]
        patches = torch.empty(size = (self.num_patches, *cdhw))
        
        for i, index in enumerate(rand_indices):
            patch_path = tomo_dir_list[index]
            patch = torch.load(patch_path, mmap = self.mmap)
            if self.transform:
                print('trasnforming check our transforms work on 2d or 3d lol dont work on batches')
                patch = self.transform(patch)
            patches[i] = patch
            
        return patches
        
        
        
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
    
    master_tomo_path = Path.cwd() / 'normalized_pt_data'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    
    dataset = PatchTomoDataset(tomo_dir_list = tomo_dir_list, num_patches=1, mmap=True, transform= None)
    
    # patches, patch_origin_metadata, labels = dataset.__getitem__(idx = 0)
    train_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory= False   , num_workers=2, persistent_workers= True, prefetch_factor= 1)
    
    thruput_tracker = ThroughputTracker()
    print('poopy')
    for patches, patch_origin_metadata, labels in train_loader._get_iterator():
        print('here')
        #data
        #metadata global coords, xyzconf
        pass

