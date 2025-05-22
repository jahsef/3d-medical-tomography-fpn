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
        test_tomo_path = self.tomo_dir_list[0]
        test_tomo_path:Path
        test_patch:torch.Tensor
        test_metadata:torch.Tensor
        
        test_patch = None
        for patch_path in test_tomo_path.iterdir():
            # print(patch_path)
            test_dict = torch.load(patch_path)
            test_patch = test_dict['data']
            
            test_metadata = test_dict['metadata']
            test_globalcoords = test_metadata['global_coords']
            test_xyzconf = test_metadata['xyzconf']
            
            break
        
        if test_patch is None:
            print('the dataset isnt loading correctly bruh')
            raise Exception(f'list of dirs: {self.tomo_dir_list}')
        
        #d,h,w OR cdhw
        if test_patch.ndim == 3:
            self.channels = 1
        elif test_patch.ndim == 4:
            self.channels = test_patch.shape[0]
        else:
            raise Exception(f'test_patch ndim should not be other than 3 or 4: {test_patch.ndim}')
        
        self.patch_size = test_patch.shape[1]
        assert int(self.patch_size ** 3) == int(torch.prod(torch.as_tensor(test_patch.shape[:]))), f'expected patch size: {self.patch_size}, shape {test_patch.shape[:]}, currently only supports patch size same for each dim'
        
        assert test_xyzconf.ndim == 2, f'xyzconf dims wrong : {test_xyzconf.ndim}'
        assert test_xyzconf.shape[1] == 4, f'xyzconf labels shape is wrong: {test_xyzconf.shape[1]}'
        assert test_globalcoords.shape == (3,) or test_globalcoords.numel() == 3, f'coords wrong shape ig: {test_globalcoords.shape}'
        
        self.max_motors = test_xyzconf.shape[0]
        
        
        self.patch_dtype = test_patch.dtype
        # assert test_patch.dtype == test_xyzconf.dtype and test_xyzconf.dtype == test_globalcoords.dtype, f'dtype mismatch: patch,xyzconf,global: {test_patch.dtype}, {test_xyzconf.dtype}, {test_globalcoords.dtype} '
        
            
            
        
        
        
    def __len__(self):
        return len(self.tomo_dir_list)

    def __getitem__(self, idx):
        """Returns a batch of random patches and corresponding label metadata from a tensor file."""
        #currently random sample, stratified/tomograph aware
        # print(type(self.__len__()))
        # print(type(self.num_patches))
        tomo_rand_indices = np.random.choice(self.__len__(), size = self.num_patches)
        #(p,c,d,h,w)
        cdhw = [self.channels, self.patch_size, self.patch_size, self.patch_size]
        patches = torch.empty(size = (self.num_patches, *cdhw), dtype = self.patch_dtype)
        xyzconf = torch.empty(size = (self.num_patches,self.max_motors, 4), dtype = torch.float32)
        global_coords = torch.empty(size = (self.num_patches,3), dtype = torch.int32)
        
        for i, rand_index in enumerate(tomo_rand_indices):
            curr_patch:torch.Tensor
            rand_tomo_dir:Path
            
            
            rand_tomo_dir = self.tomo_dir_list[rand_index]
            
            patch_paths = [x for x in rand_tomo_dir.iterdir() if x.is_file()]
            
            num_rand_patches = len(patch_paths)

            
            
            rand_patch_index = np.random.choice(num_rand_patches)
            
            rand_patch_path = patch_paths[rand_patch_index]
            
            file = torch.load(rand_patch_path, mmap = self.mmap)
            curr_patch = file['data']
            
            metadata = file['metadata']
            curr_xyzconf = metadata['xyzconf']
            curr_global_coords = metadata['global_coords']
            
            if curr_patch.ndim == 3:
                #oh we unsqueezin it
                curr_patch.unsqueeze(0)
            
            if self.transform:
                print('trasnforming check our transforms work on 2d or 3d lol dont work on batches')
                curr_patch = self.transform(curr_patch)
            # print(curr_xyzconf.shape)
            patches[i] = curr_patch
            xyzconf[i] = curr_xyzconf
            global_coords[i] = curr_global_coords
            #TODO:add some assertions here
        
        
        # print(type(patches))
        return patches, xyzconf, global_coords
        
import time

class ThroughputTracker:
    def __init__(self,name:str = None, update_interval=5):
        self.running_mb = 0
        self.updates = 0
        self.last_update = time.perf_counter()
        self.update_interval = update_interval
        self.name = name

    def update(self, mb):
        self.running_mb += mb
        self.updates +=1
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
            

if __name__ == '__main__':
    from pathlib import Path
    import utils
    import pandas as pd
    from torch.utils.data import DataLoader
    import time
    
    master_tomo_path = Path.cwd() / 'normalized_pt_data/train'
    tomo_dir_list = [dir for dir in master_tomo_path.iterdir() if dir.is_dir()]
    
    dataset = PatchTomoDataset(tomo_dir_list = tomo_dir_list, num_patches=128, mmap=True, transform= None)
    
    # patches, patch_origin_metadata, labels = dataset.__getitem__(idx = 0)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory= True   , num_workers=0, persistent_workers= False, prefetch_factor= None)
    
    main_thruput_tracker = ThroughputTracker('patch')
    # extra_thruput_tracker = ThroughputTracker('extras')
    # print('poopy')
    
    while True:
        try:
            for patches, labels, global_coords in dataloader:
                num_bytes = patches.numel() * patches.element_size()
                main_mb = num_bytes / 1024 / 1024
                main_thruput_tracker.update(main_mb)
        except StopIteration:
            continue

