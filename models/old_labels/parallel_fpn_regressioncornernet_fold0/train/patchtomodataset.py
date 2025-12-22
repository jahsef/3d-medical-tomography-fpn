from pathlib import Path
import warnings
import torch
from torch.utils.data import Dataset
import pandas as pd
import time



class PatchTomoDataset(Dataset):
    def __init__(self,
                 folds: list[int],
                 dataset_path: Path = Path('./data/processed/relabeled_patches/'),
                 labels_path: Path = Path('./data/original_data/RELABELED_DATA.csv'),
                 transform=None):
        self.dataset_path = Path(dataset_path)
        print(f'loading data from: {dataset_path}')
        print(f'loading labels from: {labels_path}')
        
        self.transform = transform

        if transform is not None:
            warnings.warn("transform is not yet implemented in this dataset")

        # Load fold info from labels CSV
        df = pd.read_csv(labels_path)
        if 'fold' in df.columns:
            tomo_folds = df.groupby('tomo_id')['fold'].first().to_dict()
        else:
            warnings.warn(f"No 'fold' column in {labels_path}, loading folds from RELABELED_DATA.csv")
            relabeled_df = pd.read_csv(Path('./data/original_data/RELABELED_DATA.csv'))
            tomo_folds = relabeled_df.groupby('tomo_id')['fold'].first().to_dict()
        
        # Collect all patch files from tomos matching requested folds
        start_time = time.perf_counter()
        self.patch_files = []
        for tomo_dir in self.dataset_path.iterdir():
            if not tomo_dir.is_dir():
                continue
            tomo_id = tomo_dir.name
            if tomo_id not in tomo_folds:
                continue
            if tomo_folds[tomo_id] not in folds:
                continue
            self.patch_files.extend(list(tomo_dir.glob('*.pt')))
        print(f'patch find time: {time.perf_counter() - time.perf_counter()} s')
    
    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        data = torch.load(self.patch_files[idx], weights_only=False)
        patch = data['patch'].unsqueeze(0).float()  # add channel dim
        gaussian = data['gaussian'].unsqueeze(0).float()  # add channel dim

        return patch, gaussian
