import torch
from pathlib import Path


class FullTomoDataset(torch.utils.data.Dataset):
    
    def __init__(self, tomo_paths:Path):
        self.tomo_list = []
        