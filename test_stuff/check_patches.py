from pathlib import Path
import torch
from tqdm import tqdm

path = Path('./data/processed/patch_pt_data')
folders = [dir for dir in  path.iterdir() if dir.is_dir()]
smallest_min = 10000
biggest_max = -10000
for folder in tqdm(folders):
    files = [file for file in folder.iterdir() if file.is_file()]

    for file in files:
        patch = torch.load(file)['patch']
        patch_min = patch.min()
        patch_max = patch.max()
        if torch.isnan(patch).any():
            print(f'WARNING NAN @ tomo: {folder}, patch: {file}')
        smallest_min = min(patch_min, smallest_min)
        biggest_max = max(patch_max, biggest_max)

print(smallest_min)
print(biggest_max)