from pathlib import Path
import torch
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count
import torchvision.transforms.v2 as t


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

transform = t.Compose([
    # t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
])

def write_tomo_pt(args):
    src, dst = args  # Unpack args for multiprocessing
    print(f'Constructing tomo file array: {src.name}')
    
    # Filter only image files
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    
    # Sort naturally
    files = natsorted(files, key=lambda x: x.name)

    # Read all images
    imgs = [iio.imread(file, mode="L") for file in files]
    tomo_array = np.stack(imgs)

    # Convert to tensor and save
    tomo_tensor = torch.as_tensor(tomo_array, dtype = torch.float16)
    
    tomo_tensor = transform(tomo_tensor)
    
    output_path = dst / f'{src.name}.pt'
    print(f'Saving tomo: {src.name} -> {output_path}')
    torch.save(tomo_tensor, output_path)


def write_whole_directory(src: Path, dst: Path, max_processes=None):
    dst.mkdir(parents=True, exist_ok=True)

    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]

    args_list = [(tomo_dir, dst) for tomo_dir in tomo_dirs]

    # Default to using all CPUs
    if max_processes is None:
        max_processes = min(cpu_count(), 16)

    with Pool(processes=max_processes) as pool:
        pool.map(write_tomo_pt, args_list)


    
if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train')

    write_whole_directory(src=src_root, dst=dst_root, max_processes=8)
