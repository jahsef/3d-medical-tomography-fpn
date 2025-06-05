from pathlib import Path
import torch
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count
import torchvision.transforms.v2 as t
import zarr
import os

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

transform = t.Compose([
    t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
])

def write_tomo_zarr(args):
    src, dst, chunk_size = args
    print(f'Processing tomogram: {src.name}')
    
    tomo_id = src.name
    
    # Load and process images
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    files = natsorted(files, key=lambda x: x.name)
    imgs = [iio.imread(file, mode="L") for file in files]
    tomo_array = np.stack(imgs)
    
    # Convert to tensor and normalize
    tomo_tensor = torch.as_tensor(tomo_array)
    tomo_tensor = transform(tomo_tensor)
    
    # Convert back to numpy for zarr storage
    tomo_np = tomo_tensor.numpy()
    
    # Create output zarr file
    zarr_path = dst / f'{tomo_id}.zarr'
    
    # Create zarr array with compression and chunking
    zarr_store = zarr.open(
        str(zarr_path),
        mode='w',
        shape=tomo_np.shape,
        dtype=tomo_np.dtype,
        chunks=chunk_size,
        codecs=[
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3, "shuffle": "bitshuffle"}}
        ]
    )
    
    # Write the tomogram data
    zarr_store[:] = tomo_np
    
    # Save metadata
    zarr_store.attrs['tomo_id'] = tomo_id
    zarr_store.attrs['original_shape'] = tomo_np.shape
    zarr_store.attrs['normalization'] = {
        'mean': 0.479915,
        'std': 0.224932
    }
    
    print(f'Finished processing: {tomo_id} - Shape: {tomo_np.shape}')

def write_whole_directory(
    src, 
    dst, 
    chunk_size=None,
    max_processes=None
):
    dst.mkdir(parents=True, exist_ok=True)
    
    # Set default chunk size if not provided
    if chunk_size is None:
        chunk_size = (64, 64, 64)
    
    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    args_list = [
        (tomo_dir, dst, chunk_size)
        for tomo_dir in tomo_dirs
    ]

    max_processes = max_processes or min(cpu_count(), 16)
    
    print(f"Starting processing of {len(tomo_dirs)} tomograms with {max_processes} workers...")
    
    with Pool(processes=max_processes) as pool:
        pool.map(write_tomo_zarr, args_list)

def prune_empty_dirs(master_tomo_dir: Path):
    print('Removing empty directories...')
    tomo_dirs = [x for x in master_tomo_dir.iterdir() if x.is_dir()]
    for dir in tomo_dirs:
        files = [x for x in dir.iterdir() if x.is_file() or x.is_dir()]
        if len(files) == 0:
            os.rmdir(dir)
    print('Finished removing empty directories')

if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_zarr_data\train')
    
    workers = 1
    chunk_size = (128, 128, 128)  # Chunk size for zarr compression and access
    
    write_whole_directory(
        src=src_root,
        dst=dst_root,
        chunk_size=chunk_size,
        max_processes=workers
    )
    
    prune_empty_dirs(dst_root)