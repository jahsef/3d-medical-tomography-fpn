from pathlib import Path
from natsort import natsorted
import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms.v2 as t
from multiprocessing import Pool, cpu_count



IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

# Same transform
transform = t.Compose([
    t.ToDtype(torch.float16, scale=True),
    t.Normalize((0.479915,), (0.224932,))
])

from pathlib import Path
import numpy as np
import zarr
from numcodecs import Blosc

def write_tomo(args):
    src, dst = args
    tomo_id = src.name
    print(f'Processing tomogram: {tomo_id}', flush=True)

    # Load images
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    ]
    files = sorted(files, key=lambda x: x.name)
    imgs = [iio.imread(file, mode="L") for file in files]
    tomo_array = np.stack(imgs)

    # Apply transforms
    tomo_tensor = torch.as_tensor(tomo_array)
    tomo_tensor = transform(tomo_tensor)
    tomo_array = tomo_tensor.numpy()

    # Define Zarr path
    zarr_path = str(dst / f'{tomo_id}.zarr')


    chunk_size = (64, 64, 64)

    # Use zarr.open_group directly (high-level API)
    root = zarr.open_group(zarr_path, mode='w')
    
    array = root.create_dataset(
        "data",
        shape=tomo_array.shape,
        dtype=tomo_array.dtype,
        chunks=chunk_size,
        compressor='lz4',
        fill_value=0
    )
    
    array[:] = tomo_array[:]
    print(f'Finished processing: {tomo_id}', flush=True)
    return Path(zarr_path)


def write_whole_directory(src, dst, max_processes=None):
    dst.mkdir(parents=True, exist_ok=True)

    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    args_list = [(src_dir, dst) for src_dir in tomo_dirs]

    max_processes = max_processes or min(cpu_count(), 16)
    with Pool(processes=max_processes) as pool:
        pool.map(write_tomo, args_list)


if __name__ == '__main__':
    src_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
    dst_root = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\zarr_data\tomos')

    write_whole_directory(
        src=src_root,
        dst=dst_root,
        max_processes=1
    )