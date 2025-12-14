from pathlib import Path
import numpy as np
from natsort import natsorted
import imageio.v3 as iio
from multiprocessing import Process, Queue
import pandas as pd
import gc

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized


def load_labels(csv_path):
    """Load motor coordinates from new RELABELED_DATA.csv format.

    Returns:
        labels: dict mapping tomo_id -> list of (z, y, x) coordinate tuples
        folds: dict mapping tomo_id -> fold number
    """
    df = pd.read_csv(csv_path)
    labels = {}
    folds = {}

    for _, row in df.iterrows():
        tomo_id = row['tomo_id']
        folds[tomo_id] = row['fold']

        z, y, x = row['z'], row['y'], row['x']
        if z >= 0 and y >= 0 and x >= 0:
            if tomo_id not in labels:
                labels[tomo_id] = []
            labels[tomo_id].append((z, y, x))

    return labels, folds


def load_tomogram(src_dir):
    """Load and normalize a single tomogram from image directory."""
    files = [f for f in src_dir.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)
    if not files:
        return None
    imgs = [iio.imread(file, mode="L") for file in files]
    return normalize_tomogram(np.stack(imgs))


def producer(input_queue, output_queue, labels, folds):
    """Load tomograms from input queue and put on output queue."""
    while True:
        tomo_dir = input_queue.get()
        if tomo_dir is None:  # Poison pill
            break

        tomo_id = tomo_dir.name
        motors = labels.get(tomo_id, [])
        fold = folds.get(tomo_id, -1)

        print(f'Loading tomogram: {tomo_id}')
        tomo_array = load_tomogram(tomo_dir)

        if tomo_array is not None:
            output_queue.put((tomo_id, tomo_array, motors, fold))
        else:
            print(f"No images found in {tomo_dir}")


def save_patches_placeholder(tomo_id, tomo_array, motors, dst):
    """Placeholder for actual patch saving logic."""
    
    #downsample, since labels we generate are gonna be in downsampled space
    
    #HARDEST CASE: multiple motor patches
    #check which motors are within patch_size of each other
    #generate all ranges of indices that have multi motors (generating all indices directly is expensive and inefficient)
    
    #single motor: generate range, motor_idx - patch_size
    
    #HARD NEGATIVE: 
    
    
    
    pass


def write_whole_directory(src, dst, csv_path, num_workers=3, queue_size=5):
    """Main function - 3 producer processes load tomograms, main process consumes."""
    labels, folds = load_labels(csv_path)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Loaded labels for {len(labels)} tomograms with motors")

    tomo_dirs = [d for d in src.iterdir() if d.is_dir()]
    num_tomos = len(tomo_dirs)
    print(f"Found {num_tomos} tomogram directories")

    input_queue = Queue()
    output_queue = Queue(maxsize=queue_size)

    # Fill input queue with work
    for tomo_dir in tomo_dirs:
        input_queue.put(tomo_dir)
    # Add poison pills for each worker
    for _ in range(num_workers):
        input_queue.put(None)

    # Start producer workers
    producers = []
    for _ in range(num_workers):
        p = Process(target=producer, args=(input_queue, output_queue, labels, folds))
        p.start()
        producers.append(p)

    # Consume from output queue in main process
    processed = 0
    while processed < num_tomos:
        alive_count = sum(1 for p in producers if p.is_alive())
        if output_queue.empty() and alive_count == 0:
            break

        try:
            item = output_queue.get(timeout=1.0)
        except:
            continue

        tomo_id, tomo_array, motors, fold = item
        D, H, W = tomo_array.shape
        print(f'Processing: {tomo_id} (fold={fold}, motors={len(motors)}, shape={D}x{H}x{W})')

        save_patches_placeholder(tomo_id, tomo_array, motors, dst)

        del tomo_array
        gc.collect()
        processed += 1

    # Wait for producers to finish
    for p in producers:
        p.join()


if __name__ == '__main__':
    src_root = Path(r'data/original_data/train')
    dst_root = Path(r'data/processed/relabeled_patch_pt_data')
    csv_path = Path(r'data/original_data/RELABELED_DATA.csv')

    write_whole_directory(src=src_root, dst=dst_root, csv_path=csv_path, num_workers=3, queue_size=3)
