from pathlib import Path
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import pandas as pd
import imageio.v3 as iio
from natsort import natsorted
import gc
import torch


IMAGE_EXTS: set[str] = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def normalize_tomogram(tomo_array: np.ndarray) -> np.ndarray:
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    return tomo_normalized

def load_tomogram(src_dir: Path) -> Optional[np.ndarray]:
    """Load and normalize a single tomogram from image directory."""
    files = [f for f in src_dir.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)
    if not files:
        return None
    imgs = [iio.imread(file, mode="L") for file in files]
    return normalize_tomogram(np.stack(imgs))


def producer(
    input_queue: Queue,
    output_queue: Queue,
    df: pd.DataFrame
) -> None:
    """Load tomograms from input queue and put on output queue."""
    while True:
        tomo_dir: Optional[Path] = input_queue.get()
        if tomo_dir is None:  # Poison pill
            break

        tomo_id = tomo_dir.name
        tomo_rows = df[df['tomo_id'] == tomo_id]

        # Extract motors (valid coordinates only)
        motors = []
        for _, row in tomo_rows.iterrows():
            z, y, x = row['z'], row['y'], row['x']
            if z >= 0 and y >= 0 and x >= 0:
                motors.append((z, y, x))

        # Get fold and voxel_spacing from first row
        first_row = tomo_rows.iloc[0]
        fold = first_row['fold']
        voxel_spacing = first_row['voxel_spacing']

        print(f'Loading tomogram: {tomo_id}')
        tomo_array = load_tomogram(tomo_dir)

        if tomo_array is not None:
            output_queue.put({
                'tomo_id': tomo_id,
                'tomo_array': tomo_array,
                'motors': motors,
                'voxel_spacing': voxel_spacing,
                'fold': fold,
            })
        else:
            print(f"No images found in {tomo_dir}")


def save_patches_placeholder(
    tomo_dict: dict,
    dst: Path,
    downsampling_factor: int = 16
) -> None:
    """Placeholder for actual patch saving logic."""
    tomo_dir = dst / tomo_dict['tomo_id']
    Path.mkdir(tomo_dir)
    tomo_array = tomo_dict['tomo_array']
    assert tomo_array.ndim == 4, "expected tomo array to be c,d,h,w"
    availability_heatmap = np.zeros(tomo_array.shape[1:] // downsampling_factor)
    
    #SAMPLING PLAN: per motor: N single, M hard neg, X random neg
    #               per tomogram: up to total_motors * Y multi-motor patches (best effort)

    #downsample, since labels we generate are gonna be in downsampled space
    
    
    #naive easy strategy (generate ranges and randomly sample until we reach our target)
    #so instead of guaranteeing ranges that are valid, we define the ranges then check validity after like is it in patch etc
    
    
    #HARDEST CASE: multiple motor patches
    #check which motors are within patch_size of each other
    #generate all ranges of indices that have multi motors (generating all indices directly is expensive and inefficient)
    
    #single motor: generate range, motor_idx - patch_size
    
    #HARD NEGATIVE (close to motor locations but not motor locations)
    
    #max range to consider 'hard', cannot be farther than 0.5*patch_size l1 distance
    #top left (start_idx): motor_idx - 1.5*patch_size
    #bottom right (start_idx): motor_idx + 0.5*patch_size 
    #min range to consider 'hard', cannot be too close otherwise its too ambiguous
    #patch center to motor idx l2 distance.
    #dist >= 1.1 * L2Norm (d/2, h/2, w/2)
    
    #random negative
    #save all ranges of the other cases
    #generate random negatives and check validity and ensure that they are not within the ranges of the other cases
    #could have a super weird edgecase where it may not be able to find any valid locations but it SHOUDLNT affect this dataset (could add like a max retry thing so that we dont get stuck)
    
    #also need to save patch_type like hard_negative, multi, single, negative in patch dict
    
    #generate gaussians
    
    #patch_D_H_W (origin)
    #patch: actual array
    #gaussian: heatmap gt
    #other metadata? probably not needed, maybe global coords if we wanted to bypass monai inferer. fold not needed since we have the label csv anyway
    
    pass

def generate_gaussian_label(motor_coords,downsampling_factor, angstrom_blob_sigma):
    """
    Generate gaussian blob and apply weighting on target device with no intermediate transfers.
    Assumes input grids are already on target_device.
    
    Returns:
        torch.Tensor: Weighted gaussian label on target_device
    """
    label_d, label_h, label_w = motor_coords
    gaussian_blob = torch.exp(-((grid_d-label_d)**2 + 
                               (grid_h-label_h)**2 + 
                               (grid_w-label_w)**2)/(2*blob_sigma_pixels**2)).to(torch.float16).unsqueeze(0) 
    return gaussian_blob
#ALL OF THESE FUNCTIONS RETURN INDICES IF APPLICABLE
#then in the save patches orchestrator function we generate gaussians there and save
#probably return motor_idx, type
#also pass reference of shape same as heatmap indicating which voxels are taken and which are available. so i guess 0 could be available nothigns there 1 is unavailable. 
# multi,single and hrad negative will populate that and the random negative function samples from that
def _multi_motor_save(multi_motor_samples:int) -> None:
    pass

def _single_motor_save(single_motor_samples:int, tomo_array: np.ndarray,
    motors: list[tuple[float, float, float]],
    tomo_dir: Path,
    availability_heatmap:np.ndarray) -> None:
    #single motor: generate range, motor_idx - patch_size
    
    for motor in motors:
        
        patch_size = tomo_array.shape[1:]
        motor_dhw = np.asarray(motor) #hopefully np can handle tuple => arr?
        #ensure valid range
        start_range = np.clip(motor_dhw - patch_size, [0,0,0], patch_size)
        end_range = motor_dhw
        
        for _ in range(single_motor_samples):
            sample_idx = np.random.randint(start_range, end_range, size = start_range.shape)
            
            patch = tomo_array[:, sample_idx:sample_idx+patch_size].copy()
            #something weird with views when you save it, it saves the whole array?
            
    
    pass

def _hard_negative_save(hard_negative_samples:int) -> None:
    #hard negative case may be harder than expected because of multi motor cases, we cant just look at  1 motor
    #i suppose an easy way to do this is we generate the hard negative bounds around 1 motor
    #then when sampling randomly from the range we just have a check of if a motor exists in that region we sampled from
    #much easier than generating a guaranteed range, i think actually we can also use the dense availability mask from earlier for this case rather than ranges
    #then with dense availability mask we actually have guaranteed voxels to sample from, no need for max retry amounts with the ranges from before
    pass

def _random_negative_save(random_negative_samples) -> None:
    pass

def write_whole_directory(
    src: Path,
    dst: Path,
    csv_path: Path,
    num_workers: int = 3,
    queue_size: int = 5
) -> None:
    """
    Docstring for write_whole_directory

    :param src: src dir for tomograms (jpgs). structure files/tomo_N/0.jpg
    :param dst: root dir for saving tomos in format dst/tomo_N/patch_N.pt
    :param csv_path: the csv path duh
    :param num_workers: yuh
    :param queue_size: yuh

    """
    #Main function - producer processes load tomograms, main process consumes."""
    df = pd.read_csv(csv_path)
    dst.mkdir(parents=True, exist_ok=False)

    print(f"Loaded {len(df)} rows from CSV")

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
        p = Process(target=producer, args=(input_queue, output_queue, df))
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
                
        tomo_dict = item
        D, H, W = tomo_dict['tomo_array'].shape
        print(f"Processing: {tomo_dict['tomo_id']} (fold={tomo_dict['fold']}, motors={len(tomo_dict['motors'])}, shape={D}x{H}x{W}, voxel_spacing={tomo_dict['voxel_spacing']})")

        save_patches_placeholder(tomo_dict, dst)

        del tomo_dict
        gc.collect()
        processed += 1

    # Wait for producers to finish
    for p in producers:
        p.join()


if __name__ == '__main__':
    src_root = Path(r'data/original_data/train')
    dst_root = Path(r'data/processed/relabeled_patches')
    csv_path = Path(r'data/original_data/RELABELED_DATA.csv')

    write_whole_directory(src=src_root, dst=dst_root, csv_path=csv_path, num_workers=3, queue_size=3)
