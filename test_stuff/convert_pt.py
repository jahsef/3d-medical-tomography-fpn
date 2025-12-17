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
    assert tomo_array.ndim == 3, "expected tomo array to be d,h,w"
    shape = np.array(tomo_array.shape)//16
    
    availability_heatmap = np.zeros((5,*shape))
    # 0 multi motor region
    # 1 single motor region
    # 2 hard negative region
    # 3 random negative region
    # 4 actual motor locations
    
    #TODO POPULATE AVAILABILITY HEATMAP ACTUAL MOTOR LOCATIONS HERE
    
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
    #top left (patch_origin): motor_idx - 1.5*patch_size
    #bottom right (patch_origin): motor_idx + 0.5*patch_size 
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

def generate_gaussian_label(downsampled_local_motor_coords,downsampled_patch_shape, angstrom_sigma, voxel_spacing):
    pixel_sigma = angstrom_sigma / voxel_spacing
    #local motor coords in realspace
    #convert to local motor coords in downsampled space
    
    label_d, label_h, label_w = downsampled_local_motor_coords
    grid_d = torch.arange(downsampled_patch_shape[0])[:, None, None]
    grid_h = torch.arange(downsampled_patch_shape[1])[None, :, None]
    grid_w = torch.arange(downsampled_patch_shape[2])[None, None, :]
    dist_sq = (grid_d-label_d)**2 + (grid_h-label_h)**2 + (grid_w-label_w)**2
    gaussian_blob = torch.exp(-dist_sq/(2*pixel_sigma**2)).to(torch.float16).unsqueeze(0) 
    return gaussian_blob

#ALL OF THESE FUNCTIONS RETURN INDICES IF APPLICABLE
#then in the save patches orchestrator function we generate gaussians there and save
#probably return motor_idx, type
#also pass reference of shape same as heatmap indicating which voxels are taken and which are available. so i guess 0 could be available nothigns there 1 is unavailable. 
# multi,single and hrad negative will populate that and the random negative function samples from that
def _multi_motor_save(
    multi_motor_samples:int,
    tomo_dict:dict,
    tomo_dir: Path,
    availability_heatmap:np.ndarray,
    downsampling_factor:int,
    angstrom_sigma:float,
    ) -> int:
    
    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    voxel_spacing = tomo_dict['voxel_spacing']
    
    for i,motor_idx_tuple_i in enumerate(motors):
        patch_size = tomo_array.shape
        motor_idx_arr_i = np.asarray(motor_idx_tuple_i) #hopefully np can handle tuple => arr?

        ### FIND VALID RANGES FOR MULTI MOTORS
        for j,motor_idx_tuple_j in enumerate(motors):
            if j <= i:
                continue#only generate valid combinations (triu indices)
            
            motor_idx_arr_j = np.asarray(motor_idx_tuple_j)
            if np.any(np.abs(motor_idx_arr_i - motor_idx_arr_j) >= patch_size):
                continue  # too far apart to fit in one patch
            
            lo = np.minimum(motor_idx_arr_i, motor_idx_arr_j)
            hi = np.maximum(motor_idx_arr_i, motor_idx_arr_j)
            #only accessing the multi motor slice
            availability_heatmap[0, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    ### IF NO VALID INDICES EXIST, EXIT
    valid_indices = np.argwhere(availability_heatmap[0]) #should be (N, heatmap.ndim (3))
    if(valid_indices.shape[0] == 0):
        return
        
    order = np.lexsort((valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]))
    sorted_indices = valid_indices[order]
    stride = min(valid_indices.shape[0] // multi_motor_samples, 1) #10 indices, 5 samples, stride 2. 4 indices 5 samples stride 1 (clamped by the loop later anyway)
    i = 0
    saved_patches = 0
    #generate all patches with stratified sampling
    for _ in range(max(multi_motor_samples, valid_indices.shape[0])):
        patch_origin = sorted_indices[i]
        i += stride
        patch_end = patch_origin+patch_size
        patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
        
        #it should be guaranteed that there are 2 motors here
        downsampled_origin = patch_origin//downsampling_factor
        downsampled_end = patch_end//downsampling_factor
        downsampled_motor_mask = availability_heatmap[4, downsampled_origin[0]:downsampled_end[0], downsampled_origin[1]:downsampled_end[1], downsampled_origin[2]:downsampled_end[2]]
        downsampled_motor_indices = np.argwhere(downsampled_motor_mask) #(N,3)
        assert downsampled_motor_indices.shape[0] >= 2, f"EXPECTED >=2 MOTORS DETECTED IN PATCH FOR MULTI MOTOR CASE, FOUND: {downsampled_motor_indices.shape[0]}"
        max_gaussian = torch.zeros(size = availability_heatmap.shape[0:])
        #get max gaussian from all motors present in the patch
        for j in range(downsampled_motor_indices.shape[0]):
            downsampled_local_motor_coords = downsampled_motor_indices[j] - patch_origin
            max_gaussian = torch.maximum(max_gaussian, generate_gaussian_label(downsampled_local_motor_coords=downsampled_local_motor_coords, downsampled_patch_shape=availability_heatmap.shape[0:], angstrom_sigma=angstrom_sigma, voxel_spacing=voxel_spacing)) #element wise max, .max() just returns actual max value

        torch.save(
            obj = {
                'patch': patch, #everything should be saved in d,h,w format, we leave it up to dataset to expand channel dimension
                'gaussian': max_gaussian,
                'patch_type': "multi_motor"
            },
            f = tomo_dir/f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
        ) 
        saved_patches+=1
    return saved_patches
        


def _single_motor_save(
    single_motor_samples:int,
    tomo_dict:dict,
    tomo_dir: Path,
    availability_heatmap:np.ndarray,
    downsampling_factor:int,
    angstrom_sigma:float,
    ) -> None:
    #single motor: generate range, motor_idx - patch_size
    motors = tomo_dict['motors']
    tomo_array = tomo_dict['tomo_array']
    voxel_spacing = tomo_dict['voxel_spacing']
    patch_size = tomo_array.shape
    #populate availability arr
    for motor_idx_tuple in motors:
        motor_idx_arr = np.array(motor_idx_tuple)
        downsampled_patch_size = patch_size//downsampling_factor
        downsampled_motor_idx = motor_idx_arr//downsampling_factor
        downsampled_start_range = downsampled_motor_idx - downsampled_patch_size
        downsampled_end_range = downsampled_motor_idx
        availability_heatmap[1,downsampled_start_range[0]:downsampled_end_range[0], downsampled_start_range[1]:downsampled_end_range[1], downsampled_start_range[2]:downsampled_end_range[2]]
    
    #now that we have availability arr populated we can do stratified sample from there
    #patch origin converted to realspace should add [0,patch_size-1] to each dimension as to avoid any grid bias
    downsampled_valid_indices = np.argwhere(availability_heatmap[1]) #N,3
    order = np.lexsort((downsampled_valid_indices[:, 2], downsampled_valid_indices[:, 1], downsampled_valid_indices[:, 0]))
    downsampled_sorted_indices = downsampled_valid_indices[order]
    stride = min(downsampled_valid_indices.shape[0] // single_motor_samples, 1) #10 indices, 5 samples, stride 2. 4 indices 5 samples stride 1 (clamped by the loop later anyway)
    i = 0
    saved_patches = 0
    #generate all patches with stratified sampling
    
    for _ in range(max(single_motor_samples, downsampled_sorted_indices.shape[0])):
        downsampled_patch_origin = downsampled_sorted_indices[i]
        patch_origin = downsampled_patch_origin*16 + np.random.randint(0,16) #add 0-15 to avoid grid bias
        i += stride
        patch_end = patch_origin+patch_size
        patch = tomo_array[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]
        
        torch.save(
            obj = {
                'patch': patch,
                'gaussian': generate_gaussian_label(downsampled_local_motor_coords=downsampled_local_motor_coords,downsampled_patch_shape=availability_heatmap.shape[1:], angstrom_sigma=angstrom_sigma, voxel_spacing=voxel_spacing),
                'patch_type': "single_motor"
            },
            f = tomo_dir/f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
        )
        
    for motor_idx_tuple in motors:
        patch_size = tomo_array.shape
        motor_idx_arr = np.asarray(motor_idx_tuple) #hopefully np can handle tuple => arr?
        #ensure valid range
        start_range = np.clip(motor_idx_arr - patch_size, [0,0,0], patch_size)
        end_range = motor_idx_arr #motor is in top left voxel of a patch
        for _ in range(single_motor_samples):
            patch_origin = np.random.randint(start_range, end_range, size = start_range.shape)
            
            patch = tomo_array[:, patch_origin:patch_origin+patch_size].copy()#need to copy otherwise saves the whole array
            #only accessing the single motor slice
            availability_heatmap[1, start_range[0]:end_range[0],start_range[1]:end_range[1], start_range[2]:end_range[2] ] = 1
            local_motor_coords = motor_idx_arr - patch_origin
            downsampled_local_motor_coords = local_motor_coords // downsampling_factor
            torch.save(
                obj = {
                    'patch': patch,
                    'gaussian': generate_gaussian_label(downsampled_local_motor_coords=downsampled_local_motor_coords,downsampled_patch_shape=availability_heatmap.shape[1:], angstrom_sigma=angstrom_sigma, voxel_spacing=voxel_spacing),
                    'patch_type': "single_motor"
                },
                f = tomo_dir/f'patch_{patch_origin[0]}_{patch_origin[1]}_{patch_origin[2]}.pt'
            )


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
