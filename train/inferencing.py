import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
from natsort import natsorted
import imageio.v3 as iio
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import multiprocessing as mp

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs.motoridentifier import MotorIdentifier

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')




# Configuration Parameters
MODEL_PATH = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models/relabel/full/lite_augs/10m3/best.pt'
MASTER_TOMO_PATH = Path.cwd() / 'original_data/train'
ORIGINAL_DATA_PATH = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
GROUND_TRUTH_CSV = r'original_data\relabel.csv'
OUTPUT_DIR = Path('inference_results')
OUTPUT_CSV_NAME = 'fart.csv'

#60,30,10,2,1
#60:1
#30:2,3,4,5,6
#10:

# Dataset Split Configuration
tomo_id_list = [dir.name for dir in MASTER_TOMO_PATH.iterdir() if dir.is_dir()]
train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.85, test_size=0.15, random_state=42)
# val_id_list = val_id_list[:len(val_id_list):]
val_id_list = train_id_list[:len(train_id_list):20]  
# val_id_list = train_id_list[len(train_id_list)//15*4:len(train_id_list)//15*8 :4]
# val_id_list = ['tomo_d7475d']

# Inference Parameters
DOWNSAMPLING_FACTOR = 16
BATCH_SIZE = 4
PATCH_SIZE = (160,288,288)
OVERLAP = 0.5
VOXEL_SPACING = 16
THRESHOLD_ANGSTROMS = 1000
# Update the threshold calculation
THRESHOLD_VOXELS = THRESHOLD_ANGSTROMS / (VOXEL_SPACING * DOWNSAMPLING_FACTOR)

# Update the pruning radius 
PRUNING_RADIUS = 5  #this should be in downscaled space

CONF_THRESHOLDS = np.arange(0.15, 0.95, 0.01)
BETA = 2

# Optimization Parameters
NUM_LOADING_THREADS = min(12, mp.cpu_count())
TOMOGRAM_QUEUE_SIZE = 3
NORMALIZATION_CONSTANTS = (0.479915, 0.224932)

def load_tomogram(src_path):
    """Load tomogram with parallel image loading."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [f for f in src_path.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)
    assert files, f"No image files found in {src_path}"
    
    with ThreadPoolExecutor(max_workers=NUM_LOADING_THREADS) as executor:
        futures = [executor.submit(iio.imread, f, mode="L") for f in files]
        images = [f.result() for f in futures]
    
    tomo_array = np.stack(images)
    tomo_normalized = (tomo_array.astype(np.float16) / 255.0 - NORMALIZATION_CONSTANTS[0]) / NORMALIZATION_CONSTANTS[1]
    return torch.as_tensor(tomo_normalized, dtype=torch.float16)

def load_ground_truth(csv_path, tomo_ids):
    """Load ground truth motor locations, optionally downsampled."""
    df = pd.read_csv(csv_path)
    gt_dict = {}
    for tomo_id in tomo_ids:
        tomo_rows = df[df['tomo_id'] == tomo_id]
        motors = []
        for _, row in tomo_rows.iterrows():
            if row['Motor axis 0'] != -1:
                # Apply downsampling to ground truth coordinates
                motor_coords = [
                    row['Motor axis 0'] / DOWNSAMPLING_FACTOR,
                    row['Motor axis 1'] / DOWNSAMPLING_FACTOR, 
                    row['Motor axis 2'] / DOWNSAMPLING_FACTOR
                ]
                motors.append(motor_coords)
        gt_dict[tomo_id] = motors
    return gt_dict



def prune_nearby_predictions_fast(heatmap: np.ndarray, r=16, min_thresh=0.55):
    """Fast pruning using precomputed candidates and dense boolean mask."""
    
    heatmap_copy = np.ascontiguousarray(heatmap.copy())
    # print(heatmap_copy.shape)
    
    D, H, W = heatmap_copy.shape
    
    # Get all candidates above threshold
    candidate_coords = np.where(heatmap_copy > min_thresh)
    candidate_values = heatmap_copy[candidate_coords]
    
    if len(candidate_values) == 0:
        return []
    
    # Sort by confidence descending
    sorted_indices = np.argsort(candidate_values)[::-1]
    
    # Dense boolean mask for O(1) lookups
    masked = np.zeros(heatmap_copy.shape, dtype=bool)
    
    kept_predictions = []
    
    # Iterate through sorted candidates
    for idx in sorted_indices:
        d = candidate_coords[0][idx]
        h = candidate_coords[1][idx] 
        w = candidate_coords[2][idx]
        
        # Skip if already masked
        if masked[d, h, w]:
            continue
            
        # Keep this prediction
        kept_predictions.append((d, h, w))
        
        # Mask neighborhood
        d_i = max(0, d - r)
        d_f = min(D, d + r + 1)
        h_i = max(0, h - r)
        h_f = min(H, h + r + 1)
        w_i = max(0, w - r)
        w_f = min(W, w + r + 1)
        
        masked[d_i:d_f, h_i:h_f, w_i:w_f] = True
    
    return kept_predictions


def hungarian_matching(predictions, ground_truth, threshold_voxels):
    """Hungarian matching with threshold."""
    if len(predictions) == 0 or len(ground_truth) == 0:
        return len(predictions), len(ground_truth)
    
    pred_array = np.array(predictions)
    gt_array = np.array(ground_truth)
    distances = np.linalg.norm(pred_array[:, None, :] - gt_array[None, :, :], axis=2)
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    tp = sum(1 for p_idx, g_idx in zip(pred_indices, gt_indices) if distances[p_idx, g_idx] <= threshold_voxels)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return fp, fn

def calculate_fbeta_score(tp, fp, fn, beta=2):
    """Calculate F-beta score."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def tomogram_thread(tomo_queue, is_tomo_ready, ground_truth):
    """Load tomograms and put in queue."""
    for tomo_id in val_id_list:
        tomo = load_tomogram(ORIGINAL_DATA_PATH / tomo_id)
        tomo_queue.put((tomo, ground_truth[tomo_id]))
        is_tomo_ready.set()
    tomo_queue.put(None)  # Sentinel

def inference_thread(tomo_queue, hungarian_queue, is_tomo_ready, model, device):
    """Run inference and pruning."""
    while True:
        logging.info('WAITING FOR TOMOGRAM')
        is_tomo_ready.wait()
        item = tomo_queue.get()
        logging.info('GOT TOMOGRAM')
        if item is None:
            hungarian_queue.put(None)
            break
        
        tomo, gt_motors = item
        
        # Run inference
        tomo_tensor = tomo.reshape(1, 1, *tomo.shape).to(device)
        logging.info('START INFERENCING')
        with torch.no_grad():
            results = model.inference(tomo_tensor, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, 
                                    overlap=OVERLAP, device=device, tqdm_progress=True)
        logging.info('END INFERENCING')
        heatmap = results.view(results.shape[2:]).cpu().numpy()
        
        # Process all thresholds
        logging.info('START CONF THRESHOLDING')
        conf_results = {}
        
        # Extract at lowest threshold and prune once
        min_thresh = CONF_THRESHOLDS.min()
        pruned_predictions = prune_nearby_predictions_fast(heatmap, PRUNING_RADIUS, min_thresh)

        # Filter pruned results by each threshold
        conf_results = {}
        if len(pruned_predictions) > 0:
            pruned_coords = np.array(pruned_predictions)
            for conf_thresh in CONF_THRESHOLDS:
                pruned_confidences = heatmap[pruned_coords[:, 0], pruned_coords[:, 1], pruned_coords[:, 2]]
                conf_results[conf_thresh] = pruned_coords[pruned_confidences > conf_thresh].tolist()
        else:
            for conf_thresh in CONF_THRESHOLDS:
                conf_results[conf_thresh] = []
        
        
        logging.info('END CONF THRESHOLDING')
        
        hungarian_queue.put((gt_motors, conf_results))
        logging.info('FINISHED PUT HUNGARIAN')
        
        if tomo_queue.empty():
            logging.info('CLEARING FLAG, NO TOMOS READY')
            is_tomo_ready.clear()

def hungarian_thread(hungarian_queue, final_metrics, metrics_lock):
    """Process Hungarian matching results."""
    while True:
        item = hungarian_queue.get()
        if item is None:
            break
            
        gt_motors, conf_results = item
        
        with metrics_lock:
            for conf_thresh, predictions in conf_results.items():
                fp, fn = hungarian_matching(predictions, gt_motors, THRESHOLD_VOXELS)
                tp = len(predictions) - fp
                
                if conf_thresh not in final_metrics:
                    final_metrics[conf_thresh] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                final_metrics[conf_thresh]['tp'] += tp
                final_metrics[conf_thresh]['fp'] += fp
                final_metrics[conf_thresh]['fn'] += fn

def main():
    # Setup
    device = torch.device('cuda')
    model = MotorIdentifier(norm_type="gn").to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Processing {len(val_id_list)} validation tomograms")
    
    # Load ground truth
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV, val_id_list)
    
    # Threading setup
    tomo_queue = queue.Queue(maxsize=TOMOGRAM_QUEUE_SIZE)
    hungarian_queue = queue.Queue()
    is_tomo_ready = threading.Event()
    final_metrics = {}
    metrics_lock = threading.Lock()
    
    # Start threads
    threads = [
        threading.Thread(target=tomogram_thread, args=(tomo_queue, is_tomo_ready, ground_truth)),
        threading.Thread(target=inference_thread, args=(tomo_queue, hungarian_queue, is_tomo_ready, model, device)),
        threading.Thread(target=hungarian_thread, args=(hungarian_queue, final_metrics, metrics_lock))
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Calculate final metrics and save
    results_data = []
    for conf_thresh in CONF_THRESHOLDS:
        metrics = final_metrics[conf_thresh]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fbeta = calculate_fbeta_score(tp, fp, fn, BETA)
        
        results_data.append({
            'threshold': round(conf_thresh, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f_beta': round(fbeta, 3),
            'tp': tp, 'fp': fp, 'fn': fn
        })
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / OUTPUT_CSV_NAME, index=False)
    
    # Print best result
    best_result = results_df.loc[results_df['f_beta'].idxmax()]
    print(f"\nBest F-beta score: {best_result['f_beta']:.4f}")
    print(f"Best threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")

if __name__ == "__main__":
    main()