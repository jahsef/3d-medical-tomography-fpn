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
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import sys
import gc
import multiprocessing as mp

current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs.motoridentifier import MotorIdentifier

# Configuration Parameters
MODEL_PATH = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\curriculum\run2\best.pt'
MASTER_TOMO_PATH = Path.cwd() / 'original_data/train'
ORIGINAL_DATA_PATH = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train')
GROUND_TRUTH_CSV = r'original_data\train_labels.csv'
OUTPUT_DIR = Path('inference_results')
OUTPUT_CSV_NAME = 'fart.csv'

# Dataset Split Configuration
tomo_id_list = [dir.name for dir in MASTER_TOMO_PATH.iterdir() if dir.is_dir()]
train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.95, test_size=0.05, random_state=42)
val_id_list = train_id_list[:len(train_id_list)//10:5]

# Inference Parameters
BATCH_SIZE = 4
PATCH_SIZE = 128
OVERLAP = 0
VOXEL_SPACING = 15
THRESHOLD_ANGSTROMS = 1000
THRESHOLD_VOXELS = THRESHOLD_ANGSTROMS / VOXEL_SPACING
PRUNING_RADIUS = 16
CONF_THRESHOLDS = np.arange(0.55, 0.80, 0.05)
BETA = 2

# Optimization Parameters
NUM_LOADING_THREADS = min(12, mp.cpu_count())
TOMOGRAM_QUEUE_SIZE = 2  # Reduced from 2 to minimize memory
NORMALIZATION_CONSTANTS = (0.479915, 0.224932)

def get_predictions_above_threshold(heatmap, conf_threshold):
    """Extract prediction coordinates above confidence threshold."""
    mask = heatmap > conf_threshold
    coords = np.where(mask)
    predictions = list(zip(coords[0], coords[1], coords[2]))
    confidences = heatmap[mask]
    return predictions, confidences

def load_single_image(file_path):
    """Load a single image file."""
    try:
        return iio.imread(file_path, mode="L")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_tomogram_vectorized(tomo_array):
    """Vectorized normalization."""
    tomo_normalized = (tomo_array.astype(np.float16) / 255.0 - NORMALIZATION_CONSTANTS[0]) / NORMALIZATION_CONSTANTS[1]
    return tomo_normalized

def load_tomogram_parallel(src: Path):
    """Load tomogram with parallel image loading."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    print(f'Loading tomogram: {src.name}')
    
    files = [f for f in src.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files = natsorted(files, key=lambda x: x.name)
    
    if not files:
        print(f"No image files found in {src}")
        return None
    
    with ThreadPoolExecutor(max_workers=NUM_LOADING_THREADS) as executor:
        future_to_idx = {executor.submit(load_single_image, file): idx for idx, file in enumerate(files)}
        
        first_img = load_single_image(files[0])
        if first_img is None:
            return None
            
        tomo_array = np.empty((len(files), *first_img.shape), dtype=np.uint8)
        tomo_array[0] = first_img
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            if idx == 0:
                continue
                
            img = future.result()
            if img is not None:
                tomo_array[idx] = img
            else:
                print(f"Failed to load image at index {idx}")
                return None
    
    tomo_array = normalize_tomogram_vectorized(tomo_array)
    return torch.as_tensor(tomo_array, dtype=torch.float16)

def load_ground_truth(csv_path, tomo_ids):
    """Load ground truth motor locations for specified tomograms."""
    df = pd.read_csv(csv_path)
    gt_dict = {}
    
    for tomo_id in tomo_ids:
        tomo_rows = df[df['tomo_id'] == tomo_id]
        if len(tomo_rows) == 0:
            gt_dict[tomo_id] = []
        else:
            motors = []
            for _, row in tomo_rows.iterrows():
                if row['Motor axis 0'] != -1:
                    motors.append([row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']])
            gt_dict[tomo_id] = motors
    
    return gt_dict

def prune_nearby_predictions(predictions, confidences, radius=25):
    """Remove predictions within radius of higher-confidence ones."""
    if len(predictions) == 0:
        return predictions
    
    sorted_indices = np.argsort(confidences)[::-1]
    kept_predictions = []
    kept_coords = []
    
    for idx in sorted_indices:
        pred_coord = np.array(predictions[idx])
        
        too_close = False
        for kept_coord in kept_coords:
            if np.linalg.norm(pred_coord - kept_coord) <= radius:
                too_close = True
                break
        
        if not too_close:
            kept_predictions.append(predictions[idx])
            kept_coords.append(pred_coord)
    
    return kept_predictions

def hungarian_matching(predictions, ground_truth, threshold_voxels):
    """Use Hungarian algorithm to match predictions to ground truth within threshold."""
    if len(predictions) == 0 or len(ground_truth) == 0:
        return [], len(predictions), len(ground_truth)
    
    pred_array = np.array(predictions)
    gt_array = np.array(ground_truth)
    
    distances = np.linalg.norm(pred_array[:, None, :] - gt_array[None, :, :], axis=2)
    pred_indices, gt_indices = linear_sum_assignment(distances)
    
    matches = []
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if distances[p_idx, g_idx] <= threshold_voxels:
            matches.append((p_idx, g_idx))
    
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    return matches, fp, fn

def process_heatmap_optimized(heatmap, gt_motors):
    """Process heatmap with immediate cleanup and no threading overhead."""
    tomo_metrics = {}
    
    # Process all thresholds sequentially and immediately calculate metrics
    for conf_thresh in CONF_THRESHOLDS:
        # Extract predictions
        predictions, confidences = get_predictions_above_threshold(heatmap, conf_thresh)
        
        # Prune predictions
        predictions = prune_nearby_predictions(predictions, confidences, PRUNING_RADIUS)
        
        # Clean up intermediate data immediately
        del confidences
        
        # Calculate metrics immediately
        _, fp, fn = hungarian_matching(predictions, gt_motors, THRESHOLD_VOXELS)
        tp = len(predictions) - fp
        
        tomo_metrics[conf_thresh] = {'tp': tp, 'fp': fp, 'fn': fn}
        
        # Clean up predictions immediately
        del predictions
    
    # Clean up heatmap immediately after processing all thresholds
    del heatmap
    gc.collect()
    
    return tomo_metrics

def load_tomogram_sequential(tomo_id):
    """Load single tomogram without threading overhead."""
    tomo_path = ORIGINAL_DATA_PATH / tomo_id
    return load_tomogram_parallel(tomo_path)

def calculate_fbeta_score(tp, fp, fn, beta=2):
    """Calculate F-beta score."""
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return fbeta

def main():
    # Setup
    device = torch.device('cuda')
    model = MotorIdentifier(max_motors=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print(f"Processing {len(val_id_list)} validation tomograms")
    print(f"Using {NUM_LOADING_THREADS} threads for image loading per tomogram")
    
    # Load ground truth
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV, val_id_list)
    
    # Initialize metrics accumulation
    threshold_metrics = {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in CONF_THRESHOLDS}
    
    # Process tomograms sequentially to avoid memory accumulation
    pbar = tqdm(total=len(val_id_list), desc="Processing tomograms")
    
    for tomo_id in val_id_list:
        # Load tomogram
        tomo = load_tomogram_sequential(tomo_id)
        
        if tomo is None:
            pbar.update(1)
            continue
        
        # Run inference
        original_shape = tomo.shape
        tomo_tensor = tomo.reshape(1, 1, *original_shape).to(device)
        
        # Clear CPU reference immediately
        del tomo
        gc.collect()
        
        with torch.no_grad():  # Ensure no gradients are computed
            results = model.inference(tomo_tensor, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, 
                                    overlap=OVERLAP, device=device, tqdm_progress=False)
        
        heatmap = results.view(results.shape[2:]).cpu().numpy()
        
        # Clean up GPU memory immediately
        del tomo_tensor, results
        torch.cuda.empty_cache()
        
        # Process heatmap with optimized function
        tomo_metrics = process_heatmap_optimized(heatmap, ground_truth[tomo_id])
        
        # Accumulate results immediately
        for conf_thresh, metrics in tomo_metrics.items():
            threshold_metrics[conf_thresh]['tp'] += metrics['tp']
            threshold_metrics[conf_thresh]['fp'] += metrics['fp']
            threshold_metrics[conf_thresh]['fn'] += metrics['fn']
        
        # Clean up tomo_metrics
        del tomo_metrics
        
        pbar.update(1)
        
        # Force cleanup after each tomogram
        gc.collect()
    
    pbar.close()
    
    # Calculate final metrics
    results_data = []
    for conf_thresh in CONF_THRESHOLDS:
        metrics = threshold_metrics[conf_thresh]
        total_tp, total_fp, total_fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        fbeta = calculate_fbeta_score(total_tp, total_fp, total_fn, BETA)
        
        results_data.append({
            'threshold': round(conf_thresh, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f_beta': round(fbeta, 3),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        })
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_DIR / OUTPUT_CSV_NAME, index=False)
    
    # Print best result
    best_result = results_df.loc[results_df['f_beta'].idxmax()]
    print(f"\nBest F-beta score: {best_result['f_beta']:.4f}")
    print(f"Best threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
    print(f"TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")

if __name__ == "__main__":
    main()