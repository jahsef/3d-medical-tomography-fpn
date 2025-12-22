import torch
from dataclasses import dataclass
from typing import Dict
import csv
import os
import math


class LossTracker:
    """Track loss values over batches for epoch-level reporting"""
    def __init__(self, is_mean_loss: bool):
        self.total_loss = 0
        self.total_samples = 0
        self.is_mean_loss = is_mean_loss

    def update(self, batch_loss, batch_size):
        if self.is_mean_loss:
            self.total_loss += batch_loss * batch_size 
        else:
            self.total_loss += batch_loss
        
        self.total_samples += batch_size

    def get_epoch_loss(self):
        return self.total_loss / self.total_samples if self.total_samples > 0 else 0




def soft_dice_score(pred, target, smooth=1e-6):
    """Compute soft Dice score for heatmaps"""
    pred = torch.sigmoid(pred)
    
    batch_size = pred.size(0)
    dice_scores = []
    
    for i in range(batch_size):
        pred_flat = pred[i].view(-1).cpu()
        target_flat = target[i].view(-1).cpu()
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_scores.append(dice.item())
    
    return sum(dice_scores) / len(dice_scores)  # Return average dice score


def peak_distance(pred_logits, target):
    """
    Compute normalized Euclidean distance between predicted and target peaks.

    NOTE: Only compares single argmax peaks - doesn't handle multi-motor patches correctly.
    Fine for diagnostics since multi-motor is a small fraction of data.

    Args:
        pred_logits: Predicted heatmap logits [B, D, H, W]
        target: Target heatmap (same shape as pred)

    Returns:
        score: [0, 1] where 1 = perfect overlap, 0 = max distance
    """
    pred = torch.sigmoid(pred_logits)
    batch_size = pred.shape[0]

    pred = pred.cpu()
    target = target.cpu()

    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    spatial_shape = pred.shape[1:]  # (D, H, W)
    pred_peaks_flat = pred_flat.argmax(dim=1)
    target_peaks_flat = target_flat.argmax(dim=1)

    # Unravel flat indices to 3D coordinates
    pred_coords = torch.stack(torch.unravel_index(pred_peaks_flat, spatial_shape))  # (3, batch)
    target_coords = torch.stack(torch.unravel_index(target_peaks_flat, spatial_shape))  # (3, batch)

    # Euclidean distance between predicted and target peaks
    distance = torch.sqrt(((pred_coords - target_coords).float() ** 2).sum(dim=0))
    max_distance = torch.sqrt(torch.tensor(spatial_shape, dtype=torch.float32).pow(2).sum())

    score = 1.0 - (distance / max_distance)
    return torch.clamp(score, 0.0, 1.0).mean().item()


def peak_sharpness(pred_logits):
    """
    Compute peak sharpness/confidence measure.

    Args:
        pred_logits: Predicted heatmap logits [B, D, H, W]

    Returns:
        score: [0, 1] where higher = sharper peak
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.shape[0], -1).cpu()

    pred_max = pred_flat.max(dim=1)[0]
    pred_mean = pred_flat.mean(dim=1)
    
    sharpness = torch.sigmoid(pred_max / (69*pred_mean + 1e-8))#consider dividing by 100, sigmoid is close to 1 at about 4, so to get to 1 sharpness, peak needs to be about 400x larger than avg
    return torch.clamp(sharpness, 0.0, 1.0).mean().item()


def topk_percent_accuracy(pred, target, k_percents):
    """
    Check if target peak is in top-k% predictions for each sample.

    Only evaluates samples with actual GT peaks (max > 0.1).
    k_percents are percentages of total voxels, converted to counts via ceil.
    """
    pred = torch.sigmoid(pred)

    batch_size = pred.size(0)
    k_accuracies = {k: [] for k in k_percents}

    for i in range(batch_size):
        pred_sample = pred[i].view(-1).cpu()
        target_sample = target[i].view(-1).cpu()

        # Skip samples with no GT peak
        if target_sample.max() < 0.1:
            continue

        target_peak_idx = target_sample.argmax()
        total_voxels = pred_sample.size(0)

        for k_pct in k_percents:
            # Convert percentage to count using ceil
            k_count = math.ceil(total_voxels * k_pct / 100)
            _, topk_indices = torch.topk(pred_sample, min(k_count, total_voxels))

            is_in_topk = (topk_indices == target_peak_idx).any()
            k_accuracies[k_pct].append(is_in_topk.item())

    # Return average accuracies (or 0 if no valid samples)
    return {k: (sum(accs) / len(accs) if accs else 0.0) for k, accs in k_accuracies.items()}


@dataclass
class MetricsResult:
    """Structured container for all training/validation metrics"""
    conf_loss: float = 0.0
    dice_score: float = 0.0
    peak_dist: float = 0.0
    peak_sharp: float = 0.0
    topk_results: Dict[int, float] = None

    def __post_init__(self):
        if self.topk_results is None:
            self.topk_results = {}


def compute_epoch_metrics(preds, gt, conf_loss, topk_percent_values) -> MetricsResult:
    """Compute all metrics from full epoch predictions"""
    with torch.no_grad():
        return MetricsResult(
            conf_loss=conf_loss,
            dice_score=soft_dice_score(preds, gt),
            peak_dist=peak_distance(preds, gt),
            peak_sharp=peak_sharpness(preds),
            topk_results=topk_percent_accuracy(preds, gt, topk_percent_values)
        )


def save_csv(epoch, filename="train_results.csv", **kwargs):
    """Save training metrics to CSV file, appending to existing file"""
    fieldnames = list(kwargs.keys())
    values = [round(float(v), 6) for v in kwargs.values()]
    
    file_exists = os.path.isfile(filename)
    
    # Append to file (or create if doesn't exist)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow(['epoch'] + fieldnames)
        
        # Write data row
        writer.writerow([epoch] + values)