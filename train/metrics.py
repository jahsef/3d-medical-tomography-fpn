import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional
import csv
import os


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


class TopKTracker:
    """Utility class to track multiple top-k accuracies"""
    def __init__(self, k_values=[1, 5, 10, 50]):
        self.k_values = k_values
        self.trackers = {k: LossTracker(is_mean_loss=True) for k in k_values}
    
    def update(self, topk_dict, batch_size):
        """Update all trackers with batch results"""
        for k in self.k_values:
            if k in topk_dict:
                self.trackers[k].update(batch_loss=topk_dict[k], batch_size=batch_size)

    def get_epoch_results(self):
        """Get epoch results for all k values"""
        return {k: tracker.get_epoch_loss() for k, tracker in self.trackers.items()}


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


def topk_accuracy(pred, target, k_values=[1, 5, 10, 50]):
    """Check if target peak is in top-k predictions for each sample across multiple k values"""
    pred = torch.sigmoid(pred)
    
    batch_size = pred.size(0)
    k_accuracies = {k: [] for k in k_values}
    
    for i in range(batch_size):
        # Get single sample
        pred_sample = pred[i].view(-1).cpu()
        target_sample = target[i].view(-1).cpu()
        # Find target peak location
        target_peak_idx = target_sample.argmax()
        
        # Calculate accuracy for each k value
        for k in k_values:
            # Get top-k predictions
            _, topk_indices = torch.topk(pred_sample, min(k, pred_sample.size(0)))
            
            # Check if target peak is in top-k
            is_in_topk = (topk_indices == target_peak_idx).any()
            k_accuracies[k].append(is_in_topk.item())
    
    # Return average accuracies for each k
    return {k: sum(accs) / len(accs) for k, accs in k_accuracies.items()}


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


class MetricsEvaluator:
    """Unified metrics evaluation with performance control"""

    def __init__(self,
                 topk_values: List[int] = [1, 5, 10, 50],
                 enable_dice: bool = True,
                 enable_peak_metrics: bool = True,
                 enable_topk: bool = True):
        """
        Args:
            topk_values: List of k values for top-k accuracy
            enable_dice: Whether to compute dice score
            enable_peak_metrics: Whether to compute peak distance and sharpness
            enable_topk: Whether to compute top-k accuracies
        """
        self.topk_values = topk_values
        self.enable_dice = enable_dice
        self.enable_peak_metrics = enable_peak_metrics
        self.enable_topk = enable_topk

        # Initialize trackers
        self.reset_trackers()

    def reset_trackers(self):
        """Reset all internal trackers for new epoch"""
        self.conf_loss_tracker = LossTracker(is_mean_loss=True)

        if self.enable_dice:
            self.dice_tracker = LossTracker(is_mean_loss=True)

        if self.enable_peak_metrics:
            self.peak_dist_tracker = LossTracker(is_mean_loss=True)
            self.peak_sharp_tracker = LossTracker(is_mean_loss=True)

        if self.enable_topk:
            self.topk_tracker = TopKTracker(self.topk_values)

    def update_batch(self, outputs: torch.Tensor, labels: torch.Tensor,
                    conf_loss: float, batch_size: int) -> None:
        """Update metrics with single batch results"""
        # Always track confidence loss
        self.conf_loss_tracker.update(batch_loss=conf_loss, batch_size=batch_size)

        # Compute additional metrics with no_grad for performance
        with torch.no_grad():
            if self.enable_dice:
                batch_dice = soft_dice_score(outputs, labels)
                self.dice_tracker.update(batch_loss=batch_dice, batch_size=batch_size)

            if self.enable_peak_metrics:
                batch_peak_dist = peak_distance(outputs, labels)
                batch_peak_sharp = peak_sharpness(outputs)
                self.peak_dist_tracker.update(batch_loss=batch_peak_dist, batch_size=batch_size)
                self.peak_sharp_tracker.update(batch_loss=batch_peak_sharp, batch_size=batch_size)

            if self.enable_topk:
                batch_topk = topk_accuracy(outputs, labels, self.topk_values)
                self.topk_tracker.update(batch_topk, batch_size)

    def get_epoch_results(self) -> MetricsResult:
        """Get accumulated results for the epoch"""
        return MetricsResult(
            conf_loss=self.conf_loss_tracker.get_epoch_loss(),
            dice_score=self.dice_tracker.get_epoch_loss() if self.enable_dice else 0.0,
            peak_dist=self.peak_dist_tracker.get_epoch_loss() if self.enable_peak_metrics else 0.0,
            peak_sharp=self.peak_sharp_tracker.get_epoch_loss() if self.enable_peak_metrics else 0.0,
            topk_results=self.topk_tracker.get_epoch_results() if self.enable_topk else {k: 0.0 for k in self.topk_values}
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