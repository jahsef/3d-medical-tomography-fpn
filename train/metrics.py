import torch
import torch.nn.functional as F


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


def comprehensive_heatmap_metric(pred, target, distance_threshold=60):
    """
    Fast and simple heatmap evaluation metric.
    
    Args:
        pred: Predicted heatmap [B, C, H, W, D] or [B, H, W, D]
        target: Target heatmap (same shape as pred)
        distance_threshold: Distance threshold in voxels
    
    Returns:
        score: Combined metric score [0, 1] where higher is better
    """
    pred = torch.sigmoid(pred)
    batch_size = pred.shape[0]
    
    # Move to CPU first
    pred = pred.cpu()
    target = target.cpu()
    
    # Flatten spatial dimensions for vectorized operations
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    # 1. Correlation (vectorized across batch)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = pred_centered.norm(dim=1)
    target_std = target_centered.norm(dim=1)
    
    # Avoid division by zero and NaN
    correlation = numerator / (pred_std * target_std + 1e-8)
    correlation = torch.clamp(correlation, -1.0, 1.0)
    correlation = torch.where(torch.isnan(correlation), torch.zeros_like(correlation), correlation)
    
    # 2. Peak distance (simplified - just flat index distance)
    pred_peaks = pred_flat.argmax(dim=1)
    target_peaks = target_flat.argmax(dim=1)
    
    # Normalize distance by total voxels for rough spatial score
    total_voxels = pred_flat.shape[1]
    peak_distance = torch.abs(pred_peaks - target_peaks).float()
    spatial_score = torch.clamp(1.0 - peak_distance / distance_threshold, 0.0, 1.0)
    
    # 3. Peak sharpness (confidence measure)
    pred_max = pred_flat.max(dim=1)[0]
    pred_mean_val = pred_flat.mean(dim=1)
    
    # Avoid division by zero and NaN
    sharpness = pred_max / (pred_mean_val + 1e-8)
    confidence = torch.clamp(sharpness / 10.0, 0.0, 1.0)
    confidence = torch.where(torch.isnan(confidence), torch.zeros_like(confidence), confidence)
    
    # 4. Combine metrics (vectorized)
    score = 0.5 * correlation + 0.3 * spatial_score + 0.2 * confidence
    score = torch.clamp(score, 0.0, 1.0)
    score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    
    return score.mean().item()


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