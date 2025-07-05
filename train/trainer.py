import torch
import time
import csv
import logging
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import utils
import gc
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
def log_metrics(epoch, filename="logs.csv", **kwargs):
    """Log training metrics to CSV file"""
    fieldnames = list(kwargs.keys())
    values = [round(float(v), 6) for v in kwargs.values()]

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['epoch'] + fieldnames)

        writer.writerow([epoch] + values)

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

class TopKTracker:
    """Utility class to track multiple top-k accuracies"""
    def __init__(self, k_values=[1, 5, 10, 50]):
        self.k_values = k_values
        self.trackers = {k: utils.LossTracker(is_mean_loss=True) for k in k_values}
    
    def update(self, topk_dict, batch_size):
        """Update all trackers with batch results"""
        for k in self.k_values:
            if k in topk_dict:
                self.trackers[k].update(batch_loss=topk_dict[k], batch_size=batch_size)

    def get_epoch_results(self):
        """Get epoch results for all k values"""
        return {k: tracker.get_epoch_loss() for k, tracker in self.trackers.items()}
        
class Trainer:

    def __init__(self, model, batches_per_step, train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn, device, save_dir='./models/', topk_values=[1, 5, 10, 50]):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        self.conf_loss_fn = conf_loss_fn
        self.topk_values = topk_values
        
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir

        # Send model to device
        self.model.to(self.device)
        
        # Per-step loss tracking for graphing
        self.step_losses = []
        self.step_count = 0
        
    def _compute_batch_loss(self, patches, labels):
        """Compute loss for a single batch"""
        patches = patches.to(self.device)
        labels = labels.to(self.device)
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = self.model(patches)
            conf_loss = self.conf_loss_fn(outputs, labels)

        return conf_loss, outputs
    
    def _train_one_epoch(self, epoch_index):
        """Train for one epoch"""
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)
        dice_tracker = utils.LossTracker(is_mean_loss=True)
        comp_tracker = utils.LossTracker(is_mean_loss=True)
        topk_tracker = TopKTracker(self.topk_values)

        self.model.train()
        total_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                            desc=f"Epoch {epoch_index}")
        progress_bar.ncols = 100

        for batch_idx, (patches, labels, global_coords) in progress_bar:
            
            # Zero gradients at start of accumulation cycle
            if batch_idx % self.batches_per_step == 0:
                self.optimizer.zero_grad()

            conf_loss, outputs = self._compute_batch_loss(patches, labels)
            conf_loss.backward()
            
            # Compute metrics for tracking
            with torch.no_grad():
                batch_dice = soft_dice_score(outputs, labels)
                dice_tracker.update(batch_loss=batch_dice, batch_size=patches.shape[0])
                
                batch_comp = comprehensive_heatmap_metric(outputs, labels)
                comp_tracker.update(batch_loss=batch_comp, batch_size=patches.shape[0])
                
                batch_topk = topk_accuracy(outputs, labels, self.topk_values)
                topk_tracker.update(batch_topk, patches.shape[0])
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % self.batches_per_step == 0:
                self.optimizer.step()
                # Log step loss for graphing
                self.step_losses.append(conf_loss.item())
                self.step_count += 1
                self.scheduler.step()
                    
            

            conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])

            # Update progress bar - only show loss
            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == total_batches:
                progress_bar.set_postfix(
                    loss=f"{conf_loss_tracker.get_epoch_loss():.6f}"
                )

        conf_loss = conf_loss_tracker.get_epoch_loss()
        dice_score = dice_tracker.get_epoch_loss()
        comp_score = comp_tracker.get_epoch_loss()
        topk_results = topk_tracker.get_epoch_results()
        return conf_loss, dice_score, comp_score, topk_results

    def _validate_one_epoch(self):
        """Validate for one epoch"""
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)
        dice_tracker = utils.LossTracker(is_mean_loss=True)
        comp_tracker = utils.LossTracker(is_mean_loss=True)
        topk_tracker = TopKTracker(self.topk_values)
        self.model.eval()

        with torch.no_grad():
            for patches, labels, global_coords in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                conf_loss, outputs = self._compute_batch_loss(patches, labels)
                conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])
                
                # Compute metrics
                batch_dice = soft_dice_score(outputs, labels)
                dice_tracker.update(batch_loss=batch_dice, batch_size=patches.shape[0])
                
                batch_comp = comprehensive_heatmap_metric(outputs, labels)
                comp_tracker.update(batch_loss=batch_comp, batch_size=patches.shape[0])
                
                batch_topk = topk_accuracy(outputs, labels, self.topk_values)
                topk_tracker.update(batch_topk, patches.shape[0])
                
        conf_loss = conf_loss_tracker.get_epoch_loss()
        dice_score = dice_tracker.get_epoch_loss()
        comp_score = comp_tracker.get_epoch_loss()
        topk_results = topk_tracker.get_epoch_results()
        return conf_loss, dice_score, comp_score, topk_results

    def train(self, epochs=50, save_period=0):
        """Main training loop"""
        csv_filepath = os.path.join(self.save_dir, "train_results.csv")
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=self.save_dir)
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, 'best.pt')
        optimizer_path = os.path.join(self.save_dir, 'best_optimizer.pt')
        
        for epoch in range(epochs):
            train_conf_loss, train_dice, train_comp, train_topk = self._train_one_epoch(epoch)
            val_conf_loss, val_dice, val_comp, val_topk = self._validate_one_epoch()
            
            print(f"Epoch {epoch}:")
            print(f"  Train Conf Loss: {train_conf_loss:.6f} | Train Dice: {train_dice:.4f} | Train Comp: {train_comp:.4f}")
            
            # Print train top-k results
            train_topk_str = " | ".join([f"Train Top-{k}: {acc:.3f}" for k, acc in train_topk.items()])
            print(f"  {train_topk_str}")
            
            print(f"  Val Conf Loss: {val_conf_loss:.6f} | Val Dice: {val_dice:.4f} | Val Comp: {val_comp:.4f}")
            
            # Print validation top-k results
            val_topk_str = " | ".join([f"Val Top-{k}: {acc:.3f}" for k, acc in val_topk.items()])
            print(f"  {val_topk_str}")
            
            print("-" * 60)

            # Prepare metrics for CSV logging
            log_data = {
                'train_conf_loss': train_conf_loss,
                'train_dice': train_dice,
                'train_comp': train_comp,
                'val_conf_loss': val_conf_loss,
                'val_dice': val_dice,
                'val_comp': val_comp,
            }
            
            # Add top-k metrics to log data
            for k, acc in train_topk.items():
                log_data[f'train_top{k}'] = acc
            for k, acc in val_topk.items():
                log_data[f'val_top{k}'] = acc

            # Log metrics to CSV
            log_metrics(epoch, filename=csv_filepath, **log_data)

            # Save best model
            if val_conf_loss < best_val_loss or val_conf_loss == 0:
                best_val_loss = val_conf_loss
                logging.info(f"Validation loss improved. Saving model to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(self.optimizer.state_dict(), optimizer_path)

            # Periodic saves
            if save_period != 0 and (epoch + 1) % save_period == 0:
                periodic_save_path = os.path.join(self.save_dir, f"epoch{epoch}.pt")
                periodic_optimizer_save_path = os.path.join(self.save_dir, f"epoch{epoch}_optimizer.pt")
                torch.save(self.model.state_dict(), periodic_save_path)
                torch.save(self.optimizer.state_dict(), periodic_optimizer_save_path)
        
        # Jank per-step loss graphing
        self._plot_step_losses()

    def _plot_step_losses(self):
        """Plot per-step training losses"""
        if not self.step_losses:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.step_losses, alpha=0.7, linewidth=0.8)
        plt.title('Training Loss per Step')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'step_losses.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Step loss plot saved to {plot_path}")

if __name__ == '__main__':
    # Test code would go here
    pass