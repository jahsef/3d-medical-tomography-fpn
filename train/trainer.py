import torch
import time
import csv
import logging
from tqdm import tqdm
import os
from pathlib import Path
import utils
import gc

from metrics import (
    LossTracker, 
    TopKTracker, 
    soft_dice_score, 
    comprehensive_heatmap_metric, 
    topk_accuracy
)
from grapher import TrainingGrapher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_metrics(epoch, filename="train_results.csv", **kwargs):
    """Log training metrics to CSV file"""
    fieldnames = list(kwargs.keys())
    values = [round(float(v), 6) for v in kwargs.values()]

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['epoch'] + fieldnames)

        writer.writerow([epoch] + values)
        
class Trainer:

    def __init__(self, model, batches_per_step, train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn, device, run_dir='./models/', topk_values=[1, 5, 10, 50]):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        self.conf_loss_fn = conf_loss_fn
        self.topk_values = topk_values
        
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup directory structure
        self.run_dir = Path(run_dir)
        self.weights_dir = self.run_dir / 'weights'
        self.logs_dir = self.run_dir / 'logs'
        
        # Create directories
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize grapher
        self.grapher = TrainingGrapher(self.run_dir)

        # Send model to device
        self.model.to(self.device)
        
        # Initialize GradScaler for AMP
        self.scaler = torch.amp.GradScaler("cuda")
        
        # Per-step loss tracking for graphing
        self.step_losses = []
        self.step_count = 0
        
    def _compute_batch_loss(self, patches, labels):
        """Compute loss for a single batch"""
        patches = patches.to(self.device)
        labels = labels.to(self.device)
        
        with torch.amp.autocast("cuda"):
            outputs = self.model(patches)
            
        conf_loss = self.conf_loss_fn(outputs, labels)

        return conf_loss, outputs
    
    def _train_one_epoch(self, epoch_index):
        """Train for one epoch"""
        conf_loss_tracker = LossTracker(is_mean_loss=True)
        # dice_tracker = LossTracker(is_mean_loss=True)  # DISABLED FOR SPEED TEST
        # comp_tracker = LossTracker(is_mean_loss=True)  # DISABLED FOR SPEED TEST
        # topk_tracker = TopKTracker(self.topk_values)   # DISABLED FOR SPEED TEST

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
            self.scaler.scale(conf_loss).backward()
            
            # Compute metrics for tracking
            # with torch.no_grad():
                # batch_dice = soft_dice_score(outputs, labels)
                # dice_tracker.update(batch_loss=batch_dice, batch_size=patches.shape[0])
                
                # batch_comp = comprehensive_heatmap_metric(outputs, labels)
                # comp_tracker.update(batch_loss=batch_comp, batch_size=patches.shape[0])
                
                # batch_topk = topk_accuracy(outputs, labels, self.topk_values)
                # topk_tracker.update(batch_topk, patches.shape[0])
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % self.batches_per_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
        # Return dummy values for disabled metrics
        dice_score = 0.0  # dice_tracker.get_epoch_loss()
        comp_score = 0.0  # comp_tracker.get_epoch_loss() 
        topk_results = {k: 0.0 for k in self.topk_values}  # topk_tracker.get_epoch_results()
        return conf_loss, dice_score, comp_score, topk_results

    def _validate_one_epoch(self):
        """Validate for one epoch"""
        conf_loss_tracker = LossTracker(is_mean_loss=True)
        dice_tracker = LossTracker(is_mean_loss=True)
        comp_tracker = LossTracker(is_mean_loss=True)
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
        csv_filepath = self.logs_dir / "train_results.csv"
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=str(self.run_dir))
        
        best_val_loss = float('inf')
        best_model_path = self.weights_dir / 'best.pt'
        optimizer_path = self.weights_dir / 'best_optimizer.pt'
        
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
            log_metrics(epoch, filename=str(csv_filepath), **log_data)

            # Save best model
            if val_conf_loss < best_val_loss or val_conf_loss == 0:
                best_val_loss = val_conf_loss
                logging.info(f"Validation loss improved. Saving model to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)
                torch.save(self.optimizer.state_dict(), optimizer_path)

            # Periodic saves
            if save_period != 0 and (epoch + 1) % save_period == 0:
                periodic_save_path = self.weights_dir / f"epoch{epoch}.pt"
                periodic_optimizer_save_path = self.weights_dir / f"epoch{epoch}_optimizer.pt"
                torch.save(self.model.state_dict(), periodic_save_path)
                torch.save(self.optimizer.state_dict(), periodic_optimizer_save_path)
        
        # Create comprehensive training plots
        self.grapher.plot_training_progress(
            step_losses=self.step_losses,
            csv_path=csv_filepath
        )

if __name__ == '__main__':
    # Test code would go here
    pass