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
        
class Trainer:

    def __init__(self, model, batches_per_step, train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn, device, save_dir='./models/'):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        self.conf_loss_fn = conf_loss_fn
        
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

        return conf_loss 
    
    def _train_one_epoch(self, epoch_index):
        """Train for one epoch"""
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)

        self.model.train()
        total_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                            desc=f"Epoch {epoch_index}")
        progress_bar.ncols = 100

        for batch_idx, (patches, labels, global_coords) in progress_bar:
            
            # Zero gradients at start of accumulation cycle
            if batch_idx % self.batches_per_step == 0:
                self.optimizer.zero_grad()

            conf_loss = self._compute_batch_loss(patches, labels)
            conf_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % self.batches_per_step == 0:
                self.optimizer.step()
                # Log step loss for graphing
                self.step_losses.append(conf_loss.item())
                self.step_count += 1
                    
            self.scheduler.step()

            conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])

            # Update progress bar
            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == total_batches:
                progress_bar.set_postfix(
                    loss=f"conf loss: {conf_loss_tracker.get_epoch_loss():.6f}"
                )

        conf_loss = conf_loss_tracker.get_epoch_loss()
        return conf_loss

    def _validate_one_epoch(self):
        """Validate for one epoch"""
        conf_loss_tracker = utils.LossTracker(is_mean_loss=True)
        self.model.eval()

        with torch.no_grad():
            for patches, labels, global_coords in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                conf_loss = self._compute_batch_loss(patches, labels)
                conf_loss_tracker.update(batch_loss=conf_loss.item(), batch_size=patches.shape[0])
                
        conf_loss = conf_loss_tracker.get_epoch_loss()
        return conf_loss

    def train(self, epochs=50, save_period=0):
        """Main training loop"""
        csv_filepath = os.path.join(self.save_dir, "train_results.csv")
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=self.save_dir)
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, 'best.pt')
        optimizer_path = os.path.join(self.save_dir, 'best_optimizer.pt')
        
        for epoch in range(epochs):
            train_conf_loss = self._train_one_epoch(epoch)
            val_conf_loss = self._validate_one_epoch()
            
            print(f"Epoch {epoch}:")
            print(f"  Train Conf Loss: {train_conf_loss:.6f} | Val Conf Loss: {val_conf_loss:.6f}")
            print("-" * 60)

            # Log metrics to CSV
            log_metrics(epoch,
                train_conf_loss=train_conf_loss,
                val_conf_loss=val_conf_loss,
                filename=csv_filepath
            )

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