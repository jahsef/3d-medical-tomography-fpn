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
    peak_distance,
    peak_sharpness,
    topk_accuracy,
    MetricsEvaluator,
    MetricsResult,
    save_csv
)
from grapher import TrainingGrapher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:

    def __init__(self, model, batches_per_step, train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn, device, run_dir='./models/', topk_values=[1, 5, 10, 50],
                enable_train_metrics=True, enable_val_metrics=True):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        self.conf_loss_fn = conf_loss_fn
        self.topk_values = topk_values
        
        # Initialize metrics evaluators
        self.train_metrics = MetricsEvaluator(
            topk_values=topk_values,
            enable_dice=enable_train_metrics,
            enable_peak_metrics=enable_train_metrics,
            enable_topk=enable_train_metrics
        )

        self.val_metrics = MetricsEvaluator(
            topk_values=topk_values,
            enable_dice=enable_val_metrics,
            enable_peak_metrics=enable_val_metrics,
            enable_topk=enable_val_metrics
        )
        
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        
        # Per-step tracking for graphing
        self.step_losses = []
        self.step_lrs = []
        self.step_count = 0
        
    def _compute_batch_loss(self, patches, labels):
        """Compute loss for a single batch"""
        patches = patches.to(self.device)
        labels = labels.to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.model(patches)
            
        conf_loss = self.conf_loss_fn(outputs, labels)

        return conf_loss, outputs
    
    def _train_one_epoch(self, epoch_index):
        """Train for one epoch"""
        # Reset metrics for new epoch
        self.train_metrics.reset_trackers()
        
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
            
            # if batch_idx % 1 == 0:
            #     stem_grad_norm = self.model.stem.weight.grad.norm().item() if self.model.stem.weight.grad is not None else 0
            #     print(f"Stem grad norm: {stem_grad_norm:.4f}")
            # Update metrics with batch results
            self.train_metrics.update_batch(
                outputs=outputs,
                labels=labels, 
                conf_loss=conf_loss.item(),
                batch_size=patches.shape[0]
            )
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=6.7)

            # Step optimizer after accumulating gradients (or at end of epoch)
            if (batch_idx + 1) % self.batches_per_step == 0 or (batch_idx + 1) == total_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Log step data for graphing
                self.step_losses.append(conf_loss.item())
                self.step_lrs.append(self.scheduler.get_last_lr()[0])
                self.step_count += 1
                self.scheduler.step()
                
            # Update progress bar - only show loss
            if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == total_batches:
                current_metrics = self.train_metrics.get_epoch_results()
                progress_bar.set_postfix(
                    loss=f"{current_metrics.conf_loss:.6f}"
                )

        # Get final epoch results
        metrics_result = self.train_metrics.get_epoch_results()
        return metrics_result.conf_loss, metrics_result.dice_score, metrics_result.peak_dist, metrics_result.peak_sharp, metrics_result.topk_results

    def _validate_one_epoch(self):
        """Validate for one epoch"""
        # Reset metrics for new epoch
        self.val_metrics.reset_trackers()
        
        self.model.eval()

        with torch.no_grad():
            for patches, labels, global_coords in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                conf_loss, outputs = self._compute_batch_loss(patches, labels)
                
                # Update metrics with batch results
                self.val_metrics.update_batch(
                    outputs=outputs,
                    labels=labels,
                    conf_loss=conf_loss.item(),
                    batch_size=patches.shape[0]
                )
                
        # Get final epoch results
        metrics_result = self.val_metrics.get_epoch_results()
        return metrics_result.conf_loss, metrics_result.dice_score, metrics_result.peak_dist, metrics_result.peak_sharp, metrics_result.topk_results

    def train(self, epochs=50, save_period=0):
        """Main training loop"""
        csv_filepath = self.logs_dir / "train_results.csv"
        
        # Clean start - delete existing CSV for fresh experiment
        csv_filepath.unlink(missing_ok=True)
        
        logger = utils.Logger()
        logger.log_training_settings(model=self.model, save_dir=str(self.run_dir))
        
        best_val_loss = float('inf')
        best_model_path = self.weights_dir / 'best.pt'
        optimizer_path = self.weights_dir / 'best_optimizer.pt'
        
        for epoch in range(epochs):
            train_conf_loss, train_dice, train_peak_dist, train_peak_sharp, train_topk = self._train_one_epoch(epoch)
            val_conf_loss, val_dice, val_peak_dist, val_peak_sharp, val_topk = self._validate_one_epoch()

            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_conf_loss:.6f} | Dice: {train_dice:.4f} | PeakDist: {train_peak_dist:.4f} | PeakSharp: {train_peak_sharp:.4f}")

            # Print train top-k results
            train_topk_str = " | ".join([f"Top-{k}: {acc:.3f}" for k, acc in train_topk.items()])
            print(f"  {train_topk_str}")

            print(f"  Val Loss: {val_conf_loss:.6f} | Dice: {val_dice:.4f} | PeakDist: {val_peak_dist:.4f} | PeakSharp: {val_peak_sharp:.4f}")

            # Print validation top-k results
            val_topk_str = " | ".join([f"Top-{k}: {acc:.3f}" for k, acc in val_topk.items()])
            print(f"  {val_topk_str}")

            print("-" * 60)

            # Prepare metrics for CSV logging
            log_data = {
                'train_conf_loss': train_conf_loss,
                'train_dice': train_dice,
                'train_peak_dist': train_peak_dist,
                'train_peak_sharp': train_peak_sharp,
                'val_conf_loss': val_conf_loss,
                'val_dice': val_dice,
                'val_peak_dist': val_peak_dist,
                'val_peak_sharp': val_peak_sharp,
            }
            
            # Add top-k metrics to log data
            for k, acc in train_topk.items():
                log_data[f'train_top{k}'] = acc
            for k, acc in val_topk.items():
                log_data[f'val_top{k}'] = acc

            # Log metrics to CSV
            save_csv(epoch, filename=str(csv_filepath), **log_data)

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
        
            #create graphs after every epoch lol
            self.grapher.plot_training_progress(
                step_losses=self.step_losses,
                step_lrs=self.step_lrs,
                csv_path=csv_filepath
            )

if __name__ == '__main__':
    # Test code would go here
    pass