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
    compute_epoch_metrics,
    save_csv
)
from grapher import TrainingGrapher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:

    def __init__(self, detector, batches_per_step, train_loader, val_loader, optimizer, scheduler,
                conf_loss_fn, device, run_dir, topk_percent_values):

        self.detector = detector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batches_per_step = batches_per_step
        self.conf_loss_fn = conf_loss_fn
        self.topk_percent_values = topk_percent_values

        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Preallocate buffers for epoch metrics
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset) if val_loader else 0
        heatmap_shape = train_loader.dataset[0][1].shape
        self.train_pred_buffer = torch.zeros((train_size, *heatmap_shape), dtype=torch.float32)
        self.train_gt_buffer = torch.zeros((train_size, *heatmap_shape), dtype=torch.float32)
        self.val_pred_buffer = torch.zeros((val_size, *heatmap_shape), dtype=torch.float32) if val_size > 0 else None
        self.val_gt_buffer = torch.zeros((val_size, *heatmap_shape), dtype=torch.float32) if val_size > 0 else None

        # Setup directory structure
        self.run_dir = Path(run_dir)
        self.weights_dir = self.run_dir / 'weights'
        self.logs_dir = self.run_dir / 'logs'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.grapher = TrainingGrapher(self.run_dir)
        self.detector.to(self.device)
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
            outputs = self.detector(patches)

        conf_loss = self.conf_loss_fn(outputs, labels)

        return conf_loss, outputs
    
    def _train_one_epoch(self, epoch_index):
        """Train for one epoch, returns MetricsResult"""
        self.detector.train()
        total_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_batches,
                            desc=f"Epoch {epoch_index}")
        progress_bar.ncols = 100

        loss_tracker = LossTracker(is_mean_loss=True)
        buffer_idx = 0

        for batch_idx, (patches, labels) in progress_bar:
            if batch_idx % self.batches_per_step == 0:
                self.optimizer.zero_grad()

            conf_loss, outputs = self._compute_batch_loss(patches, labels)
            self.scaler.scale(conf_loss).backward()

            batch_size = patches.shape[0]
            loss_tracker.update(conf_loss.item(), batch_size)

            # Fill buffers
            self.train_pred_buffer[buffer_idx:buffer_idx + batch_size] = outputs.detach().cpu()
            self.train_gt_buffer[buffer_idx:buffer_idx + batch_size] = labels.cpu()
            buffer_idx += batch_size

            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm=6.7)

            if (batch_idx + 1) % self.batches_per_step == 0 or (batch_idx + 1) == total_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step_losses.append(conf_loss.item())
                self.step_lrs.append(self.scheduler.get_last_lr()[0])
                self.step_count += 1
                self.scheduler.step()

            progress_bar.set_postfix(loss=f"{loss_tracker.get_epoch_loss():.3e}")

        return compute_epoch_metrics(
            self.train_pred_buffer, self.train_gt_buffer,
            loss_tracker.get_epoch_loss(), self.topk_percent_values
        )

    def _validate_one_epoch(self):
        """Validate for one epoch, returns MetricsResult"""
        if self.val_pred_buffer is None:
            return None
        
        self.detector.eval()
        loss_tracker = LossTracker(is_mean_loss=True)
        buffer_idx = 0

        with torch.no_grad():
            for patches, labels in tqdm(self.val_loader, desc="Validating", leave=True, ncols=100):
                conf_loss, outputs = self._compute_batch_loss(patches, labels)
                batch_size = patches.shape[0]
                loss_tracker.update(conf_loss.item(), batch_size)

                self.val_pred_buffer[buffer_idx:buffer_idx + batch_size] = outputs.cpu()
                self.val_gt_buffer[buffer_idx:buffer_idx + batch_size] = labels.cpu()
                buffer_idx += batch_size

        return compute_epoch_metrics(
            self.val_pred_buffer, self.val_gt_buffer,
            loss_tracker.get_epoch_loss(), self.topk_percent_values
        )

    def train(self, epochs, save_period):
        """Main training loop"""
        csv_filepath = self.logs_dir / "train_results.csv"
        
        # Clean start - delete existing CSV for fresh experiment
        csv_filepath.unlink(missing_ok=True)
        
        logger = utils.Logger()
        logger.log_training_settings(save_dir=str(self.run_dir))
        
        best_val_loss = float('inf')
        best_model_path = self.weights_dir / 'best.pt'

        for epoch in range(epochs):
            train = self._train_one_epoch(epoch)
            val = self._validate_one_epoch()

            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train.conf_loss:.6f} | Dice: {train.dice_score:.4f} | PeakDist: {train.peak_dist:.4f} | PeakSharp: {train.peak_sharp:.4f}")
            train_topk_str = " | ".join([f"Top-{k}%: {acc:.3f}" for k, acc in train.topk_results.items()])
            print(f"  {train_topk_str}")

            if val:
                print(f"  Val Loss: {val.conf_loss:.6f} | Dice: {val.dice_score:.4f} | PeakDist: {val.peak_dist:.4f} | PeakSharp: {val.peak_sharp:.4f}")
                val_topk_str = " | ".join([f"Top-{k}%: {acc:.3f}" for k, acc in val.topk_results.items()])
                print(f"  {val_topk_str}")

            print("-" * 60)

            log_data = {
                'train_conf_loss': train.conf_loss,
                'train_dice': train.dice_score,
                'train_peak_dist': train.peak_dist,
                'train_peak_sharp': train.peak_sharp,
                'val_conf_loss': val.conf_loss if val else 0.0,
                'val_dice': val.dice_score if val else 0.0,
                'val_peak_dist': val.peak_dist if val else 0.0,
                'val_peak_sharp': val.peak_sharp if val else 0.0,
            }
            for k, acc in train.topk_results.items():
                log_data[f'train_top{k}'] = acc
            if val:
                for k, acc in val.topk_results.items():
                    log_data[f'val_top{k}'] = acc
            else:
                for k in train.topk_results.keys():
                    log_data[f'val_top{k}'] = 0.0

            save_csv(epoch, filename=str(csv_filepath), **log_data)

            val_loss = val.conf_loss if val else 0.0
            if val_loss < best_val_loss or val_loss == 0:
                best_val_loss = val_loss
                logging.info(f"Saving checkpoint to {best_model_path}")
                self.detector.save_checkpoint(best_model_path, self.optimizer)


            # Periodic saves
            if save_period != 0 and (epoch + 1) % save_period == 0:
                periodic_save_path = self.weights_dir / f"epoch{epoch}.pt"
                self.detector.save_checkpoint(periodic_save_path, self.optimizer)
                
            #create graphs after every epoch lol
            self.grapher.plot_training_progress(
                step_losses=self.step_losses,
                step_lrs=self.step_lrs,
                csv_path=csv_filepath
            )

if __name__ == '__main__':
    # Test code would go here
    pass