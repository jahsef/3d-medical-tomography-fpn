import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging


class TrainingGrapher:
    """Comprehensive training visualization with step-based plots and epoch overlays"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_progress(self, step_losses=None, step_lrs=None, csv_path=None):
        """
        Create comprehensive training plots with step losses and epoch metrics
        
        Args:
            step_losses: List of per-step loss values
            csv_path: Path to epoch-level metrics CSV file
        """
        if csv_path is None:
            csv_path = self.logs_dir / 'train_results.csv'
        
        # Load epoch-level metrics if available
        epoch_data = None
        if csv_path.exists():
            try:
                epoch_data = pd.read_csv(csv_path)
            except Exception as e:
                logging.warning(f"Could not load epoch data from {csv_path}: {e}")
        
        # Create figure with subplots (2x3 grid)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Row 1: Step losses | Epoch losses | Learning Rate  
        if step_losses:
            self._plot_step_losses(axes[0, 0], step_losses, epoch_data)
            
        if epoch_data is not None:
            self._plot_epoch_losses(axes[0, 1], epoch_data)
            
        # Plot learning rate
        if step_lrs:
            self._plot_learning_rate(axes[0, 2], step_lrs, epoch_data)
        else:
            axes[0, 2].text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Learning Rate per Step')
        
        # Row 2: Metrics | Top-K accuracies | Empty
        if epoch_data is not None:
            self._plot_metrics(axes[1, 0], epoch_data)
            self._plot_topk_accuracies(axes[1, 1], epoch_data)
            
        # Empty plot for future use
        axes[1, 2].text(0.5, 0.5, 'Reserved for\nfuture metrics', ha='center', va='center', 
                        transform=axes[1, 2].transAxes, fontsize=12, alpha=0.5)
        axes[1, 2].set_title('Future Expansion')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.logs_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training progress plot saved to {plot_path}")
    
    def _plot_step_losses(self, ax, step_losses, epoch_data):
        """Plot step-wise losses with EMA smoothing"""
        # Plot noisy step losses with low opacity
        ax.plot(step_losses, alpha=0.6, linewidth=0.5, color='lightblue', label='Raw Step Loss')
        
        # Calculate EMA (Exponential Moving Average)
        alpha = 0.1  # EMA smoothing factor (lower = smoother)
        ema = [step_losses[0]]  # Initialize with first value
        for loss in step_losses[1:]:
            ema.append(alpha * loss + (1 - alpha) * ema[-1])
        
        # Plot EMA with full opacity
        ax.plot(ema, alpha=1.0, linewidth=2.0, color='blue', label='EMA Step Loss')
        steps_per_epoch = len(step_losses) // len(epoch_data) if len(epoch_data) > 0 else 0
        ax.set_title(f'Training Loss per Step (steps/epoch: {steps_per_epoch})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

        ax.legend()
    
    def _plot_learning_rate(self, ax, step_lrs, epoch_data):
        """Plot learning rate per step with epoch boundary markers"""
        if not step_lrs:
            ax.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate per Step')
            return
            
        ax.plot(step_lrs, alpha=1.0, linewidth=1.5, color='green', label='Learning Rate')
        steps_per_epoch = len(step_lrs) // len(epoch_data) if len(epoch_data) > 0 else 0
        ax.set_title(f'Learning Rate per Step (steps/epoch: {steps_per_epoch})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        
        
        ax.legend()
    
    def _plot_epoch_losses(self, ax, epoch_data):
        """Plot epoch-level training and validation losses"""
        epochs = epoch_data['epoch']
        
        if 'train_conf_loss' in epoch_data.columns:
            ax.plot(epochs, epoch_data['train_conf_loss'], 'o-', label='Train Loss', color='blue')
        
        if 'val_conf_loss' in epoch_data.columns:
            ax.plot(epochs, epoch_data['val_conf_loss'], 's-', label='Val Loss', color='red')
        
        ax.set_title('Training vs Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_metrics(self, ax, epoch_data):
        """Plot additional metrics: Dice, Peak Distance, Peak Sharpness"""
        epochs = epoch_data['epoch']

        ax.set_ylim(0, 1)

        # Plot Dice score
        if 'train_dice' in epoch_data.columns:
            ax.plot(epochs, epoch_data['train_dice'], 'o-', label='Train Dice', color='green')
        if 'val_dice' in epoch_data.columns:
            ax.plot(epochs, epoch_data['val_dice'], 's-', label='Val Dice', color='lightgreen')

        # Plot Peak Distance
        if 'train_peak_dist' in epoch_data.columns:
            ax.plot(epochs, epoch_data['train_peak_dist'], '^-', label='Train PeakDist', color='blue')
        if 'val_peak_dist' in epoch_data.columns:
            ax.plot(epochs, epoch_data['val_peak_dist'], 'v-', label='Val PeakDist', color='lightblue')

        # Plot Peak Sharpness
        if 'train_peak_sharp' in epoch_data.columns:
            ax.plot(epochs, epoch_data['train_peak_sharp'], 'D-', label='Train PeakSharp', color='purple')
        if 'val_peak_sharp' in epoch_data.columns:
            ax.plot(epochs, epoch_data['val_peak_sharp'], 'd-', label='Val PeakSharp', color='violet')

        ax.set_title('Training Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    def _plot_topk_accuracies(self, ax, epoch_data):
        """Plot Top-K accuracies"""
        epochs = epoch_data['epoch']
        
        # Plot train top-k metrics
        train_topk_cols = [col for col in epoch_data.columns if col.startswith('train_top')]
        val_topk_cols = [col for col in epoch_data.columns if col.startswith('val_top')]
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, col in enumerate(train_topk_cols):
            k_value = col.replace('train_top', '')
            color = colors[i % len(colors)]
            ax.plot(epochs, epoch_data[col], 'o-', label=f'Train Top-{k_value}', 
                   color=color, alpha=0.8)
        
        for i, col in enumerate(val_topk_cols):
            k_value = col.replace('val_top', '')
            color = colors[i % len(colors)]
            ax.plot(epochs, epoch_data[col], 's--', label=f'Val Top-{k_value}', 
                   color=color, alpha=0.6)
        
        ax.set_title('Top-K Accuracies')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def save_step_losses_only(self, step_losses):
        """Simple fallback method to save just step losses"""
        if not step_losses:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(step_losses, alpha=0.7, linewidth=0.8)
        plt.title('Training Loss per Step')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plot_path = self.logs_dir / 'step_losses.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Step loss plot saved to {plot_path}")