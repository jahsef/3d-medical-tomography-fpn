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
        
    def plot_training_progress(self, step_losses=None, csv_path=None):
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
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Step-wise loss with epoch boundaries
        if step_losses:
            self._plot_step_losses(axes[0, 0], step_losses, epoch_data)
        
        # Plot 2: Epoch-level loss comparison
        if epoch_data is not None:
            self._plot_epoch_losses(axes[0, 1], epoch_data)
            
            # Plot 3: Metrics comparison (Dice, Comp score)
            self._plot_metrics(axes[1, 0], epoch_data)
            
            # Plot 4: Top-K accuracies
            self._plot_topk_accuracies(axes[1, 1], epoch_data)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.logs_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training progress plot saved to {plot_path}")
    
    def _plot_step_losses(self, ax, step_losses, epoch_data):
        """Plot step-wise losses with epoch boundary markers"""
        ax.plot(step_losses, alpha=0.7, linewidth=0.8, color='blue', label='Step Loss')
        ax.set_title('Training Loss per Step')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Add epoch boundary lines if we have epoch data
        if epoch_data is not None and 'epoch' in epoch_data.columns:
            steps_per_epoch = len(step_losses) // len(epoch_data) if len(epoch_data) > 0 else 0
            if steps_per_epoch > 0:
                for epoch in epoch_data['epoch']:
                    step_num = epoch * steps_per_epoch
                    if step_num < len(step_losses):
                        ax.axvline(x=step_num, color='red', linestyle='--', alpha=0.6)
                
                # Add epoch labels on the right y-axis
                ax2 = ax.twinx()
                ax2.set_ylabel('Epoch', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Set epoch ticks
                epoch_steps = [e * steps_per_epoch for e in epoch_data['epoch'] if e * steps_per_epoch < len(step_losses)]
                epoch_labels = [str(int(e)) for e in epoch_data['epoch'] if e * steps_per_epoch < len(step_losses)]
                
                if epoch_steps:
                    ax2.set_yticks(epoch_steps)
                    ax2.set_yticklabels(epoch_labels)
                    ax2.set_ylim(ax.get_ylim())
        
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
        """Plot additional metrics like Dice score and comprehensive metric"""
        epochs = epoch_data['epoch']
        
        # Use twin y-axes for different metrics
        ax2 = ax.twinx()
        
        # Plot Dice score
        if 'train_dice' in epoch_data.columns:
            line1 = ax.plot(epochs, epoch_data['train_dice'], 'o-', label='Train Dice', color='green')
        if 'val_dice' in epoch_data.columns:
            line2 = ax.plot(epochs, epoch_data['val_dice'], 's-', label='Val Dice', color='lightgreen')
        
        # Plot comprehensive metric on second y-axis
        if 'train_comp' in epoch_data.columns:
            line3 = ax2.plot(epochs, epoch_data['train_comp'], '^-', label='Train Comp', color='orange')
        if 'val_comp' in epoch_data.columns:
            line4 = ax2.plot(epochs, epoch_data['val_comp'], 'v-', label='Val Comp', color='red')
        
        ax.set_title('Training Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score', color='green')
        ax2.set_ylabel('Comprehensive Score', color='orange')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
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