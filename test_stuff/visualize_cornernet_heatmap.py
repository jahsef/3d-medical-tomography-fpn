import numpy as np
import matplotlib.pyplot as plt

def visualize_cornernet_heatmap(alpha=2.0, beta=4.0, vmin=None, vmax=None):
    """
    Visualize CornerNet loss weighting as a 2D heatmap.
    
    Args:
        alpha: focal power for hard examples
        beta: gaussian penalty reduction power
        vmin: minimum value for colorbar (None = auto)
        vmax: maximum value for colorbar (None = auto)
    """
    # Create grid
    pred_probs = np.linspace(0.001, 0.999, 500)
    targets = np.linspace(0.0, 1.0, 500)
    
    pred_grid, target_grid = np.meshgrid(pred_probs, targets)
    
    # Calculate weighting for each (pred, target) pair
    weights = np.zeros_like(pred_grid)
    
    for i in range(pred_grid.shape[0]):
        for j in range(pred_grid.shape[1]):
            pred = pred_grid[i, j]
            target = target_grid[i, j]
            
            if target >= 0.90:  # Peak region
                # Positive loss weighting: (1-p)^α
                weight = (1 - pred) ** alpha
            else:  # Background/falloff
                # Negative loss weighting: (1-y)^β * p^α
                weight = ((1 - target) ** beta) * (pred ** alpha)
            
            weights[i, j] = weight
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(weights, extent=[0, 1, 0, 1], origin='lower', 
                   aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loss Weighting', fontsize=12)
    
    # Mark the peak threshold
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Peak threshold (0.90)')
    
    # Add diagonal line (perfect prediction)
    ax.plot([0, 1], [0, 1], 'w--', linewidth=2, alpha=0.7, label='Perfect prediction')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Ground Truth (Target)', fontsize=12)
    ax.set_title(f'CornerNet Loss Weighting Heatmap (α={alpha}, β={beta})', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Try different parameters
    visualize_cornernet_heatmap(alpha=2.0, beta=0.1)
    visualize_cornernet_heatmap(alpha=2.0, beta=1)
    visualize_cornernet_heatmap(alpha=2.0, beta=4)
