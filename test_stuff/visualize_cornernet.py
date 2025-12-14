import numpy as np
import matplotlib.pyplot as plt

def visualize_cornernet_weighting(abs_error=0.1, alpha=2.0, beta=4.0):
    """
    Visualize CornerNet focal loss WEIGHTING (modulation) across different pred_prob values
    while holding absolute error constant.
    
    Shows only the focal weighting terms, not the total loss.
    
    Args:
        abs_error: Constant absolute error |pred - target|
        alpha: focal power for hard examples
        beta: gaussian penalty reduction power
    """
    # Generate pred_prob range
    pred_probs = np.linspace(0.001, 0.999, 1000)
    
    # Calculate corresponding targets for constant abs_error
    targets_above = np.clip(pred_probs - abs_error, 0, 1)  # pred > target
    targets_below = np.clip(pred_probs + abs_error, 0, 1)  # pred < target
    
    # Calculate WEIGHTING for both cases
    weights_above = []
    weights_below = []
    
    for pred, target_above, target_below in zip(pred_probs, targets_above, targets_below):
        # Case 1: pred > target (overestimation)
        if target_above >= 0.90:  # Target is peak
            # Positive loss weighting: (1-p)^α
            weight_above = (1 - pred) ** alpha
        else:  # Target is background/falloff
            # Negative loss weighting: (1-y)^β * p^α
            weight_above = ((1 - target_above) ** beta) * (pred ** alpha)
        
        # Case 2: pred < target (underestimation)
        if target_below >= 0.90:  # Target is peak
            # Positive loss weighting: (1-p)^α
            weight_below = (1 - pred) ** alpha
        else:  # Target is background/falloff
            # Negative loss weighting: (1-y)^β * p^α
            weight_below = ((1 - target_below) ** beta) * (pred ** alpha)
        
        weights_above.append(weight_above)
        weights_below.append(weight_below)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(pred_probs, weights_above, label=f'Overestimate (pred > target)', linewidth=2)
    ax.plot(pred_probs, weights_below, label=f'Underestimate (pred < target)', linewidth=2)
    
    # Mark the peak region (target >= 0.90)
    ax.axvline(x=0.90, color='red', linestyle='--', alpha=0.5, label='Peak threshold (0.90)')
    ax.axvline(x=0.90 - abs_error, color='orange', linestyle='--', alpha=0.5, label=f'Target=0.90 (overestimate)')
    ax.axvline(x=0.90 + abs_error, color='green', linestyle='--', alpha=0.5, label=f'Target=0.90 (underestimate)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Loss Weighting (Focal Modulation)', fontsize=12)
    ax.set_title(f'CornerNet Loss Weighting vs Pred Prob (abs_error={abs_error}, α={alpha}, β={beta})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Visualize weighting with different parameters
    visualize_cornernet_weighting(abs_error=0.1, alpha=2.0, beta=4.0)
    
    # Compare different beta values (affects gaussian falloff penalty)
    # with SAME Y-AXIS SCALE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # First pass: calculate max weight to set common scale
    max_weight = 0
    beta_values = [2.0, 4.0, 8.0]
    
    for beta_val in beta_values:
        pred_probs = np.linspace(0.001, 0.999, 1000)
        targets_above = np.clip(pred_probs - 0.1, 0, 1)
        
        for pred, target in zip(pred_probs, targets_above):
            if target >= 0.90:
                weight = (1 - pred) ** 2.0
            else:
                weight = ((1 - target) ** beta_val) * (pred ** 2.0)
            max_weight = max(max_weight, weight)
    
    # Second pass: plot with common scale
    for idx, beta_val in enumerate(beta_values):
        pred_probs = np.linspace(0.001, 0.999, 1000)
        targets_above = np.clip(pred_probs - 0.1, 0, 1)
        
        weights = []
        for pred, target in zip(pred_probs, targets_above):
            if target >= 0.90:
                weight = (1 - pred) ** 2.0
            else:
                weight = ((1 - target) ** beta_val) * (pred ** 2.0)
            weights.append(weight)
        
        axes[idx].plot(pred_probs, weights, linewidth=2)
        axes[idx].axvline(x=0.90, color='red', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel('Predicted Probability')
        axes[idx].set_ylabel('Loss Weighting')
        axes[idx].set_title(f'β={beta_val}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, max_weight * 1.05)  # Common scale with 5% padding
    
    plt.suptitle('Effect of β on Gaussian Falloff Penalty (abs_error=0.1, α=2.0)', fontsize=14)
    plt.tight_layout()
    plt.show()