import numpy as np
import matplotlib.pyplot as plt


def combined_loss(target, pred, alpha=2.0, beta=4.0):
    pred = np.clip(pred, 1e-3, 1 - 1e-3)

    focal_weighting = (abs(target - pred) ** alpha)
    pos_weight = (target ** beta) * focal_weighting * np.log(pred)

    high_conf_fp_weighting = (pred ** alpha)
    neg_weight = ((1 - target) ** beta) * high_conf_fp_weighting * np.log(1 - pred)

    return -(pos_weight + neg_weight)


# Alpha, beta combinations to visualize
PARAMS = [
    (0.5, 0.25),
    (0.5, 2),
    (1, 1),
    (2.0, 0.25),
    (2.0, 2),
]

# Grid resolution
RESOLUTION = 128

# Create meshgrid
targets = np.linspace(0, 1, RESOLUTION)
preds = np.linspace(0, 1, RESOLUTION)
T, P = np.meshgrid(targets, preds, indexing='ij')

# Create figure with dynamic width
n_plots = len(PARAMS)
fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

if n_plots == 1:
    axes = [axes]

for idx, (alpha, beta) in enumerate(PARAMS):
    loss = combined_loss(T, P, alpha=alpha, beta=beta)

    # Normalize to 0-1
    max_loss = loss.max()
    min_loss = loss.min()
    loss_normalized = (loss - min_loss) / (max_loss - min_loss) if max_loss > min_loss else loss

    im = axes[idx].imshow(
        loss_normalized,
        origin='lower',
        extent=[0, 1, 0, 1],
        aspect='auto',
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    axes[idx].set_xlabel('Pred')
    axes[idx].set_ylabel('Target')
    axes[idx].set_title(f'α={alpha}, β={beta}\nmax={max_loss:.3f}')
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

plt.suptitle('Combined Loss Heatmaps', fontsize=14)
plt.tight_layout()
plt.show()
