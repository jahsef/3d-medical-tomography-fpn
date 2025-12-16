"""
Visualize SE (Squeeze-and-Excitation) block channel attention weights in the pc_fpn head.
"""

import sys
from pathlib import Path
current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
# from models.fpn_comparison.parallel_fpn_cornernet9.model_defs.motor_detector import MotorDetector
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
import random
from model_defs.motor_detector import MotorDetector
from tqdm import tqdm


DEFAULT_CHECKPOINT = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn_comparison\pc_fpn_cornernet4\weights\best.pt'
master_tomo_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\processed\patch_pt_data')

N_SAMPLES = 6


def main():
    checkpoint_path = DEFAULT_CHECKPOINT
    device = torch.device('cpu')

    print(f'Loading model from: {checkpoint_path}')
    detector, _ = MotorDetector.load_checkpoint(checkpoint_path, dropout_p=0, drop_path_p=0)
    detector.eval()
    model = detector.model.to(device)

    # Get SE block
    se_block = model.parallel_neck[1]
    n_channels = se_block.fc[0].in_channels
    print(f'SE block: {n_channels} channels')
    
    
    tomo_id_list = [dir.name for dir in master_tomo_path.iterdir() if dir.is_dir()]
    train_id_list, val_id_list = train_test_split(tomo_id_list, train_size=0.25, random_state=42)
    val_id_list = train_id_list
    

    all_patches = []
    for tomo_id in val_id_list:
        tomo_dir = master_tomo_path / tomo_id
        patches = list(tomo_dir.glob('*.pt'))
        all_patches.extend(patches)

    print(f'Found {len(all_patches)} patches in {len(val_id_list)} tomograms')

    sampled_patches = random.sample(all_patches, min(N_SAMPLES, len(all_patches)))
    print(f'Sampled {len(sampled_patches)} patches')

    all_weights = []

    def hook(module, input, output):
        x = input[0]
        x_clamped = torch.clamp(x, min=-1e3, max=1e3)
        pooled = module.pool(x_clamped)
        scale = torch.sigmoid(torch.clamp(module.fc(pooled), min=1e-3, max=1000))
        scale = torch.clamp(scale, min=1e-3, max=1.0)
        all_weights.append(scale.squeeze().detach().cpu().numpy())

    handle = se_block.register_forward_hook(hook)

    print('Running forward passes...')
    with torch.no_grad():
        for i, patch_path in tqdm(enumerate(sampled_patches)):
            patch = torch.load(patch_path)['patch'].unsqueeze(0).float().to(device) #already chw, => bchw
            _ = model(patch)

    handle.remove()

    all_weights = np.array(all_weights)  # (N_SAMPLES, n_channels)
    print(f'\nCollected weights shape: {all_weights.shape}')
    print(f'First sample weights (should be 0-1): {all_weights[0][:10]}...')  # first 10 channels
    print(f'Raw all_weights min={all_weights.min()}, max={all_weights.max()}')

    # Overall stats (across all weights)
    print('\n=== OVERALL STATS (all weights flattened) ===')
    print(f'  mean:   {all_weights.mean():.4f}')
    print(f'  median: {np.median(all_weights):.4f}')
    print(f'  std:    {all_weights.std():.4f}')
    print(f'  min:    {all_weights.min():.4f}')
    print(f'  max:    {all_weights.max():.4f}')

    # Inter-sample stats (mean weight per sample, then stats on those)
    sample_means = all_weights.mean(axis=1)  # (N_SAMPLES,)
    print('\n=== INTER-SAMPLE STATS (mean weight per sample) ===')
    print(f'  mean:   {sample_means.mean():.4f}')
    print(f'  median: {np.median(sample_means):.4f}')
    print(f'  std:    {sample_means.std():.4f}')
    print(f'  min:    {sample_means.min():.4f}')
    print(f'  max:    {sample_means.max():.4f}')

    # Per-channel stats (mean across samples for each channel)
    channel_means = all_weights.mean(axis=0)  # (n_channels,)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top plot: per-channel mean weights
    ax = axes[0]
    x = np.arange(n_channels)
    colors = plt.cm.RdYlGn(channel_means)
    ax.bar(x, channel_means, color=colors, edgecolor='black', linewidth=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=channel_means.mean(), color='blue', linestyle='-', alpha=0.7, label=f'mean: {channel_means.mean():.3f}')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title(f'Per-Channel Mean Attention ({n_channels} channels, {len(sampled_patches)} samples)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    
    # Bottom plot: per-channel std across samples (which channels vary most?)
    ax = axes[1]
    channel_stds = all_weights.std(axis=0)  # (n_channels,)
    print(f'channel_stds min={channel_stds.min():.6f}, max={channel_stds.max():.6f}')
    ax.bar(x, channel_stds, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axhline(y=channel_stds.mean(), color='blue', linestyle='-', alpha=0.7, label=f'mean std: {channel_stds.mean():.6f}')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Std Across Samples')
    ax.set_title(f'Per-Channel Attention Variance ({len(sampled_patches)} samples)')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
