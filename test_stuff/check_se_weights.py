import torch

EPOCH0 = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn_comparison\pc_fpn_cornernet4\weights\epoch0.pt'
BEST = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn_comparison\pc_fpn_cornernet4\weights\best.pt'

print("Loading checkpoints...")
e0_ckpt = torch.load(EPOCH0, map_location='cpu')['weights']
best_ckpt = torch.load(BEST, map_location='cpu')['weights']

# SE block keys in head: head.2.fc.0.weight, head.2.fc.2.weight, head.2.fc.2.bias
se_keys = [
    ("parallel_neck.1.fc.0.weight", "compression conv"),
    ("parallel_neck.1.fc.2.weight", "expansion conv"),
    ("parallel_neck.1.fc.2.bias", "expansion bias"),
]

print("\n" + "="*60)
print("COMPARING EPOCH 0 vs BEST - SE FC LAYERS")
print("="*60)
print(e0_ckpt.keys())
for key, desc in se_keys:
    e0 = e0_ckpt[key]
    best = best_ckpt[key]
    diff = (best - e0).abs()
    
    print(f"\n{key} ({desc}):")
    print(f"  Shape: {e0.shape}")
    print(f"  Epoch0 - mean: {e0.mean().item():.6f}, std: {e0.std().item():.6f}, min: {e0.min().item():.6f}, max: {e0.max().item():.6f}")
    print(f"  Best   - mean: {best.mean().item():.6f}, std: {best.std().item():.6f}, min: {best.min().item():.6f}, max: {best.max().item():.6f}")
    print(f"  Diff   - mean: {diff.mean().item():.6f}, max: {diff.max().item():.6f}")

    if diff.max().item() < 1e-5:
        print(f"  >>> FROZEN (max diff < 1e-5)")
    else:
        print(f"  >>> CHANGED")
