import torch
import numpy as np

def check_pt_file(pt_file_path):
    """Simple check of a .pt file."""
    data = torch.load(pt_file_path)
    
    patch = data['patch']
    labels = data['labels'] 
    coords = data['global_coords']
    
    print(f"File: {pt_file_path}")
    print(f"Patch shape: {patch.shape}, dtype: {patch.dtype}")
    print(f"Patch stats - min: {patch.min():.4f}, max: {patch.max():.4f}")
    print(f"Patch stats - mean: {patch.mean():.4f}, std: {patch.std():.4f}")
    print(f"Labels shape: {labels.shape}")
    print(f"Coords: {coords}")

# Usage:
check_pt_file(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\patch_pt_data\tomo_00e047\patch_0_144_400.pt')