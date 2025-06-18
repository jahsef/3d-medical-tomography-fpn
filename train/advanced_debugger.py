import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add model path
current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs.trivialmodel import TrivialModel
from patchtomodataset import PatchTomoDataset

def analyze_model_behavior():
    """Deep dive into what the model is actually learning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create minimal dataset
    train_id_list = ['tomo_d7475d']
    dataset = PatchTomoDataset(
        sigma=6,
        patch_index_path=Path('_patch_index.csv'),
        transform=None,
        tomo_id_list=train_id_list
    )
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)  # Smaller batch for analysis
    
    # Get one batch for detailed analysis
    patches, targets, _ = next(iter(loader))
    patches = patches.to(device)
    targets = targets.to(device)
    
    print("=== DETAILED BATCH ANALYSIS ===")
    print(f"Batch shape: {patches.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Analyze targets in detail
    print("\n=== TARGET ANALYSIS ===")
    for i in range(min(8, patches.shape[0])):
        target = targets[i, 0]  # Remove channel dim
        pos_pixels = (target > 0.0).sum().item()
        max_val = target.max().item()
        
        print(f"Sample {i}: {pos_pixels} positive pixels, max value: {max_val:.4f}")
        
        # Check if we have actual Gaussian blobs
        if pos_pixels > 0:
            # Find peak locations
            max_indices = torch.where(target == target.max())
            if len(max_indices[0]) > 0:
                peak_coords = [(max_indices[0][j].item(), max_indices[1][j].item(), max_indices[2][j].item()) 
                              for j in range(len(max_indices[0]))]
                print(f"  Peak locations: {peak_coords}")
    
    print("\n=== MODEL TRAINING ANALYSIS ===")
    
    # Test multiple models with different complexities
    models_to_test = [
        ("Trivial", TrivialModel()),
        ("Single Conv", nn.Sequential(nn.Conv3d(1, 1, 3, padding=1))),
        ("No Activation", nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.Conv3d(8, 1, 1)
        )),
        ("With ReLU", nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        ))
    ]
    
    results = {}
    
    for model_name, model in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        model = model.to(device)
        
        # Test different loss functions
        loss_functions = [
            ("BCE", nn.BCEWithLogitsLoss()),
            ("MSE", nn.MSELoss()),
            ("L1", nn.L1Loss())
        ]
        
        for loss_name, criterion in loss_functions:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            losses = []
            output_stats = []
            
            # Train for 20 steps
            for step in range(20):
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(patches)
                    
                    # For MSE and L1, no sigmoid needed
                    if loss_name == "BCE":
                        loss = criterion(outputs, targets)
                    else:
                        # Apply sigmoid for MSE/L1 to get 0-1 range
                        outputs_sigmoid = torch.sigmoid(outputs)
                        loss = criterion(outputs_sigmoid, targets)
                
                loss.backward()
                
                # Check gradients
                total_grad = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad += param.grad.norm().item() ** 2
                total_grad = total_grad ** 0.5
                
                optimizer.step()
                
                losses.append(loss.item())
                
                # Analyze outputs
                with torch.no_grad():
                    output_range = (outputs.min().item(), outputs.max().item())
                    output_mean = outputs.mean().item()
                    output_std = outputs.std().item()
                    
                    # Check spatial variance (are outputs just constant?)
                    spatial_var = outputs.var(dim=[2,3,4]).mean().item()
                    
                    output_stats.append({
                        'range': output_range,
                        'mean': output_mean,
                        'std': output_std,
                        'spatial_var': spatial_var,
                        'grad_norm': total_grad
                    })
            
            # Analyze results
            initial_loss = losses[0]
            final_loss = losses[-1]
            loss_improvement = initial_loss - final_loss
            
            print(f"  {loss_name}: {initial_loss:.4f} → {final_loss:.4f} (Δ: {loss_improvement:.4f})")
            
            # Check if model is learning anything meaningful
            final_stats = output_stats[-1]
            if final_stats['spatial_var'] < 1e-6:
                print(f"    WARNING: Model outputting near-constant values (spatial_var: {final_stats['spatial_var']:.2e})")
            
            if final_stats['grad_norm'] < 1e-6:
                print(f"    WARNING: Very small gradients (grad_norm: {final_stats['grad_norm']:.2e})")
            
            results[f"{model_name}_{loss_name}"] = {
                'losses': losses,
                'stats': output_stats,
                'improvement': loss_improvement
            }
    
    print("\n=== SUMMARY ===")
    print("Loss improvements by model/loss combination:")
    for key, result in results.items():
        print(f"{key}: {result['improvement']:.6f}")
    
    # Find best performing combination
    best_combo = max(results.items(), key=lambda x: x[1]['improvement'])
    print(f"\nBest performing: {best_combo[0]} with improvement of {best_combo[1]['improvement']:.6f}")
    
    return results

def test_data_scaling():
    """Test if the issue is with data preprocessing/scaling"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== DATA SCALING TEST ===")
    
    # Create dataset
    train_id_list = ['tomo_d7475d']
    dataset = PatchTomoDataset(
        sigma=6,
        patch_index_path=Path('_patch_index.csv'),
        transform=None,
        tomo_id_list=train_id_list
    )
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    patches, targets, _ = next(iter(loader))
    
    print(f"Input patches range: [{patches.min():.3f}, {patches.max():.3f}]")
    print(f"Input patches mean: {patches.mean():.3f}, std: {patches.std():.3f}")
    
    # Test different input normalizations
    normalizations = [
        ("Original", patches),
        ("Zero-Mean", patches - patches.mean()),
        ("StandardScale", (patches - patches.mean()) / (patches.std() + 1e-8)),
        ("MinMax", (patches - patches.min()) / (patches.max() - patches.min() + 1e-8))
    ]
    
    model = TrivialModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    for norm_name, norm_patches in normalizations:
        print(f"\n--- Testing {norm_name} normalization ---")
        print(f"Range: [{norm_patches.min():.3f}, {norm_patches.max():.3f}]")
        
        # Fresh model for each test
        model = TrivialModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        norm_patches = norm_patches.to(device)
        targets_device = targets.to(device)
        
        initial_loss = None
        for step in range(10):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(norm_patches)
                loss = criterion(outputs, targets_device)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        print(f"Loss: {initial_loss:.4f} → {final_loss:.4f} (Δ: {initial_loss - final_loss:.4f})")

def test_learning_rates():
    """Test if learning rate is the issue"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== LEARNING RATE TEST ===")
    
    # Create dataset
    train_id_list = ['tomo_d7475d']
    dataset = PatchTomoDataset(
        sigma=6,
        patch_index_path=Path('_patch_index.csv'),
        transform=None,
        tomo_id_list=train_id_list
    )
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    patches, targets, _ = next(iter(loader))
    patches = patches.to(device)
    targets = targets.to(device)
    
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    for lr in learning_rates:
        model = TrivialModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        for step in range(15):
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(patches)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"LR {lr:1.0e}: {losses[0]:.4f} → {losses[-1]:.4f} (Δ: {losses[0] - losses[-1]:.4f})")

if __name__ == "__main__":
    # Run comprehensive analysis
    print("Starting comprehensive debugging...")
    
    # Main analysis
    results = analyze_model_behavior()
    
    # Additional tests
    test_data_scaling()
    test_learning_rates()
    
    print("\n=== DEBUGGING COMPLETE ===")
    print("Check results above to identify the root cause.")