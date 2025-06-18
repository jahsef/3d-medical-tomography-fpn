import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add model path
current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
from model_defs.trivialmodel import TrivialModel
from patchtomodataset import PatchTomoDataset

def debug_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create minimal dataset - single tomo
    train_id_list = ['tomo_d7475d']
    dataset = PatchTomoDataset(
        sigma=6,
        patch_index_path=Path('_patch_index.csv'),
        transform=None,
        tomo_id_list=train_id_list
    )
    
    # Small batch for debugging
    loader = DataLoader(dataset, batch_size=64 , shuffle=True)
    
    # Simple model
    model = TrivialModel().to(device)
    pos_weight=torch.tensor([3]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    print("Starting debug training...")
    
    epochs = 24
    loss_list = []
    for epoch_idx in range(epochs):
        for batch_idx, (patches, targets, _) in enumerate(loader):
                
            patches = patches.to(device)
            targets = targets.to(device)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(patches)
                loss = criterion(outputs, targets)
                
                # Check outputs and loss
                print(f"Epoch: {epoch_idx}, Batch {batch_idx}:")
                print(f"  Loss: {loss.item():.6f}")
                loss_list.append(round(loss.item(),4))
                print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
                print(f"  Positive pixels: {(targets > 0.0).sum().item()}/{targets.numel()}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print(f"  Gradient norm: {total_grad_norm:.6f}")
            
            optimizer.step()
            print()
    print(f'final loss outputs:\n {loss_list}')
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.plot(loss_list)
    plt.show()

if __name__ == "__main__":
    debug_training()