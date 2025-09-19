import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from model_defs._OLD_FPN import MotorIdentifier as FPNModel
from model_defs.motoridentifier import MotorIdentifier as ResNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create models
fpn_model = FPNModel(dropout_p=0.15, norm_type='gn').to(device)
resnet_model = ResNetModel(initial_features=8, dropout_p=0.15, norm_type='gn').to(device)

# Print params
print("=== FPN Model ===")
fpn_model.print_params()
print("\n=== ResNet Model ===")
resnet_model.print_params()

# Create test input - same as training (160, 288, 288)
test_input = torch.randn(1, 1, 160, 288, 288, device=device)

print(f"\nTest input shape: {test_input.shape}")

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = fpn_model(test_input)
        _ = resnet_model(test_input)

# Time forward pass only
n_runs = 20

print("\n=== Forward Pass Timing ===")

# FPN timing
torch.cuda.synchronize()
start_time = time.time()
for _ in range(n_runs):
    with torch.no_grad():
        fpn_output = fpn_model(test_input)
torch.cuda.synchronize()
fpn_forward_time = (time.time() - start_time) / n_runs

# ResNet timing
torch.cuda.synchronize()
start_time = time.time()
for _ in range(n_runs):
    with torch.no_grad():
        resnet_output = resnet_model(test_input)
torch.cuda.synchronize()
resnet_forward_time = (time.time() - start_time) / n_runs

print(f"FPN forward pass: {fpn_forward_time*1000:.2f}ms")
print(f"ResNet forward pass: {resnet_forward_time*1000:.2f}ms")
print(f"ResNet is {resnet_forward_time/fpn_forward_time:.2f}x slower" if resnet_forward_time > fpn_forward_time else f"FPN is {fpn_forward_time/resnet_forward_time:.2f}x slower")

# Time forward + backward pass
print("\n=== Forward + Backward Pass Timing ===")

fpn_model.train()
resnet_model.train()

# FPN timing
torch.cuda.synchronize()
start_time = time.time()
for _ in range(n_runs):
    fpn_output = fpn_model(test_input)
    loss = fpn_output.sum()
    loss.backward()
    fpn_model.zero_grad()
torch.cuda.synchronize()
fpn_full_time = (time.time() - start_time) / n_runs

# ResNet timing
torch.cuda.synchronize()
start_time = time.time()
for _ in range(n_runs):
    resnet_output = resnet_model(test_input)
    loss = resnet_output.sum()
    loss.backward()
    resnet_model.zero_grad()
torch.cuda.synchronize()
resnet_full_time = (time.time() - start_time) / n_runs

print(f"FPN forward+backward: {fpn_full_time*1000:.2f}ms")
print(f"ResNet forward+backward: {resnet_full_time*1000:.2f}ms")
print(f"ResNet is {resnet_full_time/fpn_full_time:.2f}x slower" if resnet_full_time > fpn_full_time else f"FPN is {resnet_full_time/fpn_full_time:.2f}x slower")

print(f"\nOutput shapes:")
print(f"FPN output: {fpn_output.shape}")
print(f"ResNet output: {resnet_output.shape}")