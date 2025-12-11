import torch

# Test FP16
x_fp16 = torch.tensor([1.0 - 1e-3], dtype=torch.float16)
print(f"FP16: 1 - 1e-3 = {x_fp16.item()}")
print(f"FP16: 1 - x = {(1.0 - x_fp16).item()}")

# Test BFloat16
x_bf16 = torch.tensor([1.0 - 1e-2], dtype=torch.bfloat16)
print(f"BF16: 1 - 1e-2 = {x_bf16.item()}")
print(f"BF16: 1 - x = {(1.0 - x_bf16).item()}")

# Test FP32
x_fp32 = torch.tensor([1.0 - 1e-6], dtype=torch.float32)
print(f"FP32: 1 - 1e-6 = {x_fp32.item()}")
print(f"FP32: 1 - x = {(1.0 - x_fp32).item()}")