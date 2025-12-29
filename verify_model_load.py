import torch
import sys
import os

model_path = "model/best_model.pth"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found")
    sys.exit(1)

print("Attempting to load as TorchScript...")
try:
    model = torch.jit.load(model_path)
    print("Success: Loaded as TorchScript")
    sys.exit(0)
except Exception as e:
    print(f"Failed TorchScript load: {e}")

print("Attempting to load as PyTorch Pickle...")
try:
    # Use map_location='cpu' to avoid GPU issues if saved on GPU
    data = torch.load(model_path, map_location='cpu')
    print(f"Success: Loaded via torch.load. Type: {type(data)}")
    if isinstance(data, dict):
        print("Detailed keys: ", data.keys())
except Exception as e:
    print(f"Failed torch.load: {e}")
