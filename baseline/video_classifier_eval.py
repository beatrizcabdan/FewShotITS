from torchvision.models.video import r3d_18
from torchinfo import summary  # pip install torchinfo
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sensor_classifier import SensorClassifier, MultiCSVSensorDataset

def get_model_param_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 4  # float32 = 4 bytes
    print(f"Total parameters: {total_params}")
    print(f"Model size: {total_bytes / 1024:.2f} KB")
    return total_bytes

def estimate_activation_memory(model, input_size):
    dummy_input = torch.randn(1, input_size)
    hooks = []
    total_activations = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            total_activations.append(output.numel() * output.element_size())  # bytes

    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.ReLU)):
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    total = sum(total_activations)
    print(f"Estimated activation memory: {total / 1024:.2f} KB")
    return total


print("BASELINE VIDEO CLASSIFIER")
model = r3d_18()
summary(model, input_size=(1, 3, 16, 480, 720), col_names=["input_size", "output_size", "num_params"])

print("BASELINE SENSOR CLASSIFIER")
dataset = MultiCSVSensorDataset(folder_path='../data/data_csv/', include_keywords=['lefturn', 'rightturn', 'lane', 'ra'])
X_train, X_val, y_train, y_val = train_test_split(dataset.X, dataset.y, test_size=0.02, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

input_dim = dataset.X.shape[1]
model = SensorClassifier(input_dim, hidden_dim=128, output_dim=4)
param_bytes = get_model_param_size(model)
activation_bytes = estimate_activation_memory(model, input_dim)

total_ram = param_bytes + activation_bytes
print(f"Estimated total RAM required: {total_ram / 1024:.2f} KB")
