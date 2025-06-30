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

# Dataset class for treating each file as a full sequence sample
class MultiCSVSensorDataset(Dataset):
    def __init__(self, folder_path, include_keywords=None):
        self.X = []
        self.y = []
        self.include_keywords = include_keywords if include_keywords else []

        all_files = glob(os.path.join(folder_path, "*.csv"))

        if self.include_keywords:
            file_paths = [
                f for f in all_files
                if any(keyword.lower() in os.path.basename(f).lower() for keyword in self.include_keywords)
            ]
            print(f"Filtered {len(file_paths)} files with keywords {self.include_keywords}")
        else:
            file_paths = all_files

        label_names = set()
        for fpath in file_paths:
            label_str = os.path.basename(fpath).split('_')[0].lower()
            label_names.add(label_str)

        label_names = sorted(list(label_names))
        self.label_map = {name: idx for idx, name in enumerate(label_names)}
        print("Label mapping:", self.label_map)

        expected_shape = None
        used, skipped = 0, 0

        for fpath in file_paths:
            label_str = os.path.basename(fpath).split('_')[0].lower()
            if label_str not in self.label_map:
                skipped += 1
                continue

            label = self.label_map[label_str]

            try:
                df = pd.read_csv(fpath)
                if df.empty or df.shape[1] == 0:
                    skipped += 1
                    continue

                sample = df.values.flatten()

                if expected_shape is None:
                    expected_shape = sample.shape
                    print(f"🧭 Expected flattened shape: {expected_shape}")
                elif sample.shape != expected_shape:
                    print(f"❌ Shape mismatch in {fpath}")
                    skipped += 1
                    continue

                self.X.append(sample)
                self.y.append(label)
                used += 1

            except Exception as e:
                print(f"💥 Error reading {fpath}: {e}")
                skipped += 1

        if not self.X:
            raise RuntimeError(f"No valid CSV files found in {folder_path}")

        # Normalize
        scaler = StandardScaler()
        self.X = torch.tensor(scaler.fit_transform(np.stack(self.X)), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        print(f"✅ Loaded {used} files, ❌ Skipped {skipped} files")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model with two hidden layers
class SensorClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(SensorClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Training function with validation
def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1:02d}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Loss = {total_loss:.4f}")

# Main
if __name__ == '__main__':
    dataset = MultiCSVSensorDataset(
        folder_path='../data/data_csv/',
        include_keywords=['lefturn', 'rightturn', 'lane', 'ra']
    )

    print("Input shape:", dataset.X.shape)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(dataset.X, dataset.y, test_size=0.02, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = dataset.X.shape[1]
    model = SensorClassifier(input_dim=input_dim, output_dim=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer, epochs=50)
