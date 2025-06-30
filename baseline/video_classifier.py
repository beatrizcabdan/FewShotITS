import cv2
from pathlib import Path
import os
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.video import r3d_18

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []

        # Load all MP4s and their labels
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                if fname.endswith(".mp4"):
                    self.samples.append((os.path.join(label_path, fname), label))

        # Label encoding
        # self.label_to_idx = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir)))} #todo: problem with labels out of range
        label_names = sorted([label for label in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, label))])
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_name = self.samples[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')  # returns (T, H, W, C)

        # Reduce to N frames (e.g., 16 evenly spaced)
        total_frames = video.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.frames_per_clip).long()
        video = video[indices]

        # Permute to (T, C, H, W) and normalize
        video = video.permute(0, 3, 1, 2).float() / 255.0

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        return video.permute(1, 0, 2, 3), self.label_to_idx[label_name]  # (C, T, H, W), label

def convert_all_frames_to_mp4(input_root, output_root, fps=15):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for label_dir in input_root.iterdir():
        if label_dir.is_dir():
            for video_dir in label_dir.iterdir():
                if video_dir.is_dir():
                    output_label_dir = output_root / label_dir.name
                    output_label_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_label_dir / f"{video_dir.name}.mp4"

                    try:
                        frames_to_mp4(str(video_dir), str(output_path), fps=fps)
                        print(f"✔ Converted: {video_dir} -> {output_path}")
                    except Exception as e:
                        print(f"✖ Failed to convert {video_dir}: {e}")

def frames_to_mp4(frame_folder, output_path, fps=15):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in folder.")

    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, file))
        out.write(frame)
    out.release()

# convert_all_frames_to_mp4("../data/data_img", "../data/data_mp4", fps=10)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.CenterCrop(112),
])

train_dataset = VideoDataset("../data/data_mp4", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
print("Labels in dataset:", set(label for _, label in train_dataset))

# Configuration
BATCH_SIZE = 4
NUM_CLASSES = 4
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load 3D ResNet-18
model = r3d_18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for videos, labels in train_loader:
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

print("Training complete.")
