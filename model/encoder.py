import torch
import torch.nn as nn

class TinyCNNEncoder(nn.Module):
    def __init__(self, input_channels=1, sequence_length=30, output_dim=32):
        super(TinyCNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(16, output_dim)

    def forward(self, x):  # x shape: (B, T, 1)
        x = x.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)
        x = self.encoder(x)     # -> (B, 16, 1)
        x = x.squeeze(-1)       # -> (B, 16)
        return self.fc(x)       # -> (B, output_dim)