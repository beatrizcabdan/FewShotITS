import torch.nn as nn

class TinyMLEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=32, out_channels=8):
        super(TinyMLEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(out_channels, out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(out_channels*2, output_dim)


    def forward(self, x):  # x shape: (B, T, 1)
        x = x.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)
        x = self.encoder(x)     # -> (B, 16, 1)
        x = x.squeeze(-1)       # -> (B, 16)
        return self.fc(x)       # -> (B, output_dim)


class TinyCNNPlusEncoder(nn.Module):
    def __init__(self, input_channels=6, output_dim=64):
        super(TinyCNNPlusEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):  # x shape: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
        x = self.encoder(x)     # → (B, 32, 1)
        x = x.squeeze(-1)       # → (B, 32)
        return self.fc(x)       # → (B, output_dim)
