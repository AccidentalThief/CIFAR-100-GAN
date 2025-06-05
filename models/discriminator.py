import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, input_size=32):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (input_channels) x 32 x 32
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 512 x 2 x 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512 * (input_size // 16) * (input_size // 16), 1)
            # No sigmoid: use BCEWithLogitsLoss for stability
        )

    def forward(self, x):
        return self.main(x)