import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes=10, input_channels=1, input_size=28):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_size
        self.label_emb = nn.Embedding(n_classes, input_size * input_size)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        ds_size = input_size // 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1)
        )

    def forward(self, img, labels):
        # Embed labels to (batch, H*W), then reshape to (batch, 1, H, W)
        label_map = self.label_emb(labels).view(labels.size(0), 1, self.input_size, self.input_size)
        d_in = torch.cat((img, label_map), 1)
        out = self.conv(d_in)
        validity = self.adv_layer(out)
        return validity