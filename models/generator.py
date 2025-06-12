import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, output_channels, img_size, alpha=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_channels = output_channels
        self.img_size = img_size
        self.alpha = alpha

        self.label_proj = nn.Sequential(
            nn.Linear(n_classes, n_classes),
            nn.LeakyReLU(alpha, inplace=True)
        )

        input_dim = latent_dim + n_classes
        self.init_size = img_size // 4  # 4x upsampling
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.LeakyReLU(alpha, inplace=True)
        )

        # Project label features for deep conditioning
        self.deep_label_proj = nn.Sequential(
            nn.Linear(n_classes, 128 * self.init_size * self.init_size),
            nn.LeakyReLU(alpha, inplace=True)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, img_size//2, img_size//2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha, inplace=True),
            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),  # (B, output_channels, img_size, img_size)
            nn.Sigmoid()
        )

    def forward(self, labels, batch_size=None, device=None):
        if batch_size is None:
            batch_size = labels.size(0)
        if device is None:
            device = labels.device

        # One-hot encode labels
        labels_onehot = F.one_hot(labels, num_classes=self.n_classes).float().to(device)
        label_input = self.label_proj(labels_onehot)

        z = torch.randn(batch_size, self.latent_dim, device=device)
        x = torch.cat((z, label_input), dim=1)
        out = self.fc(x)

        # Deep label conditioning: add projected label features to feature map
        deep_label = self.deep_label_proj(labels_onehot)
        out = out + deep_label  # shape: (batch, 128*init_size*init_size)

        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img