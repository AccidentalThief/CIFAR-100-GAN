import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from torch.utils.data import DataLoader
from models.classifier import ClassifierNet
from models.discriminator import Discriminator
from utils.dataset import get_dataset, get_transforms
from utils.train import train, get_checkpoint_path
from utils.test import test

# ---- CONFIGURATION ----
# For MNIST, you can use either "classifier" or "discriminator" mode.
mode = "classifier"        # "classifier" for digit classification, "discriminator" for GAN/discriminator training
dataset_name = "mnist"     # "mnist" for handwritten digits

# ---- DATASET SETUP ----
augment = True if mode == "classifier" else False
transform = get_transforms(dataset_name, augment=augment)
trainset, num_classes, input_channels, input_size = get_dataset(dataset_name, train=True, transform=transform)
testset, _, _, _ = get_dataset(dataset_name, train=False, transform=get_transforms(dataset_name, augment=False))
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# ---- MODEL, LOSS, OPTIMIZER ----
if mode == "classifier":
    net = ClassifierNet(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    model_name = dataset_name
    label_fn = None
    metric_fn = None  # Default: accuracy
elif mode == "discriminator":
    net = Discriminator(input_channels=input_channels, input_size=input_size)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    model_name = f"{dataset_name}_disc"
    # For real-only training (no generator yet)
    def label_fn(inputs, outputs, epoch, batch_idx):
        return torch.ones(inputs.size(0), 1)
    def metric_fn(outputs, labels):
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return (preds == labels).float().mean().item()
else:
    raise ValueError("Unknown mode. Use 'classifier' or 'discriminator'.")

# ---- TRAINING ----
# For best MNIST results, train for 20-30 epochs for classifier, 10-20 for discriminator.
train(
    net, optimizer, criterion, trainloader,
    epochs=30 if mode == "classifier" else 15,
    patience=5, resume_best=False,
    model_name=model_name,
    label_fn=label_fn,
    metric_fn=metric_fn
)

# ---- TESTING ----
checkpoint_path = get_checkpoint_path(model_name, "best")
try:
    net.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model weights from {checkpoint_path}")
except Exception as e:
    print(f"Could not load checkpoint: {e}")

test(
    net, testloader, criterion=criterion,
    metric_fn=metric_fn,
    label_fn=label_fn
)

# ----
# HOW TO USE:
# 1. Set 'mode' to "classifier" or "discriminator" at the top.
# 2. Set 'dataset_name' to "mnist".
# 3. Run: python main.py
# 4. Checkpoints are saved in the 'checkpoints/' folder.
# 5. To test, make sure the best checkpoint exists or has been trained.
# 6. For GANs, add generator logic and fake image training as needed.