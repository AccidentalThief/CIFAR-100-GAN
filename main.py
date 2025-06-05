import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from torch.utils.data import DataLoader
from models.classifier import ClassifierNet
from models.discriminator import Discriminator
from models.generator import Generator  # Make sure this exists!
from utils.dataset import get_dataset, get_transforms
from utils.train import train, get_checkpoint_path, train_cgan
from utils.test import test, test_cgan_single_digit, test_cgan_generator
import torch.nn as nn
import random
import numpy as np

seed = int.from_bytes(os.urandom(4), 'little')
print(f"[Main] Using random seed: {seed}")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # ---- CONFIGURATION ----
    # mode: "classifier", "discriminator", or "gan"
    mode = "gan"              # "classifier", "discriminator", or "gan"
    dataset_name = "mnist"    # "mnist", "cifar10", or "cifar100"
    latent_dim = 100

    print(f"[Main] Starting in mode: {mode}")
    print(f"[Main] Preparing dataset: {dataset_name}")

    # ---- DATASET SETUP ----
    augment = True if mode == "classifier" else False
    transform = get_transforms(dataset_name, augment=augment)
    trainset, num_classes, input_channels, input_size = get_dataset(dataset_name, train=True, transform=transform)
    testset, _, _, _ = get_dataset(dataset_name, train=False, transform=get_transforms(dataset_name, augment=False))
    print(f"[Main] Creating DataLoaders...")
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

    # ---- MODEL, LOSS, OPTIMIZER ----
    print(f"[Main] Setting up model and optimizer for mode: {mode}")
    if mode == "classifier":
        net = ClassifierNet(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        model_name = dataset_name
        label_fn = None
        metric_fn = None  # Default: accuracy
        generator = None
        optimizer_G = None
        gan_mode = False
        print(f"[Main] Training classifier...")
    elif mode == "discriminator":
        net = Discriminator(n_classes=num_classes, input_channels=input_channels, input_size=input_size)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        model_name = f"{dataset_name}_disc"

        # Custom label_fn for real images (all real)
        def label_fn(inputs, outputs, epoch, batch_idx):
            return torch.ones(inputs.size(0), 1)

        # Custom metric_fn for real/fake accuracy
        def metric_fn(outputs, labels):
            preds = (torch.sigmoid(outputs) > 0.5).float()
            return (preds == labels).float().mean().item()

        # Use the train utility for training the discriminator
        print(f"[Main] Training discriminator...")
        train(
            net,
            optimizer,
            criterion,
            trainloader,
            epochs=15,
            patience=5,
            resume_best=False,
            model_name=model_name,
            label_fn=label_fn,
            metric_fn=metric_fn
        )
    elif mode == "gan":
        # GAN mode: set up both generator and discriminator
        net = Discriminator(n_classes=num_classes, input_channels=input_channels, input_size=input_size)
        generator = Generator(latent_dim=latent_dim, n_classes=num_classes, output_channels=input_channels, img_size=input_size)

        # Apply weight initialization
        generator.apply(weights_init_normal)
        net.apply(weights_init_normal)

        model_name = f"cgan_{dataset_name}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        criterion = torch.nn.BCEWithLogitsLoss()
        def label_fn(inputs, outputs, epoch, batch_idx):
            return torch.ones(inputs.size(0), 1)
        def metric_fn(outputs, labels):
            preds = (torch.sigmoid(outputs) > 0.5).float()
            return (preds == labels).float().mean().item()

        print(f"[Main] Training cGAN...")
        
        train_cgan(
            generator,
            net,
            trainloader,
            latent_dim=latent_dim,
            n_classes=num_classes,
            device=device,
            epochs=100,
            lr=0.000001,
            model_name=model_name,
            resume_best=False,
            gen_rate=2,
            lr_ratio=5
        )
        
    else:
        raise ValueError("Unknown mode. Use 'classifier', 'discriminator', or 'gan'.")

    print(f"[Main] Training complete. Starting evaluation...")

    # ---- TESTING ----
    checkpoint_path = get_checkpoint_path(model_name, "disc_best")
    try:
        net.load_state_dict(torch.load(checkpoint_path))
        print(f"[Main] Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"[Main] Could not load checkpoint: {e}")

    test(
        net, testloader, criterion=criterion,
        metric_fn=metric_fn,
        label_fn=label_fn
    )

    # ---- GAN GENERATOR TESTING ----
    if mode == "gan":
        gen_ckpt = get_checkpoint_path(model_name, "gen_best")
        try:
            generator.load_state_dict(torch.load(gen_ckpt, map_location='cpu'))
            print(f"[Main] Loaded generator weights from {gen_ckpt}")
        except Exception as e:
            print(f"[Main] Could not load generator checkpoint: {e}")

        print(f"[Main] Generating sample images from generator...")
        test_cgan_generator(generator, latent_dim=100, nrow=10, ncol=10, device='cpu')

# ----
# HOW TO USE:
# 1. Set 'mode' to "classifier", "discriminator", or "gan" at the top.
# 2. Set 'dataset_name' to "mnist", "cifar10", or "cifar100".
# 3. Make sure models/generator.py exists for GAN mode.
# 4. Run: python main.py
# 5. Checkpoints are saved in the 'checkpoints/' folder.
# 6. For GANs, generated images and generator checkpoints can be added as needed.