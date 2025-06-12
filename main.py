import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from torch.utils.data import DataLoader
from models.classifier import ClassifierNet
from models.discriminator import Discriminator
from models.generator import Generator
from utils.dataset import get_dataset, get_transforms
from utils.train import train, get_checkpoint_path, train_cgan
from utils.test import test, test_cgan_generator
import torch.nn as nn
import random
import numpy as np

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
    # mode: "classifier" or "gan"
    mode = "gan"              # "classifier" or "gan"
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
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # ---- MODEL, LOSS, OPTIMIZER ----
    print(f"[Main] Setting up model and optimizer for mode: {mode}")
    if mode == "classifier":
        net = ClassifierNet(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        model_name = dataset_name
        label_fn = None
        metric_fn = None  # Default: accuracy
        print(f"[Main] Training classifier...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        train(
            net,
            optimizer,
            criterion,
            trainloader,
            epochs=50,
            patience=5,
            resume_best=False,
            model_name=model_name,
            checkpoint_subdir="checkpoints",
            label_fn=label_fn,
            metric_fn=metric_fn,
            scheduler=None,
            save_images_fn=None,
            n_classes=num_classes,
            device=device,
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
            lr=0.0002,
            model_name=model_name,
            resume_best=False,
            gen_rate=2,
            lr_ratio=14,
            metric_fn=metric_fn,
            label_fn=label_fn,
        )
    else:
        raise ValueError("Unknown mode. Use 'classifier' or 'gan'.")

    print(f"[Main] Training complete. Starting evaluation...")

    # ---- TESTING ----
    checkpoint_path = get_checkpoint_path(model_name, "disc_best" if mode == "gan" else "best")
    try:
        net.load_state_dict(torch.load(checkpoint_path))
        print(f"[Main] Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"[Main] Could not load checkpoint: {e}")

    # Move model to device before testing
    net.to(device)

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
        test_cgan_generator(generator, nrow=10, ncol=10, device='cpu')

# ----
# HOW TO USE:
# 1. Set 'mode' to "classifier" or "gan" at the top.
# 2. Set 'dataset_name' to "mnist", "cifar10", or "cifar100".
# 3. Make sure models/generator.py exists for GAN mode.
# 4. Run: python main.py
# 5. Checkpoints are saved in the 'checkpoints/' folder.
# 6. For GANs, generated images and generator checkpoints can be added as needed.