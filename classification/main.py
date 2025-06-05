import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from models.classifier import ClassifierNet
from classification.dataset import get_dataset, get_transforms
from classification.train import train, get_checkpoint_path
from classification.test import test

if __name__ == '__main__':
    dataset_name = "cifar10"  # or "cifar100", "mnist"
    transform = get_transforms(dataset_name)
    trainset, num_classes, input_channels, input_size = get_dataset(dataset_name, train=True, transform=transform)
    testset, _, _, _ = get_dataset(dataset_name, train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    net = ClassifierNet(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train(net, optimizer, criterion, trainloader, epochs=100, patience=5, resume_best=False, dataset_name=dataset_name)
    checkpoint_path = get_checkpoint_path(dataset_name, "best")
    try:
        net.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    test(net, testloader)