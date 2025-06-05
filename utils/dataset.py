import os
import torchvision
import torchvision.transforms as transforms

def get_transforms(dataset_name, augment=False):
    if dataset_name.lower() in ["cifar100", "cifar10"]:
        transform_list = []
        if augment:
            transform_list += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ]
        return transforms.Compose(transform_list)
    elif dataset_name.lower() == "mnist":
        transform_list = []
        if augment:
            transform_list += [
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
            ]
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        return transforms.Compose(transform_list)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataset(name, train, transform):
    name = name.lower()
    if name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=os.path.join('..', 'data'), train=train, download=True, transform=transform)
        num_classes = 100
        input_channels = 3
        input_size = 32
    elif name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=os.path.join('..', 'data'), train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
        input_size = 32
    elif name == "mnist":
        dataset = torchvision.datasets.MNIST(root=os.path.join('..', 'data'), train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
        input_size = 28
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset, num_classes, input_channels, input_size