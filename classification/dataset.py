import torchvision
import torchvision.transforms as transforms

def get_transforms(dataset_name):
    if dataset_name.lower() in ["cifar100", "cifar10"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return transform

def get_dataset(name, train, transform):
    if name.lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root='../data', train=train, download=True, transform=transform)
        num_classes = 100
        input_channels = 3
        input_size = 32
    elif name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='../data', train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
        input_size = 32
    elif name.lower() == "mnist":
        dataset = torchvision.datasets.MNIST(root='../data', train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
        input_size = 28
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset, num_classes, input_channels, input_size