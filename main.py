import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import your model here (Net should accept input_channels and num_classes)
from net import Net

trainset = None
testset = None
trainloader = None
testloader = None
classes = None
batch_size = 32

def get_dataset(name, train, transform):
    if name.lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        num_classes = 100
        input_channels = 3
    elif name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 3
    elif name.lower() == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset, num_classes, input_channels

def setup_data(dataset_name="cifar100"):
    global trainset, testset, trainloader, testloader, classes, num_classes, input_channels

    print(f"Setting up {dataset_name.upper()} dataset...")

    # Choose transforms based on dataset
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

    trainset, num_classes, input_channels = get_dataset(dataset_name, train=True, transform=transform)
    testset, _, _ = get_dataset(dataset_name, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = trainset.classes if hasattr(trainset, 'classes') else [str(i) for i in range(num_classes)]

    print(f"{dataset_name.upper()} dataset setup complete.")

def imshow(images, labels=None, classes=None):
    images = images / 2 + 0.5  # unnormalize (approximate for visualization)
    np_images = images.numpy()
    batch_size = np_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2.5))
    if batch_size == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        if np_images.shape[1] == 1:
            ax.imshow(np_images[idx][0], cmap='gray')
        else:
            ax.imshow(np.transpose(np_images[idx], (1, 2, 0)))
        ax.axis('off')
        if labels is not None and classes is not None:
            ax.set_title(classes[labels[idx]], fontsize=8, color='black', pad=10)
    plt.tight_layout()
    plt.show()

def get_checkpoint_path(name, kind="best"):
    return f'./{name.lower()}_net_{kind}.pth'

def train(net, optimizer, criterion, epochs=50, patience=5):
    writer = SummaryWriter()
    print("Starting training...")
    global_step = 0
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate per-batch loss and accuracy
            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = 100.0 * batch_correct / batch_total

            # Log per-batch loss and accuracy to TensorBoard
            writer.add_scalar('MiniBatch/Loss', batch_loss, global_step)
            writer.add_scalar('MiniBatch/Accuracy', batch_acc, global_step)
            global_step += 1

            # For epoch stats
            running_loss += batch_loss * batch_total
            correct += batch_correct
            total += batch_total

            # Console log every 2000 mini-batches
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / total:.3f}, accuracy: {100.0 * correct / total:.2f}%')

        # Log epoch loss and accuracy to TensorBoard
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Early stopping and checkpointing
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            epochs_no_improve = 0
            torch.save(net.state_dict(), get_checkpoint_path(dataset_name, "best"))
            print(f"Checkpoint: Saved new best model at epoch {epoch+1} with accuracy {epoch_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early at epoch {epoch+1}.")
                break

    writer.close()
    print('Finished Training')
    PATH = get_checkpoint_path(dataset_name, "last")
    torch.save(net.state_dict(), PATH)
    print(f'Last model saved to {PATH}')

def test(net):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    net.train()

if __name__ == '__main__':
    # Choose your dataset here: "cifar100", "cifar10", or "mnist"
    dataset_name = "mnist"
    setup_data(dataset_name)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"Dataset: {dataset_name.upper()}, Number of classes: {len(classes)}, Input channels: {input_channels}")

    # Dynamically create the model for the dataset
    net = Net(input_channels=input_channels, num_classes=num_classes, input_size=28 if dataset_name == "mnist" else 32)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Uncomment to train
    train(net, optimizer, criterion, epochs=10, patience=2)

    # Load best model and test
    checkpoint_path = get_checkpoint_path(dataset_name, "best")
    try:
        net.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    test(net)