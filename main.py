import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from net import Net
from torch.utils.tensorboard import SummaryWriter

trainset = None
testset = None
trainloader = None
testloader = None
classes = None
batch_size = 32

def setup_data():
    """
    Setup the data for CIFAR-100 dataset.
    Downloads the dataset if not already present.
    """
    global trainset, testset, trainloader, testloader, classes

    print("Setting up CIFAR-100 dataset...")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = trainset.classes  # Dynamically read class names

    print("CIFAR-100 dataset setup complete.")

def imshow(images, labels=None, classes=None):
    # images: tensor of shape (B, C, H, W)
    images = images / 2 + 0.5  # unnormalize
    np_images = images.numpy()
    batch_size = np_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2.5))
    if batch_size == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(np.transpose(np_images[idx], (1, 2, 0)))
        ax.axis('off')
        if labels is not None and classes is not None:
            ax.set_title(classes[labels[idx]], fontsize=8, color='black', pad=10)
    plt.tight_layout()
    plt.show()

def train(net, optimizer, criterion, epochs=10):
    writer = SummaryWriter()  # Creates a new TensorBoard log directory
    print("Starting training...")
    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log mini-batch loss and accuracy to TensorBoard
            mini_acc = 100.0 * running_correct / running_total if running_total > 0 else 0
            writer.add_scalar('MiniBatch/Loss', running_loss, global_step)
            writer.add_scalar('MiniBatch/Accuracy', mini_acc, global_step)
            global_step += 1

            # Console log every 2000 mini-batches
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, accuracy: {mini_acc:.2f}%')
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        # Log epoch loss and accuracy to TensorBoard
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100.0 * correct / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    writer.close()
    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

def test(net):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # show images with labels under each image
    imshow(images, labels, classes)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

if __name__ == '__main__':
    setup_data()
    # Ensure the dataset is set up
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Load previously trained weights
    checkpoint_path = './cifar_net.pth'
    net.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model weights from {checkpoint_path}")

    # Continue training for another 50 epochs
    train(net, optimizer, criterion, epochs=50)
    test(net)


