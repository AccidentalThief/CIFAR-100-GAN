import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.classifier import ClassifierNet
from classification.dataset import get_dataset, get_transforms

def get_checkpoint_path(name, kind="best"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f'{name.lower()}_net_{kind}.pth')

def train(net, optimizer, criterion, trainloader, epochs=50, patience=5, resume_best=False, dataset_name="cifar100"):
    writer = SummaryWriter()
    print("Starting training...")
    global_step = 0
    best_acc = 0.0
    epochs_no_improve = 0

    if resume_best:
        best_path = get_checkpoint_path(dataset_name, "best")
        try:
            net.load_state_dict(torch.load(best_path))
            print(f"Resumed training from {best_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e} (starting from scratch)")

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

            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_acc = 100.0 * batch_correct / batch_total

            writer.add_scalar('MiniBatch/Loss', batch_loss, global_step)
            writer.add_scalar('MiniBatch/Accuracy', batch_acc, global_step)
            global_step += 1

            running_loss += batch_loss * batch_total
            correct += batch_correct
            total += batch_total

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

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