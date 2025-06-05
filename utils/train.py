import os
import torch
from torch.utils.tensorboard import SummaryWriter

def get_checkpoint_path(name, kind="best", subdir="checkpoints"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', subdir))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f'{name.lower()}_{kind}.pth')

def train(
    net,
    optimizer,
    criterion,
    trainloader,
    epochs=50,
    patience=5,
    resume_best=False,
    model_name="model",
    checkpoint_subdir="checkpoints",
    label_fn=None,  # function to create labels for each batch (for GAN/discriminator)
    metric_fn=None, # function to compute accuracy or other metric
    scheduler=None
):
    writer = SummaryWriter()
    print("Starting training...")
    global_step = 0
    best_metric = float('-inf')
    epochs_no_improve = 0

    if resume_best:
        best_path = get_checkpoint_path(model_name, "best", checkpoint_subdir)
        try:
            net.load_state_dict(torch.load(best_path))
            print(f"Resumed training from {best_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e} (starting from scratch)")

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_metric = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, labels = data
            else:
                inputs = data
                labels = None

            optimizer.zero_grad()
            outputs = net(inputs)

            # For GAN/discriminator, label_fn can override labels
            if label_fn is not None:
                labels = label_fn(inputs, outputs, epoch, i)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_total = inputs.size(0)
            if metric_fn is not None:
                batch_metric = metric_fn(outputs, labels)
            else:
                # Default: accuracy for classification
                _, predicted = torch.max(outputs.data, 1)
                batch_metric = (predicted == labels).sum().item() / batch_total if labels is not None else 0

            writer.add_scalar('MiniBatch/Loss', batch_loss, global_step)
            writer.add_scalar('MiniBatch/Metric', batch_metric, global_step)
            global_step += 1

            running_loss += batch_loss * batch_total
            running_metric += batch_metric * batch_total
            total += batch_total

        epoch_loss = running_loss / total
        epoch_metric = running_metric / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Metric', epoch_metric, epoch)
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}')

        if epoch_metric > best_metric:
            best_metric = epoch_metric
            epochs_no_improve = 0
            torch.save(net.state_dict(), get_checkpoint_path(model_name, "best", checkpoint_subdir))
            print(f"Checkpoint: Saved new best model at epoch {epoch+1} with metric {epoch_metric:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping early at epoch {epoch+1}.")
                break

        if scheduler is not None:
            scheduler.step()

    writer.close()
    print('Finished Training')
    PATH = get_checkpoint_path(model_name, "last", checkpoint_subdir)
    torch.save(net.state_dict(), PATH)
    print(f'Last model saved to {PATH}')