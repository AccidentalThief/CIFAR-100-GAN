import time
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def test(
    net,
    testloader,
    criterion=None,
    metric_fn=None,
    label_fn=None,
    verbose=True
):
    net.eval()
    device = next(net.parameters()).device  # Get model device
    correct = 0
    total = 0
    test_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for data in testloader:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                images, labels = data
            else:
                images = data
                labels = None

            images = images.to(device)
            if labels is not None:
                labels = labels.to(device)

            if hasattr(net, "label_emb"):
                outputs = net(images, labels)
            else:
                outputs = net(images)

            if label_fn is not None:
                labels = label_fn(images, outputs, 0, 0)
            if criterion is not None and labels is not None:
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
            if metric_fn is not None and labels is not None:
                batch_metric = metric_fn(outputs, labels)
                correct += batch_metric * images.size(0)
            elif labels is not None:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            total += images.size(0)

    elapsed = time.time() - start_time
    avg_loss = test_loss / total if criterion is not None and total > 0 else None
    accuracy = correct / total if total > 0 else None
    time_per_image = elapsed / total if total > 0 else 0

    if verbose:
        if avg_loss is not None:
            print(f'Test set: Average loss: {avg_loss:.4f}', end=', ')
        if accuracy is not None:
            print(f'Accuracy/Metric: {accuracy:.4f}', end=', ')
        print(f'Test time: {elapsed:.2f} seconds, {time_per_image*1000:.2f} ms/image')
    net.train()
    return avg_loss, accuracy, elapsed, time_per_image

def test_cgan_generator(generator, latent_dim=100, nrow=10, ncol=10, device='cpu', save_path=None, show=True):
    """
    Each column is a digit (0-9), each row is a different random sample for that digit.
    """
    generator.eval()
    with torch.no_grad():
        # For each row, generate ncol digits (0..ncol-1)
        z = torch.randn(nrow * ncol, latent_dim, device=device)
        labels = torch.arange(ncol, device=device).repeat(nrow)
        fake_imgs = generator(labels)
        fake_imgs = (fake_imgs + 1) / 2

        import torchvision.utils as vutils
        import matplotlib.pyplot as plt

        grid = vutils.make_grid(fake_imgs, nrow=ncol, padding=2, normalize=False)
        plt.figure(figsize=(ncol, nrow))
        plt.axis("off")
        plt.title("cGAN: Each column is a digit (0-9)")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')

        # Add digit labels below each column
        img_width_on_grid = grid.shape[2] // ncol
        for i in range(ncol):
            plt.text(
                x=i * img_width_on_grid + img_width_on_grid // 2,
                y=grid.shape[1] + 5,
                s=str(i),
                ha='center',
                va='top',
                fontsize=14,
                color='black',
                fontweight='bold'
            )
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

def test_cgan_single_digit(generator, latent_dim=100, digit=0, nrow=8, ncol=8, device='cpu', save_path=None, show=True):
    """
    Generates a grid of samples for a single digit/class using a cGAN.
    """
    generator.eval()
    with torch.no_grad():
        total_imgs = nrow * ncol
        z = torch.randn(total_imgs, latent_dim, device=device)
        print("First z sample:", z[0][:5])
        labels = torch.full((total_imgs,), digit, dtype=torch.long, device=device)
        fake_imgs = generator(z, labels).detach().cpu()
        fake_imgs = (fake_imgs + 1) / 2

        import torchvision.utils as vutils
        import matplotlib.pyplot as plt

        grid = vutils.make_grid(fake_imgs, nrow=ncol, padding=2, normalize=False)
        plt.figure(figsize=(ncol, nrow))
        plt.axis("off")
        plt.title(f"cGAN: All samples are digit {digit}")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()