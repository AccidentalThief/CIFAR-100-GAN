import os
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.test import test  # Add this import at the top if not present
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def get_checkpoint_path(name, kind="best", subdir="checkpoints"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', subdir))
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f'{name.lower()}_{kind}.pth')
    print(f"[Checkpoint] Path for '{kind}': {path}")
    return path

def get_unique_logdir(base_dir, prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    counter = 1
    while os.path.exists(logdir):
        logdir = os.path.join(base_dir, f"{prefix}_{timestamp}_{counter}")
        counter += 1
    return logdir

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
    label_fn=None,
    metric_fn=None,
    scheduler=None,
    save_images_fn=None,
    n_classes=10,
    device='cpu',
):
    """
    Training function for classifiers only.
    """
    logdir = get_unique_logdir("runs", model_name)
    writer = SummaryWriter(log_dir=logdir)
    print(f"[Train] Logging to {logdir}")
    print(f"[Train] Starting training for model '{model_name}' for {epochs} epochs.")
    global_step = 0
    best_metric = float('-inf')
    epochs_no_improve = 0

    if resume_best:
        best_path = get_checkpoint_path(model_name, "best", checkpoint_subdir)
        try:
            net.load_state_dict(torch.load(best_path))
            print(f"[Train] Resumed training from {best_path}")
        except Exception as e:
            print(f"[Train] Could not load checkpoint: {e} (starting from scratch)")

    for epoch in range(epochs):
        print(f"[Train] Epoch {epoch+1}/{epochs}...")
        net.train()
        running_loss = 0.0
        running_metric = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            if i == 0:
                print(f"[Train] First batch loaded. Batch size: {len(data[0]) if isinstance(data, (list, tuple)) else len(data)}")
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, labels = data
            else:
                inputs = data
                labels = None

            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            batch_total = inputs.size(0)

            optimizer.zero_grad()
            outputs = net(inputs)
            if label_fn is not None:
                labels = label_fn(inputs, outputs, epoch, i)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            if metric_fn is not None:
                batch_metric = metric_fn(outputs, labels)
            else:
                _, predicted = torch.max(outputs.data, 1)
                batch_metric = (predicted == labels).sum().item() / batch_total if labels is not None else 0

            writer.add_scalar('MiniBatch/Loss', batch_loss, global_step)
            writer.add_scalar('MiniBatch/Metric', batch_metric, global_step)
            running_loss += batch_loss * batch_total
            running_metric += batch_metric * batch_total
            total += batch_total
            global_step += 1

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

        print(f"[Train] Finished epoch {epoch+1}")

    writer.close()
    print(f'[Train] Finished Training for model {model_name}')
    PATH = get_checkpoint_path(model_name, "last", checkpoint_subdir)
    torch.save(net.state_dict(), PATH)
    print(f'[Train] Last model saved to {PATH}')

def train_cgan(
    generator,
    discriminator,
    trainloader,
    latent_dim=100,
    n_classes=10,
    device='cpu',
    epochs=30,
    lr=0.0001,
    model_name="cgan_mnist",
    resume_best=False,
    gen_rate=1,
    testloader=None,
    criterion=None,
    metric_fn=None,
    label_fn=None,
    lr_ratio=1
):
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
    from .train import get_checkpoint_path, get_unique_logdir

    logdir = get_unique_logdir("runs", f"{model_name}_cgan")
    writer = SummaryWriter(log_dir=logdir)
    print(f"[cGAN] Logging to {logdir}")
    print(f"[cGAN] Starting cGAN training for {epochs} epochs on device {device}.")
    loss_function = F.binary_cross_entropy_with_logits
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    generator.to(device)
    discriminator.to(device)

    # ---- RESUME FROM BEST CHECKPOINT ----
    if resume_best:
        gen_ckpt = get_checkpoint_path(model_name, "gen_best")
        disc_ckpt = get_checkpoint_path(model_name, "disc_best")
        try:
            generator.load_state_dict(torch.load(gen_ckpt, map_location=device))
            print(f"[cGAN] Resumed generator from {gen_ckpt}")
        except Exception as e:
            print(f"[cGAN] Could not load generator checkpoint: {e} (starting from scratch)")
        try:
            discriminator.load_state_dict(torch.load(disc_ckpt, map_location=device))
            print(f"[cGAN] Resumed discriminator from {disc_ckpt}")
        except Exception as e:
            print(f"[cGAN] Could not load discriminator checkpoint: {e} (starting from scratch)")

    global_step = 0

    for epoch in range(epochs):
        print(f"[cGAN] Epoch {epoch+1}/{epochs}...")
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_metric = 0.0
        total = 0
        for i, (imgs, labels) in enumerate(trainloader):
            if i == 0:
                print(f"[cGAN] First batch loaded. Batch size: {imgs.size(0)}")
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            d_loss = loss_function(discriminator(real_imgs, labels), valid)
            d_loss += loss_function(discriminator(generator(labels), labels), fake)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            g_loss = loss_function(discriminator(generator(labels), labels), valid)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # Logging
            writer.add_scalar('MiniBatch/D_Loss', d_loss.item(), global_step)
            writer.add_scalar('MiniBatch/G_Loss', g_loss.item(), global_step)
            if metric_fn is not None:
                metric = metric_fn(discriminator(real_imgs, labels), valid)
                writer.add_scalar('MiniBatch/Metric', metric, global_step)
                running_metric += metric * batch_size
            else:
                metric = 0
                running_metric += metric * batch_size
            running_g_loss += g_loss.item() * batch_size
            running_d_loss += d_loss.item() * batch_size
            total += batch_size
            global_step += 1

        # Epoch logging
        epoch_g_loss = running_g_loss / total
        epoch_d_loss = running_d_loss / total
        epoch_metric = running_metric / total if total > 0 else 0
        writer.add_scalar('Epoch/G_Loss', epoch_g_loss, epoch)
        writer.add_scalar('Epoch/D_Loss', epoch_d_loss, epoch)
        writer.add_scalar('Epoch/Metric', epoch_metric, epoch)
        print(f"[cGAN] Epoch {epoch+1}: D_Loss: {epoch_d_loss:.4f}, G_Loss: {epoch_g_loss:.4f}, Metric: {epoch_metric:.4f}")

        # Save generated images
        grid = save_cgan_samples(generator, epoch, latent_dim, n_classes, device)
        writer.add_image('Generated_Images', grid, epoch)  # if you want TensorBoard

        torch.save(generator.state_dict(), get_checkpoint_path(model_name, "gen_best"))
        torch.save(discriminator.state_dict(), get_checkpoint_path(model_name, "disc_best"))
        print(f"[cGAN] Saved checkpoints for epoch {epoch+1}")

        # ---- Run test after each epoch ----
        if testloader is not None and criterion is not None:
            print(f"[cGAN] Running test after epoch {epoch+1}...")
            test(
                discriminator, testloader,
                criterion=criterion,
                metric_fn=metric_fn,
                label_fn=label_fn,
                verbose=True
            )

        # Save generated samples for visual progress
        save_cgan_samples(generator, epoch, latent_dim, n_classes, device)

    writer.close()
    print("[cGAN] Training complete!")

def save_cgan_samples(generator, epoch, latent_dim, n_classes, device, out_dir="samples", nrow=10, ncol=10):
    generator.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        labels = torch.arange(ncol, device=device).repeat(nrow)
        imgs = generator(labels).detach().cpu()  # imgs in [0,1] because of Sigmoid
        grid = vutils.make_grid(imgs, nrow=ncol, padding=2, normalize=False)
        plt.figure(figsize=(ncol, nrow))
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray' if imgs.shape[1]==1 else None, vmin=0, vmax=1)
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"), bbox_inches='tight')
        plt.close()
    generator.train()
    return grid