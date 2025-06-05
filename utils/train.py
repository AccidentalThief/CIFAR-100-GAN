import os
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.test import test  # Add this import at the top if not present

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
    generator=None,
    optimizer_G=None,
    latent_dim=100,
    gan_mode=False,
    save_images_fn=None,
    n_classes=10,
    device='cpu',
    add_noise=False
):
    """
    Unified training function for classifier, discriminator, and cGAN.
    If gan_mode is True, expects generator, optimizer_G, and uses cGAN training logic.
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
        if gan_mode and generator is not None:
            generator.train()
        running_loss = 0.0
        running_metric = 0.0
        total = 0
        running_g_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i == 0:
                print(f"[Train] First batch loaded. Batch size: {len(data[0]) if isinstance(data, (list, tuple)) else len(data)}")
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, labels = data
            else:
                inputs = data
                labels = None

            batch_total = inputs.size(0)

            if gan_mode and generator is not None and optimizer_G is not None:
                device = next(net.parameters()).device
                real_imgs = inputs.to(device)
                if add_noise:
                    real_imgs += 0.05 * torch.randn_like(real_imgs)
                valid = torch.ones(batch_total, 1, device=device)
                fake = torch.zeros(batch_total, 1, device=device)

                # 1. Train Generator (optionally more than once)
                g_loss = 0
                z = torch.randn(batch_total, latent_dim, device=device)
                gen_labels = torch.randint(0, n_classes, (batch_total,), device=device)
                gen_imgs = generator(z, gen_labels)
                g_loss = criterion(net(gen_imgs, gen_labels), valid)
                g_loss.backward()
                optimizer_G.step()

                # 2. Train Discriminator
                optimizer.zero_grad()
                real_loss = criterion(net(real_imgs, labels), valid)
                fake_loss = criterion(net(gen_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer.step()

                # Logging
                writer.add_scalar('MiniBatch/D_Loss', d_loss.item(), global_step)
                writer.add_scalar('MiniBatch/G_Loss', g_loss.item(), global_step)
                running_loss += d_loss.item() * batch_total
                running_g_loss += g_loss.item() * batch_total
                if metric_fn is not None:
                    batch_metric = metric_fn(net(real_imgs, labels), valid)
                else:
                    batch_metric = 0
                writer.add_scalar('MiniBatch/Metric', batch_metric, global_step)
                running_metric += batch_metric * batch_total
                total += batch_total
                global_step += 1
            else:
                optimizer.zero_grad()
                outputs = net(inputs, labels) if labels is not None else net(inputs)
                if label_fn is not None:
                    labels = label_fn(inputs, outputs, epoch, i)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                if metric_fn is not None:
                    # Use real images and labels for metric
                    metric = metric_fn(discriminator(real_imgs, labels), valid)
                    writer.add_scalar('MiniBatch/Metric', metric, global_step)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    batch_metric = (predicted == labels).sum().item() / batch_total if labels is not None else 0

                writer.add_scalar('MiniBatch/Loss', batch_loss, global_step)
                writer.add_scalar('MiniBatch/Metric', batch_metric, global_step)
                running_loss += batch_loss * batch_total
                running_metric += batch_metric * batch_total
                total += batch_total
                global_step += 1

        # Epoch logging
        if gan_mode and generator is not None:
            epoch_d_loss = running_loss / total
            epoch_g_loss = running_g_loss / total
            epoch_metric = running_metric / total
            writer.add_scalar('Epoch/D_Loss', epoch_d_loss, epoch)
            writer.add_scalar('Epoch/G_Loss', epoch_g_loss, epoch)
            writer.add_scalar('Epoch/Metric', epoch_metric, epoch)
            print(f'Epoch {epoch+1}: D_Loss: {epoch_d_loss:.4f}, G_Loss: {epoch_g_loss:.4f}, Metric: {epoch_metric:.4f}')
            if save_images_fn is not None:
                save_images_fn(epoch, generator)
            epoch_loss = epoch_d_loss
        else:
            epoch_loss = running_loss / total
            epoch_metric = running_metric / total
            writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
            writer.add_scalar('Epoch/Metric', epoch_metric, epoch)
            print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}')

        if epoch_metric > best_metric:
            best_metric = epoch_metric
            epochs_no_improve = 0
            torch.save(net.state_dict(), get_checkpoint_path(model_name, "best", checkpoint_subdir))
            if gan_mode and generator is not None:
                torch.save(generator.state_dict(), get_checkpoint_path(model_name, "gen_best", checkpoint_subdir))
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
    if gan_mode and generator is not None:
        torch.save(generator.state_dict(), get_checkpoint_path(model_name, "gen_last", checkpoint_subdir))
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
    from .train import get_checkpoint_path, get_unique_logdir

    logdir = get_unique_logdir("runs", f"{model_name}_cgan")
    writer = SummaryWriter(log_dir=logdir)
    print(f"[cGAN] Logging to {logdir}")
    print(f"[cGAN] Starting cGAN training for {epochs} epochs on device {device}.")
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr/lr_ratio, betas=(0.5, 0.999))

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
            real_imgs += 0.1 * torch.randn_like(real_imgs)
            labels = labels.to(device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            for _ in range(gen_rate):
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim, device=device)
                gen_labels = torch.randint(0, n_classes, (batch_size,), device=device)
                gen_imgs = generator(z, gen_labels)
                g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
                g_loss.backward()
                optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Logging
            writer.add_scalar('MiniBatch/D_Loss', d_loss.item(), global_step)
            writer.add_scalar('MiniBatch/G_Loss', g_loss.item(), global_step)
            if metric_fn is not None:
                metric = metric_fn(discriminator(real_imgs, labels), valid)
                writer.add_scalar('MiniBatch/Metric', metric, global_step)
                running_metric += metric * batch_size
            running_g_loss += g_loss.item() * batch_size
            running_d_loss += d_loss.item() * batch_size
            total += batch_size
            global_step += 1

            if i % math.ceil(batch_size / 5) == 0:
                print(f"[cGAN] [Epoch {epoch+1}/{epochs}] [Batch {i}/{len(trainloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Epoch logging
        epoch_g_loss = running_g_loss / total
        epoch_d_loss = running_d_loss / total
        epoch_metric = running_metric / total if total > 0 else 0
        writer.add_scalar('Epoch/G_Loss', epoch_g_loss, epoch)
        writer.add_scalar('Epoch/D_Loss', epoch_d_loss, epoch)
        writer.add_scalar('Epoch/Metric', epoch_metric, epoch)
        print(f"[cGAN] Epoch {epoch+1}: D_Loss: {epoch_d_loss:.4f}, G_Loss: {epoch_g_loss:.4f}, Metric: {epoch_metric:.4f}")

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

    writer.close()
    print("[cGAN] Training complete!")