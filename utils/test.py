import time
import torch
import torch.nn as nn

def test(
    net,
    testloader,
    criterion=None,
    metric_fn=None,
    label_fn=None,
    verbose=True
):
    net.eval()
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