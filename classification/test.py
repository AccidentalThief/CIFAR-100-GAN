import torch
import torch.nn as nn
import time

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    print("Starting testing...")

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    elapsed = time.time() - start_time
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    time_per_image = elapsed / total if total > 0 else 0

    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    print(f'Test time: {elapsed:.2f} seconds, {time_per_image*1000:.2f} ms/image')
    net.train()