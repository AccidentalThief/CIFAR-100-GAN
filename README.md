# CIFAR-100-GAN: Image Classification Experiments

This project is a modular image classification pipeline built in PyTorch, designed to work with multiple datasets including **CIFAR-100**, **CIFAR-10**, and **MNIST**. The codebase is organized for easy experimentation, checkpointing, and extensibility (GAN support coming soon!).

## Features

- **Dynamic dataset support:** Easily switch between CIFAR-100, CIFAR-10, and MNIST.
- **Modular codebase:** Clean separation of models, training, testing, and data loading.
- **Automatic checkpointing:** Best and last models are saved for each dataset.
- **TensorBoard integration:** Visualize training loss and accuracy in real time.
- **Reproducibility:** Paths and checkpoints are robust to different working directories.

## Results

| Dataset   | Test Accuracy |
|-----------|---------------|
| CIFAR-100 |   60.7%       |
| CIFAR-10  |   83.11%      |
| MNIST     |   99.29%      |

*All results are on the test set using the provided classifier architecture and default training settings.*

## Usage

1. **Install dependencies**  
   (Recommended: use a virtual environment)
   ```bash
   pip install torch torchvision tensorboard
   ```

2. **Train or test a model**  
   Edit `classification/main.py` and set `dataset_name` to `"cifar100"`, `"cifar10"`, or `"mnist"`.  
   To train:
   ```python
   # Uncomment this line in main.py
   train(net, optimizer, criterion, trainloader, epochs=100, patience=5, resume_best=False, dataset_name=dataset_name)
   ```
   To test:
   ```bash
   python classification/main.py
   ```

3. **View training progress**  
   Start TensorBoard:
   ```bash
   python -m tensorboard --logdir=runs
   ```
   Then open [http://localhost:6006](http://localhost:6006) in your browser.

## Notes

- The classifier is a moderately deep CNN with batch normalization and dropout.
- Checkpoints are saved in the `checkpoints/` folder, named by dataset.
- The code is written to be easy to read and modify—perfect for learning or extending to new datasets or GANs.

---

*Created by a high schooler passionate about AI and deep learning. If you have feedback or want to collaborate, feel free to reach out!*
