# MultiVision-GAN: Conditional GANs & Classifiers for Vision Datasets

This project is a modular PyTorch framework for both **conditional GANs (cGANs)** and **image classifiers**, supporting **CIFAR-100**, **CIFAR-10**, and **MNIST**. The codebase is designed for easy experimentation, robust label conditioning, checkpointing, and extensibility.

---

## Features

- **Dynamic dataset support:** Easily switch between CIFAR-100, CIFAR-10, and MNIST.
- **Conditional GANs:** Full support for cGANs with one-hot label conditioning and deep label feature injection.
- **Modular codebase:** Clean separation of models, training, testing, and data loading.
- **Automatic checkpointing:** Best and last models are saved for each dataset and mode.
- **TensorBoard integration:** Visualize training loss, accuracy, and GAN progress in real time.
- **Sample image saving:** Generated images are saved every epoch for visual inspection.
- **Reproducibility:** Paths and checkpoints are robust to different working directories.
- **Device agnostic:** Seamless CPU/GPU support.

---

## Results

| Dataset   | Classifier Accuracy | GAN Quality (visual) |
|-----------|---------------------|----------------------|
| CIFAR-100 |   60.7%             | Good class separation|
| CIFAR-10  |   83.11%            | Sharp, colorful      |
| MNIST     |   99.29%            | Crisp digits, black backgrounds |

*All results are on the test set using the provided classifier architecture and default training settings. GAN quality is based on visual inspection of generated samples.*

---

## GAN Development: Failures & Successes

### What Worked

- **One-hot label conditioning:** Switching from embeddings to one-hot encoding for labels, and injecting label features both at the input and deeper in the generator/discriminator, dramatically improved image quality and class fidelity.
- **Matching normalization to output activation:** Using `nn.Sigmoid()` in the generator and keeping dataset images in `[0, 1]` (no normalization) ensured backgrounds were truly black and images looked correct.
- **Minimalism for debugging:** Stripping the training loop to the bare essentials (as in reference code) helped isolate and fix subtle bugs.
- **Consistent device handling:** Ensuring all tensors (including targets) are always on the correct device prevented silent errors.

### What Didn’t Work

- **Label embeddings alone:** Using only embeddings for label conditioning led to poor class separation and blurry images.
- **Mismatched normalization:** Normalizing images to `[-1, 1]` while using `Sigmoid` output (or vice versa) caused gray backgrounds and poor contrast.
- **Overcomplicated training loops:** Too many features at once made debugging difficult; simplicity first, features later.

---

## Usage

1. **Install dependencies**  
   (Recommended: use a virtual environment)
   ```bash
   pip install torch torchvision matplotlib tensorboard
   ```

2. **Train a classifier or GAN**  
   Edit `main.py` and set:
   ```python
   mode = "classifier"  # or "gan"
   dataset_name = "mnist"  # or "cifar10", "cifar100"
   ```
   Then run:
   ```bash
   python main.py
   ```

3. **View training progress**  
   Start TensorBoard:
   ```bash
   python -m tensorboard --logdir=runs
   ```
   Then open [http://localhost:6006](http://localhost:6006) in your browser.

4. **View generated images**  
   Generated samples are saved in the `samples/` directory after each epoch.

---

## Notes

- The classifier is a moderately deep CNN with batch normalization and dropout.
- The cGAN uses one-hot label conditioning and deep label feature injection for best results.
- Checkpoints are saved in the `checkpoints/` folder, named by dataset and mode.
- The code is written to be easy to read and modify—perfect for learning or extending to new datasets or GANs.

---

*Created by a high schooler passionate about AI and deep learning. If you have feedback or want to collaborate, feel free to reach out!*

---
