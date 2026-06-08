# Assignment 2 — Artificial Neural Networks

## Overview
This assignment replaces the classical classifiers from Assignment 1 with neural networks — moving from MLPs through CNNs, then tackling speech recognition, autoencoders, data augmentation, and generative adversarial networks.

---

## Problem 1 — Multilayer Perceptron (MLP) on ReducedMNIST

**Dataset:** ReducedMNIST — 1,000 images/digit for training (10,000 total), 200 images/digit for testing (2,000 total). All pixels normalised to [0, 1].

**Features:** DCT (top-energy coefficients), PCA (95% variance, ≈150 components), and Autoencoder-derived 64-dimensional bottleneck representations.

**Training protocol:** Adam (lr=1e-3), batch size 64, up to 50 epochs with early stopping (patience 10), Dropout p=0.3, He uniform initialisation.

### MLP Results

| Feature Extractor | Hidden Layers | Configuration | Train Acc | Test Acc | Train Time (s) |
|---|---|---|---|---|---|
| DCT | 1 | 128 | 99.37% | 95.10% | 6.48 |
| DCT | 3 | 256-128-64 | 99.47% | 95.05% | 7.40 |
| DCT | 4 | 512-256-128-64 | 99.43% | **95.30%** | 13.80 |
| PCA | 1 | 128 | 99.49% | **95.60%** | 5.13 |
| PCA | 3 | 256-128-64 | 99.39% | 95.50% | 6.00 |
| PCA | 4 | 512-256-128-64 | 99.43% | 95.15% | 10.79 |
| AutoEncoder | 1 | 128 | 97.39% | 96.15% | 2.42 |
| AutoEncoder | 3 | 256-128-64 | 97.92% | **96.75%** | 5.11 |
| AutoEncoder | 4 | 512-256-128-64 | 99.56% | 96.55% | 18.18 |

**Key observations:**
- AutoEncoder features consistently outperform PCA and DCT across all depths, as the encoder learns a task-relevant, non-linear compression.
- For DCT and PCA, additional depth yields minimal gains (<0.5 pp between 1 and 4 layers).
- AutoEncoder features benefit most from depth (96.15% → 96.75%, 1→3 layers); the 4-layer variant shows diminishing returns.

### K-Means Results

| Feature Extractor | k | Test Acc | Train Time (s) |
|---|---|---|---|
| DCT | 1 | 10.00% | 3.11 |
| DCT | 4 | 31.10% | 0.39 |
| DCT | 16 | 53.52% | 1.28 |
| DCT | 32 | 61.15% | 2.98 |
| PCA | 1 | 10.00% | 0.07 |
| PCA | 4 | 27.65% | 0.63 |
| PCA | 16 | 58.15% | 1.34 |
| PCA | 32 | 74.10% | 2.30 |
| AutoEncoder | 1 | 10.00% | 0.06 |
| AutoEncoder | 4 | 37.40% | 0.40 |
| AutoEncoder | 16 | 81.30% | 1.08 |
| AutoEncoder | 32 | **88.80%** | 1.80 |

### SVM Results

| Feature Extractor | Kernel | Train Acc | Test Acc | Train Time (s) |
|---|---|---|---|---|
| DCT | Linear | 95.24% | 91.25% | 5.08 |
| DCT | RBF | 99.96% | 95.00% | 57.19 |
| PCA | Linear | 97.70% | 93.60% | 3.95 |
| PCA | RBF | 99.99% | **97.10%** | 64.46 |
| AutoEncoder | Linear | 95.60% | 95.15% | 1.89 |
| AutoEncoder | RBF | 99.96% | **97.25%** | 22.13 |

---

## Problem 2 — CNN (LeNet-5) on ReducedMNIST

**Baseline architecture:** LeNet-5 adapted for 28×28 input:
- Conv1: 6 filters, 5×5, pad=2 → AvgPool 2×2
- Conv2: 16 filters, 5×5 → AvgPool 2×2
- FC layers: 120 → 84 → 10

**Training:** Adam (lr=1e-3), 20 epochs, batch size 32, all seeds set to 42.

### Results

| Variant | Test Acc | Train Time | Test Time | Notes |
|---|---|---|---|---|
| Baseline (AvgPool, 6/16) | 97.8% | 51,683 ms | 6,140 ms | Reference |
| Var 1: Max Pooling | 97.1% | 51,521 ms | 6,236 ms | −0.7 pp vs baseline |
| Var 2: Wider (16/32 filters) | 97.8% | 68,679 ms | 7,366 ms | Same acc; +33% train time |
| Var 3: Dropout (p=0.5) | 97.8% | 60,409 ms | 7,058 ms | Same acc; smoother curve |
| **Var 4: Combo (Max+Wider+Drop)** | **98.2%** | 74,647 ms | 8,387 ms | **Best accuracy** |

**Key observations:**
- Max Pooling alone slightly hurt accuracy on clean MNIST — average pooling retains more spatial context for simple digit shapes.
- Wider filters added 33% training time with no accuracy gain at this dataset size.
- Dropout slows early convergence (Epoch 1: 56.5% vs baseline 77.1%) but produces a smoother curve.
- Combining all three modifications compounds gains: Max+Wider+Dropout reaches 98.2%.

---

## Problem 3 — Speech Digit Recognition from Spectrograms

**Dataset:** Spoken digits (0–9), converted to 64×64 log-Mel spectrograms (512-pt FFT, hop=160, 64 mel bins).

**Architecture:** 3-block CNN (16→32→64 channels, 3×3 conv, MaxPool) + FC head (256→10) with Dropout(0.5). Adam (lr=1e-3), 15 epochs, batch size 16.

| Part | Configuration | Test Acc | Train Time | Test Time |
|---|---|---|---|---|
| (a) | Baseline — No Augmentation | **95.7%** | 49,134 ms | 8,764 ms |
| (b) | Audio Aug (Speed ±10% + Noise σ=0.005) | 95.0% | 57,654 ms | 9,021 ms |
| (c) | Image Aug (Squeeze 3–20% + Noise σ=0.1) | 95.0% | 54,432 ms | 8,834 ms |
| (d) | Combined Audio & Image Augmentation | 92.0% | 57,543 ms | 8,094 ms |

**Key observations:**
- Augmentation decreases accuracy here because the test set is clean, matching the un-augmented training distribution (train–test distribution mismatch).
- Combined augmentation (Part d) is worst (92.0%) — the 15-epoch budget is insufficient for the harder joint distribution.
- Image-domain augmentation shows faster early convergence than audio-domain augmentation (Epoch 1: 68.3% vs 51.7%).

---

## Problem 4 — Autoencoder for Speech Utterance Representation

**Approach:** Each utterance is split into 15 ms frames → 40 MFCC coefficients/frame → zero-padded to 53 frames max → 2120-dimensional input.

**Autoencoder:** 2120 → 1024 → 512 → 256 (bottleneck) → 512 → 1024 → 2120. Trained with MSE loss, 30 epochs.

| AE Epoch | Train MSE | Test MSE |
|---|---|---|
| 1 | 0.2404 | 0.0829 |
| 10 | 0.0322 | 0.0387 |
| 20 | 0.0255 | 0.0352 |
| 30 | 0.0215 | 0.0335 |

### Classifier Performance

| Method | Representation | Test Acc | Train Time | Test Time |
|---|---|---|---|---|
| Baseline (Average Frame) | 40D mean vector | 46.0% | 48,678 ms | 11,595 ms |
| **AE Bottleneck** | **256D encoded** | **93.0%** | **40,126 ms** | **9,537 ms** |

**Key observations:**
- Averaging frames discards all temporal order, which carries the majority of discriminative speech content. The 46% accuracy reflects a bag-of-frames representation that overfits despite weak signal.
- The AE bottleneck (256D) preserves sequential structure and improves accuracy by +47 pp, while also being faster to train.

---

## Problem 5 — Data Augmentation Study

**Model:** LeNet-5 (AvgPool, 6/16 filters), 10 epochs, Adam (lr=1e-3), batch size 32.

**Augmentation:** Random Affine ±5°, translation ±10%, Gaussian noise (σ=0.05, clamped to [0,1]).

### Test Accuracy Matrix (%)

| Generated/digit | 350 real | 750 real | 1,000 real |
|---|---|---|---|
| 0 (no augmentation) | 93.5% | 97.3% | 96.0% |
| 1,000 generated | 96.8% | 96.8% | 97.9% |
| 1,500 generated | 97.2% | 98.0% | **98.5%** |
| 2,000 generated | 97.3% | 98.2% | 98.3% |

**Key observations:**
- Augmentation is most effective when real data is scarce: +3.3 pp at 350 real/0→1,000 gen.
- Best configuration: 1,000 real + 1,500 generated = **98.5%**.
- Diminishing returns beyond 1,500 generated examples.
- Anomaly: 750 real (97.3%) outperforms 1,000 real (96.0%) without augmentation — a random-seed sampling effect.

---

## Problem 6 — GAN Synthetic Data

**Model:** DCGAN (Radford et al.) trained on 350 real examples/digit, 100 epochs. Generator updated 2× per discriminator update. Adam (lr=2e-4, β1=0.5). Training time: 96 min 6 sec (Intel Core i7 CPU).

### GAN Augmentation Results

| Generated/digit | 350 real | 750 real | 1,000 real |
|---|---|---|---|
| 0 (no augmentation) | 93.5% | 97.8% | **98.0%** |
| 1,000 generated | 97.0% | 97.8% | 97.8% |
| 1,500 generated | **97.8%** | 97.8% | 97.2% |
| 2,000 generated | 96.9% | 97.6% | 97.5% |

### Best Reduced-Data Configuration (350 real + Aug + GAN)

| Configuration | Aug/class | GAN/class | Total/class | Test Acc |
|---|---|---|---|---|
| 350 real only | — | — | 350 | 94.31% |
| 350 real + aug only | 1,000 | — | 1,350 | 96.80% |
| 350 real + GAN only | — | 1,000 | 1,350 | 97.80% |
| **350 real + aug + GAN** | **500** | **1,000** | **1,850** | **97.81%** |
| 350 real + aug + GAN | 1,000 | 1,000 | 2,350 | 97.63% |
| 1,000 real baseline | — | — | 1,000 | 98.63% |

**Key observations:**
- GAN augmentation is most valuable in the low-data regime: +4.3 pp at 350 real + 1,500 GAN vs no augmentation.
- Best reduced-data config (500 aug + 1,000 GAN) recovers 81% of the gap to the 1,000-real baseline.
- GAN samples provide diminishing returns once augmented real data already densifies the neighbourhood.
- Mild synthetic overfitting observed at 1,000 aug + 1,000 GAN (train–val gap: 2.1% vs 1.4%).

---

## Full Cross-Assignment Results Table

| Feature / Method | Classifier | Test Acc | Train Time | Test Time |
|---|---|---|---|---|
| HOG (Assign. 1) | K-Means k=32 | 99.2% | 10.52 s | — |
| HOG (Assign. 1) | SVM Linear | 99.4% | 1.62 s | — |
| HOG (Assign. 1) | SVM RBF | 99.95% | 4.48 s | — |
| DCT | SVM RBF | 95.00% | 57.19 s | — |
| PCA | SVM RBF | 97.10% | 64.46 s | — |
| AutoEncoder | SVM RBF | 97.25% | 22.13 s | — |
| AutoEncoder | MLP (3 hidden) | 96.75% | 5.11 s | — |
| Raw image | CNN Baseline (LeNet-5) | 97.8% | 51.7 s | 6.1 s |
| Raw image | CNN Var 4 (Max+Wide+Drop) | 98.2% | 74.6 s | 8.4 s |
| Raw image | LeNet-5 + 1,500 Aug | 98.5% | — | — |
| Raw image | LeNet-5 + 1,500 GAN | 97.8% | — | — |

---

## Files
```
Assignment2/
├── Problem1_MLP/
├── Problem2_CNN/
├── Problem3_SpeechRecognition/
├── Problem4_Autoencoder/
├── Problem5_Augmentation/
└── Problem6_GAN/
```

---

## Key Observations
- CNNs consistently outperform MLPs on image data when working directly on raw pixels, requiring no handcrafted feature engineering.
- Data augmentation (geometric transforms) provides the strongest gains at 350 real examples/digit (+3.3–4.3 pp) and gives the best overall result of 98.5% (1,000 real + 1,500 augmented).
- GAN-generated data can nearly match augmentation gains in the 350-real regime (97.8% vs 97.3%) but adds training complexity (~96 min CPU).
- The AE bottleneck representation improves speech digit classification from 46% (average-frame baseline) to 93%, confirming that preserving temporal order is critical for speech.
- Augmentation hurts performance on the clean speech dataset (Problem 3) due to train–test distribution mismatch — the test set remains unaugmented.