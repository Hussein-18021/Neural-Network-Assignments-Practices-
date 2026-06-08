# Assignment 2 — Artificial Neural Networks

## Overview
This assignment replaces the classical classifiers from Assignment 1 with neural networks — moving from MLPs through CNNs, then tackling speech recognition, autoencoders, data augmentation, and generative adversarial networks.

---

## Problem 1 — Multilayer Perceptron (MLP) on ReducedMNIST

**Dataset:** ReducedMNIST (same as Assignment 1)  
**Features:** DCT, PCA, and Autoencoder-derived representations  
**Architecture variants:** 1, 3, and 4 hidden layers  

All hyperparameters (learning rate, layer sizes, activation functions, batch size) are chosen freely and documented in the notebook. Results are compared directly against the Assignment 1 SVM/K-means baselines in the shared results table.

---

## Problem 2 — CNN (LeNet-5) on ReducedMNIST

**Baseline architecture:** LeNet-5 adapted for 28×28 input (rather than the original 32×32):
- Conv1: 5×5, stride 1 → AvgPool 2×2
- Conv2: 5×5, stride 1 → AvgPool 2×2
- FC layers: 120 → 84 → 10 (softmax)
- Activation: ReLU throughout

**Variations explored** (at least 4):
- Changing number of filters in convolutional layers
- Swapping activation functions (ReLU vs. others)
- Adding/removing layers
- Adjusting pooling strategy

Results (accuracy, training time, testing time) are reported for each variation. Direct comparison against MLP and Assignment 1 classifiers is provided in the full results table.

---

## Problem 3 — Speech Digit Recognition from Spectrograms

**Dataset:** Spoken digits (0–9), multiple speakers; converted to spectrogram images.  
**Approach:** Treat speech recognition as an image classification problem on spectrograms.

Four experimental conditions:
| Condition | Description |
|---|---|
| a | Baseline: CNN trained on raw spectrograms |
| b | Speech augmentation: speed ±3%, added speech noise |
| c | Image augmentation: horizontal squeeze/expand ±3%, image noise |
| d | Combined speech + image augmentation |

Starting architecture is the MLP/CNN from Problem 1/2, extended with ImageNet-style modifications as needed.

---

## Problem 4 — Autoencoder for Speech Utterance Representation

**Goal:** Compress variable-length speech utterances into fixed-length vectors for classification.

**Approach:**
1. Baseline: average all frames per utterance → fixed vector → classify
2. Autoencoder: concatenate all frames (zero-pad shorter utterances to max length) → AE encoder → single latent vector → classify

Each utterance is divided into 15ms frames; the AE is trained to reconstruct the full concatenated frame sequence from a compact bottleneck representation.

---

## Problem 5 — Data Augmentation Study

**Model:** LeNet-5 (fixed architecture)  
**Dataset:** ReducedMNIST at three real-data sizes: 350, 750, 1000 examples per digit  
**Augmentation techniques:** rotation (left/right, varying angles), random translation (Δx, Δy), noise addition

Results table (accuracy vs. real data size vs. generated examples per digit):

| Generated/digit | 350 real | 750 real | 1000 real |
|---|---|---|---|
| 0 | baseline | baseline | baseline |
| 1000 | 350+1000 gen | 750+1000 gen | 1000+1000 gen |
| 1500 | 350+1500 gen | 750+1500 gen | 1000+1500 gen |
| 2000 | 350+2000 gen | 750+2000 gen | 1000+2000 gen |

*Values filled from experimental results.*

---

## Problem 6 — GAN Synthetic Data

**Model:** Conditional DCGAN (cDCGAN) — reference: Radford et al. (DCGAN paper, arXiv:1511.06434)  
**Training data:** 350 real examples per digit only  
**Generation:** 3 sample examples per digit (qualitative evaluation) + full augmentation table

Three comparisons:
1. Visual quality of GAN-generated vs. real digits
2. Classifier accuracy when training on real + GAN-generated data (same table structure as Problem 5)
3. Best combination of augmented + synthetic data starting from 350 real examples, benchmarked against the 1000-real baseline

**Note:** All runs include timestamps; hardware setup is documented in the notebook.

---

## Full Results Table

| Feature | Classifier | Accuracy | Training Time | Testing Time |
|---|---|---|---|---|
| DCT | K-means (1/4/16/32 clusters) | | | |
| PCA | K-means (1/4/16/32 clusters) | | | |
| Autoencoder | K-means (1/4/16/32 clusters) | | | |
| DCT | SVM (linear / RBF) | | | |
| PCA | SVM (linear / RBF) | | | |
| Autoencoder | SVM (linear / RBF) | | | |
| DCT | MLP (1/3/4 hidden) | | | |
| PCA | MLP (1/3/4 hidden) | | | |
| Autoencoder | MLP (1/3/4 hidden) | | | |
| — | CNN (LeNet-5 + variations) | | | |

*Full table with values in the notebook/report.*

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
- CNNs consistently outperform MLPs on image data, especially with limited training examples.
- Data augmentation provides meaningful accuracy gains at low real-data sizes (350 examples), but returns diminish as real data increases toward 1000 examples.
- GAN-generated data quality is sensitive to training stability; augmenting the 350-example training set before GAN training significantly helps convergence.