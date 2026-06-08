# Assignment 1 — Classical Machine Learning Methods

## Overview
This assignment covers the foundational ML techniques that precede neural networks: regression modelling, feature-based classification on image data, and semi-automatic data labelling pipelines.

---

## Part 1 — Regression Problems

### Problem 1: Insurance Data (Univariate Regression)
**Dataset:** Number of insured persons at an insurance company, years 1987–1996.  
**Task:** Fit linear, quadratic, and cubic models; detect and remove the outlier (year 1990: value 1050, clearly a data entry error for 10500); compare R² scores; predict 1997.

**Key steps:**
- Scatter plot with x = years since 1987
- Polynomial regression (degrees 1, 2, 3) with and without the outlier
- Model selection using R² (prefer simpler model if gain is marginal)
- Prediction for year 10 (1997) using the best-fit cleaned model

### Problem 2: Heating Oil Consumption (Multivariate Regression)
**Dataset:** 15 samples; features are average January temperature (°F) and insulation thickness (inches); target is heating oil usage.  
**Task:** Fit linear and quadratic multivariate models; identify and remove outlier (insulation=40 is unrealistic); predict oil usage at temp=15°F, insulation=5 inches.

**Key steps:**
- Linear regression: `oil = b0 + b1*temp + b2*insulation`
- Quadratic regression with interaction terms
- Outlier removal and model re-fitting
- Prediction using both cleaned models

---

## Part 2 — Classical ML on ReducedMNIST

**Dataset:** ReducedMNIST — 1000 training + 200 test examples per digit (10 classes).

### Feature Extraction
Three feature representations are computed for each image:
- **DCT** — 225-dimensional discrete cosine transform features
- **PCA** — enough components to retain ≥95% of total variance
- **HOG** — histogram of oriented gradients (via `extractHOGFeatures`)

### Classifiers
| Classifier | Variants |
|---|---|
| K-means (per class) | 1, 4, 16, 32 clusters |
| SVM | Linear kernel, RBF (nonlinear) kernel |

### Results Summary
Full accuracy and processing time table is provided in the notebook/report, covering all feature × classifier combinations. Confusion matrices are shown for the best result of each classifier type.

---

## Part 3 — Semi-Automatic Labelling Pipelines

**Dataset:** Indian Digits — 10,000 unlabelled 28×28 grayscale images (digits 0–9, 1000 per class).  
**Target:** ≥99% labelling accuracy with minimum human time.  
**Baseline:** 10,000 images × 10 s/image = 27.8 hours of fully manual labelling.

### Pipeline 1 — K-Means Bootstrapping + Active SVM
1. Feature extraction (DCT / PCA / HOG)
2. K-means clustering (K = 40–80); human labels each cluster via 8-image sample view (20s per cluster)
3. Initial SVM (RBF kernel) trained on cluster-derived labels
4. Active refinement: identify 20–40 lowest-confidence boundary images → human labels them (10s each, weight=100)
5. Retrain weighted SVM; iterate until ≥99% accuracy or convergence

### Pipeline 2 — Manual Seed + Self-Training
1. Manually label 300 random images (~30/class) as trusted seed (weight=100)
2. Augment seed set (rotation ±5°, Gaussian noise, spatial shifts)
3. Train SVM-1 on seed + augmented data
4. Active: label 20 boundary images per iteration (weight=100)
5. Self-training: add top-50/class high-confidence pseudo-labels (margin > 75th percentile, weight=1)
6. Iterate until ≥99% or convergence

### Pipeline 3 — LLM-Based Zero-Shot Labelling
1. Benchmark 5 vision LLMs on the 500-image practical ground truth set
2. Select top-2 LLMs; run both independently over all 10,000 images
3. Accept agreed labels (verified ≥99%); manually resolve disagreements (10s/image)
4. Corrective options if accuracy falls short: prompt refinement, few-shot examples, tie-breaker LLM

### Pipeline Comparison
| Pipeline | Manual Images Handled | Estimated Manual Time |
|---|---|---|
| 1 (K-Means + SVM) | clusters×8 + boundary×iterations | ~minutes |
| 2 (Seed + Self-train) | 300 + 20×iterations | ~hours |
| 3 (LLM agreement) | 500 (benchmark) + disagreements | ~minutes–hours |

*Exact values filled in from experimental results.*

---

## Files
```
Assignment1/
├── Part1_Regression/
├── Part2_ClassicalML/
└── Part3_Labelling/
```

---

## Key Observations
- Outlier detection significantly improves regression model quality; the 1990 data point (1050 vs ~9500 expected) is a clear entry error.
- HOG features tend to produce more class-coherent clusters than raw pixels or DCT for K-means.
- Pipeline 3 offers the lowest manual time but depends heavily on LLM vision quality for small 28×28 grayscale images; upscaling to 224×224 before LLM inference is recommended.