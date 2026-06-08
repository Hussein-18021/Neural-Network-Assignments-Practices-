# Assignment 1 — Classical Machine Learning Methods

**Course:** Artificial Neural Networks — Cairo University, Faculty of Engineering, Fourth Year, Spring 2026
**Supervisors:** Dr. Mohsen Rashwan & Dr. Mohamed Moataz
**Team:** Hussein Moustafa Amin Hassan (9221585), Mohanad Hassan Aly (9221293), Serag Khaled Ahmed (9221268)

---

## Part I — Regression Problems

### Problem 1: Insurance Persons — Time-Series Curve Fitting (1987–1996)

**Dataset:** Number of insured persons at an insurance company, years 1987–1996. `x` = years since 1987 (0–9).

**Outlier:** `y = 1,050` at `x = 3` (year 1990) — a clear typographical error for 10,500, confirmed by residual and visual inspection.

**Models fitted (all data, n=10):**

| Model | Equation | R² |
|---|---|---|
| Linear | ŷ = 9,989.09 − 367.58x | 0.1345 |
| Quadratic | ŷ = 11,627.73 − 1,596.55x + 136.55x² | 0.2534 |
| Cubic | ŷ = 13,113.15 − 4,313.93x + 932.31x² − 58.95x³ | 0.3829 |

All three fits on the original data are unacceptably poor (R² < 0.40) due to the outlier's dominance.

**Models fitted (cleaned data, n=9, outlier removed):**

| Model | Equation | R² |
|---|---|---|
| Linear | ŷ = 11,621.67 − 530.83x | 0.9439 |
| Quadratic | ŷ = 12,024.17 − 863.13x + 37.44x² | 0.9725 |
| **Cubic (best)** | **ŷ = 12,204.42 − 1,221.74x + 140.17x² − 7.46x³** | **0.9787** |

The cubic model was selected as best fit (highest R² = 0.9787; the quadratic-to-cubic gain of +0.0062 was deemed sufficient to justify the extra parameter).

**1997 Prediction (x = 10):** ≈ **6,544 insured persons**

---

### Problem 2: Heating-Oil Consumption (Multivariate Regression)

**Dataset:** 15 observations; predictors are average January temperature (°F) and attic insulation thickness (inches); target is monthly heating oil usage (gallons).

**Outlier:** Observation 6 — insulation = 40 inches, physically implausible for residential attic insulation (all other values are 4–10 in.). Identified via domain knowledge (not residual criterion alone; its standardised residual was only +0.894).

**Models on full data (n=15):**

| Model | Equation | R² | R²_adj |
|---|---|---|---|
| Linear | Oil = 426.97 − 5.27T + 1.97I | 0.6967 | 0.6461 |
| Quadratic | Oil = 680.03 − 6.17T − 41.72I − 0.0125T² + 0.659I² + 0.239T·I | 0.9742 | 0.9598 |

Note: the full-data linear model yields a counter-intuitive positive insulation coefficient (+1.97), a direct consequence of the outlier.

**Models on cleaned data (n=14):**

| Model | Equation | R² | R²_adj |
|---|---|---|---|
| **Linear (preferred)** | **Oil = 601.60 − 5.48T − 23.23I** | **0.9606** | **0.9535** |
| Quadratic | Oil = 785.52 − 6.96T − 72.16I − 0.0035T² + 2.744I² + 0.257T·I | 0.9769 | 0.9624 |

The linear cleaned model is preferred by the parsimony principle (ΔR² = +0.0163 < 0.05 threshold; quadratic doubles the parameter count for negligible gain). Both coefficients now carry physically correct negative signs.

**Prediction for T = 15°F, I = 5 in.:**
- Linear model: **≈ 403 gallons**
- Quadratic model: ≈ 407 gallons (confirming linear adequacy)

---

## Part II — Classical ML on ReducedMNIST

**Dataset:** Official OpenML MNIST — 1,000 training + 200 test images per digit (10 classes), 28×28 grayscale, normalised to [0,1]. No augmentation or upsampling.

### Feature Extraction

| Feature | Method | Dimensions |
|---|---|---|
| DCT | 2-D DCT-II, top-left 15×15 low-frequency block | 225 |
| PCA | Minimum components for ≥95% variance (standardised inputs) | 38 (95.19% variance) |
| HOG | Histogram of Oriented Gradients, 4×4 px/cell, 2×2 cells/block, 9 orientations | 1,296 |

### Results

| Classifier | DCT Acc | PCA Acc | HOG Acc |
|---|---|---|---|
| K-Means k=1 | 81.4% | 79.0% | 91.3% |
| K-Means k=4 | 92.5% | 87.6% | 96.5% |
| K-Means k=16 | 96.3% | 94.1% | 98.5% |
| K-Means k=32 | 97.6% | 95.9% | **99.2%** |
| SVM Linear (C=1) | 94.0% | 96.3% | 99.4% (1.62s) |
| SVM RBF (C=10) | 98.3% | 99.5% | **99.95%** (4.48s) |

**Best configuration: SVM-RBF + HOG → 99.95% accuracy** (∼1 error across all 2,000 test images).

### Key Observations

- HOG consistently outperforms DCT and PCA at every k and for both SVM kernels.
- PCA is the fastest representation (38D, 0.25–2.15s total); SVM-RBF+PCA reaches 99.50% — best accuracy-speed trade-off.
- SVM-Linear+HOG achieves 99.35% in 1.62s — comparable to RBF at only 36% of training time.
- K-Means accuracy grows monotonically with k; the largest jump is k=1→4; gains plateau after k=16.

---

## Part III — Semi-Automatic Labelling Pipelines

**Dataset:** Indian Digits — 10,000 unlabelled 28×28 grayscale images (digits 0–9, 1,000/class).
**Baseline:** 10,000 × 10s = 100,000s ≈ 27.8 hours (fully manual).
**Target:** ≥ 99% oracle accuracy with minimum human time.

### Pipeline 1 — K-Means Bootstrapping + Active SVM Refinement

**Architecture:** HOG (1,296D, standardised) → K-Means (K=60) → human bootstrap (8 images/cluster, 20s/cluster) → RBF SVM (C=1, γ=scale) with sample weights (human: w=1000, cluster: w=1) → disagreement-first active boundary selection (40 images/iter, 10s each) → self-training → retrain loop.

**Results:**

| Metric | Value |
|---|---|
| Active iterations to ≥99% | 8 |
| Oracle accuracy | 99.07% |
| Bootstrap manual time | 1,200s (20.0 min) |
| Boundary labelling time | 3,200s (53.3 min) |
| **Total manual time** | **4,400s (73.3 min)** |
| Time saved vs. baseline | 95.6% |

### Pipeline 2 — Manual Seed + Augmentation + Self-Training

**Architecture:** Random 300-image seed (w=100) → 7× augmentation per image (rotations ±5°, Gaussian noise σ=0.05, shifts ±2px; w=1 for augmented) → HOG (64D) → RBF SVM (C=10, OVO) → margin-based active learning (20 images/iter, w=100) → self-training (top-50/class above 75th-percentile margin, w=1) → repeat.

**Results:**

| Metric | Value |
|---|---|
| Active iterations to ≥99% | 4 |
| Oracle accuracy | 99.02% |
| Practical GT accuracy (500-image held-out) | 98.40% |
| Pseudo-labels accepted / rejected | 1,332 / 20,154 (93.8% rejection rate) |
| Seed labelling time | 3,000s (50.0 min) |
| Boundary labelling time | 600s (10.0 min) |
| **Total manual time** | **3,600s (60.0 min)** |
| Time saved vs. baseline | 96.4% |

### Pipeline 3 — LLM-Based Zero-Shot Labelling with Agreement Validation

**Architecture:** 5 vision LLMs benchmarked on 500-image ground truth set (pre-existing, 0 extra manual seconds) → top-2 selected (gpt-5.4: 91.0%, gpt-4.1: 88.0%) → both independently label all 10,000 images → agreed labels accepted (9,566 images, 99.01% accuracy on agreed subset after 2 prompt refinement iterations) → 434 disagreements resolved manually (10s each).

**Prompt design:** System prompt framing LLM as Eastern Arabic-Indic numeral specialist; user prompt with visual shape guide, critical confusion-pair rules (e.g. 6 vs 7, 0 vs 5), ordered 10-step decision procedure, and 10 few-shot image examples. Two prompt refinement iterations raised agreed-label accuracy from ≈96% to 99.01%.

**Results:**

| Metric | Value |
|---|---|
| Agreement rate | 95.43% (9,543/10,000) |
| Accuracy on agreed labels | 99.01% |
| Disagreements resolved manually | 434 images |
| **Total manual time** | **4,340s (≈72.3 min)** |
| API cost | ≈$30 (combined gpt-5.4 + gpt-4.1) |
| Time saved vs. baseline | 97.4% |

### Overall Pipeline Comparison

| Metric | Pipeline 1 | Pipeline 2 | Pipeline 3 |
|---|---|---|---|
| Final oracle accuracy | 99.07% | 99.02% | 99.01% |
| Images manually handled | 800 | 360 | 434 |
| Total manual time | 4,400s (73.3 min) | **3,600s (60.0 min)** | 4,340s (72.3 min) |
| Time saved | 95.6% | 96.4% | **97.4%** |

All three pipelines achieve near-identical accuracy (≥99%) with broadly similar manual effort. Pipeline 2 requires the least hands-on time; Pipeline 3 requires the least manual image-by-image labelling but incurs API cost.

---

## Files

```
Assignment1/
├── Part1_Regression/
│   ├── insurance_regression.py
│   └── heating_oil_regression.py
├── Part2_ClassicalML/
│   └── Classical_ML_Modeling.py
└── Part3_Labelling/
    ├── Problem3
    |   └── Pipeline1.py
    ├── Problem4
    |   └── Pipeline2.py
    └── Problem5
        └── Pipeline3.py
        └── Agreement.py
        └── bench_gpt-4.1.csv
        └── bench_gpt-4.1_10k.csv
        └── bench_gpt-5.4.csv
        └── bench_gpt-5.4_10k.csv
        └── disagreed.csv
        └── pipelin3_log.txt
```