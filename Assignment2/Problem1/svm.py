"""
================================================================================
  ReducedMNIST Classification — Script 5 of 5: SVM (Linear + RBF)
================================================================================

REQUIREMENTS (requirements.txt):
    numpy>=1.24
    scipy>=1.10
    scikit-learn>=1.3
    torch>=2.0
    Pillow>=10.0
    tabulate>=0.9

WHAT THIS SCRIPT DOES:
    Loads ReducedMNIST, extracts features via DCT / PCA / Autoencoder, then
    trains two SVM classifiers per feature set:
      ● Linear SVM  — fast, interpretable, strong on high-dim features
      ● RBF SVM     — non-linear kernel; hyperparameters tuned via GridSearchCV

HOW TO RUN:
    1. Set TRAIN_DIR and TEST_DIR in the CONFIG block below.
    2. pip install numpy scipy scikit-learn torch Pillow tabulate
    3. python script5_svm.py
================================================================================
"""

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================
import os
import time
import warnings
import numpy as np
from scipy.fftpack import dct
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION BLOCK
# ============================================================
"""
All hyperparameters are gathered here. Change these values — not the
code below — to experiment with different settings.
"""
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Expected layout: TRAIN_DIR/0/, TRAIN_DIR/1/, ..., TRAIN_DIR/9/
    "TRAIN_DIR": "ReducedMNIST/Reduced MNIST Data/Reduced Training data",
    "TEST_DIR":  "ReducedMNIST/Reduced MNIST Data/Reduced Testing data",

    # ── DCT ────────────────────────────────────────────────────────────────
    "N_DCT_COEFFS": 100,

    # ── PCA ────────────────────────────────────────────────────────────────
    "N_PCA_COMPONENTS": 100,
    "PCA_WHITEN":        True,

    # ── Autoencoder ────────────────────────────────────────────────────────
    "LATENT_DIM":     64,
    "ENCODER_HIDDEN": [512, 256],
    "AE_EPOCHS":      50,
    "AE_BATCH_SIZE":  128,
    "AE_LR":          1e-3,

    # ── Linear SVM ─────────────────────────────────────────────────────────
    # C controls the regularization strength:
    #   Small C → wider margin, more misclassifications tolerated (underfits).
    #   Large C → narrower margin, penalizes misclassifications heavily (may overfit).
    # C=1.0 is the default and a strong starting point for most datasets.
    "SVM_LINEAR_C": 1.0,

    # ── RBF SVM Grid Search ────────────────────────────────────────────────
    # C and gamma are the two critical hyperparameters for the RBF kernel.
    # GridSearchCV will try every combination using 3-fold cross-validation
    # on the training set, then refit the winner on all training data.
    #
    # C:     Regularization (same as linear). Larger C = less regularization.
    # gamma: Kernel bandwidth.
    #   'scale' → 1 / (n_features × X.var())   — adapts to feature scale.
    #   'auto'  → 1 / n_features                — simpler heuristic.
    #   0.01    → fixed small value, wide kernel (smoother decision boundary).
    "SVM_RBF_PARAM_GRID": {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.01],
    },
    "SVM_CV_FOLDS": 3,

    # ── Global ─────────────────────────────────────────────────────────────
    "RANDOM_SEED": 42,
}

# Fix random seeds
np.random.seed(CONFIG["RANDOM_SEED"])
torch.manual_seed(CONFIG["RANDOM_SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SECTION 2 — DATA LOADING
# ============================================================
"""
CONCEPT — Loading images from a folder hierarchy:
    We walk each class subdirectory (named "0"–"9"), open images with PIL,
    convert to grayscale, and flatten to a 784-element float32 vector.
    Labels are inferred from the folder name.

NORMALIZATION (÷255):
    Pixels are scaled to [0, 1] before feature extraction. This is
    especially important for SVM because the kernel functions compute
    dot products and distances — unscaled pixel values [0, 255] would
    create artificially large dot products, confusing the optimizer.
"""

def load_dataset(root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    class_dirs = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    print(f"  Found classes: {class_dirs}")
    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        label = int(class_name)
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        for fname in files:
            img = Image.open(os.path.join(class_path, fname)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr.flatten())
            labels.append(label)
        print(f"    Class {class_name}: {len(files)} images")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================
# SECTION 3 — FEATURE EXTRACTION: DCT
# ============================================================
"""
CONCEPT — Discrete Cosine Transform (DCT):
    DCT converts each flattened image into frequency-domain coefficients.
    The first N low-frequency coefficients capture overall digit shape;
    high-frequency coefficients capture noise. By retaining only N_DCT_COEFFS
    coefficients we get a compact, robust representation.

    For SVM, DCT is particularly valuable because:
    ● It reduces dimension (784 → 100), speeding up kernel computation.
    ● Low-frequency features are more discriminative than raw pixels.
    ● The linear SVM can exploit the structured frequency representation
      without needing a non-linear kernel.
"""

def extract_dct_features(X: np.ndarray, n_coeffs: int) -> np.ndarray:
    dct_out = dct(X, type=2, axis=1, norm="ortho")
    return dct_out[:, :n_coeffs].astype(np.float32)


# ============================================================
# SECTION 4 — FEATURE EXTRACTION: PCA
# ============================================================
"""
CONCEPT — Principal Component Analysis (PCA):
    PCA finds the directions of maximum variance in the training data and
    projects all samples onto the top-K of these directions. The result
    is a K-dimensional representation where dimensions are decorrelated
    and ordered by importance.

    PCA + SVM is a classic and powerful pipeline:
    ● PCA decorrelates features, which benefits both linear and RBF SVMs.
    ● Whitening makes each PC unit-variance, so the RBF kernel treats all
      dimensions equally (no single PC can dominate the distance metric).
    ● Reduced dimension (784 → 100) makes the SVM quadratic program much
      faster to solve.

CRITICAL: PCA is fit on TRAINING data only. The same transform is then
applied to test data without refitting (prevents data leakage).
"""

def fit_pca(X_train: np.ndarray, X_test: np.ndarray,
            n_components: int, whiten: bool,
            random_seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_seed)
    t0 = time.perf_counter()
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    elapsed = time.perf_counter() - t0
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components explain {var*100:.1f}% variance  "
          f"| fit time: {elapsed:.2f}s")
    return X_tr.astype(np.float32), X_te.astype(np.float32), elapsed


# ============================================================
# SECTION 5 — FEATURE EXTRACTION: AUTOENCODER (PyTorch)
# ============================================================
"""
CONCEPT — Autoencoder Features:
    A fully-connected autoencoder is trained to reconstruct its pixel input
    through a narrow bottleneck layer. The bottleneck's output (LATENT_DIM
    values) is a compact, learned representation of each digit.

    Why use autoencoder features with SVM?
    ● The autoencoder can capture non-linear structure that PCA and DCT miss.
    ● The bottleneck is small (64-d), making SVM training fast.
    ● The learned representation groups visually similar digits together,
      which benefits both linear and RBF SVMs.

    Training is UNSUPERVISED — the autoencoder sees images, not labels.
    Labels are used only by the SVM classifier downstream.

ARCHITECTURE:
    Encoder: 784 → 512 → 256 → 64   (ReLU activations)
    Decoder: 64  → 256 → 512 → 784  (ReLU + Sigmoid)
    Loss:    MSE(input, reconstruction)
"""

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_train: np.ndarray, config: dict) -> tuple[Autoencoder, float]:
    model = Autoencoder(
        X_train.shape[1], config["ENCODER_HIDDEN"], config["LATENT_DIM"]
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["AE_LR"])
    loss_fn   = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=config["AE_BATCH_SIZE"], shuffle=True,
        generator=torch.Generator().manual_seed(config["RANDOM_SEED"]),
    )
    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, config["AE_EPOCHS"] + 1):
        losses = []
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            _, xr = model(bx)
            loss = loss_fn(xr, bx)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{config['AE_EPOCHS']}  "
                  f"MSE: {np.mean(losses):.5f}  "
                  f"elapsed: {time.perf_counter()-t0:.1f}s")
    elapsed = time.perf_counter() - t0
    model.eval().to("cpu")
    return model, elapsed


@torch.no_grad()
def extract_latent(model: Autoencoder, X: np.ndarray,
                   batch_size: int = 256) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32)
    parts = []
    for i in range(0, len(X_t), batch_size):
        parts.append(model.encode(X_t[i:i+batch_size]).numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ============================================================
# SECTION 6 — SVM CLASSIFIERS
# ============================================================
"""
CONCEPT — Support Vector Machines (SVM):
    An SVM finds the hyperplane that separates classes with the MAXIMUM
    MARGIN — the largest possible gap between the nearest data points of
    each class (called support vectors). This max-margin principle makes
    SVMs highly effective, especially on small datasets where overfitting
    is a real concern.

    For multi-class problems (10 digits), scikit-learn's SVC uses a
    one-vs-one strategy: it trains C(10,2)=45 binary SVMs and combines
    their votes to predict the final class.

WHY SVM OFTEN OUTPERFORMS MLP ON SMALL DATASETS:
    ● SVM's solution depends only on the support vectors (a subset of
      training points), making it less sensitive to dataset size.
    ● The max-margin objective is a strong inductive bias that generalizes
      well from limited examples.
    ● MLPs need many examples to reliably learn non-linear boundaries via
      gradient descent; SVMs can find optimal boundaries analytically.

NORMALIZATION IS CRITICAL FOR SVM:
    SVM computes distances and dot products between feature vectors. If
    one feature has range [0, 1000] and another [0, 1], the large-range
    feature will dominate the kernel function, effectively ignoring the
    small-range feature. StandardScaler (zero mean, unit variance, fitted
    on TRAIN only) puts all features on an equal footing.

──────────────────────────────────────────────────────────────
LINEAR KERNEL: K(x, z) = x · z
    The decision boundary is a hyperplane in the original feature space.
    Works well when classes are (approximately) linearly separable.
    For DCT and PCA features — which are already in a structured, compact
    space — linear separation is often sufficient.
    Speed advantage: training time scales roughly as O(N × d), much
    faster than RBF for high-dimensional data.

RBF (Gaussian) KERNEL: K(x, z) = exp(-gamma × ||x - z||²)
    Implicitly maps data to an infinite-dimensional feature space where
    a linear hyperplane becomes a curved decision boundary in the original
    space. This allows the SVM to model arbitrarily complex class shapes.

    ● Small gamma → wide Gaussian, smooth boundary (may underfit).
    ● Large gamma → narrow Gaussian, boundary hugs training points (may overfit).
    ● Small C     → wide margin, more misclassifications allowed.
    ● Large C     → narrow margin, fewer misclassifications (may overfit).

    GridSearchCV exhaustively tries all (C, gamma) combinations with
    3-fold cross-validation and returns the combination with the highest
    mean validation accuracy.
"""

def train_linear_svm(
    X_train, y_train, X_test, y_test, config
) -> dict:
    """Fit a linear SVM and return metrics."""
    svm = SVC(
        kernel="linear",
        C=config["SVM_LINEAR_C"],
        random_state=config["RANDOM_SEED"],
    )
    t0 = time.perf_counter()
    svm.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    train_acc = accuracy_score(y_train, svm.predict(X_train))
    test_acc  = accuracy_score(y_test,  svm.predict(X_test))

    return {
        "kernel":     "Linear",
        "best_params": f"C={config['SVM_LINEAR_C']}",
        "train_acc":  train_acc,
        "test_acc":   test_acc,
        "fit_time":   fit_time,
    }


def train_rbf_svm(
    X_train, y_train, X_test, y_test, config
) -> dict:
    """
    Fit an RBF SVM with GridSearchCV for hyperparameter tuning.
    GridSearchCV trains n_folds × n_param_combos SVMs internally,
    then refits the best model on the full training set.
    """
    print(f"    Running GridSearchCV ({config['SVM_CV_FOLDS']}-fold, "
          f"{len(config['SVM_RBF_PARAM_GRID']['C']) * len(config['SVM_RBF_PARAM_GRID']['gamma'])} "
          f"param combos)...")

    base_svm = SVC(kernel="rbf", random_state=config["RANDOM_SEED"])
    grid_search = GridSearchCV(
        base_svm,
        param_grid=config["SVM_RBF_PARAM_GRID"],
        cv=config["SVM_CV_FOLDS"],
        scoring="accuracy",
        n_jobs=-1,       # use all available CPU cores
        verbose=0,
    )

    t0 = time.perf_counter()
    grid_search.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    best_params = grid_search.best_params_
    best_svm    = grid_search.best_estimator_
    print(f"    Best params: C={best_params['C']}  gamma={best_params['gamma']}  "
          f"| CV score: {grid_search.best_score_*100:.2f}%")

    train_acc = accuracy_score(y_train, best_svm.predict(X_train))
    test_acc  = accuracy_score(y_test,  best_svm.predict(X_test))

    return {
        "kernel":      "RBF",
        "best_params": f"C={best_params['C']}, γ={best_params['gamma']}",
        "train_acc":   train_acc,
        "test_acc":    test_acc,
        "fit_time":    fit_time,
    }


# ============================================================
# SECTION 7 — PIPELINE RUNNER
# ============================================================

def run_svms_on_features(
    feature_name: str,
    X_train_feat: np.ndarray, y_train: np.ndarray,
    X_test_feat:  np.ndarray, y_test:  np.ndarray,
    config: dict,
    results: list,
):
    """
    Normalize features then fit both Linear and RBF SVMs.

    Normalization note:
        StandardScaler is fit on TRAINING features only. Applying it to
        test features uses the training mean/std — this is the correct
        procedure and avoids data leakage. Even a small amount of test
        set information leaking into the scaler can inflate test accuracy
        on small datasets like ReducedMNIST.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_feat)
    X_te = scaler.transform(X_test_feat)
    print(f"  Features normalized  |  "
          f"mean={X_tr.mean():.4f}  std={X_tr.std():.4f}")

    # ── Linear SVM ────────────────────────────────────────────────────────
    print(f"  ▶ Linear SVM (C={config['SVM_LINEAR_C']})...")
    lin = train_linear_svm(X_tr, y_train, X_te, y_test, config)
    results.append({
        "Feature Method": feature_name,
        "Kernel":         lin["kernel"],
        "Best Params":    lin["best_params"],
        "Train Acc (%)":  f"{lin['train_acc']*100:.2f}",
        "Test Acc (%)":   f"{lin['test_acc']*100:.2f}",
        "Fit Time (s)":   f"{lin['fit_time']:.2f}",
    })
    print(f"    Train: {lin['train_acc']*100:.2f}%  |  "
          f"Test: {lin['test_acc']*100:.2f}%  |  "
          f"Time: {lin['fit_time']:.2f}s")

    # ── RBF SVM ───────────────────────────────────────────────────────────
    print(f"  ▶ RBF SVM (GridSearchCV)...")
    rbf = train_rbf_svm(X_tr, y_train, X_te, y_test, config)
    results.append({
        "Feature Method": feature_name,
        "Kernel":         rbf["kernel"],
        "Best Params":    rbf["best_params"],
        "Train Acc (%)":  f"{rbf['train_acc']*100:.2f}",
        "Test Acc (%)":   f"{rbf['test_acc']*100:.2f}",
        "Fit Time (s)":   f"{rbf['fit_time']:.2f}",
    })
    print(f"    Train: {rbf['train_acc']*100:.2f}%  |  "
          f"Test: {rbf['test_acc']*100:.2f}%  |  "
          f"Time: {rbf['fit_time']:.2f}s")


# ============================================================
# SECTION 8 — MAIN PIPELINE
# ============================================================

def main():
    print("=" * 72)
    print("  Script 5 / 5 — SVM Classification (Linear + RBF)")
    print("  Feature methods: DCT | PCA | Autoencoder")
    print("=" * 72)
    print(f"  Device (for AE): {DEVICE}\n")

    # ── Load Data ─────────────────────────────────────────────────────────
    print("[1/7] Loading training data...")
    t0 = time.perf_counter()
    X_train_raw, y_train = load_dataset(CONFIG["TRAIN_DIR"])
    print("\n[1/7] Loading test data...")
    X_test_raw, y_test = load_dataset(CONFIG["TEST_DIR"])
    print(f"\n  Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}  "
          f"|  Load time: {time.perf_counter()-t0:.2f}s")

    results = []

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE A — DCT + SVM
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PIPELINE A: DCT Features ({CONFIG['N_DCT_COEFFS']} coefficients)")
    print(f"{'='*72}")
    print("[2/7] Extracting DCT features...")
    X_tr_dct = extract_dct_features(X_train_raw, CONFIG["N_DCT_COEFFS"])
    X_te_dct = extract_dct_features(X_test_raw,  CONFIG["N_DCT_COEFFS"])
    print(f"  Shape: {X_tr_dct.shape}")

    print("\n[3/7] Training SVMs on DCT features...")
    run_svms_on_features("DCT", X_tr_dct, y_train, X_te_dct, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE B — PCA + SVM
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PIPELINE B: PCA Features ({CONFIG['N_PCA_COMPONENTS']} components, "
          f"whiten={CONFIG['PCA_WHITEN']})")
    print(f"{'='*72}")
    print("[4/7] Fitting PCA...")
    X_tr_pca, X_te_pca, _ = fit_pca(
        X_train_raw, X_test_raw,
        CONFIG["N_PCA_COMPONENTS"], CONFIG["PCA_WHITEN"], CONFIG["RANDOM_SEED"]
    )

    print("\n[5/7] Training SVMs on PCA features...")
    run_svms_on_features("PCA", X_tr_pca, y_train, X_te_pca, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE C — AUTOENCODER + SVM
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PIPELINE C: Autoencoder Features (latent dim = {CONFIG['LATENT_DIM']})")
    print(f"{'='*72}")
    print(f"[6/7] Training autoencoder ({CONFIG['AE_EPOCHS']} epochs)...")
    ae_model, ae_time = train_autoencoder(X_train_raw, CONFIG)
    print(f"  AE training complete in {ae_time:.1f}s")
    print("  Extracting latent features...")
    X_tr_ae = extract_latent(ae_model, X_train_raw)
    X_te_ae = extract_latent(ae_model, X_test_raw)
    print(f"  Latent shape: {X_tr_ae.shape}")

    print("\n[7/7] Training SVMs on Autoencoder features...")
    run_svms_on_features("Autoencoder", X_tr_ae, y_train, X_te_ae, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # FINAL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FINAL RESULTS — SVM (Linear + RBF) across all Feature Methods")
    print(f"{'='*72}")
    headers = [
        "Feature Method", "Kernel", "Best Params",
        "Train Acc (%)", "Test Acc (%)", "Fit Time (s)"
    ]
    rows = [
        [r["Feature Method"], r["Kernel"], r["Best Params"],
         r["Train Acc (%)"], r["Test Acc (%)"], r["Fit Time (s)"]]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Best result
    best = max(results, key=lambda r: float(r["Test Acc (%)"]))
    print(f"\n  ★ Best overall:  Feature={best['Feature Method']}  "
          f"Kernel={best['Kernel']}  Params={best['Best Params']}")
    print(f"    → Test Acc: {best['Test Acc (%)']}%  |  "
          f"Train Acc: {best['Train Acc (%)']}%  |  "
          f"Fit Time: {best['Fit Time (s)']}s")

    # Linear vs RBF summary
    print(f"\n  Linear SVM results:")
    for r in [x for x in results if x["Kernel"] == "Linear"]:
        print(f"    {r['Feature Method']:12s}  Test: {r['Test Acc (%)']}%")
    print(f"\n  RBF SVM results:")
    for r in [x for x in results if x["Kernel"] == "RBF"]:
        print(f"    {r['Feature Method']:12s}  Test: {r['Test Acc (%)']}%  "
              f"Best params: {r['Best Params']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
