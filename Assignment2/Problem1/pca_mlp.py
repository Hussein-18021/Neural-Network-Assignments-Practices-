"""
================================================================================
  ReducedMNIST Classification — Script 2 of 3: PCA Features + MLP
================================================================================

REQUIREMENTS (requirements.txt):
    numpy>=1.24
    scikit-learn>=1.3
    Pillow>=10.0
    tabulate>=0.9

WHAT THIS SCRIPT DOES:
    Loads the ReducedMNIST dataset, reduces dimensionality with Principal
    Component Analysis (PCA), and trains/evaluates scikit-learn MLPClassifiers
    with 1, 3, and 4 hidden-layer architectures.

HOW TO RUN:
    1. Set TRAIN_DIR and TEST_DIR in the CONFIG block below.
    2. pip install numpy scikit-learn Pillow tabulate
    3. python script2_pca_mlp.py
================================================================================
"""

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================
import os
import time
import warnings
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION BLOCK
# ============================================================
"""
All hyperparameters live here. Edit this block before running;
do not scatter magic numbers through the rest of the code.
"""
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Each directory must contain subdirectories named "0" through "9",
    # each holding image files for that digit class.
    "TRAIN_DIR": "ReducedMNIST/Reduced MNIST Data/Reduced Training data",
    "TEST_DIR":  "ReducedMNIST/Reduced MNIST Data/Reduced Testing data",

    # ── PCA Configuration ──────────────────────────────────────────────────
    # N_COMPONENTS: number of principal components to retain.
    #   ● Too few  → loses discriminative information, lower accuracy.
    #   ● Too many → captures noise, may overfit.
    #   A common heuristic: choose enough PCs to explain ≥ 95% variance.
    #   For MNIST-like data, ~50–150 components are typically sufficient.
    #
    # WHITEN: if True, each component is divided by its singular value,
    #   making the projected features unit variance. This often helps
    #   downstream linear/MLP classifiers converge faster.
    "N_COMPONENTS": 100,
    "WHITEN":       True,

    # ── MLP Architectures to Try ──────────────────────────────────────────
    # Tuple format matches scikit-learn's hidden_layer_sizes parameter.
    # We test shallow (1 layer), medium (3 layers), and deep (4 layers).
    "MLP_ARCHITECTURES": {
        "1-hidden  (128)":            (128,),
        "3-hidden  (256-128-64)":     (256, 128, 64),
        "4-hidden  (512-256-128-64)": (512, 256, 128, 64),
    },

    # ── MLP Training ──────────────────────────────────────────────────────
    "MAX_ITER":           300,
    "RANDOM_SEED":        42,
    "LEARNING_RATE_INIT": 1e-3,
}

# Fix global random seeds
np.random.seed(CONFIG["RANDOM_SEED"])


# ============================================================
# SECTION 2 — DATA LOADING
# ============================================================
"""
CONCEPT — Loading images from a folder hierarchy:
    We traverse TRAIN_DIR (or TEST_DIR) and for each subdirectory (named
    "0"–"9") we open every image, convert to grayscale, and flatten to a
    1-D vector of 784 floats.

NORMALIZATION (÷ 255):
    PCA works on variance in the data. Pixel values must be on a consistent
    scale first. Dividing by 255 brings all values into [0, 1], which is
    the natural range before applying PCA.

    Note: PCA will FURTHER center the data (subtract column means) during
    fitting — this is part of what 'fit_transform' does internally.
"""

def load_dataset(root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from root_dir/<class_label>/ subdirectories.

    Args:
        root_dir: Path to train or test split root.

    Returns:
        X: float32 array of shape (N, 784), pixel values in [0, 1].
        y: int64  array of shape (N,),      class labels 0–9.
    """
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

        print(f"    Class {class_name}: loaded {len(files)} images")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================
# SECTION 3 — PCA FEATURE EXTRACTION
# ============================================================
"""
CONCEPT — Principal Component Analysis (PCA):
    PCA finds the directions (principal components) in feature space along
    which the data varies the most. By projecting the data onto the top-K
    principal components, we get a K-dimensional representation that
    captures maximum variance with minimum dimensions.

INTUITION FOR IMAGES:
    ● PC 1 typically encodes overall brightness.
    ● PC 2 might encode a horizontal intensity gradient.
    ● Higher PCs capture increasingly fine-grained patterns.
    ● Low-variance PCs mostly capture noise — discarding them is beneficial.

    Think of PCA as finding the "eigenfaces" of handwritten digits: a set
    of prototype digit shapes that can be linearly combined to reconstruct
    any digit in the dataset.

FITTING RULE — Train Only:
    PCA is fitted ONLY on training data. We then apply the learned
    transformation to test data. This is critical to avoid data leakage:
    the test set must be invisible during all fitting steps.

WHITENING:
    When whiten=True, each projected dimension is normalized by its
    standard deviation. This removes correlations between PCs and makes
    the MLP's weight initialization more effective (each input dimension
    contributes equally at the start).

VARIANCE EXPLAINED:
    We print the cumulative explained variance ratio after fitting, which
    tells us what percentage of the original information is retained.
    A value near 1.0 means very little information is lost.
"""

def fit_pca_and_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    whiten: bool,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on training data and transform both splits.

    Returns:
        X_train_pca, X_test_pca: projected arrays of shape (N, n_components).
        pca: fitted PCA object (for inspecting explained variance).
    """
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_seed)

    t0 = time.perf_counter()
    X_train_pca = pca.fit_transform(X_train)   # fit + project train
    X_test_pca  = pca.transform(X_test)         # project test only
    pca_time = time.perf_counter() - t0

    cumvar = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA fitted: {n_components} components explain "
          f"{cumvar*100:.2f}% of training variance  |  Time: {pca_time:.3f}s")

    return X_train_pca.astype(np.float32), X_test_pca.astype(np.float32), pca, pca_time


# ============================================================
# SECTION 4 — MLP TRAINING & EVALUATION
# ============================================================
"""
CONCEPT — Multilayer Perceptron (MLP):
    An MLP is a directed acyclic graph of neurons arranged in layers.
    Each layer applies a linear transformation (weights × inputs + bias)
    followed by a non-linear activation function (ReLU).

    Training minimizes cross-entropy loss via backpropagation + Adam.

ARCHITECTURE RATIONALE:
    ● 1 hidden layer  : Shallow network. Acts like a generalized linear
                        model with one non-linear transform. Fast; good
                        baseline.
    ● 3 hidden layers : Can model more complex decision boundaries. Each
                        layer can compose features from the previous layer,
                        building increasingly abstract representations.
    ● 4 hidden layers : Highest capacity. Risk of overfitting on small
                        datasets (1000 examples/class). Early stopping
                        and validation monitoring mitigate this.

NORMALIZATION REQUIREMENT:
    PCA with whitening already produces zero-mean, unit-variance features.
    We apply StandardScaler anyway (as a safeguard) fitted only on train
    features. When whitening is on, the scaler is nearly a no-op.
"""

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layers: tuple,
    config: dict,
) -> dict:
    """
    Train MLPClassifier and measure train/test accuracy and wall time.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=config["LEARNING_RATE_INIT"],
        max_iter=config["MAX_ITER"],
        early_stopping=True,
        validation_fraction=0.1,
        random_state=config["RANDOM_SEED"],
        verbose=False,
    )

    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    return {
        "train_acc":      accuracy_score(y_train, mlp.predict(X_train)),
        "test_acc":       accuracy_score(y_test,  mlp.predict(X_test)),
        "train_time_sec": train_time,
    }


# ============================================================
# SECTION 5 — MAIN PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("  Script 2 / 3 — PCA Feature Extraction + MLP Classification")
    print("=" * 65)

    # ── 5.1 Load Data ────────────────────────────────────────────────────
    print("\n[1/4] Loading training data...")
    t_load = time.perf_counter()
    X_train_raw, y_train = load_dataset(CONFIG["TRAIN_DIR"])
    print("\n[1/4] Loading test data...")
    X_test_raw, y_test = load_dataset(CONFIG["TEST_DIR"])
    load_time = time.perf_counter() - t_load
    print(f"\n  Train: {X_train_raw.shape}  |  "
          f"Test: {X_test_raw.shape}  |  Load time: {load_time:.2f}s")

    # ── 5.2 PCA Extraction ───────────────────────────────────────────────
    print(f"\n[2/4] Fitting PCA "
          f"({CONFIG['N_COMPONENTS']} components, whiten={CONFIG['WHITEN']})...")
    X_train_pca, X_test_pca, pca_model, pca_time = fit_pca_and_transform(
        X_train_raw, X_test_raw,
        CONFIG["N_COMPONENTS"],
        CONFIG["WHITEN"],
        CONFIG["RANDOM_SEED"],
    )

    # Print per-component variance breakdown (first 10 PCs)
    var_ratios = pca_model.explained_variance_ratio_
    print(f"  Variance explained by first 10 PCs: "
          f"{var_ratios[:10].sum()*100:.1f}%")
    print(f"  Variance explained by all {CONFIG['N_COMPONENTS']} PCs: "
          f"{var_ratios.sum()*100:.1f}%")

    # ── 5.3 Normalize PCA Features ───────────────────────────────────────
    """
    Even with PCA whitening, we apply StandardScaler for consistency.
    When whiten=True, this is essentially a no-op (features are already
    ~unit variance). When whiten=False, this step is important.
    """
    print("\n[3/4] Normalizing PCA features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled  = scaler.transform(X_test_pca)
    print(f"  Post-scaling mean ≈ {X_train_scaled.mean():.5f}  "
          f"(should be ~0.0)")

    # ── 5.4 Train All MLP Architectures ──────────────────────────────────
    print("\n[4/4] Training MLP classifiers...\n")
    results = []

    for arch_name, hidden_layers in CONFIG["MLP_ARCHITECTURES"].items():
        print(f"  ▶ Architecture: {arch_name} — hidden layers: {hidden_layers}")
        metrics = train_and_evaluate(
            X_train_scaled, y_train,
            X_test_scaled,  y_test,
            hidden_layers,  CONFIG,
        )
        results.append({
            "Architecture":   arch_name,
            "Hidden Layers":  str(hidden_layers),
            "Train Acc (%)":  f"{metrics['train_acc']*100:.2f}",
            "Test Acc (%)":   f"{metrics['test_acc']*100:.2f}",
            "Train Time (s)": f"{metrics['train_time_sec']:.2f}",
        })
        print(f"     Train Acc: {metrics['train_acc']*100:.2f}%  |  "
              f"Test Acc: {metrics['test_acc']*100:.2f}%  |  "
              f"Time: {metrics['train_time_sec']:.2f}s\n")

    # ── 5.5 Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS — PCA + MLP")
    print("=" * 65)
    headers = ["Architecture", "Hidden Layers", "Train Acc (%)", "Test Acc (%)", "Train Time (s)"]
    rows = [[r["Architecture"], r["Hidden Layers"],
             r["Train Acc (%)"], r["Test Acc (%)"], r["Train Time (s)"]]
            for r in results]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    best = max(results, key=lambda r: float(r["Test Acc (%)"]))
    print(f"\n  ★ Best configuration: {best['Architecture']}  "
          f"→  Test Acc: {best['Test Acc (%)']}%\n")

    print("  Feature details:")
    print(f"    Input pixels     : 784")
    print(f"    PCA components   : {CONFIG['N_COMPONENTS']}")
    print(f"    Whitened         : {CONFIG['WHITEN']}")
    print(f"    Variance retained: "
          f"{pca_model.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"    PCA fit+transform: {pca_time:.3f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
