"""
================================================================================
  ReducedMNIST Classification — Script 1 of 3: DCT Features + MLP
================================================================================

REQUIREMENTS (requirements.txt):
    numpy>=1.24
    scipy>=1.10
    scikit-learn>=1.3
    Pillow>=10.0
    tabulate>=0.9

WHAT THIS SCRIPT DOES:
    Loads the ReducedMNIST dataset, extracts Discrete Cosine Transform (DCT)
    features from each image, and trains/evaluates scikit-learn MLPClassifiers
    with 1, 3, and 4 hidden-layer architectures.

HOW TO RUN:
    1. Set TRAIN_DIR and TEST_DIR in the CONFIG block below.
    2. pip install numpy scipy scikit-learn Pillow tabulate
    3. python script1_dct_mlp.py
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION BLOCK
# ============================================================
"""
All hyperparameters are centralized here. Change these values to
adapt the script to your folder layout and experimental needs.
"""
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Each directory must contain 10 subdirectories named "0" through "9".
    # Example layout:
    #   TRAIN_DIR/0/img_001.png
    #   TRAIN_DIR/1/img_002.png  ...
    "TRAIN_DIR": "ReducedMNIST/Reduced MNIST Data/Reduced Training data",
    "TEST_DIR":  "ReducedMNIST/Reduced MNIST Data/Reduced Testing data",

    # ── DCT Feature Extraction ─────────────────────────────────────────────
    # Number of DCT coefficients to keep (after zigzag ordering).
    # Images are 28×28 = 784 pixels. Keeping ~100–200 low-frequency
    # coefficients captures most structural information.
    # Fewer → faster, possibly less accurate.
    # More  → richer features, but may include noise.
    "N_DCT_COEFFS": 100,

    # ── MLP Architectures to Try ──────────────────────────────────────────
    # Each entry is a tuple of hidden-layer sizes passed to MLPClassifier.
    # (128,)           → 1 hidden layer  with 128 neurons
    # (256, 128, 64)   → 3 hidden layers with decreasing widths
    # (512, 256, 128, 64) → 4 hidden layers
    "MLP_ARCHITECTURES": {
        "1-hidden  (128)":          (128,),
        "3-hidden  (256-128-64)":   (256, 128, 64),
        "4-hidden  (512-256-128-64)": (512, 256, 128, 64),
    },

    # ── MLP Training ──────────────────────────────────────────────────────
    "MAX_ITER":    300,   # Max training epochs
    "RANDOM_SEED": 42,    # For full reproducibility
    "LEARNING_RATE_INIT": 1e-3,
}

# Fix random seeds globally
np.random.seed(CONFIG["RANDOM_SEED"])


# ============================================================
# SECTION 2 — DATA LOADING
# ============================================================
"""
CONCEPT — Loading images from a folder hierarchy:
    The dataset lives on disk as image files organized by class label.
    We walk each class sub-folder, open every image with PIL, convert to
    grayscale, and flatten to a 1-D numpy array. Labels are inferred from
    the sub-folder name (the digit character "0"–"9").

WHY GRAYSCALE?
    MNIST digits are grayscale by nature. Converting explicitly ensures
    consistent behavior regardless of whether images were saved as RGB.

NORMALIZATION (÷ 255):
    Pixel values [0, 255] → [0.0, 1.0]. This stabilizes gradient-based
    learning and is required before DCT so coefficient magnitudes are
    on a consistent scale across images.
"""

def load_dataset(root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from root_dir/<class_label>/ subdirectories.

    Args:
        root_dir: Path to the dataset split (train or test).

    Returns:
        X: float32 array of shape (N, 784) — pixel values in [0, 1].
        y: int array   of shape (N,)       — class labels 0–9.
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
            img_path = os.path.join(class_path, fname)
            img = Image.open(img_path).convert("L")   # grayscale
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr.flatten())              # 28×28 → 784
            labels.append(label)

        print(f"    Class {class_name}: loaded {len(files)} images")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ============================================================
# SECTION 3 — DCT FEATURE EXTRACTION
# ============================================================
"""
CONCEPT — Discrete Cosine Transform (DCT):
    DCT decomposes a signal into a sum of cosine waves at different
    frequencies. For images, it is applied to the flattened pixel vector
    (1-D DCT-II), transforming spatial intensities into frequency space.

    The most important visual information lives in LOW-frequency
    coefficients (slow variations = shapes, edges). HIGH-frequency
    coefficients encode fine-grained texture and noise.

ZIGZAG ORDERING:
    When the DCT is applied to the full 784-element vector, the
    coefficients are already ordered from low to high frequency.
    We simply keep the first N_DCT_COEFFS — this is equivalent to
    zigzag selection applied to the 1-D DCT.

WHY DCT FOR MNIST?
    ● Compression analogy: JPEG uses DCT and discards high-frequency
      coefficients with minimal perceptual loss — the same logic applies.
    ● Dimensionality reduction: 784 → N_DCT_COEFFS with minimal
      information loss.
    ● Robustness: DCT features are less sensitive to small pixel shifts
      than raw pixels.

NORMALIZATION:
    After extracting DCT features, we apply StandardScaler (zero mean,
    unit variance). DCT coefficients can span large ranges — the DC
    component (index 0) encodes mean brightness and can be much larger
    than AC components. Scaling prevents the MLP from being dominated
    by the DC term.
"""

def extract_dct_features(X: np.ndarray, n_coeffs: int) -> np.ndarray:
    """
    Apply 1-D DCT-II to each image and keep the n_coeffs lowest-frequency
    coefficients.

    Args:
        X:        (N, 784) pixel array, values in [0, 1].
        n_coeffs: Number of DCT coefficients to retain.

    Returns:
        (N, n_coeffs) float32 feature array.
    """
    # scipy's dct with norm='ortho' produces orthonormal basis vectors,
    # ensuring coefficients are on a comparable scale.
    dct_features = dct(X, type=2, axis=1, norm="ortho")
    return dct_features[:, :n_coeffs].astype(np.float32)


# ============================================================
# SECTION 4 — MLP TRAINING & EVALUATION
# ============================================================
"""
CONCEPT — Multilayer Perceptron (MLP):
    An MLP is a stack of fully-connected layers. Each neuron computes a
    weighted sum of its inputs, adds a bias, then passes the result through
    a non-linear activation function (ReLU here). By composing many such
    layers, the MLP can learn hierarchical representations.

ARCHITECTURE CHOICES:
    ● 1 hidden layer:  A linear classifier with some non-linearity.
                       Fast to train; may underfit complex patterns.
    ● 3 hidden layers: Standard deep architecture. Good capacity/speed
                       trade-off for most tabular/feature datasets.
    ● 4 hidden layers: Higher capacity; can model more complex decision
                       boundaries. Risk of overfitting on small datasets;
                       mitigated here by the limited 1000-example train set.

ACTIVATION — ReLU:
    Rectified Linear Unit: f(x) = max(0, x). Avoids vanishing gradients
    and trains faster than sigmoid/tanh for deep nets.

SOLVER — Adam:
    Adaptive Moment Estimation. Adjusts learning rates per-parameter
    using first and second moment estimates. Superior to vanilla SGD for
    most deep learning tasks.

EARLY STOPPING:
    Monitors a held-out validation split (10%). Training halts when
    validation loss stops improving, preventing overfitting.
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
    Train MLPClassifier and return accuracy + timing metrics.

    Returns:
        dict with keys: train_acc, test_acc, train_time_sec
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

    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc  = accuracy_score(y_test,  mlp.predict(X_test))

    return {
        "train_acc":      train_acc,
        "test_acc":       test_acc,
        "train_time_sec": train_time,
    }


# ============================================================
# SECTION 5 — MAIN PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("  Script 1 / 3 — DCT Feature Extraction + MLP Classification")
    print("=" * 65)

    # ── 5.1 Load Data ────────────────────────────────────────────────────
    print("\n[1/4] Loading training data...")
    t_load = time.perf_counter()
    X_train_raw, y_train = load_dataset(CONFIG["TRAIN_DIR"])
    print(f"\n[1/4] Loading test data...")
    X_test_raw, y_test = load_dataset(CONFIG["TEST_DIR"])
    load_time = time.perf_counter() - t_load

    print(f"\n  Training set : {X_train_raw.shape[0]} images  |  "
          f"Test set: {X_test_raw.shape[0]} images  |  "
          f"Load time: {load_time:.2f}s")

    # ── 5.2 DCT Extraction ───────────────────────────────────────────────
    print(f"\n[2/4] Extracting DCT features "
          f"(keeping {CONFIG['N_DCT_COEFFS']} coefficients)...")
    t_dct = time.perf_counter()
    X_train_dct = extract_dct_features(X_train_raw, CONFIG["N_DCT_COEFFS"])
    X_test_dct  = extract_dct_features(X_test_raw,  CONFIG["N_DCT_COEFFS"])
    dct_time = time.perf_counter() - t_dct
    print(f"  DCT shape: {X_train_dct.shape}  |  Extraction time: {dct_time:.3f}s")

    # ── 5.3 Feature Normalization ────────────────────────────────────────
    """
    StandardScaler: fits mean and std on TRAINING data only, then applies
    the same transform to test data. This prevents data leakage (the model
    must never "see" statistics of the test set during training).
    """
    print("\n[3/4] Normalizing DCT features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dct)
    X_test_scaled  = scaler.transform(X_test_dct)
    print(f"  Train mean ≈ {X_train_scaled.mean():.4f}  "
          f"(should be ~0.0 after scaling)")

    # ── 5.4 Train MLP Architectures ──────────────────────────────────────
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
            "Feature Method":  "DCT",
            "Architecture":    arch_name,
            "Hidden Layers":   str(hidden_layers),
            "Train Acc (%)":   f"{metrics['train_acc']*100:.2f}",
            "Test Acc (%)":    f"{metrics['test_acc']*100:.2f}",
            "Train Time (s)":  f"{metrics['train_time_sec']:.2f}",
        })
        print(f"     Train Acc: {metrics['train_acc']*100:.2f}%  |  "
              f"Test Acc: {metrics['test_acc']*100:.2f}%  |  "
              f"Time: {metrics['train_time_sec']:.2f}s\n")

    # ── 5.5 Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS — DCT + MLP")
    print("=" * 65)
    headers = ["Architecture", "Hidden Layers", "Train Acc (%)", "Test Acc (%)", "Train Time (s)"]
    rows = [[r["Architecture"], r["Hidden Layers"],
             r["Train Acc (%)"], r["Test Acc (%)"], r["Train Time (s)"]]
            for r in results]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Best model
    best = max(results, key=lambda r: float(r["Test Acc (%)"]))
    print(f"\n  ★ Best configuration: {best['Architecture']}  "
          f"→  Test Acc: {best['Test Acc (%)']}%\n")

    print("  Feature details:")
    print(f"    Input pixels   : 784")
    print(f"    DCT coeffs kept: {CONFIG['N_DCT_COEFFS']}")
    print(f"    Compression    : {CONFIG['N_DCT_COEFFS']/784*100:.1f}% of original features retained")
    print(f"    DCT extract time: {dct_time:.3f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
