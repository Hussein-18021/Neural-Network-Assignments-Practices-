"""
================================================================================
  ReducedMNIST Classification — Script 4 of 5: K-Means Clustering
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
    applies K-Means clustering with K ∈ {1, 4, 16, 32} to each feature set.
    Cluster IDs are mapped to digit classes via majority-vote, and classification
    accuracy is reported for every (feature method × K) combination.

HOW TO RUN:
    1. Set TRAIN_DIR and TEST_DIR in the CONFIG block below.
    2. pip install numpy scipy scikit-learn torch Pillow tabulate
    3. python script4_kmeans.py
================================================================================
"""

# ============================================================
# SECTION 0 — IMPORTS
# ============================================================
import os
import time
import warnings
import numpy as np
from collections import Counter
from scipy.fftpack import dct
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION BLOCK
# ============================================================
"""
All hyperparameters are centralized here. Edit this block before running.
No magic numbers are scattered elsewhere in the script.
"""
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Folder layout: TRAIN_DIR/0/, TRAIN_DIR/1/, ..., TRAIN_DIR/9/
    "TRAIN_DIR": "ReducedMNIST/Reduced MNIST Data/Reduced Training data",
    "TEST_DIR":  "ReducedMNIST/Reduced MNIST Data/Reduced Testing data",

    # ── K-Means ────────────────────────────────────────────────────────────
    # K=1:  Degenerate baseline — one centroid covers all data. The majority-
    #       vote mapping assigns every sample to the single most common digit
    #       in the training set (expected accuracy ~10% on balanced data).
    # K=4:  Coarse groupings. Each cluster may mix several digit classes.
    # K=16: More refined. Some clusters may specialize in one digit variant.
    # K=32: ~3 sub-clusters per class on average. Highest expected accuracy.
    "K_VALUES": [1, 4, 16, 32],

    # KMeans training options
    "KMEANS_INIT":    "k-means++",  # Smart init reduces bad local minima
    "KMEANS_N_INIT":  10,           # Re-runs with different seeds, keeps best
    "KMEANS_MAX_ITER": 300,

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

    # ── Global ─────────────────────────────────────────────────────────────
    "RANDOM_SEED": 42,
}

# Fix all random seeds
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
    We walk each class subdirectory (named "0"–"9"), open every image
    with PIL, convert to grayscale, and flatten to a 784-element float32
    vector. Labels are inferred from the folder name.

NORMALIZATION (÷255):
    All pixel values are scaled to [0, 1] before any feature extraction.
    This is required so that:
      ● DCT coefficients are on a natural scale.
      ● PCA computes meaningful variance.
      ● The autoencoder's Sigmoid output matches the input range.
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
    DCT decomposes each flattened image vector into cosine-frequency
    components. The first N coefficients capture the lowest frequencies
    (overall shape, broad strokes), while later coefficients encode
    fine texture and noise. Keeping only the low-frequency coefficients
    achieves compact, noise-robust representation — the same principle
    used in JPEG compression.

    We use scipy's 1-D DCT-II with orthonormal normalization (norm='ortho')
    applied row-wise, then slice the first N_DCT_COEFFS columns.
"""

def extract_dct_features(X: np.ndarray, n_coeffs: int) -> np.ndarray:
    dct_out = dct(X, type=2, axis=1, norm="ortho")
    return dct_out[:, :n_coeffs].astype(np.float32)


# ============================================================
# SECTION 4 — FEATURE EXTRACTION: PCA
# ============================================================
"""
CONCEPT — Principal Component Analysis (PCA):
    PCA rotates the feature space so that the new axes (principal
    components) are ordered by the amount of variance they explain.
    By keeping only the top K components we discard dimensions that
    mostly encode noise, achieving dimensionality reduction with
    minimal information loss.

    CRITICAL: PCA is fit on TRAINING data only. Applying the learned
    projection to the test set avoids data leakage — the model never
    sees test-set statistics during training.

    Whitening (dividing each PC by its standard deviation) normalizes
    all components to unit variance, which helps distance-based methods
    like K-Means treat every dimension equally.
"""

def fit_pca(X_train: np.ndarray, X_test: np.ndarray,
            n_components: int, whiten: bool,
            random_seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_seed)
    t0 = time.perf_counter()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    elapsed = time.perf_counter() - t0
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components explain {var*100:.1f}% variance  "
          f"| fit time: {elapsed:.2f}s")
    return X_train_pca.astype(np.float32), X_test_pca.astype(np.float32), elapsed


# ============================================================
# SECTION 5 — FEATURE EXTRACTION: AUTOENCODER (PyTorch)
# ============================================================
"""
CONCEPT — Autoencoder Feature Learning:
    An autoencoder is trained to compress each image into a small latent
    vector (the bottleneck) and then reconstruct the original pixel values.
    Because the bottleneck is much smaller than the input, the encoder
    is forced to learn a compact, information-dense representation.

    After training, we discard the decoder and use the encoder's output
    (LATENT_DIM-dimensional vector) as the feature for each image.
    This is an UNSUPERVISED approach — the autoencoder sees no labels.

ARCHITECTURE:
    Encoder: 784 → 512 → 256 → LATENT_DIM  (ReLU activations)
    Decoder: LATENT_DIM → 256 → 512 → 784  (ReLU + Sigmoid output)

LOSS: MSE between input pixels and reconstructed pixels.
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
# SECTION 6 — K-MEANS CLUSTERING + MAJORITY-VOTE MAPPING
# ============================================================
"""
CONCEPT — K-Means Clustering:
    K-Means partitions N data points into K clusters by iteratively:
      1. Assigning each point to its nearest centroid (Euclidean distance).
      2. Recomputing each centroid as the mean of its assigned points.
    This repeats until assignments stop changing (convergence).

    K-Means is UNSUPERVISED — it sees no class labels. It groups points
    by geometric proximity in feature space, not by semantic meaning.

WHY K MATTERS:
    ● K=1  → Single centroid = grand mean of all data. Every sample maps
              to the same cluster → predicted as the most common digit.
              Accuracy ≈ 10% on a balanced 10-class dataset (the trivial
              baseline). Included purely to show what "no clustering" means.
    ● K=4  → Rough groupings. Each cluster is a broad region of feature
              space that likely blends multiple digit classes.
    ● K=16 → Finer resolution. Some clusters may specialize (e.g., one
              cluster captures most "1"s, another captures "0"s).
    ● K=32 → ~3 sub-clusters per digit class. Best expected accuracy
              because different writing styles of the same digit (upright vs.
              slanted "7", looped vs. open "4") can occupy separate clusters.

MAJORITY-VOTE LABEL MAPPING:
    K-Means produces cluster IDs (0 … K-1), not digit labels (0 … 9).
    To evaluate classification accuracy we need to convert cluster IDs
    to digit predictions. The majority-vote rule:
      For each cluster c:
        predicted_class[c] = most common true label among training
                             members assigned to cluster c.
    This is done on training data only. The mapping is then frozen and
    applied to test data. Note that multiple clusters can map to the
    same digit class — this is intentional and expected when K > 10.

WHY ACCURACY IS LOWER THAN SUPERVISED METHODS:
    K-Means optimizes for geometric compactness, NOT class separability.
    A perfect cluster (all same digit) is geometrically tight, but K-Means
    has no incentive to find it — it just minimizes within-cluster variance.
    Supervised methods (MLP, SVM) directly optimize a classification
    objective, giving them a decisive accuracy advantage.
"""

def kmeans_classify(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    k: int, config: dict,
) -> dict:
    """
    Fit K-Means, build majority-vote mapping, compute train/test accuracy.
    """
    # ── K=1 edge case note ────────────────────────────────────────────────
    # With K=1 there is exactly one cluster containing all training samples.
    # The majority-vote assigns it the label of the most frequent digit.
    # On a balanced dataset (1000 examples per digit) this is essentially
    # random tie-breaking among digits → expected accuracy ≈ 10%.
    # The code below handles K=1 without special-casing — it just works.

    km = KMeans(
        n_clusters=k,
        init=config["KMEANS_INIT"],
        n_init=config["KMEANS_N_INIT"],
        max_iter=config["KMEANS_MAX_ITER"],
        random_state=config["RANDOM_SEED"],
    )

    t0 = time.perf_counter()
    train_cluster_ids = km.fit_predict(X_train)  # fit + assign train
    test_cluster_ids  = km.predict(X_test)        # assign test (no refit)
    elapsed = time.perf_counter() - t0

    # ── Build majority-vote map: cluster_id → digit class ─────────────────
    cluster_to_label = {}
    for cid in range(k):
        mask = train_cluster_ids == cid
        if mask.sum() == 0:
            # Empty cluster (rare with k-means++): default to class 0
            cluster_to_label[cid] = 0
        else:
            cluster_to_label[cid] = Counter(y_train[mask]).most_common(1)[0][0]

    # ── Apply mapping to get predicted labels ─────────────────────────────
    y_train_pred = np.array([cluster_to_label[cid] for cid in train_cluster_ids])
    y_test_pred  = np.array([cluster_to_label[cid] for cid in test_cluster_ids])

    return {
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc":  accuracy_score(y_test,  y_test_pred),
        "fit_time":  elapsed,
    }


# ============================================================
# SECTION 7 — MAIN PIPELINE
# ============================================================

def run_kmeans_on_features(
    feature_name: str,
    X_train_feat: np.ndarray, y_train: np.ndarray,
    X_test_feat:  np.ndarray, y_test:  np.ndarray,
    config: dict,
    results: list,
):
    """Normalize features and run K-Means for all K values."""
    # ── Normalize ─────────────────────────────────────────────────────────
    """
    K-Means uses Euclidean distance. If one feature dimension has a much
    larger range than others, it will dominate the distance calculation
    and bias cluster assignments. StandardScaler (zero mean, unit std)
    fitted on training data ensures all dimensions contribute equally.
    The same scaler parameters are applied to test data (no leakage).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_feat)
    X_te = scaler.transform(X_test_feat)

    for k in config["K_VALUES"]:
        print(f"    K={k:<3}  ", end="", flush=True)
        metrics = kmeans_classify(X_tr, y_train, X_te, y_test, k, config)
        results.append({
            "Feature Method": feature_name,
            "K":              k,
            "Train Acc (%)":  f"{metrics['train_acc']*100:.2f}",
            "Test Acc (%)":   f"{metrics['test_acc']*100:.2f}",
            "Fit Time (s)":   f"{metrics['fit_time']:.2f}",
        })
        print(f"Train: {metrics['train_acc']*100:.2f}%  "
              f"Test: {metrics['test_acc']*100:.2f}%  "
              f"Time: {metrics['fit_time']:.2f}s")


def main():
    print("=" * 68)
    print("  Script 4 / 5 — K-Means Clustering (K ∈ {1, 4, 16, 32})")
    print("  Feature methods: DCT | PCA | Autoencoder")
    print("=" * 68)
    print(f"  Device (for AE): {DEVICE}\n")

    # ── Load Data ─────────────────────────────────────────────────────────
    print("[1/7] Loading training data...")
    t0 = time.perf_counter()
    X_train_raw, y_train = load_dataset(CONFIG["TRAIN_DIR"])
    print("\n[1/7] Loading test data...")
    X_test_raw, y_test = load_dataset(CONFIG["TEST_DIR"])
    print(f"\n  Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}  "
          f"|  Load: {time.perf_counter()-t0:.2f}s")

    results = []

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE A — DCT FEATURES
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  PIPELINE A: DCT Features ({CONFIG['N_DCT_COEFFS']} coefficients)")
    print(f"{'='*68}")
    print("[2/7] Extracting DCT features...")
    X_tr_dct = extract_dct_features(X_train_raw, CONFIG["N_DCT_COEFFS"])
    X_te_dct = extract_dct_features(X_test_raw,  CONFIG["N_DCT_COEFFS"])
    print(f"  DCT shape: {X_tr_dct.shape}")

    print("[3/7] Running K-Means on DCT features...")
    run_kmeans_on_features("DCT", X_tr_dct, y_train, X_te_dct, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE B — PCA FEATURES
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  PIPELINE B: PCA Features ({CONFIG['N_PCA_COMPONENTS']} components)")
    print(f"{'='*68}")
    print("[4/7] Fitting PCA...")
    X_tr_pca, X_te_pca, _ = fit_pca(
        X_train_raw, X_test_raw,
        CONFIG["N_PCA_COMPONENTS"], CONFIG["PCA_WHITEN"], CONFIG["RANDOM_SEED"]
    )

    print("[5/7] Running K-Means on PCA features...")
    run_kmeans_on_features("PCA", X_tr_pca, y_train, X_te_pca, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE C — AUTOENCODER FEATURES
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  PIPELINE C: Autoencoder Features (latent dim = {CONFIG['LATENT_DIM']})")
    print(f"{'='*68}")
    print(f"[6/7] Training autoencoder ({CONFIG['AE_EPOCHS']} epochs)...")
    ae_model, ae_time = train_autoencoder(X_train_raw, CONFIG)
    print(f"  AE trained in {ae_time:.1f}s  |  extracting latent features...")
    X_tr_ae = extract_latent(ae_model, X_train_raw)
    X_te_ae = extract_latent(ae_model, X_test_raw)
    print(f"  Latent shape: {X_tr_ae.shape}")

    print("[7/7] Running K-Means on Autoencoder features...")
    run_kmeans_on_features("Autoencoder", X_tr_ae, y_train, X_te_ae, y_test, CONFIG, results)

    # ─────────────────────────────────────────────────────────────────────
    # FINAL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  FINAL RESULTS — K-Means across all Feature Methods and K values")
    print(f"{'='*68}")
    headers = ["Feature Method", "K", "Train Acc (%)", "Test Acc (%)", "Fit Time (s)"]
    rows = [[r["Feature Method"], r["K"], r["Train Acc (%)"],
             r["Test Acc (%)"], r["Fit Time (s)"]] for r in results]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Best result
    best = max(results, key=lambda r: float(r["Test Acc (%)"]))
    print(f"\n  ★ Best combination: Feature={best['Feature Method']}  "
          f"K={best['K']}  →  Test Acc: {best['Test Acc (%)']}%")

    # K=1 reminder
    k1_rows = [r for r in results if r["K"] == 1]
    print(f"\n  ℹ  K=1 results (trivial baseline — all samples → 1 cluster):")
    for r in k1_rows:
        print(f"     {r['Feature Method']:12s}  Test Acc: {r['Test Acc (%)']}%  "
              f"(expected ≈10% on balanced 10-class data)")
    print("=" * 68)


if __name__ == "__main__":
    main()
