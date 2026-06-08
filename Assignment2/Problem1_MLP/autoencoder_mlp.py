"""
================================================================================
  ReducedMNIST Classification — Script 3 of 3: Autoencoder Features + MLP
================================================================================

REQUIREMENTS (requirements.txt):
    numpy>=1.24
    torch>=2.0
    scikit-learn>=1.3
    Pillow>=10.0
    tabulate>=0.9

WHAT THIS SCRIPT DOES:
    Loads the ReducedMNIST dataset, trains a PyTorch autoencoder to learn a
    compact latent representation, then uses those latent features to train
    and evaluate scikit-learn MLPClassifiers with 1, 3, and 4 hidden-layer
    architectures.

HOW TO RUN:
    1. Set TRAIN_DIR and TEST_DIR in the CONFIG block below.
    2. pip install numpy torch scikit-learn Pillow tabulate
    3. python script3_autoencoder_mlp.py
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1 — CONFIGURATION BLOCK
# ============================================================
"""
Centralizing all hyperparameters makes experimentation safe and
reproducible. Change values here rather than hunting through the code.
"""
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Subdirectory structure: TRAIN_DIR/0/, TRAIN_DIR/1/, ..., TRAIN_DIR/9/
    "TRAIN_DIR": "ReducedMNIST/Reduced MNIST Data/Reduced Training data",
    "TEST_DIR":  "ReducedMNIST/Reduced MNIST Data/Reduced Testing data",

    # ── Autoencoder Architecture ───────────────────────────────────────────
    # LATENT_DIM: size of the bottleneck layer — the encoded representation.
    #   ● Smaller → more compression, loses details, faster MLP training.
    #   ● Larger  → richer features, but may approach raw pixel quality.
    #   64–128 is a good range for MNIST-sized images.
    #
    # ENCODER_HIDDEN: list of hidden layer widths in the encoder half.
    #   The decoder mirrors this in reverse (symmetric autoencoder).
    #   Example: [512, 256] means encoder is 784 → 512 → 256 → LATENT_DIM
    #            and decoder is LATENT_DIM → 256 → 512 → 784.
    "LATENT_DIM":     64,
    "ENCODER_HIDDEN": [512, 256],

    # ── Autoencoder Training ───────────────────────────────────────────────
    # AE_EPOCHS: training epochs for the autoencoder (pre-training step).
    #   More epochs → better reconstruction → better features; but slower.
    # AE_BATCH_SIZE: mini-batch size. Larger batches = more stable gradients
    #   but higher memory use. 128 is a safe default.
    # AE_LR: learning rate for Adam optimizer.
    "AE_EPOCHS":     50,
    "AE_BATCH_SIZE": 128,
    "AE_LR":         1e-3,

    # ── MLP Architectures to Try ──────────────────────────────────────────
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

# Fix random seeds for full reproducibility
np.random.seed(CONFIG["RANDOM_SEED"])
torch.manual_seed(CONFIG["RANDOM_SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["RANDOM_SEED"])

# Select device: GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SECTION 2 — DATA LOADING
# ============================================================
"""
CONCEPT — Loading images from a folder hierarchy:
    We traverse the class subdirectories, load each image, convert to
    grayscale, and flatten to a 784-element float vector.

NORMALIZATION (÷ 255):
    The autoencoder's output layer uses Sigmoid activation, which maps
    values to (0, 1). Pixel inputs must also be in [0, 1] for the
    reconstruction loss (Binary Cross-Entropy or MSE) to be meaningful.
"""

def load_dataset(root_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from root_dir/<class_label>/ subdirectories.

    Returns:
        X: float32 array of shape (N, 784), pixels in [0, 1].
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
# SECTION 3 — AUTOENCODER DEFINITION
# ============================================================
"""
CONCEPT — What is an Autoencoder?
    An autoencoder is a neural network trained to reproduce its own input.
    It has two components:

    ENCODER: compresses input X (784-dim) → latent vector Z (LATENT_DIM-dim)
    DECODER: reconstructs input from Z: Z → X_hat ≈ X

    By forcing all information through the narrow bottleneck (LATENT_DIM
    dimensions), the encoder must learn the most compact, informative
    representation of the input. This bottleneck output Z is what we use
    as features for the downstream MLP classifier.

ARCHITECTURE DETAIL:
    Encoder:  784 → 512 → 256 → LATENT_DIM  (with ReLU activations)
    Decoder:  LATENT_DIM → 256 → 512 → 784  (with ReLU + final Sigmoid)

    ● ReLU in hidden layers: avoids vanishing gradients, fast convergence.
    ● Sigmoid at decoder output: squashes reconstructed values to [0, 1],
      matching the normalized pixel input range.

LOSS FUNCTION — MSE (Mean Squared Error):
    MSE = mean((X - X_hat)²) measures how well the autoencoder reconstructs
    each pixel. We minimize this during training. Binary Cross-Entropy (BCE)
    is an alternative — BCE treats each pixel as an independent Bernoulli
    probability and often produces sharper reconstructions, but MSE is
    simpler and works well for continuous pixel values.

TRAINING STRATEGY:
    ● The autoencoder is trained UNSUPERVISED: it sees only the images, not
      the class labels. It learns to compress digits generically.
    ● After training, we FREEZE the encoder weights and extract Z vectors
      for all training and test images. These Z vectors are the features
      fed to the MLP classifier.
    ● The MLP then learns the classification boundary in the latent space.
"""

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        """
        Symmetric fully-connected autoencoder.

        Args:
            input_dim:   Flattened image size (784 for 28×28 MNIST).
            hidden_dims: Widths of encoder hidden layers (decoder mirrors).
            latent_dim:  Bottleneck size (number of latent features).
        """
        super().__init__()

        # ── Encoder ────────────────────────────────────────────────────────
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        # No activation at bottleneck → allows unrestricted latent values
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Decoder ────────────────────────────────────────────────────────
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        decoder_layers += [nn.Linear(prev_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass.

        Returns:
            z:       Latent vector (bottleneck output).
            x_recon: Reconstructed input.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent encoding (used for feature extraction)."""
        return self.encoder(x)


# ============================================================
# SECTION 4 — AUTOENCODER TRAINING
# ============================================================
"""
CONCEPT — Pre-training the Autoencoder:
    We train the autoencoder on the TRAINING images only (unsupervised).
    The goal is to learn a compact representation before the downstream
    classification task begins.

TRAINING LOOP:
    For each epoch:
      1. Shuffle training data into mini-batches.
      2. Forward pass: encoder → latent Z → decoder → reconstruction X_hat.
      3. Compute MSE loss between X and X_hat.
      4. Backpropagate gradients through both decoder and encoder.
      5. Adam optimizer updates all weights.

MINI-BATCHES:
    Processing one sample at a time is slow; processing all at once may
    exceed GPU memory. Mini-batches of 128 are a common compromise.

ADAM OPTIMIZER:
    Adam adapts the learning rate for each parameter using exponential
    moving averages of the gradient (m₁) and its square (m₂). This
    typically converges faster than plain SGD.
"""

def train_autoencoder(
    X_train: np.ndarray,
    config: dict,
) -> tuple[Autoencoder, list[float], float]:
    """
    Train the autoencoder on the training images.

    Args:
        X_train: (N, 784) pixel array in [0, 1].
        config:  Global configuration dict.

    Returns:
        model:        Trained Autoencoder (moved to CPU for inference).
        epoch_losses: MSE loss per epoch (for monitoring convergence).
        total_time:   Wall time for the full training run (seconds).
    """
    input_dim = X_train.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=config["ENCODER_HIDDEN"],
        latent_dim=config["LATENT_DIM"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["AE_LR"])
    loss_fn   = nn.MSELoss()

    # Wrap training data in a DataLoader for efficient mini-batch iteration
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(
        dataset,
        batch_size=config["AE_BATCH_SIZE"],
        shuffle=True,
        # pin_memory speeds up CPU→GPU transfers (ignored if device is CPU)
        pin_memory=(DEVICE.type == "cuda"),
        generator=torch.Generator().manual_seed(config["RANDOM_SEED"]),
    )

    epoch_losses = []
    t0 = time.perf_counter()

    print(f"  Training autoencoder on {DEVICE} for {config['AE_EPOCHS']} epochs...")
    model.train()
    for epoch in range(1, config["AE_EPOCHS"] + 1):
        batch_losses = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            _, x_recon = model(batch_x)
            loss = loss_fn(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(f"    Epoch {epoch:>3}/{config['AE_EPOCHS']}  "
                  f"MSE Loss: {epoch_loss:.5f}  "
                  f"Elapsed: {elapsed:.1f}s")

    total_time = time.perf_counter() - t0
    model.eval()
    model.to("cpu")   # move to CPU for sklearn-compatible extraction
    return model, epoch_losses, total_time


# ============================================================
# SECTION 5 — LATENT FEATURE EXTRACTION
# ============================================================
"""
CONCEPT — Extracting Latent Features from the Trained Encoder:
    After training, we discard the decoder and use only the encoder to map
    each image to its latent vector Z.

    This Z vector is the autoencoder's learned representation of the image.
    It should capture the most relevant structure in the digits (stroke
    direction, curvature, loop shape) because the encoder was trained to
    preserve all information needed for pixel-perfect reconstruction.

    We run inference in torch.no_grad() mode because we don't need
    gradients — this saves memory and speeds up the pass.

IMPORTANT:
    The encoder was ONLY trained on training images. We now apply it to
    test images too. If the autoencoder has learned a general digit
    representation (not just memorized training images), it will produce
    meaningful latent vectors for unseen test digits.
"""

@torch.no_grad()
def extract_latent_features(
    model: Autoencoder,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Pass images through the encoder to get latent vectors.

    Args:
        model:      Trained Autoencoder (on CPU).
        X:          (N, 784) pixel array in [0, 1].
        batch_size: Batch size for efficient encoding.

    Returns:
        (N, LATENT_DIM) float32 array of latent features.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    latents = []

    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i : i + batch_size]
        z = model.encode(batch)
        latents.append(z.numpy())

    return np.concatenate(latents, axis=0).astype(np.float32)


# ============================================================
# SECTION 6 — MLP TRAINING & EVALUATION
# ============================================================
"""
CONCEPT — MLP Classifier on Latent Features:
    We now treat the LATENT_DIM-dimensional vectors as the input features
    to a standard MLP classifier (scikit-learn MLPClassifier).

    The latent space has already been shaped by the autoencoder to be
    maximally informative, so even a shallow MLP can achieve good accuracy.
    Deeper MLPs may further refine the decision boundary.

NORMALIZATION:
    The encoder's output (latent vectors) has no guaranteed range or scale.
    We apply StandardScaler to normalize each latent dimension to zero mean
    and unit variance. This is important because:
      ● Adam's learning rate applies equally to all parameters.
      ● Features with larger magnitude would otherwise dominate gradients.
      ● Normalized inputs allow better weight initialization strategies.
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
    Train scikit-learn MLPClassifier and return performance metrics.
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
# SECTION 7 — MAIN PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("  Script 3 / 3 — Autoencoder Features + MLP Classification")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Autoencoder: 784 → {CONFIG['ENCODER_HIDDEN']} → "
          f"{CONFIG['LATENT_DIM']} (bottleneck) → "
          f"{list(reversed(CONFIG['ENCODER_HIDDEN']))} → 784")

    # ── 7.1 Load Data ────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")
    t_load = time.perf_counter()
    X_train_raw, y_train = load_dataset(CONFIG["TRAIN_DIR"])
    print("\n[1/5] Loading test data...")
    X_test_raw, y_test = load_dataset(CONFIG["TEST_DIR"])
    load_time = time.perf_counter() - t_load
    print(f"\n  Train: {X_train_raw.shape}  |  "
          f"Test: {X_test_raw.shape}  |  Load time: {load_time:.2f}s")

    # ── 7.2 Train Autoencoder ────────────────────────────────────────────
    print(f"\n[2/5] Training autoencoder "
          f"({CONFIG['AE_EPOCHS']} epochs, "
          f"batch size {CONFIG['AE_BATCH_SIZE']})...")
    ae_model, ae_losses, ae_train_time = train_autoencoder(X_train_raw, CONFIG)
    print(f"\n  Autoencoder training complete in {ae_train_time:.1f}s")
    print(f"  Initial MSE: {ae_losses[0]:.5f}  →  "
          f"Final MSE: {ae_losses[-1]:.5f}  "
          f"(improvement: {(1 - ae_losses[-1]/ae_losses[0])*100:.1f}%)")

    # ── 7.3 Extract Latent Features ──────────────────────────────────────
    print(f"\n[3/5] Extracting latent features "
          f"(bottleneck dim = {CONFIG['LATENT_DIM']})...")
    t_enc = time.perf_counter()
    X_train_latent = extract_latent_features(ae_model, X_train_raw)
    X_test_latent  = extract_latent_features(ae_model, X_test_raw)
    enc_time = time.perf_counter() - t_enc
    print(f"  Latent shapes — Train: {X_train_latent.shape}  |  "
          f"Test: {X_test_latent.shape}  |  Encode time: {enc_time:.3f}s")

    # ── 7.4 Normalize Latent Features ────────────────────────────────────
    print("\n[4/5] Normalizing latent features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_latent)
    X_test_scaled  = scaler.transform(X_test_latent)
    print(f"  Post-scaling mean ≈ {X_train_scaled.mean():.5f}  "
          f"std ≈ {X_train_scaled.std():.5f}")

    # ── 7.5 Train MLP Architectures ──────────────────────────────────────
    print("\n[5/5] Training MLP classifiers...\n")
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

    # ── 7.6 Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS — Autoencoder + MLP")
    print("=" * 65)
    headers = ["Architecture", "Hidden Layers", "Train Acc (%)", "Test Acc (%)", "Train Time (s)"]
    rows = [[r["Architecture"], r["Hidden Layers"],
             r["Train Acc (%)"], r["Test Acc (%)"], r["Train Time (s)"]]
            for r in results]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    best = max(results, key=lambda r: float(r["Test Acc (%)"]))
    print(f"\n  ★ Best configuration: {best['Architecture']}  "
          f"→  Test Acc: {best['Test Acc (%)']}%\n")

    print("  Autoencoder details:")
    print(f"    Input dim       : 784 pixels")
    print(f"    Encoder hidden  : {CONFIG['ENCODER_HIDDEN']}")
    print(f"    Latent dim      : {CONFIG['LATENT_DIM']}")
    print(f"    Compression     : {CONFIG['LATENT_DIM']/784*100:.1f}% of input dims")
    print(f"    AE train time   : {ae_train_time:.1f}s  ({CONFIG['AE_EPOCHS']} epochs)")
    print(f"    AE final MSE    : {ae_losses[-1]:.5f}")
    print(f"    Feature extract : {enc_time:.3f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
