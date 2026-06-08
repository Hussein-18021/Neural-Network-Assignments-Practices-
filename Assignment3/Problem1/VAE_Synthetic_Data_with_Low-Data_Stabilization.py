"""
VAE Synthetic Data with Low-Data Stabilization — Extended with Set D Control
==============================================================================
Identical to the original pipeline, plus Set D: a size-matched high-confidence
control set that allows isolating whether Set C's advantage (if any) comes from
mid-confidence DIVERSITY or simply from using FEWER synthetic samples (less
dilution of the 350 real examples).

Set summary:
  Set A  — All 50,000 generated samples (no filter)
  Set B  — High-confidence samples (conf >= 0.9), balanced
  Set C  — Mid-confidence samples (0.6 <= conf <= 0.9), balanced
  Set D  — High-confidence samples, subsampled to same size as Set C  [NEW]

Interpreting Set C vs Set D:
  |C - D| < 0.3pp  →  SIZE RATIO drives Set C's result, not diversity
  C > D by > 0.3pp →  MID-CONFIDENCE DIVERSITY genuinely helps
  D > C by > 0.3pp →  HIGH-CONFIDENCE QUALITY wins at equal sample count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
import ssl
import certifi
import os

# ── Fix SSL certificate issue for MNIST download ────────────────────────────
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION — ReducedMNIST
# ══════════════════════════════════════════════════════════════════════════════

def load_reduced_mnist(train_per_digit=1000, test_per_digit=200):
    """Load ReducedMNIST: subset of MNIST with fixed examples per digit."""
    full_train = datasets.MNIST(root="./data", train=True, download=True,
                                transform=transforms.ToTensor())
    full_test = datasets.MNIST(root="./data", train=False, download=True,
                               transform=transforms.ToTensor())

    def subsample(dataset, per_digit):
        data, targets = dataset.data.float() / 255.0, dataset.targets
        selected_idx = []
        for d in range(10):
            idx = (targets == d).nonzero(as_tuple=True)[0].numpy()
            chosen = np.random.choice(idx, size=per_digit, replace=False)
            selected_idx.extend(chosen)
        np.random.shuffle(selected_idx)
        selected_idx = torch.tensor(selected_idx)
        return data[selected_idx].unsqueeze(1), targets[selected_idx]

    train_x, train_y = subsample(full_train, train_per_digit)
    test_x, test_y = subsample(full_test, test_per_digit)
    return train_x, train_y, test_x, test_y


def select_n_per_digit(images, labels, n):
    """Second-level subsampling: pick n examples per digit from an already-
    reduced dataset (e.g. 350 from the 1000-per-digit ReducedMNIST)."""
    selected_x, selected_y = [], []
    for d in range(10):
        idx = (labels == d).nonzero(as_tuple=True)[0].numpy()
        chosen = np.random.choice(idx, size=n, replace=False)
        selected_x.append(images[chosen])
        selected_y.append(labels[chosen])
    return torch.cat(selected_x), torch.cat(selected_y)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA AUGMENTATION (vectorised with torchvision v2)
# ══════════════════════════════════════════════════════════════════════════════

class Clamp(nn.Module):
    """Clamp pixel values to [0, 1] — pickling-safe alternative to T.Lambda."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


def augment_dataset(images, labels, multiplier=15, chunk_size=64):
    """Augment dataset by `multiplier` times using per-image random transforms.
    Processes in chunks to ensure each image gets independent random params.
    images: (N, 1, 28, 28) float tensor in [0, 1]."""
    try:
        noise_t = T.GaussianNoise(mean=0.0, sigma=0.02)
    except AttributeError:
        class _GaussianNoise(nn.Module):
            def forward(self, x):
                return x + torch.randn_like(x) * 0.02
        noise_t = _GaussianNoise()

    transform = T.Compose([
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.9, 1.1)),
        noise_t,
        Clamp(),
    ])

    aug_imgs, aug_labels = [], []
    for _ in range(multiplier):
        pass_imgs = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            pass_imgs.append(transform(chunk))
        aug_imgs.append(torch.cat(pass_imgs))
        aug_labels.append(labels)
    return torch.cat(aug_imgs), torch.cat(aug_labels)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONDITIONAL VAE
# ══════════════════════════════════════════════════════════════════════════════

LATENT_DIM = 20
NUM_CLASSES = 10
IMG_CHANNELS = 1


class ConditionalVAE(nn.Module):
    """Conditional VAE with convolutional encoder/decoder for 28×28 images."""

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Separate label embeddings for encoder and decoder
        self.enc_label_emb = nn.Embedding(num_classes, num_classes)
        self.dec_label_emb = nn.Embedding(num_classes, num_classes)

        # ── Encoder ──
        # Input: (IMG_CHANNELS + num_classes) × 28 × 28 (image + label map)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS + num_classes, 32, 3, stride=2, padding=1),   # → 32×14×14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),               # → 64×7×7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),              # → 128×7×7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

        # ── Decoder ──
        self.dec_fc = nn.Linear(latent_dim + num_classes, 128 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),     # → 64×14×14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),     # → 32×28×28
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),               # → 1×28×28
            nn.Sigmoid(),
        )

    def _enc_label_map(self, labels, h, w):
        """Create (B, num_classes, h, w) encoder label conditioning map."""
        emb = self.enc_label_emb(labels)                 # (B, num_classes)
        return emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    def encode(self, x, labels):
        label_map = self._enc_label_map(labels, x.size(2), x.size(3))
        x_cond = torch.cat([x, label_map], dim=1)
        h = self.enc_conv(x_cond)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        emb = self.dec_label_emb(labels)
        z_cond = torch.cat([z, emb], dim=1)
        h = self.dec_fc(z_cond).view(-1, 128, 7, 7)
        return self.dec_conv(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE ELBO loss: reconstruction (BCE) + KL divergence, normalised by batch size."""
    B = x.size(0)
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / B
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return recon_loss + beta * kl_loss


def train_vae(vae, train_loader, epochs=80, lr=1e-3):
    """Train VAE with KL annealing (warm-up over first 20 epochs)."""
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    vae.train()

    for epoch in range(1, epochs + 1):
        total_loss, n_samples = 0.0, 0
        beta = min(1.0, epoch / 20.0)  # KL annealing
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch_x, batch_y)
            loss = vae_loss(recon, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        scheduler.step()
        avg = total_loss / n_samples
        if epoch % 10 == 0 or epoch == 1:
            print(f"  VAE Epoch {epoch:3d}/{epochs}  loss={avg:.4f}  β={beta:.2f}")
    return vae


# ══════════════════════════════════════════════════════════════════════════════
# 4. LeNet-5 CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),    # → 6×28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                  # → 6×14×14
            nn.Conv2d(6, 16, 5),              # → 16×10×10
            nn.ReLU(),
            nn.MaxPool2d(2),                  # → 16×5×5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_lenet(model, train_loader, epochs=30, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        correct, total, running_loss = 0, 0, 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  LeNet Epoch {epoch:3d}/{epochs}  loss={running_loss/total:.4f}  "
                  f"acc={100*correct/total:.1f}%")
    return model


def evaluate_lenet(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
    acc = 100.0 * correct / total
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYNTHETIC GENERATION & CONFIDENCE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def generate_from_vae(vae, num_per_digit=1000):
    """Generate synthetic samples by sampling z ~ N(0,I) for each digit."""
    vae.eval()
    all_imgs, all_labels = [], []
    with torch.no_grad():
        for d in range(10):
            z = torch.randn(num_per_digit, LATENT_DIM).to(DEVICE)
            labels = torch.full((num_per_digit,), d, dtype=torch.long).to(DEVICE)
            imgs = vae.decode(z, labels).cpu()
            all_imgs.append(imgs)
            all_labels.append(labels.cpu())
    return torch.cat(all_imgs), torch.cat(all_labels)


def compute_confidences(classifier, images, labels, batch_size=256):
    """Compute max softmax confidence for each sample using the classifier."""
    classifier.eval()
    confs = []
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(DEVICE)
            logits = classifier(bx)
            probs = F.softmax(logits, dim=1)
            max_conf, _ = probs.max(dim=1)
            confs.append(max_conf.cpu())
    return torch.cat(confs)


def filter_by_confidence(images, labels, confidences, low=None, high=None):
    """Filter samples by confidence range [low, high]."""
    mask = torch.ones(len(confidences), dtype=torch.bool)
    if low is not None:
        mask &= (confidences >= low)
    if high is not None:
        mask &= (confidences <= high)
    return images[mask], labels[mask], confidences[mask]


def balance_per_digit(images, labels, max_per_digit):
    """Cap each digit class to max_per_digit samples to prevent imbalance."""
    sel_x, sel_y = [], []
    for d in range(10):
        idx = (labels == d).nonzero(as_tuple=True)[0]
        n = min(len(idx), max_per_digit)
        chosen = idx[torch.randperm(len(idx))[:n]]
        sel_x.append(images[chosen])
        sel_y.append(labels[chosen])
    return torch.cat(sel_x), torch.cat(sel_y)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("VAE Synthetic Data with Low-Data Stabilization")
    print("=" * 70)

    # ── Step 1: Load ReducedMNIST ─────────────────────────────────────────
    print("\n[1] Loading ReducedMNIST (1000 train / 200 test per digit)...")
    train_x, train_y, test_x, test_y = load_reduced_mnist(
        train_per_digit=1000, test_per_digit=200
    )
    print(f"    Full train: {train_x.shape}  Full test: {test_x.shape}")

    # Select 350 real examples per digit
    real350_x, real350_y = select_n_per_digit(train_x, train_y, n=350)
    print(f"    350-real subset: {real350_x.shape}")

    # Also keep a 1000-per-digit set for baseline
    real1000_x, real1000_y = train_x, train_y

    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256, shuffle=False)

    # ── Step 2: Augment 350 real examples (15× augmentation) ──────────────
    print("\n[2] Augmenting 350-real set by 15× ...")
    aug_x, aug_y = augment_dataset(real350_x, real350_y, multiplier=15)
    print(f"    Augmented: {aug_x.shape}  (original + augmented combined below)")

    # Combine real + augmented for VAE training
    vae_train_x = torch.cat([real350_x, aug_x])
    vae_train_y = torch.cat([real350_y, aug_y])
    print(f"    VAE training data (real+aug): {vae_train_x.shape}")

    vae_train_loader = DataLoader(
        TensorDataset(vae_train_x, vae_train_y),
        batch_size=128, shuffle=True, drop_last=True
    )

    # ── Step 3: Train Conditional VAE ─────────────────────────────────────
    print("\n[3] Training Conditional VAE on real+augmented data...")
    # We will train 5 independent VAEs with different random seeds
    NUM_RUNS = 5
    all_gen_imgs, all_gen_labels = [], []

    for run_i in range(1, NUM_RUNS + 1):
        print(f"\n  --- VAE Run {run_i}/{NUM_RUNS} ---")
        # Different random init for each run
        torch.manual_seed(SEED + run_i * 1000)
        vae = ConditionalVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
        vae = train_vae(vae, vae_train_loader, epochs=80, lr=1e-3)

        # Generate 1000 samples per digit from this run
        gen_imgs, gen_labels = generate_from_vae(vae, num_per_digit=1000)
        all_gen_imgs.append(gen_imgs)
        all_gen_labels.append(gen_labels)
        print(f"    Generated: {gen_imgs.shape}")

    # Combine all 5 runs: 5 × 10000 = 50000 total
    all_gen_imgs = torch.cat(all_gen_imgs)
    all_gen_labels = torch.cat(all_gen_labels)
    print(f"\n    Total generated (5 runs): {all_gen_imgs.shape}")

    # ── Step 4: Train LeNet-5 on 350 real (for confidence scoring) ────────
    print("\n[4] Training LeNet-5 classifier on 350-real subset...")
    torch.manual_seed(SEED)
    lenet_scorer = LeNet5().to(DEVICE)
    scorer_loader = DataLoader(
        TensorDataset(real350_x, real350_y),
        batch_size=64,
        shuffle=True,
        drop_last=False   # LeNet has no BatchNorm — safe to keep all samples
    )
    lenet_scorer = train_lenet(lenet_scorer, scorer_loader, epochs=30, lr=1e-3)
    scorer_test_acc = evaluate_lenet(lenet_scorer, test_loader)
    print(f"    Scorer LeNet-5 test accuracy: {scorer_test_acc:.2f}%")

    # ── Step 5: Compute confidences ───────────────────────────────────────
    print("\n[5] Computing confidences on generated samples...")
    confidences = compute_confidences(lenet_scorer, all_gen_imgs, all_gen_labels)
    print(f"    Confidence stats: min={confidences.min():.4f}  "
          f"max={confidences.max():.4f}  mean={confidences.mean():.4f}")

    # ── Step 6: Create three synthetic datasets ───────────────────────────
    print("\n[6] Creating synthetic datasets A, B, C...")

    # Set A: all generated samples
    set_a_x, set_a_y = all_gen_imgs, all_gen_labels
    print(f"    Set A (all):            {set_a_x.shape[0]} samples")

    # Set B: high-confidence ≥ 0.9, balanced per digit
    set_b_x, set_b_y, _ = filter_by_confidence(all_gen_imgs, all_gen_labels,
                                                 confidences, low=0.9)
    min_b = min((set_b_y == d).sum().item() for d in range(10))
    if min_b == 0:
        print("    [!] WARNING: Set B has 0 samples for at least one digit — "
            "skipping balance. Results may be unreliable.")
        balance_note_b = "unbalanced"
    else:
        set_b_x, set_b_y = balance_per_digit(set_b_x, set_b_y, max_per_digit=min_b)
        balance_note_b = f"balanced to {min_b}/digit"
    print(f"    Set B (conf ≥ 0.9):  {set_b_x.shape[0]} samples  ({balance_note_b})")

    # Set C: mid-confidence 0.6 ≤ conf ≤ 0.9, balanced per digit
    set_c_x, set_c_y, _ = filter_by_confidence(all_gen_imgs, all_gen_labels,
                                                 confidences, low=0.6, high=0.9)
    min_c = min((set_c_y == d).sum().item() for d in range(10))
    if min_c == 0:
        print("    [!] WARNING: Set C has 0 samples for at least one digit — "
            "skipping balance. Results may be unreliable.")
        balance_note_c = "unbalanced"
    else:
        set_c_x, set_c_y = balance_per_digit(set_c_x, set_c_y, max_per_digit=min_c)
        balance_note_c = f"balanced to {min_c}/digit"
    print(f"    Set C (0.6 ≤ conf ≤ 0.9): {set_c_x.shape[0]} samples  ({balance_note_c})")

    # Set D: size-matched high-confidence control
    # Same number of samples per digit as Set C, but drawn from the high-confidence pool.
    # Purpose: isolates the effect of diversity vs. dataset size.
    #   If Set C ≈ Set D  → size ratio (less dilution of real data) explains Set C's advantage.
    #   If Set C > Set D  → mid-confidence diversity genuinely helps.
    #   If Set D > Set C  → high-confidence quality wins even at equal size.
    torch.manual_seed(SEED)
    set_d_x, set_d_y, _ = filter_by_confidence(all_gen_imgs, all_gen_labels,
                                                 confidences, low=0.9)
    if min_c == 0 or min((set_d_y == d).sum().item() for d in range(10)) == 0:
        print("    [!] WARNING: Set D cannot be created (Set C or high-conf pool is empty).")
        set_d_x, set_d_y = set_c_x, set_c_y  # fallback: identical to Set C
        balance_note_d = "fallback"
    else:
        set_d_x, set_d_y = balance_per_digit(set_d_x, set_d_y, max_per_digit=min_c)
        balance_note_d = f"balanced to {min_c}/digit (same as Set C)"
    print(f"    Set D (high-conf, size-matched): {set_d_x.shape[0]} samples  ({balance_note_d})")

    # Print per-digit breakdown
    for name, sx, sy in [("A", set_a_x, set_a_y),
                          ("B", set_b_x, set_b_y),
                          ("C", set_c_x, set_c_y),
                          ("D", set_d_x, set_d_y)]:
        counts = [(sy == d).sum().item() for d in range(10)]
        print(f"    Set {name} per-digit: {counts}")

    # ── Step 7: Train LeNet-5 on each set + 350 real ─────────────────────
    print("\n[7] Training LeNet-5 classifiers on various training sets...")

    results = {}

    def train_and_eval(name, extra_x=None, extra_y=None):
        torch.manual_seed(SEED)
        model = LeNet5().to(DEVICE)
        if extra_x is not None and len(extra_x) > 0:
            combined_x = torch.cat([real350_x, extra_x])
            combined_y = torch.cat([real350_y, extra_y])
        else:
            combined_x, combined_y = real350_x, real350_y
        loader = DataLoader(TensorDataset(combined_x, combined_y),
                            batch_size=128, shuffle=True)
        model = train_lenet(model, loader, epochs=30, lr=1e-3)
        acc = evaluate_lenet(model, test_loader)
        results[name] = acc
        print(f"    {name}: {acc:.2f}%  (trained on {len(combined_x)} samples)")
        return acc

    # Baseline: 350 real only
    print("\n  --- Baseline: 350-real ---")
    train_and_eval("350-real baseline")

    # Baseline: 1000 real
    print("\n  --- Baseline: 1000-real ---")
    torch.manual_seed(SEED)
    model_1000 = LeNet5().to(DEVICE)
    loader_1000 = DataLoader(TensorDataset(real1000_x, real1000_y),
                             batch_size=128, shuffle=True)
    model_1000 = train_lenet(model_1000, loader_1000, epochs=30, lr=1e-3)
    acc_1000 = evaluate_lenet(model_1000, test_loader)
    results["1000-real baseline"] = acc_1000
    print(f"    1000-real baseline: {acc_1000:.2f}%")

    # Baseline: 350 real + augmentation only (no VAE)
    print("\n  --- Baseline: 350-real + augmentation only ---")
    train_and_eval("350-real + augmentation", aug_x, aug_y)

    # Set A: 350 real + all generated
    print("\n  --- Set A: 350-real + all VAE generated ---")
    train_and_eval("Set A (all VAE)", set_a_x, set_a_y)

    # Set B: 350 real + high-confidence
    print("\n  --- Set B: 350-real + high-confidence VAE ---")
    train_and_eval("Set B (conf≥0.9)", set_b_x, set_b_y)

    # Set C: 350 real + mid-confidence
    print("\n  --- Set C: 350-real + mid-confidence VAE ---")
    train_and_eval("Set C (0.6≤conf≤0.9)", set_c_x, set_c_y)

    # Set D: 350 real + size-matched high-confidence control
    print("\n  --- Set D: 350-real + size-matched high-confidence (control) ---")
    train_and_eval("Set D (size-matched high-conf)", set_d_x, set_d_y)

    # ── Step 8: Summary & Comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Training Set':<35} {'Test Accuracy':>15}")
    print("-" * 52)
    for name in ["350-real baseline", "1000-real baseline",
                  "350-real + augmentation",
                  "Set A (all VAE)", "Set B (conf≥0.9)",
                  "Set C (0.6≤conf≤0.9)", "Set D (size-matched high-conf)"]:
        print(f"{name:<40} {results[name]:>14.2f}%")
    print("-" * 56)

    # ── Diversity vs Size control ────────────────────────────────────────
    c_vs_d = results["Set C (0.6≤conf≤0.9)"] - results["Set D (size-matched high-conf)"]
    print(f"\n  Set C vs Set D (same size, diff quality): {c_vs_d:+.2f}pp")
    if abs(c_vs_d) < 0.3:
        print("  → SIZE RATIO is the main driver; diversity effect is negligible.")
    elif c_vs_d > 0:
        print("  → MID-CONFIDENCE DIVERSITY genuinely helps beyond size effects.")
    else:
        print("  → HIGH-CONFIDENCE QUALITY wins even at equal sample count.")

    # ── Visualization ─────────────────────────────────────────────────────
    print("\n[8] Generating visualization plots...")

    # Plot 1: Sample generated images from best VAE
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fig.suptitle("VAE Generated Samples (random from all runs)", fontsize=14)
    for d in range(10):
        idx = (all_gen_labels == d).nonzero(as_tuple=True)[0]
        n_show = min(2, len(idx))
        chosen = idx[torch.randperm(len(idx))[:2]]
        for row in range(n_show):
            axes[row, d].imshow(all_gen_imgs[chosen[row], 0].numpy(), cmap='gray')
            axes[row, d].set_title(str(d))
            axes[row, d].axis('off')
        for row in range(n_show, 2):      # hide unused subplot slots
            axes[row, d].axis('off')
    plt.tight_layout()
    plt.savefig("vae_generated_samples.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidences.numpy(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0.9, color='red', linestyle='--', linewidth=2, label='High conf threshold (0.9)')
    ax.axvline(0.6, color='orange', linestyle='--', linewidth=2, label='Mid conf threshold (0.6)')
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution of VAE-Generated Samples")
    ax.legend()
    plt.tight_layout()
    plt.savefig("vae_confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    accs = [results[n] for n in names]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#00BCD4', '#795548']
    bars = ax.bar(range(len(names)), accs, color=colors[:len(names)], edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("LeNet-5 Test Accuracy: VAE Synthetic Data Comparison")
    ax.set_ylim(min(accs) - 3, max(accs) + 3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig("vae_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPlots saved:")
    print("  - vae_generated_samples.png")
    print("  - vae_confidence_distribution.png")
    print("  - vae_accuracy_comparison.png")

    # ── Analysis ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_set = max(["Set A (all VAE)", "Set B (conf≥0.9)",
                    "Set C (0.6≤conf≤0.9)", "Set D (size-matched high-conf)"],
                   key=lambda k: results[k])
    print(f"\nBest VAE synthetic set: {best_set} ({results[best_set]:.2f}%)")
    print(f"Improvement over 350-real baseline: "
          f"{results[best_set] - results['350-real baseline']:+.2f}%")
    print(f"Gap to 1000-real baseline: "
          f"{results[best_set] - results['1000-real baseline']:+.2f}%")

    if results["Set B (conf≥0.9)"] > results["Set C (0.6≤conf≤0.9)"]:
        print("\n→ High-confidence selection (Set B) outperforms mid-confidence (Set C).")
        print("  This suggests the classifier benefits more from 'clean' synthetic samples")
        print("  that closely match the learned decision boundary.")
    else:
        print("\n→ Mid-confidence/diverse samples (Set C) outperform high-confidence (Set B).")
        print("  This suggests diversity in the training set is more valuable than")
        print("  conformity to the existing classifier's predictions.")

    if results["Set A (all VAE)"] >= max(results["Set B (conf≥0.9)"],
                                          results["Set C (0.6≤conf≤0.9)"]):
        print("→ Using ALL generated samples (Set A) works best — volume matters.")
    else:
        print("→ Confidence-based filtering improves over using all samples.")

    # ── Key causal test: diversity vs. size ──────────────────────────────
    c_vs_d = results["Set C (0.6≤conf≤0.9)"] - results["Set D (size-matched high-conf)"]
    print(f"\n→ Set C vs Set D (same size, different confidence): {c_vs_d:+.2f}pp")
    if abs(c_vs_d) < 0.3:
        print("  Verdict: SIZE RATIO is the main driver of Set C's performance.")
        print("  The 'diversity' benefit is not distinguishable from less data dilution.")
    elif c_vs_d > 0:
        print("  Verdict: MID-CONFIDENCE DIVERSITY genuinely helps beyond size effects.")
        print("  Boundary-region samples provide information high-conf samples cannot.")
    else:
        print("  Verdict: HIGH-CONFIDENCE QUALITY wins at equal sample count.")
        print("  Clean samples are preferable when total volume is held constant.")

    print(f"\nAugmentation-only vs VAE: "
          f"{results['350-real + augmentation']:.2f}% vs {results[best_set]:.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
