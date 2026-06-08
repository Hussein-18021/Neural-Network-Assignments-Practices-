import os, time, csv, platform, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ==============================================================================
# CONSTANTS — edit here only
# ==============================================================================
SEED              = 42
LATENT_DIM        = 100
EMBED_DIM         = 50
NUM_CLASSES       = 10
CGAN_EPOCHS       = 30
BATCH_SIZE        = 128
CLF_EPOCHS        = 15
LR                = 0.0002
BETA1             = 0.5
BETA2             = 0.999

AUG_FACTOR        = 10          # 10–20×: how many augmented copies per real image
REAL_PER_DIGIT    = 350         # real examples per digit for GAN training
BASELINE_1000     = 1000        # real examples per digit for Baseline-1000
TEST_PER_DIGIT    = 200
GEN_BATCHES       = 5           # independent generation passes per digit
GEN_PER_BATCH     = 1000        # samples per batch → 5000/digit, 50000 total
CONF_HIGH         = 0.90        # Set B threshold
CONF_LOW          = 0.60        # Set C lower threshold
LENET_EPOCHS      = 20
LENET_LR          = 0.001
LENET_BATCH       = 128

# ==============================================================================
# SETUP
# ==============================================================================
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# ORIGINAL DATA HELPERS (unchanged from attached code)
# ==============================================================================
def get_reduced_mnist(root='./data', train_per_digit=1000, test_per_digit=200):
    train_path = os.path.join(root, 'ReducedMNIST', 'train')
    test_path  = os.path.join(root, 'ReducedMNIST', 'test')
    transform  = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if os.path.exists(train_path) and os.path.exists(test_path):
        return (datasets.ImageFolder(train_path, transform=transform),
                datasets.ImageFolder(test_path,  transform=transform))

    print("Creating ReducedMNIST from full MNIST ...")
    full_train = datasets.MNIST(root, train=True,  download=True,
                                transform=transforms.ToTensor())
    full_test  = datasets.MNIST(root, train=False, download=True,
                                transform=transforms.ToTensor())
    for full_ds, per_d, path in [(full_train, train_per_digit, train_path),
                                  (full_test,  test_per_digit,  test_path)]:
        os.makedirs(path, exist_ok=True)
        idx_by = {d: [] for d in range(10)}
        for i, (_, lbl) in enumerate(full_ds):
            idx_by[lbl].append(i)
        for d in range(10):
            cdir = os.path.join(path, str(d))
            os.makedirs(cdir, exist_ok=True)
            for j, idx in enumerate(
                np.random.choice(idx_by[d], per_d, replace=False)
            ):
                utils.save_image(full_ds[idx][0],
                                 os.path.join(cdir, f"{j:04d}.png"))
    return (datasets.ImageFolder(train_path, transform=transform),
            datasets.ImageFolder(test_path,  transform=transform))


def balanced_subset(dataset, per_class):
    indices = []
    for c in range(10):
        cls_idx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == c]
        chosen  = np.random.choice(cls_idx, per_class, replace=False)
        indices.extend(chosen)
    return Subset(dataset, indices)


# ==============================================================================
# GENERATOR (unchanged from attached code)
# ==============================================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.fc  = nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 7 * 7)
        self.bn0 = nn.BatchNorm1d(256 * 7 * 7)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        e = self.emb(labels)
        x = F.relu(self.bn0(self.fc(torch.cat([z, e], 1))))
        x = x.view(-1, 256, 7, 7)
        return self.out(self.up2(self.up1(x)))


# ==============================================================================
# DISCRIMINATOR (unchanged from attached code)
# ==============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        SN = nn.utils.spectral_norm

        self.c1 = SN(nn.Conv2d(1,   64,  4, stride=2, padding=1))
        self.c2 = SN(nn.Conv2d(64,  128, 4, stride=2, padding=1))
        self.b2 = nn.BatchNorm2d(128)
        self.c3 = SN(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        self.b3 = nn.BatchNorm2d(256)

        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = SN(nn.Linear(256, 1))
        self.embed  = SN(nn.Embedding(NUM_CLASSES, 256))

    def forward(self, img, labels):
        x   = F.leaky_relu(self.c1(img), 0.2, True)
        x   = F.leaky_relu(self.b2(self.c2(x)), 0.2, True)
        x   = F.leaky_relu(self.b3(self.c3(x)), 0.2, True)
        phi = self.pool(x).view(x.size(0), -1)
        return self.linear(phi) + (phi * self.embed(labels)).sum(1, keepdim=True)


# ==============================================================================
# GAN TRAINING (unchanged from attached code — only data loader is swapped)
# ==============================================================================
def train_cgan(real_subset, epochs=CGAN_EPOCHS, batch_size=BATCH_SIZE):
    loader = DataLoader(real_subset, batch_size, shuffle=True, drop_last=True)
    G = Generator().to(device)
    D = Discriminator().to(device)

    g_opt = optim.Adam(G.parameters(), lr=LR,     betas=(BETA1, BETA2))
    d_opt = optim.Adam(D.parameters(), lr=LR*2,   betas=(BETA1, BETA2))
    g_sch = optim.lr_scheduler.CosineAnnealingLR(g_opt, epochs, eta_min=1e-5)
    d_sch = optim.lr_scheduler.CosineAnnealingLR(d_opt, epochs, eta_min=2e-5)

    start = time.time()
    step  = 0

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        noise_std = max(0.0, 0.05 * (1 - epoch / epochs))

        for real_imgs, labels in loader:
            bs         = real_imgs.size(0)
            real_imgs  = real_imgs.to(device)
            labels     = labels.to(device)
            real_noisy = real_imgs + noise_std * torch.randn_like(real_imgs)

            d_opt.zero_grad()
            z          = torch.randn(bs, LATENT_DIM, device=device)
            fake_lbls  = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_imgs  = G(z, fake_lbls).detach()
            fake_noisy = fake_imgs + noise_std * torch.randn_like(fake_imgs)

            d_real = D(real_noisy, labels)
            d_fake = D(fake_noisy, fake_lbls)
            d_loss = F.relu(1. - d_real).mean() + F.relu(1. + d_fake).mean()

            if step % 16 == 0:
                real_imgs.requires_grad_(True)
                grad = torch.autograd.grad(
                    D(real_imgs, labels).sum(), real_imgs,
                    create_graph=True)[0]
                d_loss = d_loss + 5.0 * grad.pow(2).sum([1, 2, 3]).mean()
                real_imgs.requires_grad_(False)

            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            z         = torch.randn(bs, LATENT_DIM, device=device)
            fake_lbls = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            g_loss    = -D(G(z, fake_lbls), fake_lbls).mean()
            g_loss.backward()
            g_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            step += 1

        g_sch.step(); d_sch.step()

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"D: {np.mean(d_losses):.3f} | "
                  f"G: {np.mean(g_losses):.3f} | "
                  f"Elapsed: {elapsed:.0f}s")
            G.eval()
            with torch.no_grad():
                z      = torch.randn(30, LATENT_DIM, device=device)
                labels = torch.arange(30, device=device) % 10
                imgs   = (G(z, labels) + 1) / 2
                grid   = utils.make_grid(imgs.cpu(), nrow=10)
            plt.figure(figsize=(12, 4))
            plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Epoch {epoch + 1}")
            plt.savefig(os.path.join(OUT_DIR, f"gen_epoch_{epoch+1}.png"),
                        bbox_inches='tight')
            plt.close()
            G.train()

    t_total = time.time() - start
    print(f"  CGAN done in {t_total:.1f}s ({t_total/60:.1f} min)")
    G.eval()
    return G, t_total


# ==============================================================================
# GENERATION (unchanged from attached code)
# ==============================================================================
def generate_synthetic(G, n_per_class):
    G.eval()
    imgs, lbls = [], []
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            z      = torch.randn(n_per_class, LATENT_DIM, device=device)
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            imgs.append(G(z, labels).cpu())
            lbls.append(labels.cpu())
    return torch.cat(imgs), torch.cat(lbls)


# ==============================================================================
# VISUALIZATION (unchanged from attached code)
# ==============================================================================
def save_real_vs_generated(real_subset, G, n=5):
    G.eval()
    real_by = {c: [] for c in range(10)}
    for idx in real_subset.indices:
        img, lbl = real_subset.dataset[idx]
        if len(real_by[lbl]) < n:
            real_by[lbl].append((img + 1) / 2)

    gen_by = {}
    with torch.no_grad():
        for c in range(10):
            z      = torch.randn(n, LATENT_DIM, device=device)
            labels = torch.full((n,), c, dtype=torch.long, device=device)
            gen_by[c] = [(G(z, labels)[i] + 1) / 2 for i in range(n)]

    fig, axes = plt.subplots(10, n*2+1, figsize=(n*2*1.1+1.5, 22))
    fig.patch.set_facecolor('#111')

    for row in range(10):
        axes[row, n].axis('off')
        axes[row, n].text(0.5, 0.5, str(row), ha='center', va='center',
                          fontsize=15, fontweight='bold', color='white',
                          transform=axes[row, n].transAxes)
        for col in range(n):
            for ax, data, color in [
                (axes[row, col],       real_by[row][col], '#4ecdc4'),
                (axes[row, n+1+col],   gen_by[row][col],  '#ff6b6b'),
            ]:
                ax.imshow(data.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(color); s.set_linewidth(1)
                ax.set_facecolor('#111')

    fig.text(0.27, 0.997, 'REAL  (teal border)',
             ha='center', va='top', color='#4ecdc4', fontsize=11, fontweight='bold')
    fig.text(0.73, 0.997, 'GENERATED  (red border)',
             ha='center', va='top', color='#ff6b6b', fontsize=11, fontweight='bold')
    fig.text(0.5, 1.0,
             'Part 1 – Real vs CGAN Generated  |  350 real images/digit',
             ha='center', va='top', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.996])
    plt.savefig(os.path.join(OUT_DIR, "part1_real_vs_generated.png"),
                dpi=120, bbox_inches='tight', facecolor='#111')
    plt.close()
    print("Saved: part1_real_vs_generated.png")


# ==============================================================================
# LeNet-5 (standard architecture: C1→S2→C3→S4→C5→F6→output)
# ==============================================================================
class LeNet5(nn.Module):
    """
    Standard LeNet-5:
      C1  : Conv2d(1,  6,  5) → 28x28 → 24x24
      S2  : AvgPool2d(2,2)   → 12x12
      C3  : Conv2d(6,  16, 5) → 8x8
      S4  : AvgPool2d(2,2)   → 4x4
      C5  : Conv2d(16, 120, 5) on padded input → 1x1  (use linear for 4x4→120)
      F6  : Linear(120, 84)
      Out : Linear(84,  10)
    """
    def __init__(self):
        super().__init__()
        self.c1  = nn.Conv2d(1,  6,  5, padding=2)   # pad to keep 28→28 then pool→14
        self.s2  = nn.AvgPool2d(2, 2)
        self.c3  = nn.Conv2d(6,  16, 5)               # 14→10 after pool→5
        self.s4  = nn.AvgPool2d(2, 2)
        self.c5  = nn.Conv2d(16, 120, 5)              # 5→1
        self.f6  = nn.Linear(120, 84)
        self.out = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = torch.tanh(self.c1(x))   # C1
        x = self.s2(x)               # S2  → 14×14
        x = torch.tanh(self.c3(x))   # C3  → 10×10
        x = self.s4(x)               # S4  → 5×5
        x = torch.tanh(self.c5(x))   # C5  → 1×1
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.f6(x))   # F6
        return self.out(x)            # logits


def train_lenet(train_ds, epochs=LENET_EPOCHS, lr=LENET_LR, batch_size=LENET_BATCH,
                desc="LeNet-5"):
    """Train a fresh LeNet-5 on train_ds. Returns trained model."""
    torch.manual_seed(SEED)
    model  = LeNet5().to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), lbls)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    [{desc}] Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    model.eval()
    return model


def evaluate_lenet(model, test_ds, batch_size=LENET_BATCH):
    """Returns (accuracy, macro_f1) on test_ds."""
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1


# ==============================================================================
# AUGMENTED DATASET (defined here so it's available to stage functions)
# ==============================================================================
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps a Subset and produces AUG_FACTOR augmented copies of every sample.
    Images are in [-1, 1]. Augmentation rescales to [0,1], applies RandomAffine,
    rescales back, then adds Gaussian noise (σ=0.05).
    """
    def __init__(self, subset, aug_factor, aug_tf):
        self.aug_factor = aug_factor
        self.aug_tf     = aug_tf
        self.samples    = [(subset[i][0], subset[i][1]) for i in range(len(subset))]
        self.length     = len(self.samples) * (1 + aug_factor)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = len(self.samples)
        img, lbl = self.samples[idx % n]
        if idx < n:
            return img, lbl                          # original, no augmentation
        img_01  = (img + 1.0) / 2.0
        img_aug = self.aug_tf(img_01)
        img_n1  = img_aug * 2.0 - 1.0
        noise   = torch.randn_like(img_n1) * 0.05   # σ ≤ 0.05
        return (img_n1 + noise).clamp(-1.0, 1.0), lbl


# ==============================================================================
# HELPER used in Stages 6 & 7
# ==============================================================================
def subset_to_tensordataset(subset):
    imgs = torch.stack([subset[i][0] for i in range(len(subset))])
    lbls = torch.tensor([subset[i][1] for i in range(len(subset))])
    return TensorDataset(imgs, lbls)

def make_combined_ds(real_imgs, real_lbls, gen_imgs, gen_lbls):
    return TensorDataset(
        torch.cat([real_imgs, gen_imgs]),
        torch.cat([real_lbls, gen_lbls])
    )


# ==============================================================================
# === STAGE 1: DATA AUGMENTATION ===
# ==============================================================================
def stage1_augment():
    print("\n" + "=" * 65)
    print("STAGE 1: DATA AUGMENTATION")
    print("=" * 65)

    train_full, test_ds = get_reduced_mnist(
        root='./data',
        train_per_digit=max(BASELINE_1000, REAL_PER_DIGIT),
        test_per_digit=TEST_PER_DIGIT
    )
    real_350_subset = balanced_subset(train_full, REAL_PER_DIGIT)

    aug_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,               # ±15° rotation
            translate=(0.10, 0.10),   # ±10% shift
            scale=(0.9, 1.1),         # 0.9–1.1× scaling
        ),
    ])
    aug_dataset = AugmentedDataset(real_350_subset, AUG_FACTOR, aug_transform)
    n_aug = len(aug_dataset)

    print(f"  Originals : {len(real_350_subset):,}  ({REAL_PER_DIGIT}/digit × 10)")
    print(f"  AUG_FACTOR: {AUG_FACTOR}×")
    print(f"  Total aug dataset size: {n_aug:,}")
    print(f"✅ Stage 1 complete — {n_aug:,} samples ready "
          f"({REAL_PER_DIGIT} real + {AUG_FACTOR}× augmented per digit)")

    return train_full, test_ds, real_350_subset, aug_dataset


# ==============================================================================
# === STAGE 2: cDCGAN TRAINING ===
# ==============================================================================
def stage2_train_gan(aug_dataset, real_350_subset):
    print("\n" + "=" * 65)
    print("STAGE 2: cDCGAN TRAINING")
    print("=" * 65)
    print(f"  Training on augmented dataset ({len(aug_dataset):,} samples) "
          f"for {CGAN_EPOCHS} epochs ...")

    G, gan_time = train_cgan(aug_dataset, epochs=CGAN_EPOCHS, batch_size=BATCH_SIZE)

    gan_ckpt = os.path.join(OUT_DIR, "generator.pt")
    torch.save(G.state_dict(), gan_ckpt)
    print(f"  Generator saved → {gan_ckpt}")

    save_real_vs_generated(real_350_subset, G)

    print(f"✅ Stage 2 complete — cDCGAN trained for {CGAN_EPOCHS} epochs "
          f"in {gan_time/60:.1f} min on {len(aug_dataset):,} samples")

    return G


# ==============================================================================
# === STAGE 3: SYNTHETIC SAMPLE GENERATION ===
# ==============================================================================
def stage3_generate(G):
    print("\n" + "=" * 65)
    print("STAGE 3: SYNTHETIC SAMPLE GENERATION")
    print("=" * 65)
    total = GEN_BATCHES * GEN_PER_BATCH * NUM_CLASSES
    print(f"  Generating {GEN_BATCHES} batches × {GEN_PER_BATCH}/digit "
          f"= {GEN_BATCHES * GEN_PER_BATCH}/digit, {total:,} total ...")

    all_imgs, all_lbls = [], []
    t0 = time.time()
    for batch_idx in range(GEN_BATCHES):
        imgs, lbls = generate_synthetic(G, GEN_PER_BATCH)
        all_imgs.append(imgs)
        all_lbls.append(lbls)
        print(f"  Generation batch {batch_idx+1}/{GEN_BATCHES} done "
              f"({imgs.shape[0]:,} samples)")

    all_gen_imgs = torch.cat(all_imgs)   # [50000, 1, 28, 28]
    all_gen_lbls = torch.cat(all_lbls)   # [50000]

    print(f"  Total generated: {all_gen_imgs.shape[0]:,} images")
    print(f"✅ Stage 3 complete — {all_gen_imgs.shape[0]:,} synthetic samples "
          f"generated in {time.time()-t0:.1f}s")

    return all_gen_imgs, all_gen_lbls


# ==============================================================================
# === STAGE 4: CONFIDENCE FILTERING VIA LeNet-5 ===
# ==============================================================================
def stage4_score(real_350_subset, all_gen_imgs, all_gen_lbls):
    print("\n" + "=" * 65)
    print("STAGE 4: CONFIDENCE FILTERING VIA LeNet-5")
    print("=" * 65)

    real_350_ds = TensorDataset(
        torch.stack([real_350_subset[i][0] for i in range(len(real_350_subset))]),
        torch.tensor([real_350_subset[i][1] for i in range(len(real_350_subset))])
    )

    print(f"  Training LeNet-5 on {len(real_350_ds)} real examples "
          f"({LENET_EPOCHS} epochs) ...")
    lenet_filter = train_lenet(real_350_ds, epochs=LENET_EPOCHS,
                                desc="LeNet-5 (filter)")

    print(f"  Scoring {all_gen_imgs.shape[0]:,} generated samples ...")
    score_loader = DataLoader(
        TensorDataset(all_gen_imgs, all_gen_lbls), batch_size=512, shuffle=False
    )
    all_conf, all_pred = [], []
    lenet_filter.eval()
    with torch.no_grad():
        for imgs, _ in score_loader:
            probs      = F.softmax(lenet_filter(imgs.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            all_conf.append(conf.cpu())
            all_pred.append(pred.cpu())

    all_conf = torch.cat(all_conf)
    all_pred = torch.cat(all_pred)

    print(f"  Mean confidence: {all_conf.mean():.4f}  |  "
          f"Min: {all_conf.min():.4f}  |  Max: {all_conf.max():.4f}")
    print(f"✅ Stage 4 complete — confidence scores computed for all "
          f"{all_gen_imgs.shape[0]:,} generated samples")

    return real_350_ds, all_conf, all_pred


# ==============================================================================
# === STAGE 5: SYNTHETIC DATASET CONSTRUCTION ===
# ==============================================================================
def stage5_build_sets(all_gen_imgs, all_gen_lbls, all_conf):
    print("\n" + "=" * 65)
    print("STAGE 5: SYNTHETIC DATASET CONSTRUCTION")
    print("=" * 65)

    mask_B = all_conf >= CONF_HIGH
    mask_C = (all_conf >= CONF_LOW) & (all_conf < CONF_HIGH)

    imgs_A, lbls_A = all_gen_imgs,          all_gen_lbls            # Set A: all
    imgs_B, lbls_B = all_gen_imgs[mask_B],  all_gen_lbls[mask_B]    # Set B: high-conf
    imgs_C, lbls_C = all_gen_imgs[mask_C],  all_gen_lbls[mask_C]    # Set C: mid-conf

    print(f"\n  {'Digit':<8} {'Set A':>8} {'Set B (≥0.9)':>13} {'Set C (0.6–0.9)':>16}")
    print("  " + "-" * 50)
    for d in range(NUM_CLASSES):
        cA = (lbls_A == d).sum().item()
        cB = (lbls_B == d).sum().item()
        cC = (lbls_C == d).sum().item()
        print(f"  {d:<8} {cA:>8,} {cB:>13,} {cC:>16,}")
    print("  " + "-" * 50)
    print(f"  {'TOTAL':<8} {len(lbls_A):>8,} {len(lbls_B):>13,} {len(lbls_C):>16,}")

    print(f"\n✅ Stage 5 complete — Set A: {len(lbls_A):,} | "
          f"Set B: {len(lbls_B):,} | Set C: {len(lbls_C):,}")

    return (imgs_A, lbls_A), (imgs_B, lbls_B), (imgs_C, lbls_C)


# ==============================================================================
# === STAGE 6: LeNet-5 COMPARATIVE TRAINING ===
# ==============================================================================
def stage6_train_classifiers(train_full, real_350_ds,
                              set_A, set_B, set_C):
    print("\n" + "=" * 65)
    print("STAGE 6: LeNet-5 COMPARATIVE TRAINING")
    print("=" * 65)

    imgs_A, lbls_A = set_A
    imgs_B, lbls_B = set_B
    imgs_C, lbls_C = set_C
    real_350_imgs, real_350_lbls = real_350_ds.tensors

    real_1000_ds = subset_to_tensordataset(balanced_subset(train_full, BASELINE_1000))
    print(f"  Baseline-1000 dataset: {len(real_1000_ds):,} samples")

    ds_350_A = make_combined_ds(real_350_imgs, real_350_lbls, imgs_A, lbls_A)
    ds_350_B = make_combined_ds(real_350_imgs, real_350_lbls, imgs_B, lbls_B)
    ds_350_C = make_combined_ds(real_350_imgs, real_350_lbls, imgs_C, lbls_C)

    configs = [
        ("Baseline-350",     real_350_ds),
        ("Baseline-1000",    real_1000_ds),
        ("Real-350 + Set A", ds_350_A),
        ("Real-350 + Set B", ds_350_B),
        ("Real-350 + Set C", ds_350_C),
    ]

    trained_models = {}
    for name, ds in configs:
        print(f"\n  ── Training: {name}  ({len(ds):,} samples) ──")
        trained_models[name] = train_lenet(ds, epochs=LENET_EPOCHS, desc=name)

    print(f"\n✅ Stage 6 complete — 5 LeNet-5 models trained "
          f"({LENET_EPOCHS} epochs each)")

    return configs, trained_models


# ==============================================================================
# === STAGE 7: RESULTS COMPARISON ===
# ==============================================================================
def stage7_compare(test_ds, configs, trained_models):
    print("\n" + "=" * 65)
    print("STAGE 7: RESULTS COMPARISON")
    print("=" * 65)

    # Convert ImageFolder test set to TensorDataset for uniform evaluation
    test_imgs_list, test_lbls_list = [], []
    for imgs, lbls in DataLoader(test_ds, batch_size=512, shuffle=False):
        test_imgs_list.append(imgs)
        test_lbls_list.append(lbls)
    test_tensor_ds = TensorDataset(
        torch.cat(test_imgs_list),
        torch.cat(test_lbls_list)
    )
    print(f"  Test set size: {len(test_tensor_ds):,} samples\n")

    results = []
    for name, ds in configs:
        acc, f1 = evaluate_lenet(trained_models[name], test_tensor_ds)
        results.append({"name": name, "train_size": len(ds),
                         "accuracy": acc, "f1": f1})

    # Print comparison table
    header = (f"{'Dataset Config':<22} {'Train Size':>12} "
              f"{'Test Acc':>10} {'F1 (macro)':>12}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"  {r['name']:<20} {r['train_size']:>12,} "
              f"{r['accuracy']*100:>9.2f}%  {r['f1']:>11.4f}")

    # Improvements over Baseline-350
    baseline_acc = results[0]["accuracy"]
    print(f"\n  Baseline-350 accuracy: {baseline_acc*100:.2f}%")
    print("\n  Improvements over Baseline-350:")
    improved = False
    for r in results[1:]:
        delta = r["accuracy"] - baseline_acc
        sign  = "+" if delta >= 0 else ""
        flag  = "  ✅ BETTER" if delta > 0 else ("  ⚠ WORSE" if delta < 0 else "  = SAME")
        print(f"    {r['name']:<22}  {sign}{delta*100:.2f}%{flag}")
        if delta > 0:
            improved = True
    if not improved:
        print("    No configuration outperformed the 350-real baseline.")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "results_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "train_size", "accuracy", "f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved → {csv_path}")

    print(f"\n✅ Stage 7 complete — comparison table printed and "
          f"saved to results_comparison.csv")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 65)
    print("COMPUTER SETUP")
    print("=" * 65)
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"Processor : {platform.processor()}")
    print(f"Device    : {device}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    try:
        import psutil
        print(f"CPU cores : {psutil.cpu_count(logical=True)}")
        print(f"RAM       : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        pass
    print(f"PyTorch   : {torch.__version__}")
    print("=" * 65)

    # Stage 1 — Data Augmentation
    train_full, test_ds, real_350_subset, aug_dataset = stage1_augment()

    # Stage 2 — cDCGAN Training
    G = stage2_train_gan(aug_dataset, real_350_subset)

    # Stage 3 — Synthetic Sample Generation
    all_gen_imgs, all_gen_lbls = stage3_generate(G)

    # Stage 4 — Confidence Filtering via LeNet-5
    real_350_ds, all_conf, _ = stage4_score(real_350_subset, all_gen_imgs, all_gen_lbls)

    # Stage 5 — Synthetic Dataset Construction
    set_A, set_B, set_C = stage5_build_sets(all_gen_imgs, all_gen_lbls, all_conf)

    # Stage 6 — LeNet-5 Comparative Training
    configs, trained_models = stage6_train_classifiers(
        train_full, real_350_ds, set_A, set_B, set_C
    )

    # Stage 7 — Results Comparison
    stage7_compare(test_ds, configs, trained_models)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()