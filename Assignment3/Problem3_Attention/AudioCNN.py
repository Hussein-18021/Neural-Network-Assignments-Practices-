import os
import re
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader

import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cnn_attention_comparison.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. Reproducibility
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 3. Dataset: Spectrogram Image Dataset
#    Produces a 2-D mel-spectrogram per utterance
#    (same audio files as Assignment 2, but fed
#     to a CNN as a fixed-size image instead of
#     the averaged-MFCC vector used previously)
# ==========================================
class SpectrogramDataset(Dataset):
    """
    Reads every .wav file in `data_dir`, extracts a
    log-mel spectrogram, resizes it to `img_size x img_size`,
    and returns (1 x H x W tensor, digit label).

    Label is parsed from the filename exactly as in Ass-2:
      *_<digit>.wav   or   *<digit>.wav
    """
    def __init__(self, data_dir, img_size=64, fixed_width=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.file_paths = []
        self.labels = []

        # Detect sample rate from the first file
        sample_file = next(f for f in os.listdir(data_dir) if f.endswith('.wav'))
        _, self.sr = sf.read(os.path.join(data_dir, sample_file))

        self.mel_transform = AT.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=512,
            hop_length=128,
            n_mels=64,
            f_min=0,
            f_max=self.sr // 2,
        )
        self.db_transform = AT.AmplitudeToDB(top_db=80)

        for file in sorted(os.listdir(data_dir)):
            if not file.endswith('.wav'):
                continue
            match = re.search(r'_(\d)\.wav$', file) or re.search(r'(\d)\.wav$', file)
            if match:
                self.file_paths.append(os.path.join(data_dir, file))
                self.labels.append(int(match.group(1)))

        # Compute fixed time-axis width from the training set
        if fixed_width is None:
            max_w = 0
            for p in self.file_paths:
                wav, _ = sf.read(p)
                waveform = torch.tensor(wav, dtype=torch.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=-1)
                waveform = waveform.unsqueeze(0)
                spec = self.mel_transform(waveform)
                max_w = max(max_w, spec.shape[-1])
            self.fixed_width = max_w
            logging.info(f"[{data_dir}] max spectrogram width = {self.fixed_width} frames")
        else:
            self.fixed_width = fixed_width

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_data, _ = sf.read(self.file_paths[idx])
        waveform = torch.tensor(wav_data, dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)
        waveform = waveform.unsqueeze(0)                    # (1, T)

        spec = self.mel_transform(waveform)                 # (1, n_mels, W)
        spec = self.db_transform(spec)

        # Pad / trim time axis
        w = spec.shape[-1]
        if w < self.fixed_width:
            spec = F.pad(spec, (0, self.fixed_width - w))
        else:
            spec = spec[..., :self.fixed_width]

        # Resize to img_size × img_size
        spec = F.interpolate(
            spec.unsqueeze(0),                              # (1,1,H,W)
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                                        # (1, H, W)

        # Per-sample normalisation
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        return spec, self.labels[idx]


# ==========================================
# 4. Model A — CNN (Ass-2 style, no attention)
# ==========================================
class CNN(nn.Module):
    """
    Straightforward CNN for spectrogram digit classification.
    Architecture mirrors the spirit of the Ass-2 baseline but
    operates on 2-D spectrogram images instead of 1-D vectors.
    """
    def __init__(self, num_classes=10, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 64→32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 32→16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16→8
        )
        feat_dim = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 5. Spatial Attention Module (CBAM-style)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B,1,H,W)
        attn = torch.cat([avg_out, max_out], dim=1)         # (B,2,H,W)
        attn = self.sigmoid(self.conv(attn))                # (B,1,H,W)
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * attn


# ==========================================
# 6. Model B — CNN + Attention (CBAM after each block)
# ==========================================
class CNNWithAttention(nn.Module):
    """
    Same CNN backbone as Model A, with a Channel + Spatial
    attention gate (CBAM) inserted after every conv block.
    """
    def __init__(self, num_classes=10, img_size=64):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.attn1_ch = ChannelAttention(32)
        self.attn1_sp = SpatialAttention()
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.attn2_ch = ChannelAttention(64)
        self.attn2_sp = SpatialAttention()
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.attn3_ch = ChannelAttention(128)
        self.attn3_sp = SpatialAttention()
        self.pool3 = nn.MaxPool2d(2)

        feat_dim = 128 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.attn1_ch(x)
        x = self.attn1_sp(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.attn2_ch(x)
        x = self.attn2_sp(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.attn3_ch(x)
        x = self.attn3_sp(x)
        x = self.pool3(x)

        return self.classifier(x)


# ==========================================
# 7. Training & Evaluation Engine
# ==========================================
def train_model(model, train_loader, test_loader, epochs=25, exp_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train_loss': [], 'test_loss': [],
               'train_acc':  [], 'test_acc':  []}
    total_train_time = 0.0

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * inputs.size(0)
            _, pred     = torch.max(out, 1)
            tr_correct += (pred == labels).sum().item()
            tr_total   += labels.size(0)

        total_train_time += time.time() - t0
        scheduler.step()

        ep_tr_loss = tr_loss / tr_total
        ep_tr_acc  = 100.0 * tr_correct / tr_total

        # ---- Evaluate ----
        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out  = model(inputs)
                loss = criterion(out, labels)
                te_loss    += loss.item() * inputs.size(0)
                _, pred     = torch.max(out, 1)
                te_correct += (pred == labels).sum().item()
                te_total   += labels.size(0)

        ep_te_loss = te_loss / te_total
        ep_te_acc  = 100.0 * te_correct / te_total

        history['train_loss'].append(ep_tr_loss)
        history['test_loss'].append(ep_te_loss)
        history['train_acc'].append(ep_tr_acc)
        history['test_acc'].append(ep_te_acc)

        logging.info(
            f"[{exp_name}] Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {ep_tr_loss:.4f}, Acc: {ep_tr_acc:.1f}% | "
            f"Test  Loss: {ep_te_loss:.4f}, Acc: {ep_te_acc:.1f}%"
        )

    # ---- Per-model plots ----
    safe = exp_name.replace(" ", "_").replace("+", "plus")
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['train_acc'], marker='o', label='Train Acc')
    plt.plot(epochs_range, history['test_acc'],  marker='s', label='Test Acc')
    plt.title(f'Accuracy — {exp_name}')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cnn_ACC_{safe}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['train_loss'], marker='o', label='Train Loss')
    plt.plot(epochs_range, history['test_loss'],  marker='s', label='Test Loss')
    plt.title(f'Loss — {exp_name}')
    plt.xlabel('Epoch'); plt.ylabel('CrossEntropy Loss')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cnn_LOSS_{safe}.png", dpi=150)
    plt.close()

    final_test_acc  = history['test_acc'][-1]
    best_epoch      = int(np.argmax(history['test_acc'])) + 1
    logging.info(f"[{exp_name}] FINAL TEST ACC: {final_test_acc:.1f}% | "
                 f"Best Epoch: {best_epoch} | Total Train Time: {total_train_time:.1f}s")

    return history, total_train_time, final_test_acc, best_epoch


# ==========================================
# 8. Comparison Plot (both models, same axes)
# ==========================================
def plot_comparison(hist_a, hist_b, name_a, name_b, epochs):
    er = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CNN vs CNN + Attention — Spoken Digit Recognition', fontsize=13, fontweight='bold')

    ax1.plot(er, hist_a['train_loss'], '--', label=f'{name_a} Train')
    ax1.plot(er, hist_a['test_loss'],        label=f'{name_a} Test')
    ax1.plot(er, hist_b['train_loss'], '--', label=f'{name_b} Train')
    ax1.plot(er, hist_b['test_loss'],        label=f'{name_b} Test')
    ax1.set_title('Training & Test Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(er, hist_a['train_acc'], '--', label=f'{name_a} Train')
    ax2.plot(er, hist_a['test_acc'],        label=f'{name_a} Test')
    ax2.plot(er, hist_b['train_acc'], '--', label=f'{name_b} Train')
    ax2.plot(er, hist_b['test_acc'],        label=f'{name_b} Test')
    ax2.set_title('Training & Test Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('cnn_comparison_plot.png', dpi=150)
    plt.close()
    logging.info("Saved: cnn_comparison_plot.png")


# ==========================================
# 9. Main
# ==========================================
if __name__ == "__main__":
    TRAIN_PATH = "./audio-dataset/Train"
    TEST_PATH  = "./audio-dataset/Test"
    IMG_SIZE   = 64
    EPOCHS     = 10

    # ── Datasets ──────────────────────────────────────────────
    logging.info("Loading spectrogram datasets …")
    train_ds = SpectrogramDataset(TRAIN_PATH, img_size=IMG_SIZE)
    test_ds  = SpectrogramDataset(TEST_PATH,  img_size=IMG_SIZE,
                                  fixed_width=train_ds.fixed_width)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=0, pin_memory=False)

    logging.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # ── Model A — plain CNN ────────────────────────────────────
    logging.info("\n" + "="*60)
    logging.info("MODEL A: CNN (no attention)")
    logging.info("="*60)
    torch.manual_seed(SEED)
    model_a = CNN(num_classes=10, img_size=IMG_SIZE)
    hist_a, time_a, acc_a, best_a = train_model(
        model_a, train_loader, test_loader,
        epochs=EPOCHS, exp_name="CNN"
    )

    # ── Model B — CNN + CBAM Attention ────────────────────────
    logging.info("\n" + "="*60)
    logging.info("MODEL B: CNN + Spatial Attention (CBAM)")
    logging.info("="*60)
    torch.manual_seed(SEED)
    model_b = CNNWithAttention(num_classes=10, img_size=IMG_SIZE)
    hist_b, time_b, acc_b, best_b = train_model(
        model_b, train_loader, test_loader,
        epochs=EPOCHS, exp_name="CNN_plus_Attention"
    )

    # ── Comparison Plot ────────────────────────────────────────
    plot_comparison(hist_a, hist_b, "CNN", "CNN+Attention", EPOCHS)

    # ── Summary Table ──────────────────────────────────────────
    acc_diff  = acc_b - acc_a
    time_diff = time_b - time_a
    direction = "higher" if acc_diff >= 0 else "lower"
    overhead  = "more" if time_diff >= 0 else "less"

    logging.info("\n" + "="*72)
    logging.info("FINAL COMPARISON TABLE")
    logging.info("="*72)
    logging.info(f"{'Model':<30} {'Test Acc (%)':>12} {'Train Time (s)':>15} {'Best Epoch':>11}")
    logging.info("-"*72)
    logging.info(f"{'CNN (no attention)':<30} {acc_a:>11.1f}% {time_a:>14.1f}s {best_a:>11}")
    logging.info(f"{'CNN + Attention (CBAM)':<30} {acc_b:>11.1f}% {time_b:>14.1f}s {best_b:>11}")
    logging.info("="*72)

    print(f"""
Analysis:
Model B (CNN + CBAM Attention) achieved a final test accuracy {abs(acc_diff):.1f}% {direction} than
the baseline CNN (Model A), demonstrating that the channel-and-spatial attention gates
{'help the network focus on phonetically discriminative time-frequency regions of the spectrogram, yielding a measurable accuracy gain.' if acc_diff > 0 else 'did not provide a clear benefit on this dataset, possibly because spoken digit spectrograms are already compact enough for a plain CNN to capture.'}
The attention modules increased total training time by {abs(time_diff):.1f}s
({abs(time_diff)/max(time_a,1)*100:.1f}% {overhead} than the baseline), which is a
{'modest overhead relative to the accuracy gain it provides.' if abs(time_diff)/max(time_a,1) < 0.3 else 'notable cost that should be weighed against the accuracy improvement.'}
Overall, {'the CBAM attention mechanism offers a meaningful improvement for spectrogram-based digit recognition at an acceptable computational cost.' if acc_diff > 0 else 'for this relatively simple 10-class digit task the plain CNN remains highly competitive, and attention may be more beneficial with larger or noisier datasets.'}
""")