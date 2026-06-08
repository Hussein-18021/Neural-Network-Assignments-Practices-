import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
full_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

rng = np.random.default_rng(SEED)
train_idx = rng.choice(len(full_train), size=10_000, replace=False)
test_idx  = rng.choice(len(full_test),  size=2_000,  replace=False)

train_loader = DataLoader(Subset(full_train, train_idx), batch_size=64, shuffle=True)
test_loader  = DataLoader(Subset(full_test,  test_idx),  batch_size=64, shuffle=False)

# ── Model A — LeNet-5 ────────────────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.tanh  = nn.Tanh()

    def forward(self, x):
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# ── Spatial Attention Module ─────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

# ── Model B — LeNet-5 + Spatial Attention ────────────────────────────────────
class LeNet5WithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1     = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1     = nn.AvgPool2d(2, 2)
        self.conv2     = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2     = nn.AvgPool2d(2, 2)
        self.attention = SpatialAttention()
        self.fc1       = nn.Linear(16 * 5 * 5, 120)
        self.fc2       = nn.Linear(120, 84)
        self.fc3       = nn.Linear(84, 10)
        self.tanh      = nn.Tanh()

    def forward(self, x):
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.tanh(self.conv2(x))
        x = self.attention(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# ── Training & Evaluation ─────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total

def train_model(model, name):
    torch.manual_seed(SEED)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []
    epoch_accs   = []
    start = time.time()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc      = evaluate(model, test_loader)
        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)
        print(f"[{name}] Epoch {epoch+1:02d}/10 — Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    elapsed = time.time() - start
    return epoch_losses, epoch_accs, elapsed

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training Model A — LeNet-5")
print("=" * 60)
model_a = LeNet5().to(device)
losses_a, accs_a, time_a = train_model(model_a, "Model A")

print()
print("=" * 60)
print("Training Model B — LeNet-5 + Spatial Attention")
print("=" * 60)
model_b = LeNet5WithAttention().to(device)
losses_b, accs_b, time_b = train_model(model_b, "Model B")

# ── Results Table ─────────────────────────────────────────────────────────────
best_epoch_a = int(np.argmax(accs_a)) + 1
best_epoch_b = int(np.argmax(accs_b)) + 1

print()
print("=" * 60)
print(f"{'Model':<30} {'Test Acc (%)':>12} {'Train Time (s)':>15} {'Best Epoch':>11}")
print("-" * 60)
print(f"{'Model A — LeNet-5':<30} {accs_a[-1]:>11.2f}% {time_a:>14.1f}s {best_epoch_a:>11}")
print(f"{'Model B — LeNet-5 + Attention':<30} {accs_b[-1]:>11.2f}% {time_b:>14.1f}s {best_epoch_b:>11}")
print("=" * 60)

# ── Plot ──────────────────────────────────────────────────────────────────────
epochs = range(1, 11)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("LeNet-5 vs LeNet-5 + Spatial Attention — ReducedMNIST", fontsize=13, fontweight="bold")

ax1.plot(epochs, losses_a, marker="o", label="Model A — LeNet-5")
ax1.plot(epochs, losses_b, marker="s", label="Model B — LeNet-5 + Attention")
ax1.set_title("Training Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(epochs, accs_a, marker="o", label="Model A — LeNet-5")
ax2.plot(epochs, accs_b, marker="s", label="Model B — LeNet-5 + Attention")
ax2.set_title("Validation Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
print("\nPlot saved to comparison_plot.png")

# ── Analysis ──────────────────────────────────────────────────────────────────
acc_diff  = accs_b[-1] - accs_a[-1]
time_diff = time_b - time_a
direction = "higher" if acc_diff >= 0 else "lower"

print(f"""
Analysis:
Model B (LeNet-5 + Spatial Attention) achieved a final test accuracy {abs(acc_diff):.2f}% {direction} than
Model A (LeNet-5), suggesting the attention mechanism {'improved' if acc_diff >= 0 else 'did not improve'}
the model's ability to focus on digit-relevant spatial regions. The added attention module
increased total training time by {abs(time_diff):.1f}s ({abs(time_diff)/time_a*100:.1f}% {'more' if time_diff >= 0 else 'less'}),
a {'modest' if abs(time_diff)/time_a < 0.2 else 'notable'} overhead for the architectural complexity it introduces.
Overall, {'the spatial attention mechanism offers a favourable accuracy-to-cost trade-off on this task.' if acc_diff > 0 else 'on this relatively simple dataset the baseline LeNet-5 is competitive without attention.'}
""")