import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt 

# ==========================================
# 1. Logging & Reproducibility
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("assignment_2_problem_5_augmentation.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 2. Augmentation Pipeline
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.05) 
])

class AugmentedDataset(Dataset):
    def __init__(self, real_subset, num_generated_per_digit):
        self.real_subset = real_subset
        self.num_real = len(real_subset)
        self.num_generated = num_generated_per_digit * 10 
        self.total_size = self.num_real + self.num_generated

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < self.num_real:
            return self.real_subset[idx]
        else:
            random_real_idx = random.randint(0, self.num_real - 1)
            image, label = self.real_subset[random_real_idx]
            aug_image = augmentation_transforms(image)
            return aug_image, label

# ==========================================
# 3. Data Extraction 
# ==========================================
def extract_balanced_subset(dataset, num_per_digit):
    targets = dataset.targets.numpy()
    indices = []
    for i in range(10):
        digit_indices = np.where(targets == i)[0]
        np.random.shuffle(digit_indices)
        indices.extend(digit_indices[:num_per_digit])
    return Subset(dataset, indices)

def get_datasets(num_real_per_digit, num_generated_per_digit):
    base_transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    full_test = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    test_subset = extract_balanced_subset(full_test, 200)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    real_train_subset = extract_balanced_subset(full_train, num_real_per_digit)
    
    if num_generated_per_digit > 0:
        final_train_dataset = AugmentedDataset(real_train_subset, num_generated_per_digit)
    else:
        final_train_dataset = real_train_subset

    train_loader = DataLoader(final_train_dataset, batch_size=64, shuffle=True)
    return train_loader, test_loader

# ==========================================
# 4. The LeNet-5 Model (FIXED ARCHITECTURE)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # FIXED: Added padding=2 so spatial dimensions exactly match Problem 2!
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # FIXED: Because of padding=2, the tensor is now 16*5*5 instead of 16*4*4
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        
        # FIXED: Flatten to 16*5*5
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 5. Upgraded Training Engine with Graphs
# ==========================================
def train_and_test(model, train_loader, test_loader, epochs=10, exp_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- TEST PHASE ---
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100.0 * test_correct / test_total

        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_acc'].append(epoch_test_acc)

        logging.info(f"Epoch {epoch+1}/{epochs} | "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.1f}% | "
                     f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.1f}%")

    # ==========================================
    # Draw and Save the Graphs!
    # ==========================================
    epochs_range = range(1, epochs + 1)
    
    # Graph 1: Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(epochs_range, history['test_acc'], label='Testing Accuracy', marker='s')
    plt.title(f'Accuracy vs. Epochs\n({exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prob5_graph_ACC_{exp_name}.png") 
    plt.close() 

    # Graph 2: Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs_range, history['test_loss'], label='Testing Loss', marker='s')
    plt.title(f'Loss vs. Epochs\n({exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prob5_graph_LOSS_{exp_name}.png")
    plt.close()

    return history['test_acc'][-1]

# ==========================================
# 6. Main Experimental Matrix
# ==========================================
if __name__ == "__main__":
    experiments = [
        # 0 Generated (Baseline)
        (350, 0), (750, 0), (1000, 0),
        # 1000 Generated
        (350, 1000), (750, 1000), (1000, 1000),
        # 1500 Generated
        (350, 1500), (750, 1500), (1000, 1500),
        # 2000 Generated
        (350, 2000), (750, 2000), (1000, 2000)
    ]

    results = {}

    logging.info("Starting Problem 5 Experimental Matrix...")
    logging.info("=" * 80)

    for real_count, gen_count in experiments:
        logging.info(f"\n--- Preparing Data: {real_count} Real + {gen_count} Generated per digit ---")
        train_loader, test_loader = get_datasets(real_count, gen_count)
        
        experiment_name = f"{real_count}Real_{gen_count}Gen"
        
        model = LeNet5()
        accuracy = train_and_test(model, train_loader, test_loader, epochs=10, exp_name=experiment_name)
        
        results[(real_count, gen_count)] = accuracy
        logging.info(f"FINAL RESULT -> Real: {real_count}, Gen: {gen_count} | Accuracy: {accuracy:.1f}%\n")

    # Final Output Matrix
    logging.info("=" * 80)
    logging.info("FINAL PROBLEM 5 ACCURACY TABLE (For your report):")
    logging.info(f"{'Real Data per digit ->':<25} | {'350':<15} | {'750':<15} | {'1000':<15}")
    logging.info("-" * 80)
    
    gen_levels = [0, 1000, 1500, 2000]
    for gen in gen_levels:
        row = f"{gen} Generated per digit   |"
        for real in [350, 750, 1000]:
            acc = results.get((real, gen), 0.0)
            row += f" {acc:>5.1f}%          |"
        logging.info(row)
    logging.info("=" * 80)