import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import logging
import time
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. Logging & Reproducibility Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("assignment_2_problem_2_graphs.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed=42):
    """Guarantees exact reproducibility for your report."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 2. Dataset Preparation
# ==========================================
def get_reduced_mnist(data_dir='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_indices, test_indices = [], []
    for i in range(10):
        # Find all indices for digit 'i'
        all_train_idx = np.where(full_train.targets.numpy() == i)[0]
        all_test_idx = np.where(full_test.targets.numpy() == i)[0]
        
        # Randomly select 1000 train and 200 test examples per digit
        selected_train = np.random.choice(all_train_idx, 1000, replace=False)
        selected_test = np.random.choice(all_test_idx, 200, replace=False)
        
        train_indices.extend(selected_train)
        test_indices.extend(selected_test)
        
    train_loader = DataLoader(Subset(full_train, train_indices), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(full_test, test_indices), batch_size=32, shuffle=False)
    return train_loader, test_loader

# ==========================================
# 3. Model Architectures
# ==========================================

# BASELINE (Strictly matches the PDF diagram with AvgPool)
class LeNet5_Base(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjusted padding to fit 28x28 images instead of 32x32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# VARIATION 1: Max Pooling (To increase accuracy)
class LeNet5_Var1_MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2) # Changed to MaxPool
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2) # Changed to MaxPool
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# VARIATION 2: Wider Network (More Filters)
class LeNet5_Var2_Wider(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2) # 6 -> 16
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5) # 16 -> 32
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# VARIATION 3: Dropout Regularization (To reduce overfitting)
class LeNet5_Var3_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Added Dropout

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(self.relu(self.fc1(x))) 
        x = self.dropout(self.relu(self.fc2(x))) 
        return self.fc3(x)

# VARIATION 4: Combo (Max Pooling + Wider + Dropout)
class LeNet5_Var4_Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

# ==========================================
# 4. Upgraded Training Engine with Graphs
# ==========================================
def train_and_test(model, train_loader, test_loader, epochs=20, exp_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_train_time = 0.0
    total_test_time = 0.0
    
    # Track metrics for graphing
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        total_train_time += (time.time() - train_start_time)
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- TEST PHASE ---
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        test_start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        total_test_time += (time.time() - test_start_time)
        
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100.0 * test_correct / test_total

        # Save to history for plotting
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
    # Clean string so it can be saved as a valid filename
    safe_exp_name = exp_name.replace(" ", "_").replace(":", "").replace("/", "_").replace("(", "").replace(")", "").replace(",", "")
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
    plt.savefig(f"prob2_graph_ACC_{safe_exp_name}.png") 
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
    plt.savefig(f"prob2_graph_LOSS_{safe_exp_name}.png")
    plt.close()
            
    final_test_accuracy = history['test_acc'][-1]
    logging.info(f"--> FINAL TEST ACCURACY ({exp_name}): {final_test_accuracy:.1f}%")
    return total_train_time * 1000, total_test_time * 1000, final_test_accuracy

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    logging.info("Loading Data with Random Selection (Seed 42)...")
    train_loader, test_loader = get_reduced_mnist()
    
    models_to_test = {
        "Baseline (Avg Pool, 6/16 filters)": LeNet5_Base(),
        "Var 1: Max Pooling": LeNet5_Var1_MaxPool(),
        "Var 2: Wider (16/32 filters)": LeNet5_Var2_Wider(),
        "Var 3: Dropout (0.5)": LeNet5_Var3_Dropout(),
        "Var 4: Combo (Max, Wider, Drop)": LeNet5_Var4_Combo()
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        logging.info("\n" + "="*50)
        logging.info(f"--- STARTING TRAINING: {name} ---")
        logging.info("="*50)
        
        # We pass the 'name' variable as the exp_name so it gets used in the graph title and filename!
        t_train, t_test, acc = train_and_test(model, train_loader, test_loader, epochs=20, exp_name=name)
        results[name] = {'train_time': t_train, 'test_time': t_test, 'acc': acc}
        
    logging.info("\n" + "="*80)
    logging.info("FINAL COMPARISON TABLE (For your report)")
    logging.info("="*80)
    for name, metrics in results.items():
        logging.info(f"{name:<35} | Acc: {metrics['acc']:.1f}% | Train Time: {metrics['train_time']:.1f}ms | Test Time: {metrics['test_time']:.1f}ms")