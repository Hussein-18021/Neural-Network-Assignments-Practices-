import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torch.utils.data import Dataset, DataLoader
import logging
import time
import random
import soundfile as sf  
import matplotlib.pyplot as plt # <--- Added for Graph Drawing!

# ==========================================
# 1. Fully Detailed Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("assignment_2_problem_3_graphs.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. Smart Audio-to-Image Dataset 
# ==========================================
class AugmentedSpectrogramDataset(Dataset):
    def __init__(self, data_dir, mode='A'):
        self.data_dir = data_dir
        self.mode = mode
        self.file_paths = []
        self.labels = []
        
        sample_file = next(f for f in os.listdir(data_dir) if f.endswith('.wav'))
        _, self.true_sr = sf.read(os.path.join(data_dir, sample_file))
        logging.info(f"Directory: {data_dir} | Detected Sample Rate: {self.true_sr} Hz")
        
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=self.true_sr,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )
        self.amplitude_to_db = AT.AmplitudeToDB()
        self.resize = VT.Resize((64, 64)) 

        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                match = re.search(r'_(\d)\.wav$', file)
                if match:
                    self.file_paths.append(os.path.join(data_dir, file))
                    self.labels.append(int(match.group(1)))
                else:
                    match_fallback = re.search(r'(\d)\.wav$', file)
                    if match_fallback:
                        self.file_paths.append(os.path.join(data_dir, file))
                        self.labels.append(int(match_fallback.group(1)))

    def apply_audio_augmentation(self, waveform):
        # 1. Speed Perturbation
        speed_factor = random.choice([0.97, 1.03])
        waveform = F.interpolate(waveform.unsqueeze(0), scale_factor=1.0/speed_factor, mode='linear', align_corners=False).squeeze(0)
        # 2. White Noise Injection
        noise = torch.randn_like(waveform) * 0.005 
        waveform = waveform + noise
        return waveform

    def apply_image_augmentation(self, spectrogram):
        # 1. Random Squeeze (Width)
        squeeze_factor = random.uniform(0.8, 1.0)
        new_width = int(64 * squeeze_factor)
        spectrogram = VT.Resize((64, new_width))(spectrogram)
        spectrogram = VT.Pad((0, 0, 64 - new_width, 0))(spectrogram) 
        # 2. Random Noise
        noise = torch.randn_like(spectrogram) * 0.1 
        spectrogram = spectrogram + noise
        return spectrogram

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_data, sr = sf.read(self.file_paths[idx])
        waveform = torch.tensor(wav_data, dtype=torch.float32)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        else:
            waveform = waveform.t() 
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # APPLY AUDIO AUGMENTATION (Parts B and D)
        if self.mode in ['B', 'D']:
            waveform = self.apply_audio_augmentation(waveform)

        spectrogram = self.mel_transform(waveform)
        spectrogram = self.amplitude_to_db(spectrogram)
        
        # Normalize strictly to [-1, 1] range to fix color washing
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram = (spectrogram * 2) - 1.0 
        
        spectrogram = self.resize(spectrogram) 

        # APPLY IMAGE AUGMENTATION (Parts C and D)
        if self.mode in ['C', 'D']:
            spectrogram = self.apply_image_augmentation(spectrogram)

        return spectrogram, self.labels[idx]

# ==========================================
# 3. Convolutional Neural Network
# ==========================================
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. Upgraded Training Engine with Graphs!
# ==========================================
def train_and_test(model, train_loader, test_loader, epochs=15, exp_name=""):
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
    safe_exp_name = exp_name.replace(" ", "_").replace(":", "").replace("/", "_")
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
    plt.savefig(f"prob3_graph_ACC_{safe_exp_name}.png") 
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
    plt.savefig(f"prob3_graph_LOSS_{safe_exp_name}.png")
    plt.close()
            
    final_test_accuracy = history['test_acc'][-1]
    logging.info(f"--> FINAL TEST ACCURACY ({exp_name}): {final_test_accuracy:.1f}%")
    return total_train_time * 1000, total_test_time * 1000, final_test_accuracy

# ==========================================
# 5. Main Execution Matrix
# ==========================================
if __name__ == "__main__":
    # ---> UPDATE THESE PATHS <---
    TRAIN_PATH = "./speech_data/train" 
    TEST_PATH = "./speech_data/test"
    
    logging.info("Initializing Test Dataset (No Augmentation)...")
    test_dataset = AugmentedSpectrogramDataset(TEST_PATH, mode='A')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    experiments = {
        "Part A: Baseline (No Augmentation)": 'A',
        "Part B: Audio Augmentation (Speed Noise)": 'B',
        "Part C: Image Augmentation (Squeeze Noise)": 'C',
        "Part D: Combined Audio & Image Aug": 'D'
    }
    
    results = {}
    
    for name, mode in experiments.items():
        logging.info("\n" + "="*60)
        logging.info(f"--- STARTING: {name} ---")
        logging.info("="*60)
        
        train_dataset = AugmentedSpectrogramDataset(TRAIN_PATH, mode=mode)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = AudioCNN()
        
        t_train, t_test, acc = train_and_test(model, train_loader, test_loader, epochs=15, exp_name=name)
        results[name] = {'acc': acc, 'train_time': t_train, 'test_time': t_test}
        
    logging.info("\n" + "="*80)
    logging.info("FINAL PROBLEM 3 COMPARISON TABLE")
    logging.info("="*80)
    for name, metrics in results.items():
        logging.info(f"{name:<45} | Acc: {metrics['acc']:.1f}% | Train Time: {metrics['train_time']:.1f}ms | Test Time: {metrics['test_time']:.1f}ms")