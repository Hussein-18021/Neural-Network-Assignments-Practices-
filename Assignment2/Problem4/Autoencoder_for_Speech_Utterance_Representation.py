import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import Dataset, DataLoader
import logging
import time
import soundfile as sf  
import matplotlib.pyplot as plt # <--- Added for Graph Drawing!

# ==========================================
# 1. Fully Detailed Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("assignment_2_problem_4_graphs.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. Dataset: 15ms Frame Extractor with Normalization
# ==========================================
class AudioFrameDataset(Dataset):
    def __init__(self, data_dir, mode='average', max_frames=None):
        self.data_dir = data_dir
        self.mode = mode
        self.file_paths = []
        self.labels = []
        
        sample_file = next(f for f in os.listdir(data_dir) if f.endswith('.wav'))
        _, self.true_sr = sf.read(os.path.join(data_dir, sample_file))
        
        self.frame_size = int(self.true_sr * 0.015) 
        
        self.mfcc_transform = AT.MFCC(
            sample_rate=self.true_sr,
            n_mfcc=40,
            melkwargs={
                'n_fft': self.frame_size,
                'hop_length': self.frame_size, 
                'center': False
            }
        )

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

        self.max_frames = max_frames
        if self.mode == 'flatten' and self.max_frames is None:
            logging.info(f"Scanning {data_dir} to find the maximum utterance length...")
            max_len = 0
            for path in self.file_paths:
                wav, _ = sf.read(path)
                num_frames = len(wav) // self.frame_size
                if num_frames > max_len:
                    max_len = num_frames
            self.max_frames = max_len
            logging.info(f"Maximum utterance length found: {self.max_frames} frames.")

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
            
        mfcc = self.mfcc_transform(waveform).squeeze(0) 
        mfcc = (mfcc - torch.mean(mfcc)) / (torch.std(mfcc) + 1e-6)
        
        if self.mode == 'average':
            feature_vector = torch.mean(mfcc, dim=1) 
            return feature_vector, self.labels[idx]
            
        elif self.mode == 'flatten':
            current_frames = mfcc.shape[1]
            if current_frames < self.max_frames:
                pad_amount = self.max_frames - current_frames
                mfcc = F.pad(mfcc, (0, pad_amount))
            elif current_frames > self.max_frames:
                mfcc = mfcc[:, :self.max_frames]
                
            feature_vector = mfcc.flatten() 
            return feature_vector, self.labels[idx]

# ==========================================
# 3. Enhanced Models
# ==========================================
class BaselineClassifier(nn.Module):
    def __init__(self, input_size=40): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

class SpeechAutoencoder(nn.Module):
    def __init__(self, input_size, bottleneck_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_size)
        )
        
    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return reconstructed, bottleneck

class AE_Classifier(nn.Module):
    def __init__(self, bottleneck_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Training Engines (With Graphs and Test Evaluators)
# ==========================================
def train_classifier(model, train_loader, test_loader, epochs=25, is_ae_classifier=False, encoder=None, exp_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if is_ae_classifier and encoder is not None:
        encoder.to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_train_time = 0.0
    total_test_time = 0.0
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_start_time = time.time()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_ae_classifier and encoder is not None:
                encoder.eval() 
                with torch.no_grad():
                    inputs = encoder(inputs) 
                    
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
        test_start_time = time.time()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if is_ae_classifier and encoder is not None:
                    inputs = encoder(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        total_test_time += (time.time() - test_start_time)
        epoch_test_loss = test_loss / test_total
        epoch_test_acc = 100.0 * test_correct / test_total

        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_acc'].append(epoch_test_acc)
        
        logging.info(f"Classifier Epoch {epoch+1}/{epochs} | "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.1f}% | "
                     f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.1f}%")
                     
    # ==========================================
    # Draw and Save the Graphs!
    # ==========================================
    safe_exp_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
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
    plt.savefig(f"prob4_graph_ACC_{safe_exp_name}.png") 
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
    plt.savefig(f"prob4_graph_LOSS_{safe_exp_name}.png")
    plt.close()

    acc = history['test_acc'][-1]
    logging.info(f"--> FINAL TEST ACCURACY ({exp_name}): {acc:.1f}%")
    return total_train_time * 1000, total_test_time * 1000, acc

def train_autoencoder(ae_model, train_loader, test_loader, epochs=25, exp_name="Autoencoder_MSE"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model.to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'test_loss': []}
    
    logging.info("--- Training Autoencoder (Minimizing ||F - F'||_2) ---")
    for epoch in range(epochs):
        ae_model.train()
        train_loss = 0.0
        for inputs, _ in train_loader: 
            inputs = inputs.to(device)
            optimizer.zero_grad()
            reconstructed, _ = ae_model(inputs)
            loss = criterion(reconstructed, inputs) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        ae_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                reconstructed, _ = ae_model(inputs)
                loss = criterion(reconstructed, inputs)
                test_loss += loss.item() * inputs.size(0)
                
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_test_loss = test_loss / len(test_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['test_loss'].append(epoch_test_loss)
        
        logging.info(f"AE Epoch {epoch+1}/{epochs} | "
                     f"Train MSE: {epoch_train_loss:.4f} | Test MSE: {epoch_test_loss:.4f}")

    # Plot the AE Reconstruction Graph
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training MSE', marker='o')
    plt.plot(epochs_range, history['test_loss'], label='Testing MSE', marker='s')
    plt.title(f'MSE Loss vs. Epochs\n({exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prob4_graph_LOSS_{exp_name}.png")
    plt.close()
    
    return ae_model

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    TRAIN_PATH = "./speech_data/train" 
    TEST_PATH = "./speech_data/test"
    
    results = {}
    
    # ---------------------------------------------------------
    # PART 1: The Baseline (Average Frame)
    # ---------------------------------------------------------
    logging.info("\n" + "="*50 + "\nPART 1: BASELINE (AVERAGE FRAME)\n" + "="*50)
    train_data_avg = AudioFrameDataset(TRAIN_PATH, mode='average')
    test_data_avg = AudioFrameDataset(TEST_PATH, mode='average')
    
    train_loader_avg = DataLoader(train_data_avg, batch_size=32, shuffle=True)
    test_loader_avg = DataLoader(test_data_avg, batch_size=32, shuffle=False)
    
    baseline_model = BaselineClassifier(input_size=40)
    t_train, t_test, acc = train_classifier(
        baseline_model, train_loader_avg, test_loader_avg, epochs=25, exp_name="Baseline_Average_Frame"
    )
    results["Baseline (Average Frame)"] = {'acc': acc, 'train_time': t_train, 'test_time': t_test}

    # ---------------------------------------------------------
    # PART 2: The Autoencoder (Concatenated & Padded Frames)
    # ---------------------------------------------------------
    logging.info("\n" + "="*50 + "\nPART 2: AUTOENCODER REPRESENTATION\n" + "="*50)
    
    train_data_ae = AudioFrameDataset(TRAIN_PATH, mode='flatten')
    max_train_frames = train_data_ae.max_frames 
    test_data_ae = AudioFrameDataset(TEST_PATH, mode='flatten', max_frames=max_train_frames)
    
    train_loader_ae = DataLoader(train_data_ae, batch_size=32, shuffle=True)
    test_loader_ae = DataLoader(test_data_ae, batch_size=32, shuffle=False)
    
    input_size_ae = 40 * max_train_frames 
    logging.info(f"Concatenated Vector Size for AE: {input_size_ae}")
    
    # 2A. Train the Autoencoder
    autoencoder = SpeechAutoencoder(input_size=input_size_ae, bottleneck_size=256)
    autoencoder = train_autoencoder(autoencoder, train_loader_ae, test_loader_ae, epochs=30, exp_name="AE_Reconstruction")
    
    # 2B. Train the Classifier on the AE features
    logging.info("\n--- Training Final Classifier on Autoencoder Features ---")
    ae_classifier = AE_Classifier(bottleneck_size=256)
    
    t_train, t_test, acc = train_classifier(
        ae_classifier, train_loader_ae, test_loader_ae, epochs=25, 
        is_ae_classifier=True, encoder=autoencoder.encoder, exp_name="Autoencoder_Bottleneck"
    )
    results["Autoencoder Bottleneck"] = {'acc': acc, 'train_time': t_train, 'test_time': t_test}
    
    # --- Final Output ---
    logging.info("\n" + "="*80)
    logging.info("FINAL PROBLEM 4 COMPARISON TABLE")
    logging.info("="*80)
    for name, metrics in results.items():
        logging.info(f"{name:<25} | Acc: {metrics['acc']:.1f}% | Train Time: {metrics['train_time']:.1f}ms | Test Time: {metrics['test_time']:.1f}ms")