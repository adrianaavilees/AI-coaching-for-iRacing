"""
Train the LSTM Autoencoder on expert lap telemetry
 
Architecture: sequence-to-sequence LSTM Autoencoder
    Encoder: LSTM(input=10, hidden=64, layers=1) → latent vector (16,)
    Decoder: LSTM(input=16, hidden=64, layers=1) → reconstructed sequence (1000, 10)
 
Outputs (inside models/):
    - autoencoder_best.pt       Best checkpoint (lowest val loss)
    - scaler_params.npz         Per-channel mean + std for inference
    - training_report.txt       Loss curves + hyperparameter log
    - loss_curve.png            Visual training / validation loss plot
"""

import json 
import time 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from config import DATA_DIR, MODELS_DIR, FEATURE_COLS, N_POINTS, STRIDE, HIDDEN_SIZE, LATENT_DIM, N_LAYERS,TRAIN_HP

TRAIN_META_PATH = DATA_DIR / "train_metadata.csv"

#-------------------------------- Dataset --------------------------------#
class LapDataset(Dataset):
    """Custom Dataset for loading telemetry laps with optional noise augmentation."""
    def __init__(self, telemetry, noise_std=0.0, n_augments=0, device="cpu"):
        self.data = torch.tensor(telemetry, dtype=torch.float32).to(device)
        self.noise_std = noise_std
        self.n_augments = n_augments
        self.device = device

    def __len__(self):
        return len(self.data) * (1 + self.n_augments) # Each lap + its augmented versions

    def __getitem__(self, idx):
        lap_idx = idx // (1 + self.n_augments)
        base_lap = self.data[lap_idx][::STRIDE]  # subsample: 1000 → n_points
        
        if self.noise_std > 0 and (idx % (1 + self.n_augments)) > 0:
            noise = torch.normal(0, self.noise_std, size=base_lap.shape, device=self.device)
            augmented_lap = base_lap + noise
            return augmented_lap
        
        return base_lap
    

#-------------------------------- Model --------------------------------#
class LSTMEncoder(nn.Module):
    """Encodes a telemetry sequence into a compact latent vector."""
    def __init__(self, input_size, hidden_size, latent_dim, n_layers=1, dropout=0.0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0) # batch_first=True → input shape: (batch, seq_len, features)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n shape: (n_layers, batch_size, hidden_size)
        z = self.fc(h_n[-1])        # Compress to latent vector (true bottleneck)
        return z

class LSTMDecoder(nn.Module):
    """Decodes a latent vector back into a telemetry sequence."""
    def __init__(self, latent_dim, hidden_size, output_size, n_points, n_layers=1, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_points = n_points
        self.fc = nn.Linear(latent_dim, hidden_size)  # project z → initial hidden state
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # Project z to hidden state this is the true 32-dim bottleneck
        h_init = self.fc(z).unsqueeze(0)              # (1, batch, hidden_size) z(32) → h_init(128)
        c_init = torch.zeros_like(h_init)
        inp = torch.zeros(z.size(0), self.n_points, self.hidden_size, device=z.device) # (batch, seq_len, hidden_size) zero input to start decoding
        lstm_out, _ = self.lstm(inp, (h_init, c_init))
        return self.output_layer(lstm_out)
        
class LSTMAutoencoder(nn.Module):
    """Combines the LSTMEncoder and LSTMDecoder into a full autoencoder."""
    def __init__(self, input_size, hidden_size, latent_dim, n_points, n_layers=1, dropout=0.0):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, n_points, n_layers, dropout)

    def forward(self, x):
        """Encode the input sequence and then decode it back to the original space."""
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x):
        """Encode a telemetry sequence into the latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode a latent vector back to telemetry space."""
        return self.decoder(z)
    
# --------------------------- Normalization --------------------------#
def compute_scaler_params(telemetry):
    """Compute per-channel mean and std for normalization."""
    flat = telemetry.reshape(-1, telemetry.shape[-1])  # Flatten to (n_samples * seq_len, n_features)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8  # Add small value to prevent division by zero
    return mean, std

def apply_normalization(telemetry, mean, std):
    return (telemetry - mean) / std


# --------------------------- Training Loop --------------------------#
def train_autoencoder(model, train_loader, val_loader, hp, device):
    """Train the LSTM Autoencoder with early stopping and learning rate scheduling."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=hp["patience"]//2)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent exploding gradients
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch}/{hp['epochs']} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "autoencoder_best.pt")
            print("  → New best model saved.")
        else:
            epochs_no_improve += 1
        
        scheduler.step(avg_val_loss)

        # Early stopping
        if epochs_no_improve >= hp["patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    return training_history


# --------------------------- Main --------------------------#
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(TRAIN_HP["seed"])
    np.random.seed(TRAIN_HP["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess training data
    train_meta = pd.read_csv(TRAIN_META_PATH)
    train_telemetry = np.load(DATA_DIR / "train_telemetry.npy")  # Shape: (n_laps, n_points, n_features)
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # Apply normalization
    mean, std = compute_scaler_params(train_telemetry)
    train_telemetry_norm = apply_normalization(train_telemetry, mean, std)
    np.savez(MODELS_DIR / "scaler_params.npz", mean=mean, std=std)

    # Create Dataset and DataLoader
    full_dataset = LapDataset(train_telemetry_norm, noise_std=TRAIN_HP["noise_std"], n_augments=TRAIN_HP["n_augments"], device=device)
    
    val_size = int(len(full_dataset) * TRAIN_HP["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=False)

    print(f"Training samples: {len(train_dataset)} (including augmented), Validation samples: {len(val_dataset)}")

    # Initialize model
    model = LSTMAutoencoder(input_size=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM, n_points=N_POINTS, n_layers=N_LAYERS, dropout=TRAIN_HP["dropout"]).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train = train_autoencoder(model, train_loader, val_loader, TRAIN_HP, device)
    print("Training completed")
    print(f"Best validation loss: {min(train['val_loss']):.6f}")

    # Save training report
    with open(MODELS_DIR / "training_report.txt", "w") as f:
        json.dump({
            "hyperparameters": TRAIN_HP,
            "train": train,
        }, f, indent=4)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train["train_loss"], label="Train Loss")
    plt.plot(train["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Autoencoder Training Loss")
    plt.legend()
    plt.grid()
    plt.savefig(MODELS_DIR / "loss_curve.png")  
    plt.show()

if __name__ == "__main__":
    main()