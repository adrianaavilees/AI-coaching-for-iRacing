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
import sys
import time 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
from utils.config import DATA_DIR, MODELS_DIR, FEATURE_COLS, N_POINTS, HIDDEN_SIZE, LATENT_DIM, N_LAYERS,TRAIN_HP

TRAIN_META_PATH = DATA_DIR / "train_metadata.csv"

#* -------------------------------- Dataset -------------------------------- #
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
        base_lap = self.data[lap_idx]
        
        if self.noise_std > 0 and (idx % (1 + self.n_augments)) > 0:
            noise = torch.normal(0, self.noise_std, size=base_lap.shape, device=self.device)
            augmented_lap = base_lap + noise
            return augmented_lap
        
        return base_lap
    

#* -------------------------------- Model --------------------------------#
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
        self.n_layers = n_layers
        self.fc = nn.Linear(latent_dim, hidden_size)  # project z → initial hidden state
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # Project z to hidden state this is the true 32-dim bottleneck
        h_init = self.fc(z).unsqueeze(0).repeat(self.n_layers, 1, 1)  # (n_layers, batch, hidden_size)
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
    
#* --------------------------- Normalization --------------------------#
def compute_scaler_params(telemetry):
    """Compute per-channel mean and std for normalization."""
    flat = telemetry.reshape(-1, telemetry.shape[-1])  # Flatten to (n_samples * seq_len, n_features)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8  # Add small value to prevent division by zero
    return mean, std

def apply_normalization(telemetry, mean, std):
    return (telemetry - mean) / std


#* --------------------------- Training Loop --------------------------#
def train_autoencoder(model, train_loader, val_loader, hp, device, save_path):
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
            torch.save(model.state_dict(), save_path)
            print("  → New best model saved.")
        else:
            epochs_no_improve += 1
        
        scheduler.step(avg_val_loss)

        # Early stopping
        if epochs_no_improve >= hp["patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    return training_history, best_val_loss


#* ------------------------------- Cross-Validation ------------------------------- #

def run_cross_validation(train_telemetry, device, k=5):
    """
    Run K-Fold Cross-Validation to validate hyperparameters.
    """
    print(f"\n{'='*55}")
    print(f" {k}-Fold Cross-Validation")
    print(f"{'='*55}")

    kf = KFold(n_splits=k, shuffle=True, random_state=TRAIN_HP["seed"])

    fold_results = []
    all_fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_telemetry)):
        print(f"\n--- Fold {fold + 1}/{k} (train: {len(train_idx)}, val: {len(val_idx)}) ---")

        train_data = train_telemetry[train_idx]
        val_data = train_telemetry[val_idx]

        # Scaler fitted on train fold only (no data leakage)
        mean, std = compute_scaler_params(train_data)
        train_data = apply_normalization(train_data, mean, std)
        val_data = apply_normalization(val_data, mean, std)

        train_dataset = LapDataset(train_data, noise_std=TRAIN_HP["noise_std"], n_augments=TRAIN_HP["n_augments"], device=device)
        val_dataset = LapDataset(val_data, noise_std=0.0, n_augments=0, device=device)
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=False)

        model = LSTMAutoencoder(input_size=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM, n_points=N_POINTS, n_layers=N_LAYERS, dropout=TRAIN_HP["dropout"]).to(device)
        save_path = MODELS_DIR / f"autoencoder_fold{fold + 1}.pt"

        history, best_val_loss = train_autoencoder(model, train_loader, val_loader, TRAIN_HP, device, save_path)
        all_fold_histories.append(history)
        fold_results.append({"fold": fold + 1, "best_val_loss": best_val_loss})

    losses = [r["best_val_loss"] for r in fold_results]

    print(f"\n{'='*55}")
    print(f" Cross Validation Results:")
    for r in fold_results:
        print(f"   Fold {r['fold']}: {r['best_val_loss']:.6f}")
    print(f"   Mean ± Std: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
    print(f"{'='*55}")

    # Save CV report
    report = {
        "hyperparameters": TRAIN_HP,
        "fold_results": fold_results,
        "mean_val_loss": float(np.mean(losses)),
        "std_val_loss": float(np.std(losses)),
    }
    with open(MODELS_DIR / "cross_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Plot per-fold val loss
    plt.figure(figsize=(10, 6))
    for fold, history in enumerate(all_fold_histories):
        plt.plot(history["val_loss"], label=f"Fold {fold + 1} Val Loss")
    plt.title("LSTM Autoencoder Validation Loss Across Folds")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.savefig(MODELS_DIR / "cross_validation_loss_curve.png")
    plt.show()

    return report


#* ------------------------------- Final Model ------------------------------- #
def train_final_model(train_telemetry, device):
    """
    Train the final model on 100% of the train set.
    """
    print(f"\n{'='*55}")
    print(f" Training Final Model...")
    print(f"{'='*55}")

    # Scaler on the FULL train set → this is what evaluation will use
    mean, std = compute_scaler_params(train_telemetry)
    np.savez(MODELS_DIR / "scaler_params.npz", mean=mean, std=std)

    # Normalize
    all_norm = apply_normalization(train_telemetry, mean, std)

    # Random split for early stopping (no data leakage since scaler is fitted on full train set)
    indices = np.random.permutation(len(all_norm))
    val_size = max(1, int(len(all_norm) * TRAIN_HP["val_split"]))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = LapDataset(all_norm[train_idx], noise_std=TRAIN_HP["noise_std"], n_augments=TRAIN_HP["n_augments"], device=device)
    val_dataset = LapDataset(all_norm[val_idx], noise_std=0.0, n_augments=0, device=device)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_HP["batch_size"], shuffle=False)

    print(f"Train: {len(train_idx)} laps ({len(train_dataset)} w/ augment), Val (early-stop): {len(val_idx)} laps")

    model = LSTMAutoencoder(input_size=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE, latent_dim=LATENT_DIM, n_points=N_POINTS, n_layers=N_LAYERS, dropout=TRAIN_HP["dropout"]).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    save_path = MODELS_DIR / "autoencoder_best.pt"
    history, best_val_loss = train_autoencoder(model, train_loader, val_loader, TRAIN_HP, device, save_path)

    print(f"\nFinal model best val loss: {best_val_loss:.6f}")
    print(f"Saved: autoencoder_best.pt + scaler_params.npz")

    # Save report
    report = {
        "hyperparameters": TRAIN_HP,
        "best_val_loss": best_val_loss,
        "train_history": history,
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Autoencoder — Final Model Training")
    plt.legend()
    plt.grid()
    plt.savefig(MODELS_DIR / "loss_curve.png")
    plt.show()

    return report


#* ------------------------------- Main -------------------------------#
def main():
    torch.manual_seed(TRAIN_HP["seed"])
    np.random.seed(TRAIN_HP["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_telemetry = np.load(DATA_DIR / "train_telemetry.npy")
    print(f"Loaded {len(train_telemetry)} laps | Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # --- Cross-validation ---
    run_cross_validation(train_telemetry, device, k=5)

    # --- Train final model ---
    #train_final_model(train_telemetry, device)


if __name__ == "__main__":
    main()