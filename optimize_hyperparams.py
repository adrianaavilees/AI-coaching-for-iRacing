"""
Hyperparameter optimization for the LSTM Autoencoder using Optuna (Bayesian TPE).

Objective: 
    - Minimize 5 k-fold CV validation loss on the training set
    - Maximize robustness 
"""

import json
import numpy as np
from optuna import trial
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import optuna

from config import DATA_DIR, MODELS_DIR, FEATURE_COLS, N_POINTS, TRAIN_HP
from train_autoencoder import (
    LapDataset,
    LSTMAutoencoder,
    compute_scaler_params,
    apply_normalization,
)

# ----------------------------- Settings -------------------------------- #
N_TRIALS       = 80       # Number of Optuna trials
SEARCH_FOLDS   = 3        # Folds used during search (fast)
FINAL_FOLDS    = 5        # Folds used for final evaluation
SEARCH_EPOCHS  = 100      # Fewer epochs during search (enough to rank HPs)
SEARCH_PATIENCE = 20      # Faster early stopping during search
OPTUNA_DIR     = MODELS_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_META_PATH = DATA_DIR / "train_metadata.csv"

# Load data ONCE (avoid reloading every trial)
TRAIN_TELEMETRY = np.load(DATA_DIR / "train_telemetry.npy")


def objective(trial):

        # Hyperparameter search space
    HP_SPACE = {
        "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout":      trial.suggest_float("dropout", 0.0, 0.4),
        "hidden_size":  trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "latent_dim":   trial.suggest_categorical("latent_dim", [16, 32, 64]),
        "n_layers":     trial.suggest_int("n_layers", 1, 2),
        "batch_size":   trial.suggest_categorical("batch_size", [4, 6, 8, 12]),
        "noise_std":    trial.suggest_float("noise_std", 0.005, 0.05, log=True),
        "n_augments":   trial.suggest_int("n_augments", 3, 15),
        "epochs":       SEARCH_EPOCHS,   # Reduced for search speed
        "patience":     SEARCH_PATIENCE, # Reduced for search speed
        "seed":         TRAIN_HP["seed"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_telemetry = TRAIN_TELEMETRY  # Pre-loaded at module level

    kf = KFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=HP_SPACE["seed"])

    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_telemetry)):
        # Split data
        train_data = train_telemetry[train_idx]
        val_data   = train_telemetry[val_idx]

        # Compute scaler params on training fold
        mean, std = compute_scaler_params(train_data)
        # Apply normalization
        train_data = apply_normalization(train_data, mean, std)
        val_data   = apply_normalization(val_data, mean, std)

        # Create datasets and dataloaders
        train_dataset = LapDataset(train_data, noise_std=HP_SPACE["noise_std"], n_augments=HP_SPACE["n_augments"])
        val_dataset   = LapDataset(val_data, noise_std=0.0, n_augments=0)  # No augmentation for validation

        train_loader = DataLoader(train_dataset, batch_size=HP_SPACE["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=HP_SPACE["batch_size"], shuffle=False)


        # Initialize model with trial hyperparameters
        model = LSTMAutoencoder(
            input_size=len(FEATURE_COLS),
            hidden_size=HP_SPACE["hidden_size"],
            latent_dim=HP_SPACE["latent_dim"],
            n_points=N_POINTS,
            n_layers=HP_SPACE["n_layers"],
            dropout=HP_SPACE["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=HP_SPACE["lr"], weight_decay=HP_SPACE["weight_decay"])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        best_val_loss = float("inf")
        epochs_no_improve = 0  

        for epoch in range(HP_SPACE["epochs"]):
            model.train()

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    val_loss += loss.item() * batch.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            # Report to Optuna for pruning
            trial.report(val_loss, epoch + fold * HP_SPACE["epochs"])
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= HP_SPACE["patience"]:
                break

        val_losses.append(best_val_loss)

    avg_val_loss = np.mean(val_losses)
    return avg_val_loss


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=TRAIN_HP["seed"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="lstm_autoencoder_hp_optimization")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value (CV loss): {best_trial.value:.6f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    # Save best hyperparameters to JSON
    with open(MODELS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)

    print(f"Best hyperparameters saved to {MODELS_DIR / 'best_hyperparameters.json'}")





