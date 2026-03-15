"""
Train the LSTM Autoencoder on expert lap telemetry
 
Architecture: sequence-to-sequence LSTM Autoencoder
    Encoder: LSTM(input=9, hidden=64, layers=1) → latent vector (16,)
    Decoder: LSTM(input=16, hidden=64, layers=1) → reconstructed sequence (1000, 9)
 
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

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_META_PATH = DATA_DIR / "train_metadata.csv"
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "SteeringWheelAngle","Gear", "LatAccel", "LongAccel", "YawRate"]


# Hyperparameters
HP = {
    "n_points":      1000,   # Interpolated points per lap (must match matrix)
    "hidden_size":   64,     # LSTM hidden dim — small enough to avoid memorisation
    "latent_dim":    16,     # Bottleneck: forces learning of compact driving patterns
    "n_lstm_layers": 1,      # Single layer — extra layers add capacity we can't afford
    "dropout":       0.2,    # Light regularisation; heavier dropout collapses training
    "noise_std":     0.01,   # Gaussian noise std for data augmentation
    "n_augments":    9,      # Each lap → 9 noisy copies → effective dataset: 18×10=180
    "batch_size":    6,      # Small batches suit small datasets; gradient is noisier but generalises better
    "lr":            1e-3,   # Adam default; will be reduced by scheduler
    "weight_decay":  1e-4,   # L2 regularisation on weights — critical with few samples
    "epochs":        300,    
    "patience":      30,     # Early stopping patience (epochs without val improvement)
    "val_split":     0.15,   # ~3 laps held out for validation loss monitoring
    "seed":          42,
}


