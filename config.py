"""
config.py - Shared configuration for training and evaluating the autoencoder.

This file contains hyperparameters and settings. By centralizing these configurations, 
we can ensure consistency between training and evaluation, and make it easier to update parameters in one place.

Never hardcode these values in individual scripts.
"""

from pathlib import Path

# Paths
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR   = MODELS_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Telemetry 
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "SteeringWheelAngle","Gear", "LatAccel", "LongAccel", "VertAccel", "YawRate"]
LATLON_COLS  = ["Lat", "Lon"]

# Channel groups
DRIVER_INPUT_COLS    = ["Throttle", "Brake", "SteeringWheelAngle", "Gear"]
VEHICLE_DYNAMIC_COLS = ["Speed", "RPM", "LatAccel", "LongAccel", "VertAccel", "YawRate"]

RAW_POINTS = 1000              # Points per lap after LapDistPct interpolation
#STRIDE     = 5                 # Subsample factor: every Nth point fed to the LSTM
N_POINTS   = 1000 

HIDDEN_SIZE = 256     
LATENT_DIM  = 32        
N_LAYERS    = 2

# Train hyperparameters
TRAIN_HP = {
    "dropout":      0.3314950036607718,    # Light regularisation; heavier dropout collapses training
    "noise_std":    0.03269124292259021,   # Gaussian noise std for data augmentation
    "n_augments":   8,      # Each lap → 8 noisy copies → effective dataset: 18×10=180
    "batch_size":   8,      # Small batch size for better generalisation
    "lr":           0.00045745782054754043,   # Adam default; will be reduced by scheduler
    "weight_decay": 3.488976654890367e-05,   # L2 regularisation on weights — critical with few samples
    "epochs":       300,
    "patience":     50,     # Early stopping patience (epochs without val improvement)
    "val_split":    0.15,   # ~3 laps held out for validation loss monitoring
    "seed":         42,
}
