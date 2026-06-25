"""
Data loading and caching layer for the Streamlit app.

Supports two flows:
    1. User uploads a raw Garage61 CSV lap → processes through the full pipeline
    2. Loads pre-computed feedback from disk (eval directories)

All heavy I/O is cached with st.cache_data / st.cache_resource.
Feedback results are cached to disk by telemetry hash to avoid redundant LLM calls.
"""

import io
import re
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import asdict

import streamlit as st
import sys

# Add project root to path so we can import existing modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import (
    DATA_DIR, MODELS_DIR, FEATURE_COLS, LATLON_COLS, N_POINTS,
    HIDDEN_SIZE, LATENT_DIM, N_LAYERS,
    CHANNEL_DISPLAY_SCALE, CHANNEL_UNITS,
    DRIVER_INPUT_COLS, VEHICLE_DYNAMIC_COLS,
)
from model.train_autoencoder import LSTMAutoencoder, apply_normalization
from model.evaluate_autoencoder import load_model, reconstruct_errors
from coaching.feedback_engine import generate_feedback
from data.create_final_dataset import interpolate_lap


N_FEATURES = len(FEATURE_COLS)
LAP_DIST = np.linspace(0, 100, N_POINTS)

# Regex for parsing Garage61 filenames
FILENAME_RE = re.compile(
    r"Garage 61 - (.+?) - Ferrari 296 GT3 - .+? - (\d{2})\.(\d{2})\.(\d{3}) - (.+?)\.csv"
)

# ─── Feedback disk cache ────────────────────────────────────────────────────
FEEDBACK_CACHE_DIR = DATA_DIR / "feedback_cache"
FEEDBACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _telemetry_hash(telemetry_array: np.ndarray) -> str:
    """Create a fast hash of a telemetry array for cache lookup."""
    return hashlib.sha256(telemetry_array.tobytes()).hexdigest()[:16]


def _get_cached_feedback(cache_key: str):
    """Load cached feedback report dict from disk, or return None."""
    cache_path = FEEDBACK_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_feedback_cache(cache_key: str, report_dict: dict):
    """Save feedback report dict to disk cache."""
    cache_path = FEEDBACK_CACHE_DIR / f"{cache_key}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False)


# ─── Cached Loaders ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def get_model():
    device = get_device()
    return load_model(device)


@st.cache_data(show_spinner=False)
def load_scaler():
    scaler = np.load(MODELS_DIR / "scaler_params.npz")
    return scaler["mean"], scaler["std"]


@st.cache_data(show_spinner=False)
def load_train_data():
    """Load expert training data + metadata."""
    telemetry = np.load(DATA_DIR / "train_telemetry.npy")
    latlon = np.load(DATA_DIR / "train_latlon.npy")
    meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
    return telemetry, latlon, meta


@st.cache_data(show_spinner=False)
def compute_expert_baseline():
    """Reconstruct expert laps and compute baseline MSE."""
    telemetry, _, _ = load_train_data()
    mean, std = load_scaler()
    device = get_device()
    model = get_model()

    train_norm = apply_normalization(telemetry, mean, std)
    recon, err, mse = reconstruct_errors(model, train_norm, device)

    # Expert baseline squared error (mean across all expert laps)
    expert_baseline_sq = err.mean(axis=0)  # (N_POINTS, N_FEATURES)

    return recon, err, mse, expert_baseline_sq


# ─── CSV Upload Processing ──────────────────────────────────────────────────

def parse_filename(filename: str):
    """
    Extract driver name and lap time from a Garage61 CSV filename.
    Returns (driver, lap_time_s, lap_time_str) or (None, None, None).
    """
    match = FILENAME_RE.match(filename)
    if match:
        driver = match.group(1)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4))
        lap_time_s = minutes * 60 + seconds + millis / 1000
        lap_time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
        return driver, lap_time_s, lap_time_str
    return None, None, None


def process_uploaded_lap(uploaded_file) -> dict:
    """
    Full coaching pipeline for a single uploaded CSV lap.

    Steps:
        1. Parse filename → extract driver / lap time
        2. Read CSV → validate required columns
        3. Interpolate to 1000-point grid
        4. Normalize with expert scaler
        5. Reconstruct through autoencoder
        6. Denormalize for physical-space analysis
        7. Generate coaching feedback (statistical + LLM)

    Returns a dict with all data needed by the UI pages.
    """
    filename = uploaded_file.name

    # 1. Parse metadata from filename
    driver, lap_time_s, lap_time_str = parse_filename(filename)
    if driver is None:
        # Fallback: try to extract from file content or use generic
        driver = "Unknown Driver"
        lap_time_s = 0.0
        lap_time_str = "N/A"

    # 2. Read CSV
    df = pd.read_csv(uploaded_file)

    required_cols = set(FEATURE_COLS + LATLON_COLS + ["LapDistPct"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # 3. Interpolate
    uniform_dist = np.linspace(0, 1, N_POINTS)
    telemetry_interp = interpolate_lap(df, uniform_dist, FEATURE_COLS)  # (1000, 9)
    latlon_interp = interpolate_lap(df, uniform_dist, LATLON_COLS)      # (1000, 2)

    telemetry_interp = np.nan_to_num(telemetry_interp, nan=0.0)
    latlon_interp = np.nan_to_num(latlon_interp, nan=0.0)

    # Reshape to (1, 1000, N) for batch processing
    telemetry_batch = telemetry_interp[np.newaxis, ...]  # (1, 1000, 9)
    latlon_batch = latlon_interp[np.newaxis, ...]        # (1, 1000, 2)

    # 4. Normalize
    mean, std = load_scaler()
    norm = apply_normalization(telemetry_batch, mean, std)

    # 5. Reconstruct
    device = get_device()
    model = get_model()
    recon_norm, err, mse = reconstruct_errors(model, norm, device)

    # 6. Denormalize
    denorm = norm * std + mean
    recon_denorm = recon_norm * std + mean

    # 7. Expert baseline for feedback
    _, _, expert_mse, expert_baseline_sq = compute_expert_baseline()

    # 8. Generate feedback (with disk cache)
    cache_key = _telemetry_hash(denorm[0])
    report_dict = _get_cached_feedback(cache_key)

    if report_dict is None:
        report = generate_feedback(
            amateur_raw=denorm[0],
            expert_recon_raw=recon_denorm[0],
            expert_baseline_sq_error=expert_baseline_sq,
            lap_time_s=lap_time_s,
            top_k=5,
            llm_provider="groq",
        )
        report_dict = report.to_dict()
        _save_feedback_cache(cache_key, report_dict)

    # Build summary
    summary = {
        "driver": driver,
        "n_laps": 1,
        "laps": [
            {
                "lap_number": 1,
                "lap_time": lap_time_str,
                "lap_time_s": lap_time_s,
                "mse_normalised": round(float(mse[0]), 6),
                "n_coaching_zones": report_dict.get("n_zones", 0),
                "overall_severity": report_dict.get("overall_severity", 0),
                "estimated_time_loss_s": report_dict.get("estimated_time_loss_s", 0),
            }
        ],
        "expert_baseline": {
            "mse_mean": round(float(expert_mse.mean()), 6),
            "mse_std": round(float(expert_mse.std()), 6),
            "n_laps": int(len(expert_mse)),
        },
    }

    _, train_latlon, train_meta = load_train_data()

    return {
        "summary": summary,
        "reports": {1: report_dict},
        "amateur_result": {
            "norm": norm,
            "recon_norm": recon_norm,
            "denorm": denorm,
            "recon_denorm": recon_denorm,
            "error": err,
            "mse_per_lap": mse,
        },
        "expert_mse": float(expert_mse.mean()),
        "expert_mse_std": float(expert_mse.std()),
        "train_latlon": train_latlon,
        "train_meta": train_meta,
        "latlon": latlon_batch,
        "driver": driver,
        "lap_time_s": lap_time_s,
        "lap_time_str": lap_time_str,
    }


def process_multiple_laps(uploaded_files: list) -> dict:
    """
    Process multiple uploaded CSV laps through the full pipeline.
    Returns a unified session dict for the UI.
    """
    mean, std = load_scaler()
    device = get_device()
    model = get_model()
    _, _, expert_mse_arr, expert_baseline_sq = compute_expert_baseline()
    _, train_latlon, train_meta = load_train_data()

    uniform_dist = np.linspace(0, 1, N_POINTS)

    all_telemetry = []
    all_latlon = []
    meta_rows = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        driver, lap_time_s, lap_time_str = parse_filename(filename)
        if driver is None:
            driver, lap_time_s, lap_time_str = "Unknown", 0.0, "N/A"

        df = pd.read_csv(uploaded_file)
        required_cols = set(FEATURE_COLS + LATLON_COLS + ["LapDistPct"])
        missing = required_cols - set(df.columns)
        if missing:
            st.warning(f"Skipping {filename}: missing columns {missing}")
            continue

        telem = interpolate_lap(df, uniform_dist, FEATURE_COLS)
        ll = interpolate_lap(df, uniform_dist, LATLON_COLS)
        telem = np.nan_to_num(telem, nan=0.0)
        ll = np.nan_to_num(ll, nan=0.0)

        all_telemetry.append(telem)
        all_latlon.append(ll)
        meta_rows.append({"driver": driver, "lap_time_s": lap_time_s, "lap_time_str": lap_time_str})

    if not all_telemetry:
        raise ValueError("No valid laps could be processed from uploaded files.")

    telemetry_batch = np.stack(all_telemetry)  # (n_laps, 1000, 9)
    latlon_batch = np.stack(all_latlon)        # (n_laps, 1000, 2)

    # Normalize + reconstruct
    norm = apply_normalization(telemetry_batch, mean, std)
    recon_norm, err, mse = reconstruct_errors(model, norm, device)
    denorm = norm * std + mean
    recon_denorm = recon_norm * std + mean

    # Generate feedback per lap (with disk cache)
    reports = {}
    laps_summary = []

    for i in range(len(meta_rows)):
        cache_key = _telemetry_hash(denorm[i])
        cached = _get_cached_feedback(cache_key)

        if cached is not None:
            report_dict = cached
        else:
            report = generate_feedback(
                amateur_raw=denorm[i],
                expert_recon_raw=recon_denorm[i],
                expert_baseline_sq_error=expert_baseline_sq,
                lap_time_s=meta_rows[i]["lap_time_s"],
                top_k=5,
                llm_provider="groq",
            )
            report_dict = report.to_dict()
            _save_feedback_cache(cache_key, report_dict)

        reports[i + 1] = report_dict
        laps_summary.append({
            "lap_number": i + 1,
            "lap_time": meta_rows[i]["lap_time_str"],
            "lap_time_s": meta_rows[i]["lap_time_s"],
            "mse_normalised": round(float(mse[i]), 6),
            "n_coaching_zones": report_dict.get("n_zones", 0),
            "overall_severity": report_dict.get("overall_severity", 0),
            "estimated_time_loss_s": report_dict.get("estimated_time_loss_s", 0),
        })

    driver_name = meta_rows[0]["driver"] if meta_rows else "Unknown"
    summary = {
        "driver": driver_name,
        "n_laps": len(meta_rows),
        "laps": laps_summary,
        "expert_baseline": {
            "mse_mean": round(float(expert_mse_arr.mean()), 6),
            "mse_std": round(float(expert_mse_arr.std()), 6),
            "n_laps": int(len(expert_mse_arr)),
        },
    }

    return {
        "summary": summary,
        "reports": reports,
        "amateur_result": {
            "norm": norm,
            "recon_norm": recon_norm,
            "denorm": denorm,
            "recon_denorm": recon_denorm,
            "error": err,
            "mse_per_lap": mse,
        },
        "expert_mse": float(expert_mse_arr.mean()),
        "expert_mse_std": float(expert_mse_arr.std()),
        "train_latlon": train_latlon,
        "train_meta": train_meta,
        "latlon": latlon_batch,
        "driver": driver_name,
    }


# ─── Utilities ───────────────────────────────────────────────────────────────

def denormalize_telemetry_for_display(values: np.ndarray, channel_idx: int):
    """Convert raw telemetry values to display units for a channel."""
    ch_name = FEATURE_COLS[channel_idx]
    scale = CHANNEL_DISPLAY_SCALE.get(ch_name, 1.0)
    return values * scale


def get_severity_class(severity: float) -> str:
    if severity < 0.33:
        return "low"
    elif severity < 0.66:
        return "medium"
    return "high"


def get_severity_color(severity: float) -> str:
    if severity < 0.33:
        return "#00E676"
    elif severity < 0.66:
        return "#FFD600"
    return "#FF1744"


# ─── Track info ──────────────────────────────────────────────────────────────

TRACK_INFO = {
    "name": "Autodromo Internazionale Enzo e Dino Ferrari",
    "short_name": "Imola",
    "country": "Italy",
    "length_km": 4.909,
    "corners": 19,
    "config": "Grand Prix",
}

CAR_INFO = {
    "name": "Ferrari 296 GT3",
    "class": "GT3",
    "manufacturer": "Ferrari",
    "power": "600 HP",
    "weight": "1,250 kg",
}

