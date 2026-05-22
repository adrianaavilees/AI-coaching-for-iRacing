"""
Progressive analysis pipeline.

Breaks the monolithic process_uploaded_lap() into 3 phases that can be
executed incrementally, allowing the UI to render results as each phase
completes rather than blocking until everything is done.

Phase 1 — INSTANT (<1s): Parse CSV, interpolate, extract metadata
    → The user immediately sees driver info, lap time, raw telemetry charts
Phase 2 — FAST (~2-5s):  Normalize, run autoencoder, compute errors + zones
    → Unlocks anomaly heatmaps, telemetry overlays, track map zones
Phase 3 — BACKGROUND (~5-15s): Generate LLM coaching feedback
    → Enriches selected-zone coaching with natural-language insights
"""

import hashlib
import json
import threading
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_DIR, MODELS_DIR, FEATURE_COLS, LATLON_COLS, N_POINTS,
    HIDDEN_SIZE, LATENT_DIM, N_LAYERS,
    CHANNEL_DISPLAY_SCALE, CHANNEL_UNITS,
)
from train_autoencoder import LSTMAutoencoder, apply_normalization
from evaluate_autoencoder import load_model, reconstruct_errors
from feedback_engine import generate_feedback
from create_final_dataset import interpolate_lap
from statistical_analysis import (compute_signed_error, detect_zones, analyse_zone_channels,
                                   detect_causal_chains, compute_zone_severity, estimate_time_loss)
from template_feedback_fallback import render_zone_feedback, render_summary

import re

N_FEATURES = len(FEATURE_COLS)
LAP_DIST = np.linspace(0, 100, N_POINTS)

FILENAME_RE = re.compile(
    r"Garage 61 - (.+?) - Ferrari 296 GT3 - .+? - (\d{2})\.(\d{2})\.(\d{3}) - (.+?)\.csv"
)

FEEDBACK_CACHE_DIR = DATA_DIR / "feedback_cache"
FEEDBACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Singleton cached resources (loaded once, shared across all calls) ──────

_model_cache = {}


def _get_device():
    if "device" not in _model_cache:
        _model_cache["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _model_cache["device"]


def _get_model():
    if "model" not in _model_cache:
        device = _get_device()
        model = LSTMAutoencoder(N_FEATURES, HIDDEN_SIZE, LATENT_DIM, N_POINTS, N_LAYERS).to(device)
        model.load_state_dict(torch.load(MODELS_DIR / "autoencoder_best.pt", map_location=device))
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]


def _get_scaler():
    if "scaler" not in _model_cache:
        s = np.load(MODELS_DIR / "scaler_params.npz")
        _model_cache["scaler"] = (s["mean"], s["std"])
    return _model_cache["scaler"]


def _get_expert_baseline():
    if "expert" not in _model_cache:
        baseline_mean, baseline_std, n_laps = _get_expert_baseline_stats()
        mse = np.full(n_laps, baseline_mean, dtype=np.float32)
        expert_baseline_sq = np.full((N_POINTS, N_FEATURES), baseline_mean, dtype=np.float32)
        _model_cache["expert"] = (mse, expert_baseline_sq)
    return _model_cache["expert"]


def _get_expert_baseline_stats():
    """Fast scalar baseline used for UI inference without reconstructing all expert laps."""
    if "expert_stats" not in _model_cache:
        summary_path = MODELS_DIR / "eval_test_dani" / "pilot_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            baseline = summary.get("expert_baseline", {})
            stats = (
                float(baseline.get("mse_mean", 0.104771)),
                float(baseline.get("mse_std", 0.020664)),
                int(baseline.get("n_laps", 149)),
            )
        else:
            stats = (0.104771, 0.020664, 149)
        _model_cache["expert_stats"] = stats
    return _model_cache["expert_stats"]


def _get_train_latlon():
    if "train_latlon" not in _model_cache:
        _model_cache["train_latlon"] = np.load(DATA_DIR / "train_latlon.npy")
    return _model_cache["train_latlon"]


def _get_train_meta():
    if "train_meta" not in _model_cache:
        _model_cache["train_meta"] = pd.read_csv(DATA_DIR / "train_metadata.csv")
    return _model_cache["train_meta"]


def _telemetry_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _get_cached_feedback(cache_key: str):
    path = FEEDBACK_CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_feedback_cache(cache_key: str, report_dict: dict):
    path = FEEDBACK_CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False)


def _zone_time_loss_from_template(zone: dict) -> float:
    if zone.get("estimated_time_loss_s") is not None:
        try:
            return float(zone["estimated_time_loss_s"])
        except (TypeError, ValueError):
            pass
    text = zone.get("template_feedback", "") or ""
    marker = "Estimated time impact: ~"
    if marker in text:
        try:
            return float(text.split(marker, 1)[1].split("s", 1)[0])
        except (IndexError, ValueError):
            return 0.0
    return 0.0


def _normalise_report_for_ui(report_dict: dict) -> dict:
    """Make cached/LLM reports usable for driver-facing UI priority display."""
    zones = report_dict.get("zones", []) or []
    if not zones:
        return report_dict

    strengths = []
    for zone in zones:
        time_loss = _zone_time_loss_from_template(zone)
        zone["estimated_time_loss_s"] = round(time_loss, 4)
        strengths.append(time_loss if time_loss > 0 else float(zone.get("severity_score", 0)))

    min_strength = min(strengths)
    max_strength = max(strengths)
    for zone, strength in zip(zones, strengths):
        if max_strength > min_strength:
            zone["severity_score"] = round(0.35 + 0.65 * ((strength - min_strength) / (max_strength - min_strength)), 3)
        else:
            zone["severity_score"] = round(min(float(zone.get("severity_score", 0.65)), 0.75), 3)

    report_dict["overall_severity"] = round(float(np.mean([z.get("severity_score", 0) for z in zones])), 3)
    report_dict["estimated_time_loss_s"] = round(float(sum(z.get("estimated_time_loss_s", 0) for z in zones)), 3)
    return report_dict


# ─── Filename parsing ────────────────────────────────────────────────────────

def parse_filename(filename: str):
    match = FILENAME_RE.match(filename)
    if match:
        driver = match.group(1)
        minutes, seconds, millis = int(match.group(2)), int(match.group(3)), int(match.group(4))
        lap_time_s = minutes * 60 + seconds + millis / 1000
        lap_time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
        return driver, lap_time_s, lap_time_str
    return None, None, None


# ─── Phase 1: Instant — parse + interpolate ─────────────────────────────────

def run_phase1(uploaded_file) -> dict:
    """
    Parse the CSV, interpolate to 1000-point grid, extract metadata.
    This is CPU-only numpy work and completes in <1 second.
    """
    filename = uploaded_file.name
    driver, lap_time_s, lap_time_str = parse_filename(filename)
    if driver is None:
        driver, lap_time_s, lap_time_str = "Unknown Driver", 0.0, "N/A"

    df = pd.read_csv(uploaded_file)
    required_cols = set(FEATURE_COLS + LATLON_COLS + ["LapDistPct"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    uniform_dist = np.linspace(0, 1, N_POINTS)
    telemetry_interp = np.nan_to_num(interpolate_lap(df, uniform_dist, FEATURE_COLS), nan=0.0)
    latlon_interp = np.nan_to_num(interpolate_lap(df, uniform_dist, LATLON_COLS), nan=0.0)

    telemetry_batch = telemetry_interp[np.newaxis, ...]
    latlon_batch = latlon_interp[np.newaxis, ...]

    return {
        "driver": driver,
        "lap_time_s": lap_time_s,
        "lap_time_str": lap_time_str,
        "telemetry_raw": telemetry_batch,       # (1, 1000, 9)
        "latlon": latlon_batch,                 # (1, 1000, 2)
        "train_latlon": _get_train_latlon(),
        "train_meta": _get_train_meta(),
    }


# ─── Phase 2: Fast — normalize + autoencoder + anomaly detection ─────────────

def run_phase2(phase1: dict) -> dict:
    """
    Normalize, run through autoencoder, compute errors and zones.
    Typically completes in 2-5 seconds.
    """
    telemetry_batch = phase1["telemetry_raw"]
    mean, std = _get_scaler()
    device = _get_device()
    model = _get_model()

    norm = apply_normalization(telemetry_batch, mean, std)

    with torch.no_grad():
        t = torch.tensor(norm, dtype=torch.float32).to(device)
        recon_norm = model(t).cpu().numpy()

    error_pw = (recon_norm - norm) ** 2
    mse_per_lap = error_pw.mean(axis=(1, 2))

    denorm = norm * std + mean
    recon_denorm = recon_norm * std + mean

    expert_mse_mean, expert_mse_std, expert_n_laps = _get_expert_baseline_stats()
    expert_baseline_sq = np.full((N_POINTS, N_FEATURES), expert_mse_mean, dtype=np.float32)

    # Compute zones (statistical analysis — no LLM)
    signed_error = compute_signed_error(denorm[0], recon_denorm[0])
    sq_error = signed_error ** 2
    raw_zones = detect_zones(sq_error, window=15, top_k=5)

    zones_data = []
    zone_deviations = []  # keep per-zone deviations for template summary
    zone_chains = []
    total_time_loss = 0.0
    zone_strengths = [float(v) for _, _, v in raw_zones]
    min_strength = min(zone_strengths) if zone_strengths else 0.0
    max_strength = max(zone_strengths) if zone_strengths else 0.0

    for rank, (idx_s, idx_e, zone_strength) in enumerate(raw_zones, start=1):
        deviations = analyse_zone_channels(signed_error, sq_error, idx_s, idx_e)
        causal_chains = detect_causal_chains(signed_error, idx_s, idx_e)
        if max_strength > min_strength:
            severity = 0.35 + 0.65 * ((float(zone_strength) - min_strength) / (max_strength - min_strength))
        else:
            severity = 0.65
        time_loss = estimate_time_loss(signed_error, phase1["lap_time_s"], idx_s, idx_e)
        total_time_loss += time_loss

        zone_deviations.append(deviations)
        zone_chains.append(causal_chains)

        # Template feedback (instant, no LLM)
        template_text = render_zone_feedback(
            zone_id=rank,
            lap_pct_start=LAP_DIST[idx_s],
            lap_pct_end=LAP_DIST[idx_e],
            deviations=deviations,
            causal_chains=causal_chains,
            severity_score=severity,
            time_loss_s=time_loss,
        )

        zones_data.append({
            "zone_id": rank,
            "lap_pct_start": round(float(LAP_DIST[idx_s]), 2),
            "lap_pct_end": round(float(LAP_DIST[idx_e]), 2),
            "idx_start": int(idx_s),
            "idx_end": int(idx_e),
            "severity_score": round(severity, 3),
            "dominant_channels": [
                {"channel": d.channel, "signed_mean": d.signed_mean,
                 "unit": d.unit, "direction": d.direction, "severity": d.severity}
                for d in deviations
            ],
            "causal_chains": [
                {"description": c.description, "confidence": c.confidence}
                for c in causal_chains
            ],
            "template_feedback": template_text,
            "llm_feedback": None,
            "estimated_time_loss_s": round(time_loss, 4),
        })

    overall_severity = np.mean([z["severity_score"] for z in zones_data]) if zones_data else 0.0

    # Template summary (instant) — use per-zone deviations
    class _Z:
        pass
    _zones_for_template = []
    for i, z in enumerate(zones_data):
        obj = _Z()
        obj.zone_id = z["zone_id"]
        obj.lap_pct_start = z["lap_pct_start"]
        obj.lap_pct_end = z["lap_pct_end"]
        obj.severity_score = z["severity_score"]
        obj.dominant_channels = zone_deviations[i]
        obj.causal_chains = zone_chains[i]
        _zones_for_template.append(obj)

    summary_template = render_summary(_zones_for_template, overall_severity, total_time_loss)

    # Build the partial report (with template text, without LLM)
    partial_report = {
        "n_zones": len(zones_data),
        "overall_severity": round(float(overall_severity), 3),
        "estimated_time_loss_s": round(total_time_loss, 3),
        "zones": zones_data,
        "summary_template": summary_template,
        "summary_llm": None,
    }

    summary = {
        "driver": phase1["driver"],
        "n_laps": 1,
        "laps": [{
            "lap_number": 1,
            "lap_time": phase1["lap_time_str"],
            "lap_time_s": phase1["lap_time_s"],
            "mse_normalised": round(float(mse_per_lap[0]), 6),
            "n_coaching_zones": len(zones_data),
            "overall_severity": round(float(overall_severity), 3),
            "estimated_time_loss_s": round(total_time_loss, 3),
        }],
        "expert_baseline": {
            "mse_mean": round(float(expert_mse_mean), 6),
            "mse_std": round(float(expert_mse_std), 6),
            "n_laps": int(expert_n_laps),
        },
    }

    return {
        "summary": summary,
        "reports": {1: partial_report},
        "amateur_result": {
            "norm": norm,
            "recon_norm": recon_norm,
            "denorm": denorm,
            "recon_denorm": recon_denorm,
            "error": error_pw,
            "mse_per_lap": mse_per_lap,
        },
        "expert_mse": float(expert_mse_mean),
        "expert_mse_std": float(expert_mse_std),
        "latlon": phase1["latlon"],
        "train_latlon": phase1["train_latlon"],
        "train_meta": phase1["train_meta"],
        "driver": phase1["driver"],
        "lap_time_s": phase1["lap_time_s"],
        "lap_time_str": phase1["lap_time_str"],
    }


# ─── Phase 3: Background — LLM feedback generation ──────────────────────────

def run_phase3(session_data: dict) -> dict:
    """
    Generate LLM coaching feedback. This is the slowest phase and runs
    in a background thread. Returns the updated reports dict.
    """
    denorm = session_data["amateur_result"]["denorm"]
    recon_denorm = session_data["amateur_result"]["recon_denorm"]
    expert_mse_mean, _, _ = _get_expert_baseline_stats()
    expert_baseline_sq = np.full((N_POINTS, N_FEATURES), expert_mse_mean, dtype=np.float32)
    lap_time_s = session_data["lap_time_s"]

    cache_key = _telemetry_hash(denorm[0])
    cached = _get_cached_feedback(cache_key)

    if cached is not None:
        return _normalise_report_for_ui(cached)

    report = generate_feedback(
        amateur_raw=denorm[0],
        expert_recon_raw=recon_denorm[0],
        expert_baseline_sq_error=expert_baseline_sq,
        lap_time_s=lap_time_s,
        top_k=5,
        llm_provider="groq",
    )
    report_dict = _normalise_report_for_ui(report.to_dict())
    _save_feedback_cache(cache_key, report_dict)
    return report_dict


# ─── Background thread runner ───────────────────────────────────────────────

def start_phase3_background(session_data: dict):
    """
    Launch Phase 3 in a background thread. The result is stored in
    st.session_state["_phase3_result"] when complete, and
    st.session_state["_phase3_done"] is set to True.
    """
    def _worker():
        try:
            result = run_phase3(session_data)
            # Store in a thread-safe way via the dict reference
            session_data["_phase3_result"] = result
            session_data["_phase3_done"] = True
        except Exception as e:
            session_data["_phase3_error"] = str(e)
            session_data["_phase3_done"] = True

    session_data["_phase3_done"] = False
    session_data["_phase3_result"] = None
    session_data["_phase3_error"] = None

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ─── Preload heavy resources at import time ─────────────────────────────────

def preload_resources():
    """
    Eagerly load model, scaler, and expert baseline into the module cache
    so they are ready when the user uploads a lap. Called once at app startup.
    """
    _get_scaler()
    _get_device()
    _get_model()
    _get_train_latlon()
    _get_train_meta()
    _get_expert_baseline_stats()
