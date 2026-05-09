"""
Coaching pipeline for a specific pilot.

Loads raw Garage61 CSVs → interpolates → normalises → reconstructs through the autoencoder → generates per-lap coaching feedback via feedback_engine.
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from create_final_dataset import load_raw_laps, build_telemetry_matrix
from train_autoencoder import apply_normalization
from evaluate_autoencoder import load_model, reconstruct_errors
from feedback_engine import generate_feedback
from config import (
    ROOT_DIR, DATA_DIR, MODELS_DIR, FEATURE_COLS, N_POINTS,
)

PILOT_DIR = ROOT_DIR / "data" / "test dani" #! CHANGE THIS TO OTHER PILOT FOLDER IF NEEDED
LLM_PROVIDER = "groq" #! CHANGE THIS TO "gemini" OR "groq" IF YOU WANT TO USE A DIFFERENT LLM (see feedback_engine.py for details)

N_FEATURES = len(FEATURE_COLS)
LAP_DIST = np.linspace(0, 100, N_POINTS)


#* ---------------------------------------- Visualisation ---------------------------------------- #

def plot_telemetry_overlay(pilot_raw, expert_recon_raw, lap_idx, meta_row, eval_dir):
    """Overlay pilot's actual telemetry vs expert reconstruction (denormalised)."""
    n_cols = 2
    n_rows = (N_FEATURES + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes[i]
        ax.plot(LAP_DIST, pilot_raw[:, i], color="#FF9800", lw=1.2, label="Pilot (actual)")
        ax.plot(LAP_DIST, expert_recon_raw[:, i], color="#2196F3", lw=1.2,
                alpha=0.8, label="Expert pattern (recon)")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Lap Distance (%)", fontsize=9)

    fig.suptitle(f"Telemetry Overlay — {meta_row['driver']} lap {lap_idx + 1} "
                 f"({meta_row['lap_time_str']})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(eval_dir / f"telemetry_overlay_lap{lap_idx + 1}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: telemetry_overlay_lap{lap_idx + 1}.png")


#* ---------------------------------------- Main pipeline ---------------------------------------- #

def main():
    pilot_dir = Path(PILOT_DIR)


    eval_dir = MODELS_DIR / f"eval_{pilot_dir.name.replace(' ', '_')}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #* ---------------------------- Load & interpolate ---------------------------- #
    print("\nLoading laps...")
    lap_dfs, meta = load_raw_laps(pilot_dir)
    print(f"   {len(lap_dfs)} laps found")

    print("\nInterpolating...")
    telemetry, _ = build_telemetry_matrix(lap_dfs)
    telemetry = np.nan_to_num(telemetry, nan=0.0)

    #* ---------------------------- Normalise ---------------------------- #
    print("\nNormalising...")
    scaler = np.load(MODELS_DIR / "scaler_params.npz")
    mean, std = scaler["mean"], scaler["std"]
    pilot_norm = apply_normalization(telemetry, mean, std)

    #* ---------------------------- Load expert baseline + model ---------------------------- #
    print("\nLoading model & expert baseline...")
    train_norm = apply_normalization(np.load(DATA_DIR / "train_telemetry.npy"), mean, std)
    model = load_model(device)
    expert_recon, expert_err, expert_mse = reconstruct_errors(model, train_norm, device)
    pilot_recon, pilot_err, pilot_mse = reconstruct_errors(model, pilot_norm, device)

    print(f"\n   Expert MSE: {expert_mse.mean():.6f} ± {expert_mse.std():.6f}")
    print(f"   Pilot  MSE: {pilot_mse.mean():.6f} ± {pilot_mse.std():.6f}")
    print(f"   Anomaly ratio: {pilot_mse.mean() / expert_mse.mean():.2f}×")

    #* ---------------------------- Denormalise for feedback ---------------------------- #
    pilot_denorm = pilot_norm * std + mean
    pilot_recon_denorm = pilot_recon * std + mean
    expert_baseline_sq = expert_err.mean(axis=0)

    #* ---------------------------- Feedback + visualisations per lap ---------------------------- #
    print("\nGenerating coaching feedback...\n")
    all_reports = []

    for i in range(len(pilot_mse)):
        row = meta.iloc[i]
        print(f"{'='*60}")
        print(f"  LAP {i+1}: {row['driver']} — {row['lap_time_str']} (MSE: {pilot_mse[i]:.6f})")
        print(f"{'='*60}")

        report = generate_feedback(
            amateur_raw=pilot_denorm[i],
            expert_recon_raw=pilot_recon_denorm[i],
            expert_baseline_sq_error=expert_baseline_sq,
            lap_time_s=row["lap_time_s"],
            top_k=5,
            llm_provider=LLM_PROVIDER,
        )
        all_reports.append(report)

        print(report.summary_template)
        for zone in report.zones:
            print(zone.llm_feedback or zone.template_feedback)

        report.to_json(eval_dir / f"feedback_lap{i+1}.json")
        plot_telemetry_overlay(pilot_denorm[i], pilot_recon_denorm[i], i, row, eval_dir)

    #* ---------------------------- Summary ---------------------------- #
    summary = {
        "driver": meta.iloc[0]["driver"],
        "n_laps": len(pilot_mse),
        "expert_baseline": {
            "mse_mean": round(float(expert_mse.mean()), 6),
            "mse_std": round(float(expert_mse.std()), 6),
            "n_laps": int(len(expert_mse)),
        },
        "laps": [
            {
                "lap": i + 1,
                "lap_time": meta.iloc[i]["lap_time_str"],
                "mse": round(float(pilot_mse[i]), 6),
                "severity": all_reports[i].overall_severity,
                "time_loss_s": all_reports[i].estimated_time_loss_s,
            }
            for i in range(len(pilot_mse))
        ],
    }

    with open(eval_dir / "pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to: {eval_dir}")


if __name__ == "__main__":
    main()
