"""
Evaluate the trained LSTM Autoencoder on expert (train) and amateur (test) laps.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from train_autoencoder import LSTMAutoencoder, apply_normalization
from config import (
    DATA_DIR, MODELS_DIR, EVAL_DIR, FEATURE_COLS, N_POINTS,
    HIDDEN_SIZE, LATENT_DIM, N_LAYERS,
    DRIVER_INPUT_COLS, VEHICLE_DYNAMIC_COLS,
)


N_FEATURES = len(FEATURE_COLS)
LAP_DIST = np.linspace(0, 100, N_POINTS)

DRIVER_INPUT_IDX     = [FEATURE_COLS.index(c) for c in DRIVER_INPUT_COLS]
VEHICLE_DYNAMIC_IDX  = [FEATURE_COLS.index(c) for c in VEHICLE_DYNAMIC_COLS]

def load_model(device):
    model = LSTMAutoencoder(N_FEATURES, HIDDEN_SIZE, LATENT_DIM, N_POINTS, N_LAYERS).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "autoencoder_best.pt", map_location=device))
    model.eval()
    return model

def reconstruct_errors(model, telemetry_norm, device, batch_size=8):
    """
    Returns:
        recon_norm  : (n_laps, N_POINTS, N_FEATURES) — reconstructed (normalised)
        error_pw    : (n_laps, N_POINTS, N_FEATURES) — per-point per-channel squared error
        mse_per_lap : (n_laps,)                       — mean MSE per lap
    """
    t = torch.tensor(telemetry_norm, dtype=torch.float32)
    recon_list = []
    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            batch = t[i:i+batch_size].to(device)
            recon_list.append(model(batch).cpu().numpy())
    recon_norm  = np.concatenate(recon_list, axis=0)      # (n, N_POINTS, N_FEATURES)
    error_pw    = (recon_norm - telemetry_norm) ** 2      # (n, N_POINTS, N_FEATURES)
    mse_per_lap = error_pw.mean(axis=(1, 2))               # (n,)
    return recon_norm, error_pw, mse_per_lap

def compute_metrics(expert_norm, expert_recon, expert_err, amateur_norm, amateur_recon, amateur_err, mean, std, expert_meta, amateur_meta):
    """
    Compute comprehensive metrics for comparison    
    """
    # Denormalize for physical space metrics
    expert_denorm = expert_norm * std + mean
    expert_recon_denorm = expert_recon * std + mean
    
    amateur_denorm = amateur_norm * std + mean
    amateur_recon_denorm = amateur_recon * std + mean
    
    expert_err_denorm = (expert_recon_denorm - expert_denorm) ** 2
    amateur_err_denorm = (amateur_recon_denorm - amateur_denorm) ** 2
    
    expert_mse_per_lap = expert_err.mean(axis=(1, 2))
    amateur_mse_per_lap = amateur_err.mean(axis=(1, 2))
    
    expert_mse_per_lap_denorm = expert_err_denorm.mean(axis=(1, 2))
    amateur_mse_per_lap_denorm = amateur_err_denorm.mean(axis=(1, 2))
    
    # Global metrics in normalized space
    global_metrics_norm = {
        "expert": {
            "mse": float(expert_mse_per_lap.mean()),
            "mse_std": float(expert_mse_per_lap.std()),
            "rmse": float(np.sqrt(expert_mse_per_lap.mean())),
            "mae": float(np.abs(expert_norm - expert_recon).mean()),
            "n_laps": int(len(expert_mse_per_lap)),
        },
        "amateur": {
            "mse": float(amateur_mse_per_lap.mean()),
            "mse_std": float(amateur_mse_per_lap.std()),
            "rmse": float(np.sqrt(amateur_mse_per_lap.mean())),
            "mae": float(np.abs(amateur_norm - amateur_recon).mean()),
            "n_laps": int(len(amateur_mse_per_lap)),
        },
    }
    
    # Global metrics in denormalized (physical) space
    global_metrics_denorm = {
        "expert": {
            "mse": float(expert_mse_per_lap_denorm.mean()),
            "mse_std": float(expert_mse_per_lap_denorm.std()),
            "rmse": float(np.sqrt(expert_mse_per_lap_denorm.mean())),
            "mae": float(np.abs(expert_denorm - expert_recon_denorm).mean()),
        },
        "amateur": {
            "mse": float(amateur_mse_per_lap_denorm.mean()),
            "mse_std": float(amateur_mse_per_lap_denorm.std()),
            "rmse": float(np.sqrt(amateur_mse_per_lap_denorm.mean())),
            "mae": float(np.abs(amateur_denorm - amateur_recon_denorm).mean()),
        },
    }
    
    # Per-channel metrics
    per_channel = {}
    for ch_idx, ch_name in enumerate(FEATURE_COLS):
        expert_ch_err = expert_err[:, :, ch_idx].mean(axis=1)
        amateur_ch_err = amateur_err[:, :, ch_idx].mean(axis=1)
        
        per_channel[ch_name] = {
            "expert_mse": float(expert_ch_err.mean()),
            "expert_std": float(expert_ch_err.std()),
            "amateur_mse": float(amateur_ch_err.mean()),
            "amateur_std": float(amateur_ch_err.std()),
            "ratio": float(amateur_ch_err.mean() / expert_ch_err.mean()) if expert_ch_err.mean() > 0 else 0,
        }
    
    # Per-group metrics
    per_group = {}
    
    for group_name, idxs in [("Driver Inputs", DRIVER_INPUT_IDX), 
                              ("Vehicle Dynamics", VEHICLE_DYNAMIC_IDX)]:
        expert_group_err = expert_err[:, :, idxs].mean(axis=(1, 2))
        amateur_group_err = amateur_err[:, :, idxs].mean(axis=(1, 2))
        
        per_group[group_name] = {
            "expert_mse": float(expert_group_err.mean()),
            "expert_std": float(expert_group_err.std()),
            "amateur_mse": float(amateur_group_err.mean()),
            "amateur_std": float(amateur_group_err.std()),
            "ratio": float(amateur_group_err.mean() / expert_group_err.mean()),
            "channels": [FEATURE_COLS[i] for i in idxs],
        }
    
    # Per-lap statistics
    per_lap_expert = [
        {
            "driver": row.driver,
            "lap_time_str": row.lap_time_str,
            "mse_norm": float(expert_mse_per_lap[i]),
            "mse_denorm": float(expert_mse_per_lap_denorm[i]),
        }
        for i, row in enumerate(expert_meta.itertuples())
    ]
    
    per_lap_amateur = [
        {
            "driver": row.driver,
            "lap_time_str": row.lap_time_str,
            "mse_norm": float(amateur_mse_per_lap[i]),
            "mse_denorm": float(amateur_mse_per_lap_denorm[i]),
        }
        for i, row in enumerate(amateur_meta.itertuples())
    ]

    return {
        "global_metrics_normalized": global_metrics_norm,
        "global_metrics_denormalized": global_metrics_denorm,
        "per_channel_metrics": per_channel,
        "per_group_metrics": per_group,
        "per_lap_expert": per_lap_expert,
        "per_lap_amateur": per_lap_amateur,
    }

def classification_metrics(expert_mse, amateur_mse):
    """
    Binary classification evaluation:
        expert  = 0 (normal)
        amateur = 1 (anomaly)
    Uses lap reconstruction MSE as anomaly score.
    """

    # Labels
    y_true = np.concatenate([
        np.zeros(len(expert_mse)),
        np.ones(len(amateur_mse))
    ])

    # Scores (higher = more anomalous)
    scores = np.concatenate([expert_mse, amateur_mse])

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)

    # Best threshold = Youden J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Predictions
    y_pred = (scores >= best_threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    cm        = confusion_matrix(y_true, y_pred)

    metrics = {
        "auc": float(auc_score),
        "best_threshold": float(best_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

    return metrics, fpr, tpr

#* --------------------------------- Plot functions --------------------------------- #
# Plot ROC curve
def plot_roc_curve(fpr, tpr, clf_metrics):
    "ROC curve + metrics table"
    auc_score = clf_metrics["auc"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # ROC curve
    ax1.plot(fpr, tpr, lw=2.5, label=f"AUC = {auc_score:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.8)

    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve - Expert vs Amateur", fontsize=13)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.axis("off")

    table_data = [
        ["AUC", f"{clf_metrics['auc']:.4f}"],
        ["Best threshold (MSE)", f"{clf_metrics['best_threshold']:.6f}"],
        ["Precision", f"{clf_metrics['precision']:.4f}"],
        ["Recall", f"{clf_metrics['recall']:.4f}"],
        ["F1 Score", f"{clf_metrics['f1_score']:.4f}"],
    ]

    table = ax2.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.8)

    # Header style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DCE6F1")
        else:
            cell.set_facecolor("#F9F9F9")

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("  Saved: roc_curve.png")

def align_latlon_to_seq_len(latlon_raw, target_len):
    """Align raw lat/lon arrays to the model/evaluation sequence length."""
    raw_len = latlon_raw.shape[1]
    if raw_len == target_len:
        return latlon_raw
    idx = np.round(np.linspace(0, raw_len - 1, target_len)).astype(int)
    return latlon_raw[:, idx, :]


#* --------------------------------- Plot 1: error distribution (expert vs amateur) --------------------------------- #
def plot_error_distribution(expert_mse, amateur_mse, expert_meta, amateur_meta):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter points
    jitter_e = np.random.uniform(-0.08, 0.08, len(expert_mse))
    jitter_a = np.random.uniform(-0.08, 0.08, len(amateur_mse))
    ax.scatter(np.ones(len(expert_mse))  + jitter_e, expert_mse,
               color="#2196F3", alpha=0.7, zorder=3, label="Expert laps (train)")
    ax.scatter(np.ones(len(amateur_mse)) * 2 + jitter_a, amateur_mse,
               color="#FF5722", alpha=0.7, zorder=3, label="Amateur laps (test)")

    # Boxplots
    bp = ax.boxplot([expert_mse, amateur_mse], positions=[1, 2], widths=0.3,
                    patch_artist=True, zorder=2,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#BBDEFB")
    bp["boxes"][1].set_facecolor("#FFCCBC")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Expert\n(train set)", "Amateur\n(test set)"], fontsize=12)
    ax.set_ylabel("Mean Reconstruction Error (MSE, normalised space)", fontsize=11)
    ax.set_title("Reconstruction Error Distribution — Expert vs Amateur", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    # Annotate a few amateur points with driver name + lap time
    threshold = np.percentile(amateur_mse, 70)
    for i, (mse, row) in enumerate(zip(amateur_mse, amateur_meta.itertuples())):
        if mse >= threshold:
            ax.annotate(f"{row.driver.split()[-1]}\n{row.lap_time_str}",
                        xy=(2 + jitter_a[i], mse),
                        xytext=(2.15, mse),
                        fontsize=7, color="#BF360C",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_distribution.png", dpi=150)
    plt.close()
    print("  Saved: error_distribution.png")


#* --------------------------------- Plot 2: per-channel error profile --------------------------------- #
def plot_channel_profiles(expert_err, amateur_err):
    """
    expert_err  : (n_expert, N_POINTS, N_FEATURES)
    amateur_err : (n_amateur, N_POINTS, N_FEATURES)
    """
    n_cols = 2
    n_rows = (N_FEATURES + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, feat in enumerate(FEATURE_COLS):
        ax = axes[i]
        e_mean = expert_err[:, :, i].mean(axis=0)
        a_mean = amateur_err[:, :, i].mean(axis=0)
        a_std  = amateur_err[:, :, i].std(axis=0)

        # Smooth for readability
        e_sm = uniform_filter1d(e_mean, size=7)
        a_sm = uniform_filter1d(a_mean, size=7)

        ax.fill_between(LAP_DIST, a_sm - a_std, a_sm + a_std, alpha=0.15, color="#FF5722")
        ax.plot(LAP_DIST, e_sm, color="#2196F3", lw=1.5, label="Expert")
        ax.plot(LAP_DIST, a_sm, color="#FF5722", lw=1.5, label="Amateur")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Squared error", fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("Lap Distance (%)", fontsize=9)

    fig.suptitle("Reconstruction Error Profile per Channel — Expert vs Amateur",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_profile_channels.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: error_profile_channels.png")


#* --------------------------------- Plot 3: heatmap of test-lap errors --------------------------------- #
def plot_heatmap(amateur_err, amateur_meta):
    """amateur_err: (n_amateur, N_POINTS, N_FEATURES) → averaged over features."""
    err_2d = amateur_err.mean(axis=2)   # (n_amateur, N_POINTS)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(err_2d, aspect="auto", cmap="hot", origin="upper",
                   extent=[0, 100, len(amateur_meta) - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label="Mean Squared Error (normalised)")

    driver_labels = [f"{r.driver.split()[-1]} ({r.lap_time_str})"
                     for r in amateur_meta.itertuples()]
    ax.set_yticks(range(len(driver_labels)))
    ax.set_yticklabels(driver_labels, fontsize=8)
    ax.set_xlabel("Lap Distance (%)", fontsize=11)
    ax.set_title("Reconstruction Error Heatmap — Amateur Test Laps", fontsize=13)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "heatmap_test.png", dpi=150)
    plt.close()
    print("  Saved: heatmap_test.png")


#* --------------------------------- Plot 4: coaching zones --------------------------------- #
def find_coaching_zones(error_1d, window=10, top_k=5):
    """
    error_1d : (N_POINTS,) mean error across all amateur laps and features
    Returns list of (dist_start%, dist_end%, mean_error) for top-k zones.
    """
    smoothed = uniform_filter1d(error_1d, size=window)
    threshold = np.percentile(smoothed, 75)
    in_zone = smoothed >= threshold

    zones = []
    start = None
    for i, active in enumerate(in_zone):
        if active and start is None:
            start = i
        elif not active and start is not None:
            zones.append((start, i - 1, smoothed[start:i].mean()))
            start = None
    if start is not None:
        zones.append((start, len(in_zone) - 1, smoothed[start:].mean()))

    zones.sort(key=lambda z: z[2], reverse=True)
    zones = zones[:top_k]
    # Re-sort by track position so zone numbering is consistent
    zones.sort(key=lambda z: z[0])
    return zones


def plot_coaching_zones(amateur_err, top_k=5):
    mean_err = amateur_err.mean(axis=(0, 2))   # (200,)
    zones = find_coaching_zones(mean_err, top_k=top_k)

    fig, ax = plt.subplots(figsize=(14, 5))
    smoothed = uniform_filter1d(mean_err, size=7)
    ax.plot(LAP_DIST, smoothed, color="#37474F", lw=1.8, label="Mean amateur error (smoothed)")
    ax.fill_between(LAP_DIST, 0, smoothed, alpha=0.15, color="#37474F")

    colors = cm.Reds(np.linspace(0.5, 0.9, len(zones)))
    for rank, ((s, e, val), col) in enumerate(zip(zones, colors)):
        ax.axvspan(LAP_DIST[s], LAP_DIST[e], alpha=0.35, color=col,
                   label=f"Zone {rank+1}: {LAP_DIST[s]:.1f}–{LAP_DIST[e]:.1f}%")
        ax.text((LAP_DIST[s] + LAP_DIST[e]) / 2, smoothed.max() * 0.92,
                f"Z{rank+1}", ha="center", fontsize=9, fontweight="bold", color="darkred")

    ax.set_xlabel("Lap Distance (%)", fontsize=11)
    ax.set_ylabel("Mean Reconstruction Error", fontsize=11)
    ax.set_title("Top Coaching Zones — Segments with Highest Anomaly Score", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "coaching_zones.png", dpi=150)
    plt.close()
    print("  Saved: coaching_zones.png")
    return zones


#* --------------------------------- Plot 5: per-channel contribution in coaching zones --------------------------------- #
def plot_zone_channel_breakdown(amateur_err, zones):
    """Bar chart: which channels drive the error most in each coaching zone."""
    fig, axes = plt.subplots(1, len(zones), figsize=(4 * len(zones), 5), sharey=False)
    if len(zones) == 1:
        axes = [axes]

    for ax, (s, e, _) in zip(axes, zones):
        zone_err = amateur_err[:, s:e+1, :].mean(axis=(0, 1))  # (10,)
        sorted_idx = np.argsort(zone_err)[::-1]
        colors = ["#E53935" if c > np.mean(zone_err) else "#90CAF9"
                  for c in zone_err[sorted_idx]]
        ax.barh([FEATURE_COLS[i] for i in sorted_idx], zone_err[sorted_idx], color=colors)
        ax.set_title(f"{LAP_DIST[s]:.1f}–{LAP_DIST[e]:.1f}%", fontsize=10)
        ax.set_xlabel("Mean squared error", fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Channel Contribution per Coaching Zone", fontsize=13)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "zone_channel_breakdown.png", dpi=150)
    plt.close()
    print("  Saved: zone_channel_breakdown.png")
 

#* --------------------------------- Plot 6: circuit error map --------------------------------- #
def plot_circuit_error_map(test_latlon, amateur_err):
    """
    Scatter the circuit trace coloured by mean per-point reconstruction error.
    """
    latlon = align_latlon_to_seq_len(test_latlon, amateur_err.shape[1])
    mean_err = amateur_err.mean(axis=(0, 2))       # (N_POINTS,) — avg over laps & channels

    # Smooth the trace for better visualisation
    mean_lat = latlon[0,:,0]
    mean_lon = latlon[0,:,1]

    mean_lat = uniform_filter1d(mean_lat, size=9)
    mean_lon = uniform_filter1d(mean_lon, size=9)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(mean_lon, mean_lat, color="#CCCCCC", lw=4, zorder=1)   # grey base trace

    # Adjust color scale to focus on the main range of errors
    vmin = np.percentile(mean_err, 5)
    vmax = np.percentile(mean_err, 95)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True) # clip to handle outliers

    sc = ax.scatter(mean_lon, mean_lat, c=mean_err, cmap="hot_r",
                    s=30, zorder=2, norm=norm, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mean Reconstruction Error (MSE, normalised)", fontsize=10)

    # Direction arrow
    i_arr = 10
    ax.annotate("", xy=(mean_lon[i_arr + 3], mean_lat[i_arr + 3]),
                xytext=(mean_lon[i_arr], mean_lat[i_arr]),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Imola Circuit — Reconstruction Error Projection\n(hotter = higher anomaly)",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "circuit_error_map.png", dpi=150)
    plt.close()
    print("  Saved: circuit_error_map.png")


#* --------------------------------- Plot 7: coaching zones on the circuit --------------------------------- #
def plot_circuit_zone_map(test_latlon, amateur_err, zones):
    """
    Circuit trace with each coaching zone highlighted in a distinct colour.
    test_latlon : (n_laps, raw_len, 2)
    zones       : list of (start_idx, end_idx, mean_error) at model sequence resolution
    """
    latlon = align_latlon_to_seq_len(test_latlon, amateur_err.shape[1])
    mean_lat = latlon[0,:,0]
    mean_lon = latlon[0,:,1]

    mean_lat = uniform_filter1d(mean_lat, size=9)
    mean_lon = uniform_filter1d(mean_lon, size=9)

    zone_colors = ["#D32F2F", "#F57C00", "#FBC02D", "#388E3C", "#1976D2"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(mean_lon, mean_lat, color="#CCCCCC", lw=4, zorder=1, label="Circuit")

    for rank, ((s, e, val), col) in enumerate(zip(zones, zone_colors)):
        ax.plot(mean_lon[s:e + 1], mean_lat[s:e + 1], color=col, lw=6, zorder=2,
                label=f"Zone {rank + 1}: {LAP_DIST[s]:.1f}–{LAP_DIST[e]:.1f}%  (err {val:.3f})")
        mid = (s + e) // 2
        ax.text(mean_lon[mid], mean_lat[mid], f"Z{rank + 1}",
                fontsize=10, fontweight="bold", color=col,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.5, alpha=0.9),
                zorder=3)

    # Direction arrow
    i_arr = 10
    ax.annotate("", xy=(mean_lon[i_arr + 3], mean_lat[i_arr + 3]),
                xytext=(mean_lon[i_arr], mean_lat[i_arr]),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Imola Circuit — Top Coaching Zones", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "circuit_zone_map.png", dpi=150)
    plt.close()
    print("  Saved: circuit_zone_map.png")


#* --------------------------------- Plot 8: error split by channel group --------------------------------- #
def plot_error_by_group(expert_err, amateur_err):
    """
    Two-panel figure:
      Top:    Driver Inputs  (Throttle, Brake, Steering, Gear)
      Bottom: Vehicle Dynamics (Speed, RPM, all accel channels, YawRate)
    Each panel shows individual channels (thin lines) + group mean (bold) for
    both expert and amateur, plus filled gap between the means.

    expert_err  : (n_expert,  N_POINTS, N_FEATURES)
    amateur_err : (n_amateur, N_POINTS, N_FEATURES)
    """
    groups = [
        ("Driver Inputs",    DRIVER_INPUT_IDX,    DRIVER_INPUT_COLS),
        ("Vehicle Dynamics", VEHICLE_DYNAMIC_IDX, VEHICLE_DYNAMIC_COLS),
    ]
    colors = {"Expert": "#2196F3", "Amateur": "#FF5722"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax, (group_name, idxs, cols) in zip(axes, groups):
        # Thin per-channel lines
        for feat_idx, feat_name in zip(idxs, cols):
            e_sm = uniform_filter1d(expert_err[:,  :, feat_idx].mean(axis=0), size=7)
            a_sm = uniform_filter1d(amateur_err[:, :, feat_idx].mean(axis=0), size=7)
            ax.plot(LAP_DIST, e_sm, lw=1.0, ls="--", alpha=0.45, color=colors["Expert"])
            ax.plot(LAP_DIST, a_sm, lw=1.0, alpha=0.45, color=colors["Amateur"],
                    label=feat_name)

        # Bold group mean
        e_mean = uniform_filter1d(expert_err[:,  :, idxs].mean(axis=(0, 2)), size=7)
        a_mean = uniform_filter1d(amateur_err[:, :, idxs].mean(axis=(0, 2)), size=7)
        ax.plot(LAP_DIST, e_mean, lw=2.5, color=colors["Expert"],  label="Expert mean")
        ax.plot(LAP_DIST, a_mean, lw=2.5, color=colors["Amateur"], label="Amateur mean")
        ax.fill_between(LAP_DIST, e_mean, a_mean, alpha=0.13, color="#7B1FA2",
                        label="Gap (input problem)" if group_name == "Driver Inputs"
                        else "Gap (car reaction)")

        ax.set_title(f"{group_name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Squared Error", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Lap Distance (%)", fontsize=10)
    fig.suptitle("Error Split: Driver Inputs vs Vehicle Dynamics — Expert vs Amateur",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_by_group.png", dpi=150)
    plt.close()
    print("  Saved: error_by_group.png")


#* --------------------------------- Plot 9: MSE vs Lap Time correlation --------------------------------- #
def plot_mse_vs_laptime(expert_mse, amateur_mse, expert_meta, amateur_meta):
    """
    Scatter MSE vs lap time for all laps (train + test) and compute
    Pearson and Spearman correlations to validate that reconstruction
    error actually captures driving performance.
    """
    # Combine expert + amateur
    all_mse = np.concatenate([expert_mse, amateur_mse])
    all_laptimes = np.concatenate([
        expert_meta["lap_time_s"].values,
        amateur_meta["lap_time_s"].values,
    ])
    labels = (["Expert"] * len(expert_mse)) + (["Amateur"] * len(amateur_mse))

    # Correlations on the combined dataset
    r_pearson, p_pearson = pearsonr(all_laptimes, all_mse)
    r_spearman, p_spearman = spearmanr(all_laptimes, all_mse)

    # Per-group correlations
    r_expert_p, p_expert_p = pearsonr(expert_meta["lap_time_s"].values, expert_mse)
    r_expert_s, p_expert_s = spearmanr(expert_meta["lap_time_s"].values, expert_mse)
    r_amateur_p, p_amateur_p = pearsonr(amateur_meta["lap_time_s"].values, amateur_mse)
    r_amateur_s, p_amateur_s = spearmanr(amateur_meta["lap_time_s"].values, amateur_mse)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # Scatter
    expert_mask = np.array([l == "Expert" for l in labels])
    ax1.scatter(all_laptimes[expert_mask], all_mse[expert_mask],
               color="#2196F3", alpha=0.6, s=30, label="Expert (train)", zorder=3)
    ax1.scatter(all_laptimes[~expert_mask], all_mse[~expert_mask],
               color="#FF5722", alpha=0.6, s=30, label="Amateur (test)", zorder=3)

    # Regression line (all data)
    z = np.polyfit(all_laptimes, all_mse, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(all_laptimes.min(), all_laptimes.max(), 100)
    ax1.plot(x_range, p_line(x_range), "--", color="#7B1FA2", lw=2,
             label=f"Linear fit (r={r_pearson:.3f})")

    ax1.set_xlabel("Lap Time (s)", fontsize=11)
    ax1.set_ylabel("Reconstruction MSE (normalised)", fontsize=11)
    ax1.set_title("Reconstruction Error vs Lap Time", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Stats table
    ax2.axis("off")
    table_data = [
        ["Pearson r (all)",   f"{r_pearson:.4f}"],
        ["Pearson p (all)",   f"{p_pearson:.2e}"],
        ["Spearman ρ (all)",  f"{r_spearman:.4f}"],
        ["Spearman p (all)",  f"{p_spearman:.2e}"],
        ["", ""],
        ["Pearson r (expert)",  f"{r_expert_p:.4f}"],
        ["Pearson p (expert)",  f"{p_expert_p:.2e}"],
        ["Spearman ρ (expert)", f"{r_expert_s:.4f}"],
        ["", ""],
        ["Pearson r (amateur)",  f"{r_amateur_p:.4f}"],
        ["Pearson p (amateur)",  f"{p_amateur_p:.2e}"],
        ["Spearman ρ (amateur)", f"{r_amateur_s:.4f}"],
    ]

    table = ax2.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DCE6F1")
        elif table_data[row - 1][0] == "":
            cell.set_facecolor("white")
            cell.set_edgecolor("white")
        else:
            cell.set_facecolor("#F9F9F9")

    plt.tight_layout()
    plt.savefig(EVAL_DIR / "mse_vs_laptime.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved: mse_vs_laptime.png")

    # ---- Console summary ----
    print(f"\n-- MSE vs Lap Time Correlation ---------------------------------")
    print(f"  ALL DATA  ({len(all_mse)} laps):")
    print(f"    Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
    print(f"    Spearman ρ = {r_spearman:.4f}  (p = {p_spearman:.2e})")
    print(f"  EXPERT ({len(expert_mse)} laps):")
    print(f"    Pearson  r = {r_expert_p:.4f}  (p = {p_expert_p:.2e})")
    print(f"    Spearman ρ = {r_expert_s:.4f}  (p = {p_expert_s:.2e})")
    print(f"  AMATEUR ({len(amateur_mse)} laps):")
    print(f"    Pearson  r = {r_amateur_p:.4f}  (p = {p_amateur_p:.2e})")
    print(f"    Spearman ρ = {r_amateur_s:.4f}  (p = {p_amateur_s:.2e})")

    if r_spearman >= 0.5 and p_spearman < 0.05:
        print(f"  Significant positive correlation — model captures performance.")
    elif r_spearman >= 0.3 and p_spearman < 0.05:
        print(f"  Moderate correlation — model partially captures performance.")
    else:
        print(f"  Weak/no correlation — model may need redesign.")

    # Return for inclusion in metrics JSON 
    return {
        "all": {
            "n_laps": int(len(all_mse)),
            "pearson_r": round(float(r_pearson), 4),
            "pearson_p": float(p_pearson),
            "spearman_rho": round(float(r_spearman), 4),
            "spearman_p": float(p_spearman),
        },
        "expert": {
            "n_laps": int(len(expert_mse)),
            "pearson_r": round(float(r_expert_p), 4),
            "pearson_p": float(p_expert_p),
            "spearman_rho": round(float(r_expert_s), 4),
            "spearman_p": float(p_expert_s),
        },
        "amateur": {
            "n_laps": int(len(amateur_mse)),
            "pearson_r": round(float(r_amateur_p), 4),
            "pearson_p": float(p_amateur_p),
            "spearman_rho": round(float(r_amateur_s), 4),
            "spearman_p": float(p_amateur_s),
        },
    }


# JSON report 
def save_report(expert_mse, amateur_mse, expert_meta, amateur_meta, zones):
    def zone_to_dict(z):
        s, e, val = z
        return {
            "lap_dist_start_pct": round(float(LAP_DIST[s]), 2),
            "lap_dist_end_pct":   round(float(LAP_DIST[e]), 2),
            "mean_error":         round(float(val), 6),
        }

    report = {
        "expert_laps": {
            "n": int(len(expert_mse)),
            "mse_mean": round(float(expert_mse.mean()), 6),
            "mse_std":  round(float(expert_mse.std()),  6),
            "per_lap":  [{"driver": r.driver, "lap_time": r.lap_time_str,
                          "mse": round(float(m), 6)}
                         for r, m in zip(expert_meta.itertuples(), expert_mse)],
        },
        "amateur_laps": {
            "n": int(len(amateur_mse)),
            "mse_mean": round(float(amateur_mse.mean()), 6),
            "mse_std":  round(float(amateur_mse.std()),  6),
            "per_lap":  [{"driver": r.driver, "lap_time": r.lap_time_str,
                          "mse": round(float(m), 6)}
                         for r, m in zip(amateur_meta.itertuples(), amateur_mse)],
        },
        "top_coaching_zones": [zone_to_dict(z) for z in zones],
    }

    with open(EVAL_DIR / "eval_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("  Saved: eval_report.json")
    return report


#* -------------------------------- Main -------------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load scaler
    scaler     = np.load(MODELS_DIR / "scaler_params.npz")
    mean, std  = scaler["mean"], scaler["std"]

    # Load telemetry at full model resolution.
    train_telemetry = np.load(DATA_DIR / "train_telemetry.npy")
    test_telemetry  = np.load(DATA_DIR / "test_telemetry.npy")

    # Load pre-saved Lat/Lon arrays (raw grid; aligned inside plot functions)
    test_latlon  = np.load(DATA_DIR / "test_latlon.npy")   # (21, 1000, 2)

    train_norm = apply_normalization(train_telemetry, mean, std)
    test_norm  = apply_normalization(test_telemetry,  mean, std)

    # Load metadata
    train_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
    test_meta  = pd.read_csv(DATA_DIR / "test_metadata.csv")

    # Load model and compute reconstruction errors
    print("Loading model...")
    model = load_model(device)

    print("Computing expert reconstruction errors...")
    expert_recon, expert_err, expert_mse = reconstruct_errors(model, train_norm, device)

    print("Computing amateur reconstruction errors...")
    amateur_recon, amateur_err, amateur_mse = reconstruct_errors(model, test_norm, device)

    print(f"\nExpert  MSE: {expert_mse.mean():.4f} ± {expert_mse.std():.4f}")
    print(f"Amateur MSE: {amateur_mse.mean():.4f} ± {amateur_mse.std():.4f}")
    print(f"Anomaly ratio (amateur/expert): {amateur_mse.mean()/expert_mse.mean():.2f}×\n")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(train_norm, expert_recon, expert_err,
                              test_norm, amateur_recon, amateur_err,
                              mean, std, train_meta, test_meta)
    
    print("Computing ROC / AUC / Precision / Recall / F1...")
    clf_metrics, fpr, tpr = classification_metrics(expert_mse, amateur_mse)

    metrics["classification_metrics"] = clf_metrics

    plot_roc_curve(fpr, tpr, clf_metrics)

    print("\n-- Classification Metrics (Expert vs Amateur) ----------------")
    print(f"AUC: {clf_metrics['auc']:.3f}")
    print(f"Best threshold (MSE): {clf_metrics['best_threshold']:.4f}")
    print(f"Precision: {clf_metrics['precision']:.3f}")
    print(f"Recall:    {clf_metrics['recall']:.3f}")
    print(f"F1 Score:  {clf_metrics['f1_score']:.3f}")
    print(f"Confusion Matrix:\n{np.array(clf_metrics['confusion_matrix'])}")

    
    # MSE vs Lap Time correlation analysis
    print("\nComputing MSE vs Lap Time correlation...")
    correlation_metrics = plot_mse_vs_laptime(expert_mse, amateur_mse, train_meta, test_meta)
    metrics["mse_laptime_correlation"] = correlation_metrics

    # Save metrics report (after all metrics are computed)
    with open(EVAL_DIR / "metrics_report.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("  Saved: metrics_report.json")

    # Generate plots
    print("Generating plots...")
    np.random.seed(0)
    plot_error_distribution(expert_mse, amateur_mse, train_meta, test_meta)
    plot_channel_profiles(expert_err, amateur_err)
    plot_heatmap(amateur_err, test_meta)
    zones = plot_coaching_zones(amateur_err)
    plot_zone_channel_breakdown(amateur_err, zones)
    plot_error_by_group(expert_err, amateur_err)
    plot_circuit_error_map(test_latlon, amateur_err)
    plot_circuit_zone_map(test_latlon, amateur_err, zones)

    report = save_report(expert_mse, amateur_mse, train_meta, test_meta, zones)

    print("\n-- Top coaching zones ------------------------------------------")
    for i, z in enumerate(report["top_coaching_zones"]):
        print(f"  Zone {i+1}: {z['lap_dist_start_pct']}–{z['lap_dist_end_pct']}% of lap  "
              f"(mean error {z['mean_error']:.4f})")

    print(f"\nAll outputs saved to: {EVAL_DIR}")


if __name__ == "__main__":
    main()
