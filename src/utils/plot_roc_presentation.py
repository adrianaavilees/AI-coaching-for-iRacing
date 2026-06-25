"""
Generates the ROC curve with red/black aesthetics for the presentation.
Uses real data from mse_expert_amateur_canva.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT_DIR / "data" / "mse_expert_amateur_canva.csv"
OUTPUT_PATH = ROOT_DIR / "models" / "eval" / "roc_curve_presentation.png"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
expert_mse  = df[df["Grup"] == "Expert"]["Error MSE"].values
amateur_mse = df[df["Grup"] != "Expert"]["Error MSE"].values

y_true = np.concatenate([np.zeros(len(expert_mse)), np.ones(len(amateur_mse))])
scores = np.concatenate([expert_mse, amateur_mse])

fpr, tpr, thresholds = roc_curve(y_true, scores)
auc_score = roc_auc_score(y_true, scores)

best_idx       = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]
y_pred         = (scores >= best_threshold).astype(int)

precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

# ── Presentation style: black background, red curve ─────────────────────────
BG      = "#0d0d0d"   # near-black background
RED     = "#cc0000"   # rojo Ferrari / corporativo
RED_LT  = "#ff4444"   # rojo más vivo para highlight
GRAY    = "#555555"   # reference diagonal
WHITE   = "#f0f0f0"   # text
GRID    = "#2a2a2a"   # grid lines

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.5), facecolor=BG,
    gridspec_kw={"width_ratios": [2.2, 1]}
)
ax.set_facecolor(BG)
ax2.set_facecolor(BG)

# Area under the curve – semi-transparent fill
ax.fill_between(fpr, tpr, alpha=0.18, color=RED)

# Reference diagonal
ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, alpha=0.7)

# Main ROC curve
ax.plot(fpr, tpr, color=RED, lw=2.8, label=f"AUC = {auc_score:.3f}")

# Optimal point (Youden J)
ax.scatter(fpr[best_idx], tpr[best_idx],
           color=RED_LT, s=90, zorder=5, edgecolors=WHITE, linewidths=0.8)

# Axes and labels
ax.set_xlabel("False Positive Rate", fontsize=11, color=WHITE, labelpad=8)
ax.set_ylabel("True Positive Rate", fontsize=11, color=WHITE, labelpad=8)
ax.set_title("ROC Curve – Expert vs Amateur", fontsize=13, color=WHITE,
             fontweight="bold", pad=14)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.tick_params(colors=WHITE, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(GRAY)

ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)

# Legend
legend = ax.legend(loc="lower right", fontsize=10,
                   framealpha=0.25, edgecolor=RED,
                   labelcolor=WHITE)

# ── Right panel: metrics table ───────────────────────────────────────────────
ax2.axis("off")
for spine in ax2.spines.values():
    spine.set_visible(False)

table_data = [
    ["AUC", f"{auc_score:.4f}"],
    ["Best threshold (MSE)", f"{best_threshold:.6f}"],
    ["Precision", f"{precision:.4f}"],
    ["Recall", f"{recall:.4f}"],
    ["F1 Score", f"{f1:.4f}"],
]

table = ax2.table(
    cellText=table_data,
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 2.0)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(BG)
    cell.set_edgecolor(GRAY)
    cell.set_text_props(color=WHITE)
    if row == 0:
        cell.set_facecolor(RED)
        cell.set_text_props(color=WHITE, fontproperties=None, weight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()

print(f"Guardado en: {OUTPUT_PATH}")
print(f"  AUC       : {auc_score:.4f}")
print(f"  Threshold : {best_threshold:.6f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
