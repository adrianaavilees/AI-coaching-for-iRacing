"""
Create the final filtered dataset for training the AI coaching autoencoder

Full self-contained pipeline — reads raw Garage61 CSV laps, interpolates,
builds the telemetry matrix, applies every filter, and saves the outputs.

Pipeline:
    1. Load raw CSV laps + extra metadata (weather/fuel/tire).
    2. Interpolate every lap onto a uniform 1000-point LapDistPct grid.
    3. Build the (n_laps, 1000, 9) telemetry matrix.
    4. Environmental filter  → dry, no rain, dry-compound tyres.
    5. Telemetry integrity   → no NaN rows in the interpolated matrix.
    6. Outlier filter        → remove absurd lap times (incidents, re-entries).
    7. Skill split           → Expert (train) vs Amateur (test) by lap-time percentile.
    8. Fuel-load filter      → keep laps within a reasonable fuel window (train only).
    9. Save train / test metadata CSVs + telemetry .npy + report.

Outputs (inside data/processed/):
    - train_metadata.csv        Expert-lap metadata
    - test_metadata.csv         Amateur-lap metadata
    - train_telemetry.npy       (n_laps_train, N_POINTS, n_features) → (18, 1000, 10)
    - test_telemetry.npy        (n_laps_test,  N_POINTS, n_features) → (21, 1000, 10)
    - filtering_report.txt      
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


ROOT_DIR = Path(__file__).resolve().parent
RAW_LAPS_DIR = ROOT_DIR / "data" / "Ferrari 296 GT3" / "Imola"
EXTRA_DATA_PATH = ROOT_DIR / "data" / "Ferrari 296 GT3" / "extra_data_Imola_Ferrari296.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Interpolation 
N_POINTS = 1000  # Points per lap after interpolation

# Channels that go into the telemetry matrix (order matters)
FEATURE_COLS = ["Speed", "Throttle", "Brake", "RPM", "SteeringWheelAngle","Gear", "LatAccel", "LongAccel", "VertAccel","YawRate"]
#* yaw, lat, lon are not necessary for the autoencoder to learn good representations of lap structure

FILENAME_RE = re.compile(r"Garage 61 - (.+?) - Ferrari 296 GT3 - .+? - (\d{2})\.(\d{2})\.(\d{3}) - (.+?)\.csv")


EXPERT_PERCENTILE = 25
AMATEUR_PERCENTILE = 75
MAX_LAP_TIME_S = 130.0
FUEL_LEVEL_RANGE = (10, 60)


def load_raw_laps():
    """Scan the raw CSV directory, parse filenames, return (lap_dfs, meta_df)

    Each DataFrame in *lap_dfs* is the raw telemetry for one lap.
    *meta_df* contains one row per lap with driver / time / id metadata.
    The two lists are aligned by position (index 0 ↔ row 0 of meta_df)
    """
    csv_files = sorted(RAW_LAPS_DIR.glob("*.csv"))
    lap_dfs = []
    meta_rows = []

    for f in csv_files:
        match = FILENAME_RE.match(f.name) 
        if not match:
            continue

        driver = match.group(1) 
        minutes, seconds, millis = int(match.group(2)), int(match.group(3)), int(match.group(4))
        lap_time_s = minutes * 60 + seconds + millis / 1000
        lap_id = match.group(5)

        # \\?\ prefix to handle Windows MAX_PATH limitation
        long_path = "\\\\?\\" + str(f)
        try:
            df = pd.read_csv(long_path)
        except Exception as exc:
            print(f"  WARN  Could not read {f.name}: {exc}")
            continue

        lap_dfs.append(df)
        meta_rows.append({
            "driver": driver,
            "lap_time_s": lap_time_s,
            "lap_time_str": f"{minutes:02d}:{seconds:02d}.{millis:03d}",
            "lap_id": lap_id,
            "n_samples": len(df),
            "filename": f.name,
        })

    meta_df = pd.DataFrame(meta_rows).sort_values("lap_time_s").reset_index(drop=True)

    # Re-order lap_dfs to match the sorted meta_df
    original_order = {row["filename"]: idx for idx, row in enumerate(meta_rows)}
    sorted_dfs = [lap_dfs[original_order[fn]] for fn in meta_df["filename"]]

    return sorted_dfs, meta_df


def interpolate_lap(lap_df,target_dist,cols):
    """Interpolate a lap to a uniform LapDistPct grid and return an (N, C) array"""
    x = lap_df["LapDistPct"].values

    # Remove duplicate x-values (keep last) to guarantee monotonicity
    mask = np.diff(x, prepend=-1) > 0
    x_clean = x[mask]

    out = np.empty((len(target_dist), len(cols)), dtype=np.float64)
    for j, col in enumerate(cols):
        y_clean = lap_df[col].values[mask]
        try:
            # Linear interpolation with extrapolation for out-of-bounds values
            f = interp1d(x_clean, y_clean, kind="linear", fill_value="extrapolate")
            out[:, j] = f(target_dist)
        except Exception:
            out[:, j] = np.nan

    return out

def build_telemetry_matrix(lap_dfs):
    """Interpolate every lap and stack into (n_laps, N_POINTS, n_features)"""
    uniform_dist = np.linspace(0, 1, N_POINTS)
    n_laps = len(lap_dfs)
    n_feat = len(FEATURE_COLS)

    telemetry = np.zeros((n_laps, N_POINTS, n_feat), dtype=np.float64)
    for i, lap in enumerate(lap_dfs):
        telemetry[i] = interpolate_lap(lap, uniform_dist, FEATURE_COLS)

    return telemetry, uniform_dist


def merge_extra_data(meta_df):
    """Merge the extra Garage61 metadata (weather, fuel, tyres) into meta_df"""
    extra = pd.read_csv(EXTRA_DATA_PATH)

    # Clean track_usage: -1 means unknown → NaN
    if "track_usage" in extra.columns:
        extra["track_usage"] = extra["track_usage"].replace(-1, np.nan)

    # Drop cols that already exist in meta_df 
    drop_cols = [c for c in ("driver", "lap_time_str") if c in extra.columns]
    extra = extra.drop(columns=drop_cols, errors="ignore")

    merged = meta_df.merge(extra, on="lap_id", how="left")
    return merged


# Filtering functions
def filter_environment(df):
    """Keep only laps with standard weather: dry, no rain, dry-compound tyres"""

    # No precipitation
    mask_dry = df["precipitation"] == 0

    # Dry tyre compound (1.0). Exclude wet (2.0) and unknown (NaN)
    mask_tyre = df["tire_compound"] == 1.0

    # No fog
    mask_fog = df["fog_level"].fillna(0) == 0

    combined = mask_dry & mask_tyre & mask_fog
    df_out = df[combined].copy()

    return df_out


def filter_telemetry_integrity(df, telemetry):
    """Drop laps whose interpolated telemetry row contains any NaN"""

    has_nan_mask = np.isnan(telemetry).reshape(telemetry.shape[0], -1).any(axis=1)
    nan_indices = set(np.where(has_nan_mask)[0])
    keep = ~df.index.isin(nan_indices)
    df_out = df[keep].copy()

    return df_out


def filter_outlier_lap_times(df):
    """Remove absurdly slow laps (likely incidents, not lack of skill)"""
    mask = df["lap_time_s"] <= MAX_LAP_TIME_S
    df_out = df[mask].copy()

    return df_out


def split_by_skill(df):
    """Split into expert / average / amateur by percentile thresholds"""
    p_expert = np.percentile(df["lap_time_s"], EXPERT_PERCENTILE)
    p_amateur = np.percentile(df["lap_time_s"], AMATEUR_PERCENTILE)

    experts = df[df["lap_time_s"] <= p_expert].copy()
    amateurs = df[df["lap_time_s"] >= p_amateur].copy()
    average = df[(df["lap_time_s"] > p_expert) & (df["lap_time_s"] < p_amateur)].copy()

    return experts, average, amateurs


def filter_fuel_train(df):
    """For the train set, keep laps within a same fuel window to reduce weight noise"""
    lo, hi = FUEL_LEVEL_RANGE
    mask = df["fuel_level"].between(lo, hi)
    df_out = df[mask].copy()

    return df_out

def main():
    report = []
    report.append("FINAL DATASET CREATION — FILTERING REPORT")
    report.append("")

    # Load raw CSVs + extra metadata 
    print("Loading raw laps...")
    lap_dfs, meta_df = load_raw_laps()
    report.append(f"Raw laps loaded: {len(lap_dfs)}")
    report.append(f"Unique drivers: {meta_df['driver'].nunique()}")

    # Interpolate & build telemetry matrix 
    print("Interpolating telemetry...")
    telemetry, _uniform_dist = build_telemetry_matrix(lap_dfs)
    report.append(f"Telemetry matrix: {telemetry.shape}")
    report.append("")

    # Merge extra data (weather / fuel / tyres)
    meta_df = merge_extra_data(meta_df)
    report.append(f"Extra data merged: {meta_df.columns.tolist()}")
    report.append("")

    # Environmental filter
    df = filter_environment(meta_df)

    # Telemetry integrity
    df = filter_telemetry_integrity(df, telemetry)

    # Outlier lap times
    df = filter_outlier_lap_times(df)

    # Skill split
    experts, _average, amateurs = split_by_skill(df)

    # Fuel filter (train only)
    experts = filter_fuel_train(experts)

    report.append("=" * 60)
    report.append("FINAL RESULT")
    report.append(f"  Train (expert) laps:  {len(experts)}  ({experts['driver'].nunique()} drivers)")
    report.append(f"  Test (amateur) laps:  {len(amateurs)}  ({amateurs['driver'].nunique()} drivers)")
    report.append("")

    if len(experts) > 0:
        report.append("Train lap-time range: "
                       f"{experts['lap_time_s'].min():.3f}s – {experts['lap_time_s'].max():.3f}s")
    if len(amateurs) > 0:
        report.append("Test  lap-time range: "
                       f"{amateurs['lap_time_s'].min():.3f}s – {amateurs['lap_time_s'].max():.3f}s")

    # Save
    # Metadata CSVs
    experts.to_csv(OUTPUT_DIR / "train_metadata.csv", index=False)
    amateurs.to_csv(OUTPUT_DIR / "test_metadata.csv", index=False)

    # Telemetry numpy arrays
    train_tel = telemetry[experts.index.values].astype(np.float32)
    test_tel = telemetry[amateurs.index.values].astype(np.float32)
    np.save(OUTPUT_DIR / "train_telemetry.npy", train_tel)
    np.save(OUTPUT_DIR / "test_telemetry.npy", test_tel)

    report.append("")
    report.append(f"train_telemetry.npy  →  {train_tel.shape}")
    report.append(f"test_telemetry.npy   →  {test_tel.shape}")

    report_text = "\n".join(report)
    (OUTPUT_DIR / "filtering_report.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nFiles saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()