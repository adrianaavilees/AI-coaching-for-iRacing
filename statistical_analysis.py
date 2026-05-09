"""
Feedback Engine Statistical Analysis Module

This module performs in-depth statistical analysis of the reconstruction errors between expert and pilot telemetry data. 
It computes key metrics such as mean squared error (MSE), identifies dominant error channels, and detects causal chains of errors that contribute to performance degradation.
"""
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import uniform_filter1d
from config import FEATURE_COLS, N_POINTS, CHANNEL_DISPLAY_SCALE, CHANNEL_UNITS


@dataclass
class ChannelDeviation:
    """Deviation for a single channel within a coaching zone."""
    channel: str
    signed_mean: float       # Mean signed error (recon - original) in display units
    abs_mean: float          # Mean absolute deviation in display units
    unit: str
    direction: str           # "over" | "under" — relative to expert pattern
    severity: float          # 0-1 normalised severity within this zone


@dataclass
class CausalChain:
    """A detected cause → effect relationship between channels."""
    cause_channel: str
    cause_direction: str     # "over" | "under"
    effect_channel: str
    effect_direction: str
    confidence: float        # 0-1, based on temporal correlation
    description: str         # Human-readable causal explanation

def compute_signed_error(amateur_raw, expert_recon_raw):
    """
    Compute the signed reconstruction error in raw (physical) space.
    
    Positive = model expected MORE of this channel (amateur is UNDER the expert pattern)
    Negative = model expected LESS of this channel (amateur is OVER the expert pattern)
    
    Returns: (N_POINTS, N_FEATURES) signed error array
    """
    return expert_recon_raw - amateur_raw


def detect_zones(squared_error, window=15, top_k=5, threshold_pct=75):
    """
    Detect coaching zones from per-point squared error.
    
    Args:
        squared_error: (N_POINTS, N_FEATURES) — squared error per point per channel
        window: smoothing window size
        top_k: max number of zones to return
        threshold_pct: percentile threshold for zone detection
    
    Returns: list of (idx_start, idx_end, mean_severity)
    """
    mean_err = squared_error.mean(axis=1)  # (N_POINTS,)
    smoothed = uniform_filter1d(mean_err, size=window)
    threshold = np.percentile(smoothed, threshold_pct)
    in_zone = smoothed >= threshold # True indicates points where error is above threshold meaning potential coaching zones

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

    # Filter tiny zones (< 0.5% of lap)
    min_zone_len = max(3, int(N_POINTS * 0.005))
    zones = [(s, e, v) for s, e, v in zones if (e - s + 1) >= min_zone_len]

    zones.sort(key=lambda z: z[2], reverse=True)
    return zones[:top_k]


def analyse_zone_channels(signed_error, sq_error, idx_start, idx_end):
    """
    For a given zone, compute per-channel deviations and rank by severity.
    
    Returns: list of ChannelDeviation sorted by absolute severity (descending)
    """
    zone_signed = signed_error[idx_start:idx_end + 1, :]   # (zone_len, N_FEATURES)
    zone_sq     = sq_error[idx_start:idx_end + 1, :]

    deviations = []
    max_abs = 0.0

    for ch_idx, ch_name in enumerate(FEATURE_COLS):
        raw_signed_mean = float(zone_signed[:, ch_idx].mean())
        raw_abs_mean    = float(np.abs(zone_signed[:, ch_idx]).mean())

        # Convert to display units
        scale = CHANNEL_DISPLAY_SCALE.get(ch_name, 1.0)
        display_signed = raw_signed_mean * scale
        display_abs    = raw_abs_mean * scale
        unit           = CHANNEL_UNITS.get(ch_name, "")

        # Direction: positive signed error means recon > original → model expected more → amateur is UNDER
        direction = "under" if raw_signed_mean > 0 else "over"

        deviations.append(ChannelDeviation(
            channel=ch_name,
            signed_mean=round(display_signed, 2),
            abs_mean=round(display_abs, 2),
            unit=unit,
            direction=direction,
            severity=display_abs,  # will be normalised later
        ))
        max_abs = max(max_abs, display_abs)

    # Normalise severity to 0-1 within the zone
    if max_abs > 0:
        for d in deviations:
            d.severity = round(d.severity / max_abs, 3)

    deviations.sort(key=lambda d: d.severity, reverse=True)
    return deviations


def detect_causal_chains(signed_error, idx_start, idx_end):
    """
    Detect cause → effect relationships between channels within a zone by
    analysing temporal lag correlations in the signed error.
    
    Known causal patterns in motorsport:
        1. Brake (late/weak) → Speed (too high at apex) → LatAccel (understeer)
        2. Throttle (early/aggressive) → YawRate (oversteer) → SteeringWheelAngle (correction)
        3. Speed (too low at entry) → Throttle (early to compensate) → suboptimal exit
        4. SteeringWheelAngle (excessive) → Speed (scrubbed) → LongAccel (poor exit)
        5. Brake (too early) → Speed (too low at apex) → Throttle (compensating early)
    """
    zone_signed = signed_error[idx_start:idx_end + 1, :]
    zone_len = idx_end - idx_start + 1
    chains = []

    # Channel indices
    ch_idx = {name: FEATURE_COLS.index(name) for name in FEATURE_COLS}

    # Helper: mean signed error for a channel in a sub-range of the zone
    def mean_signed(channel, frac_start=0.0, frac_end=1.0):
        s = int(zone_len * frac_start)
        e = max(s + 1, int(zone_len * frac_end))
        return float(zone_signed[s:e, ch_idx[channel]].mean())

    #* -------------------------- Pattern 1: Late/weak braking → high apex speed → understeer -------------------------- #
    brake_signed = mean_signed("Brake", 0.0, 0.4)
    speed_signed = mean_signed("Speed", 0.3, 0.7)
    lat_signed   = mean_signed("LatAccel", 0.3, 0.8)

    # brake_signed > 0 means model expected MORE brake → amateur brakes LESS (late/weak)
    # speed_signed < 0 means model expected LESS speed → amateur is FASTER
    if brake_signed > 0 and speed_signed < 0:
        conf = min(1.0, (abs(brake_signed) + abs(speed_signed)) / 2)
        desc = "Insufficient braking leads to excess speed at the apex"
        chains.append(CausalChain("Brake", "under", "Speed", "over", round(conf, 2), desc))

        if lat_signed < 0:  # more lateral load than expected → pushing
            chains.append(CausalChain("Speed", "over", "LatAccel", "over",
                                       round(conf * 0.8, 2),
                                       "Excess apex speed causes understeer (high lateral load)"))

    #* -------------------------- Pattern 2: Aggressive throttle → oversteer (yaw) -------------------------- #
    throttle_signed = mean_signed("Throttle", 0.5, 1.0)
    yaw_signed      = mean_signed("YawRate", 0.5, 1.0)
    steer_signed    = mean_signed("SteeringWheelAngle", 0.6, 1.0)

    if throttle_signed < 0 and abs(yaw_signed) > 0:
        conf = min(1.0, (abs(throttle_signed) + abs(yaw_signed)) / 2)
        chains.append(CausalChain("Throttle", "over", "YawRate", 
                                   "over" if yaw_signed < 0 else "under",
                                   round(conf, 2),
                                   "Aggressive throttle application causes rear instability"))

    #* -------------------------- Pattern 3: Early braking → low apex speed → compensating with early throttle -------------------------- #
    if brake_signed < 0 and speed_signed > 0:
        throttle_exit = mean_signed("Throttle", 0.4, 0.8)
        if throttle_exit < 0:
            conf = min(1.0, (abs(brake_signed) + abs(speed_signed)) / 2)
            chains.append(CausalChain("Brake", "over", "Speed", "under",
                                       round(conf, 2),
                                       "Braking too early causes low apex speed"))
            chains.append(CausalChain("Speed", "under", "Throttle", "over",
                                       round(conf * 0.7, 2),
                                       "Low apex speed leads to premature throttle application"))

    #* -------------------------- Pattern 4: Excessive steering → speed scrub -------------------------- #
    steer_entry = mean_signed("SteeringWheelAngle", 0.0, 0.5)
    speed_mid   = mean_signed("Speed", 0.3, 0.7)
    if steer_entry < 0 and speed_mid > 0:  # too much steering, losing speed
        conf = min(1.0, (abs(steer_entry) + abs(speed_mid)) / 2)
        chains.append(CausalChain("SteeringWheelAngle", "over", "Speed", "under",
                                   round(conf, 2),
                                   "Excessive steering input scrubs speed through the corner"))

    #* -------------------------- Pattern 5: Late throttle application → poor corner exit -------------------------- #
    throttle_exit = mean_signed("Throttle", 0.6, 1.0)
    speed_exit    = mean_signed("Speed", 0.7, 1.0)
    longaccel_exit = mean_signed("LongAccel", 0.7, 1.0)

    if throttle_exit > 0 and speed_exit > 0:
        conf = min(1.0, (abs(throttle_exit) + abs(speed_exit)) / 2)
        chains.append(CausalChain("Throttle", "under", "Speed", "under",
                                   round(conf, 2),
                                   "Late throttle application compromises corner exit speed"))

    # Sort by confidence
    chains.sort(key=lambda c: c.confidence, reverse=True)
    return chains[:4]  # Max 4 causal chains per zone


def compute_zone_severity(sq_error, expert_sq_error_baseline, idx_start, idx_end):
    """
    Compute a 0-1 severity score for a zone, relative to expert reconstruction error.
    
    severity = (amateur_zone_mse - expert_baseline_mse) / expert_baseline_mse
    Clamped to [0, 1] and scaled.
    """
    zone_mse = sq_error[idx_start:idx_end + 1].mean()
    baseline = expert_sq_error_baseline.mean() + 1e-8
    raw_ratio = (zone_mse - baseline) / baseline
    return float(np.clip(raw_ratio / 10.0, 0.0, 1.0))  # 10x expert = severity 1.0


def estimate_time_loss(signed_error_raw, lap_time_s, idx_start, idx_end):
    """
    Rough estimate of time lost in a zone based on speed deviation.
    
    ΔT ≈ (zone_fraction × lap_time) × (ΔSpeed / mean_Speed)
    
    This is a first-order approximation. Negative ΔSpeed (amateur faster than 
    expert reconstruction) in one zone may still cost time overall if it leads to poor exit speed.
    """
    speed_idx = FEATURE_COLS.index("Speed")
    zone_len = idx_end - idx_start + 1
    zone_fraction = zone_len / N_POINTS

    # Signed speed error: positive means amateur is slower than expert pattern
    speed_diff = signed_error_raw[idx_start:idx_end + 1, speed_idx]
    mean_speed_diff = float(speed_diff.mean())

    # Use absolute speed diff to estimate time impact (both faster and slower cost time
    # if they deviate from optimal)
    abs_speed_diff = float(np.abs(speed_diff).mean())

    # Assume average zone speed ~ proportional to lap time
    # Very rough: ΔT ≈ zone_fraction × lap_time × (abs_speed_diff / reference_speed)
    reference_speed = 50.0  # m/s ~ 180 km/h average for GT3 at Imola 
    time_loss = zone_fraction * lap_time_s * (abs_speed_diff / reference_speed)

    return round(time_loss, 3)
