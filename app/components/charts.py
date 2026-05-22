"""
Plotly chart builders for telemetry visualisation.

All charts use the motorsport dark theme from theme.py.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

from app.theme import COLORS, PLOTLY_LAYOUT, CHANNEL_COLORS, CHANNEL_DISPLAY_NAMES

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURE_COLS, CHANNEL_DISPLAY_SCALE, CHANNEL_UNITS, N_POINTS

LAP_DIST = np.linspace(0, 100, N_POINTS)


# ─── Telemetry Overlay ──────────────────────────────────────────────────────

def telemetry_overlay(amateur_raw: np.ndarray, expert_recon_raw: np.ndarray,
                      channels: list = None, zones: list = None,
                      height: int = 200) -> list:
    """
    Create individual Plotly figures for each telemetry channel overlay.
    
    Returns a list of (channel_name, fig) tuples.
    """
    if channels is None:
        channels = ["Speed", "Throttle", "Brake", "SteeringWheelAngle"]
    
    figures = []
    
    for ch_name in channels:
        ch_idx = FEATURE_COLS.index(ch_name)
        scale = CHANNEL_DISPLAY_SCALE.get(ch_name, 1.0)
        unit = CHANNEL_UNITS.get(ch_name, "")
        display_name = CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name)
        colors = CHANNEL_COLORS.get(ch_name, {"expert": COLORS["expert"], "amateur": COLORS["amateur"]})
        
        amateur_vals = amateur_raw[:, ch_idx] * scale
        expert_vals = expert_recon_raw[:, ch_idx] * scale
        
        fig = go.Figure()
        
        # Zone highlights
        if zones:
            for z in zones:
                start = z.get("lap_pct_start", 0)
                end = z.get("lap_pct_end", 0)
                sev = z.get("severity_score", 0.5)
                color = "rgba(255,23,68,0.08)" if sev > 0.66 else (
                        "rgba(255,214,0,0.06)" if sev > 0.33 else "rgba(0,230,118,0.05)")
                fig.add_vrect(x0=start, x1=end, fillcolor=color,
                              layer="below", line_width=0)
        
        # Expert trace
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=expert_vals, mode="lines",
            name="Expert pattern",
            line=dict(color=colors["expert"], width=2),
            hovertemplate=f"Expert: %{{y:.1f}} {unit}<extra></extra>",
        ))
        
        # Amateur trace
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=amateur_vals, mode="lines",
            name="Your lap",
            line=dict(color=colors["amateur"], width=2, dash="solid"),
            hovertemplate=f"You: %{{y:.1f}} {unit}<extra></extra>",
        ))
        
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=height,
            title=dict(text=f"{display_name} ({unit})", font=dict(size=13)),
            xaxis_title="Lap Distance (%)",
            yaxis_title=f"{display_name} ({unit})" if unit else display_name,
            showlegend=True,
        )
        
        figures.append((ch_name, fig))
    
    return figures


def telemetry_stacked(amateur_raw: np.ndarray, expert_recon_raw: np.ndarray,
                      channels: list = None, zones: list = None) -> go.Figure:
    """
    Create a single stacked subplot figure with all channels.
    """
    if channels is None:
        channels = ["Speed", "Throttle", "Brake", "SteeringWheelAngle"]
    
    n = len(channels)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=[CHANNEL_DISPLAY_NAMES.get(c, c) for c in channels],
    )
    
    for i, ch_name in enumerate(channels, 1):
        ch_idx = FEATURE_COLS.index(ch_name)
        scale = CHANNEL_DISPLAY_SCALE.get(ch_name, 1.0)
        unit = CHANNEL_UNITS.get(ch_name, "")
        colors = CHANNEL_COLORS.get(ch_name, {"expert": COLORS["expert"], "amateur": COLORS["amateur"]})
        
        amateur_vals = amateur_raw[:, ch_idx] * scale
        expert_vals = expert_recon_raw[:, ch_idx] * scale
        
        # Zone highlights
        if zones:
            for z in zones:
                start = z.get("lap_pct_start", 0)
                end = z.get("lap_pct_end", 0)
                sev = z.get("severity_score", 0.5)
                color = "rgba(255,23,68,0.08)" if sev > 0.66 else (
                        "rgba(255,214,0,0.06)" if sev > 0.33 else "rgba(0,230,118,0.05)")
                fig.add_vrect(x0=start, x1=end, fillcolor=color,
                              layer="below", line_width=0, row=i, col=1)
        
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=expert_vals, mode="lines",
            name="Expert" if i == 1 else None, showlegend=(i == 1),
            line=dict(color=colors["expert"], width=1.5),
            hovertemplate=f"%{{y:.1f}} {unit}<extra>Expert</extra>",
        ), row=i, col=1)
        
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=amateur_vals, mode="lines",
            name="You" if i == 1 else None, showlegend=(i == 1),
            line=dict(color=colors["amateur"], width=1.5),
            hovertemplate=f"%{{y:.1f}} {unit}<extra>You</extra>",
        ), row=i, col=1)
        
        fig.update_yaxes(title_text=unit, row=i, col=1,
                         gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    
    fig.update_xaxes(title_text="Lap Distance (%)", row=n, col=1,
                     gridcolor=COLORS["grid"])
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=200 * n,
        showlegend=True,
    )
    
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = COLORS["text_secondary"]
    
    return fig


# ─── Track Map ───────────────────────────────────────────────────────────────

def track_map(latlon: np.ndarray, zones: list = None,
              selected_zone: int = None) -> go.Figure:
    """
    Interactive circuit map using lat/lon coordinates.
    Zones are highlighted with performance colors.
    """
    # Use mean of expert laps for clean track outline
    if latlon.ndim == 3:
        lat = latlon.mean(axis=0)[:, 0]
        lon = latlon.mean(axis=0)[:, 1]
    else:
        lat = latlon[:, 0]
        lon = latlon[:, 1]
    
    fig = go.Figure()
    
    # Base track line
    fig.add_trace(go.Scatter(
        x=lon, y=lat, mode="lines",
        line=dict(color=COLORS["text_muted"], width=3),
        name="Track", showlegend=False,
        hoverinfo="skip",
    ))
    
    # Zone overlays
    if zones:
        for z in zones:
            idx_s = z.get("idx_start", 0)
            idx_e = z.get("idx_end", 0)
            sev = z.get("severity_score", 0.5)
            zid = z.get("zone_id", 0)
            feedback_preview = _feedback_preview(z)
            
            if sev > 0.66:
                color = COLORS["bad"]
            elif sev > 0.33:
                color = COLORS["medium"]
            else:
                color = COLORS["good"]
            
            width = 6 if selected_zone == zid else 4
            opacity = 1.0 if selected_zone == zid else 0.8
            
            zone_lat = lat[idx_s:idx_e + 1]
            zone_lon = lon[idx_s:idx_e + 1]
            
            fig.add_trace(go.Scatter(
                x=zone_lon, y=zone_lat, mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                name=f"Zone {zid}",
                hoverinfo="skip",
            ))
            
            # Zone label at midpoint
            mid = (idx_s + idx_e) // 2
            fig.add_trace(go.Scatter(
                x=[lon[mid]], y=[lat[mid]], mode="markers+text",
                marker=dict(color=color, size=14, symbol="circle",
                            line=dict(color=COLORS["bg_primary"], width=2)),
                text=[str(zid)], textposition="middle center",
                textfont=dict(size=9, color="white", family="Inter"),
                customdata=[[zid, z.get("lap_pct_start", 0), z.get("lap_pct_end", 0), sev, feedback_preview]],
                showlegend=False,
                hovertemplate=(
                    f"<b>Zone {zid}</b><br>"
                    "%{customdata[4]}<br>"
                    "<i>Click for full AI explanation</i><extra></extra>"
                ),
            ))
    
    # Start/finish marker
    fig.add_trace(go.Scatter(
        x=[lon[0]], y=[lat[0]], mode="markers",
        marker=dict(color=COLORS["accent"], size=12, symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Start/Finish", showlegend=True,
        hovertemplate="<b>Start/Finish</b><extra></extra>",
    ))
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    
    return fig


def _feedback_preview(zone: dict, max_chars: int = 96) -> str:
    text = zone.get("llm_feedback") or zone.get("template_feedback") or "Open this zone for detailed coaching."
    text = re.sub(r"[═━█]+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if 28 <= len(sentence) <= max_chars:
        text = sentence
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def zone_focus_timeline(zones: list) -> go.Figure:
    """Driver-friendly replacement for the raw reconstruction heatmap."""
    fig = go.Figure()
    for z in sorted(zones or [], key=lambda item: item.get("lap_pct_start", 0)):
        sev = z.get("severity_score", 0.0)
        if sev > 0.66:
            color = COLORS["bad"]
        elif sev > 0.33:
            color = COLORS["medium"]
        else:
            color = COLORS["good"]
        start = z.get("lap_pct_start", 0)
        end = z.get("lap_pct_end", start)
        width = max(end - start, 0.8)
        fig.add_trace(go.Bar(
            x=[width], y=["Lap focus"], base=[start], orientation="h",
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.45)", width=1)),
            text=[f"Z{z.get('zone_id', '')}"], textposition="inside",
            customdata=[[z.get("zone_id"), sev, z.get("estimated_time_loss_s", 0)]],
            hovertemplate="<b>Zone %{customdata[0]}</b><br>Severity: %{customdata[1]:.0%}<br>Time loss: %{customdata[2]:.3f}s<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        height=190,
        barmode="stack",
        margin=dict(l=20, r=20, t=25, b=35),
        xaxis=dict(title="Lap distance (%)", range=[0, 100], gridcolor=COLORS["grid"]),
        yaxis=dict(showticklabels=False, gridcolor=COLORS["grid"]),
    )
    return fig


# ─── Error Heatmap ──────────────────────────────────────────────────────────

def error_heatmap(error_per_point: np.ndarray, channels: list = None) -> go.Figure:
    """
    Heatmap of reconstruction error across lap distance and channels.
    error_per_point: (N_POINTS, N_FEATURES) squared error
    """
    if channels is None:
        channels = FEATURE_COLS
    
    ch_indices = [FEATURE_COLS.index(c) for c in channels]
    display_names = [CHANNEL_DISPLAY_NAMES.get(c, c) for c in channels]
    
    z_data = error_per_point[:, ch_indices].T  # (n_channels, N_POINTS)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=LAP_DIST,
        y=display_names,
        colorscale=[
            [0.0, "#0D1117"],
            [0.2, "#1A237E"],
            [0.4, "#4A148C"],
            [0.6, "#FF6F00"],
            [0.8, "#FF1744"],
            [1.0, "#FFEB3B"],
        ],
        colorbar=dict(
            title="Error", titlefont=dict(color=COLORS["text_secondary"]),
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        hovertemplate="Distance: %{x:.1f}%<br>Channel: %{y}<br>Error: %{z:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        xaxis_title="Lap Distance (%)",
    )
    fig.update_yaxes(gridcolor=COLORS["grid"])
    
    return fig


# ─── Radar Chart ─────────────────────────────────────────────────────────────

def radar_chart(channel_scores: dict, title: str = "Performance Profile") -> go.Figure:
    """
    Radar chart showing relative performance across telemetry dimensions.
    channel_scores: {"Speed": 0.8, "Brake": 0.4, ...} — 0=worst, 1=best
    """
    categories = list(channel_scores.keys())
    values = list(channel_scores.values())
    
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]
    
    fig = go.Figure()
    
    # Reference circle (expert = 1.0)
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(categories),
        theta=categories,
        fill="none",
        line=dict(color=COLORS["expert"], width=1, dash="dot"),
        name="Expert baseline",
    ))
    
    # Driver performance
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(255, 145, 0, 0.15)",
        line=dict(color=COLORS["amateur"], width=2),
        name="Your performance",
    ))
    
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=400,
        polar=dict(
            bgcolor=COLORS["chart_bg"],
            radialaxis=dict(
                visible=True, range=[0, 1.2],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9, color=COLORS["text_muted"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(size=11, color=COLORS["text_secondary"]),
            ),
        ),
        showlegend=True,
    )
    
    return fig


# ─── Lap Consistency ─────────────────────────────────────────────────────────

def lap_consistency_chart(mse_per_lap: np.ndarray, lap_times: list,
                          expert_mse_mean: float, expert_mse_std: float) -> go.Figure:
    """Scatter plot of MSE vs lap time with expert baseline band."""
    laps = list(range(1, len(mse_per_lap) + 1))
    
    fig = go.Figure()
    
    # Expert baseline band
    fig.add_hrect(
        y0=expert_mse_mean - expert_mse_std,
        y1=expert_mse_mean + expert_mse_std,
        fillcolor="rgba(0, 176, 255, 0.08)",
        line_width=0,
        annotation_text="Expert zone",
        annotation_position="top left",
        annotation_font=dict(color=COLORS["expert"], size=10),
    )
    
    fig.add_hline(y=expert_mse_mean, line=dict(color=COLORS["expert"], width=1, dash="dash"))
    
    # Driver laps
    fig.add_trace(go.Scatter(
        x=laps, y=mse_per_lap, mode="markers+lines",
        marker=dict(
            color=mse_per_lap, colorscale=[[0, COLORS["good"]], [0.5, COLORS["medium"]], [1, COLORS["bad"]]],
            size=12, line=dict(color=COLORS["bg_primary"], width=2),
        ),
        line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
        text=[f"Lap {i}: {t}" for i, t in zip(laps, lap_times)],
        hovertemplate="<b>%{text}</b><br>MSE: %{y:.6f}<extra></extra>",
        name="Your laps",
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        xaxis_title="Lap Number",
        yaxis_title="Reconstruction Error (MSE)",
    )
    fig.update_xaxes(tickmode="linear", dtick=1, gridcolor=COLORS["grid"])
    
    return fig


# ─── Delta Time ──────────────────────────────────────────────────────────────

def delta_time_chart(amateur_raw: np.ndarray, expert_recon_raw: np.ndarray,
                     lap_time_s: float) -> go.Figure:
    """
    Cumulative delta time chart (amateur vs expert).
    Approximation based on speed differences.
    """
    speed_idx = FEATURE_COLS.index("Speed")
    amateur_speed = amateur_raw[:, speed_idx]
    expert_speed = expert_recon_raw[:, speed_idx]
    
    # Avoid division by zero
    mean_speed = (amateur_speed + expert_speed) / 2
    mean_speed = np.clip(mean_speed, 1.0, None)
    
    # Approximate delta per segment
    segment_fraction = 1.0 / N_POINTS
    segment_time_ref = lap_time_s * segment_fraction
    
    speed_diff = expert_speed - amateur_speed  # positive = amateur slower
    delta_per_point = segment_time_ref * (speed_diff / mean_speed)
    cumulative_delta = np.cumsum(delta_per_point)
    
    fig = go.Figure()
    
    # Fill above/below zero
    fig.add_trace(go.Scatter(
        x=LAP_DIST, y=cumulative_delta,
        fill="tozeroy",
        mode="lines",
        line=dict(color=COLORS["delta"], width=2),
        fillcolor="rgba(224, 64, 251, 0.1)",
        hovertemplate="Distance: %{x:.1f}%<br>Delta: %{y:+.3f}s<extra></extra>",
        name="Delta",
    ))
    
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dash"))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
        title=dict(text="Cumulative Delta Time", font=dict(size=13)),
        xaxis_title="Lap Distance (%)",
        yaxis_title="Delta (s)",
    )
    
    return fig


# ─── Zone Bar Chart ──────────────────────────────────────────────────────────

def zone_ranking_chart(zones: list) -> go.Figure:
    """Horizontal bar chart ranking zones by severity."""
    if not zones:
        return go.Figure()
    
    zone_ids = [f"Zone {z.get('zone_id', 0)}" for z in zones]
    severities = [z.get("severity_score", 0) for z in zones]
    time_losses = [z.get("estimated_time_loss_s", 0) for z in zones]
    colors = [COLORS["bad"] if s > 0.66 else (COLORS["medium"] if s > 0.33 else COLORS["good"])
              for s in severities]
    
    # Sort by severity descending
    sorted_data = sorted(zip(zone_ids, severities, time_losses, colors),
                         key=lambda x: x[1], reverse=True)
    zone_ids, severities, time_losses, colors = zip(*sorted_data)
    
    fig = go.Figure(go.Bar(
        y=list(zone_ids), x=list(severities),
        orientation="h",
        marker=dict(color=list(colors), line=dict(color=COLORS["bg_primary"], width=1)),
        text=[f"{s:.0%} | ~{t:.3f}s" for s, t in zip(severities, time_losses)],
        textposition="inside",
        textfont=dict(color="white", size=11, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Severity: %{x:.0%}<extra></extra>",
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(zones) * 50),
        xaxis_title="Severity",
    )
    fig.update_xaxes(range=[0, 1.05], gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    
    return fig


# ─── Multi-lap Comparison ───────────────────────────────────────────────────

def multi_lap_overlay(laps_raw: np.ndarray, channel: str,
                      lap_labels: list, expert_recon: np.ndarray = None) -> go.Figure:
    """
    Overlay multiple amateur laps for a single channel.
    laps_raw: (n_laps, N_POINTS, N_FEATURES)
    """
    ch_idx = FEATURE_COLS.index(channel)
    scale = CHANNEL_DISPLAY_SCALE.get(channel, 1.0)
    unit = CHANNEL_UNITS.get(channel, "")
    display_name = CHANNEL_DISPLAY_NAMES.get(channel, channel)
    
    # Color palette for multiple laps
    lap_colors = ["#FF9100", "#E040FB", "#FFD600", "#00E676", "#FF1744",
                  "#18FFFF", "#FF6E40", "#B388FF"]
    
    fig = go.Figure()
    
    # Expert reference (if provided)
    if expert_recon is not None:
        expert_vals = expert_recon[:, ch_idx] * scale
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=expert_vals, mode="lines",
            name="Expert", line=dict(color=COLORS["expert"], width=2, dash="dot"),
        ))
    
    for i in range(min(len(laps_raw), len(lap_colors))):
        vals = laps_raw[i, :, ch_idx] * scale
        fig.add_trace(go.Scatter(
            x=LAP_DIST, y=vals, mode="lines",
            name=lap_labels[i] if i < len(lap_labels) else f"Lap {i+1}",
            line=dict(color=lap_colors[i % len(lap_colors)], width=1.5),
        ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text=f"{display_name} — Multi-Lap Comparison", font=dict(size=13)),
        xaxis_title="Lap Distance (%)",
        yaxis_title=f"{display_name} ({unit})" if unit else display_name,
    )
    
    return fig
