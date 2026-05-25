"""
Page 6: Advanced Features

Multi-lap comparison, sector analysis, lap consistency, and reconstruction error deep dive.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components.ui import section_header, empty_state
from app.components.charts import (
    multi_lap_overlay, error_heatmap, lap_consistency_chart,
)
from app.theme import COLORS, CHANNEL_DISPLAY_NAMES, CHANNEL_COLORS, PLOTLY_LAYOUT

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURE_COLS, CHANNEL_UNITS, CHANNEL_DISPLAY_SCALE, N_POINTS

LAP_DIST = np.linspace(0, 100, N_POINTS)


def render(session: dict):
    """Render the Advanced Features page."""
    reports = session.get("reports", {})
    amateur_result = session.get("amateur_result", {})
    summary = session.get("summary", {})
    expert_mse_mean = session.get("expert_mse", 0.0)
    expert_mse_std = session.get("expert_mse_std", 0.0)
    latlon = session.get("latlon")

    section_header("Advanced Analysis", "⚙️")

    if not amateur_result:
        empty_state("No data loaded for advanced analysis.", "📊")
        return

    laps_data = summary.get("laps", [])
    n_laps = len(laps_data)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_multi, tab_sector, tab_heatmap, tab_consistency = st.tabs([
        "🔄 Multi-Lap Comparison",
        "📐 Sector Analysis",
        "📈 Consistency Deep Dive",
    ])

    # ── Multi-Lap Comparison ─────────────────────────────────────────────────
    with tab_multi:
        section_header("Multi-Lap Overlay", "🔄")

        if n_laps < 2:
            st.info("Need at least 2 laps for multi-lap comparison.")
        else:
            channel = st.selectbox(
                "Channel to compare",
                FEATURE_COLS,
                format_func=lambda c: f"{CHANNEL_DISPLAY_NAMES.get(c, c)} ({CHANNEL_UNITS.get(c, '')})",
                key="multi_lap_channel",
            )

            lap_options = list(range(1, n_laps + 1))
            selected_laps = st.multiselect(
                "Select laps to overlay",
                lap_options,
                default=lap_options[:min(3, len(lap_options))],
                key="multi_lap_selection",
            )

            if len(selected_laps) < 2:
                st.info("Select at least 2 laps to compare.")
            else:
                indices = [l - 1 for l in selected_laps]
                laps_raw = amateur_result["denorm"][indices]
                lap_labels = [
                    f"Lap {l} ({laps_data[l-1].get('lap_time', '')})"
                    for l in selected_laps
                    if l - 1 < len(laps_data)
                ]

                # Use first lap's expert reconstruction as reference
                expert_ref = amateur_result["recon_denorm"][0] if len(amateur_result["recon_denorm"]) > 0 else None

                fig = multi_lap_overlay(laps_raw, channel, lap_labels, expert_recon=expert_ref)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Per-lap MSE for this channel
                ch_idx = FEATURE_COLS.index(channel)
                st.markdown(f"<div style='font-size: 0.85rem; font-weight: 600; color: {COLORS['text_secondary']}; margin: 16px 0 8px;'>PER-LAP ERROR ({CHANNEL_DISPLAY_NAMES.get(channel, channel)})</div>", unsafe_allow_html=True)

                for lap_num in selected_laps:
                    idx = lap_num - 1
                    if idx < len(amateur_result["error"]):
                        ch_mse = float(amateur_result["error"][idx, :, ch_idx].mean())
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px; padding: 6px 12px;
                                    background: {COLORS['bg_card']}; border-radius: 6px; margin-bottom: 4px;">
                            <span style="font-size: 0.85rem; color: {COLORS['text_primary']}; min-width: 80px;">Lap {lap_num}</span>
                            <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: {COLORS['text_secondary']};">{ch_mse:.6f}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Sector Analysis ──────────────────────────────────────────────────────
    with tab_sector:
        section_header("Sector Analysis", "📐")

        selected_lap_sector = st.selectbox(
            "Lap", range(1, n_laps + 1),
            format_func=lambda l: f"Lap {l} — {laps_data[l-1].get('lap_time', '')}" if l <= len(laps_data) else f"Lap {l}",
            key="sector_lap",
        )

        # Divide lap into 3 sectors (0-33%, 33-66%, 66-100%)
        n_sectors = 3
        lap_idx = selected_lap_sector - 1
        if lap_idx < len(amateur_result["error"]):
            lap_error = amateur_result["error"][lap_idx]
            sector_size = N_POINTS // n_sectors

            sector_data = []
            for s in range(n_sectors):
                s_start = s * sector_size
                s_end = (s + 1) * sector_size if s < n_sectors - 1 else N_POINTS
                sector_mse = float(lap_error[s_start:s_end].mean())
                sector_data.append({
                    "sector": s + 1,
                    "start_pct": round(s_start / N_POINTS * 100, 1),
                    "end_pct": round(s_end / N_POINTS * 100, 1),
                    "mse": sector_mse,
                })

            # Sector cards
            cols = st.columns(n_sectors)
            best_sector = min(sector_data, key=lambda x: x["mse"])
            worst_sector = max(sector_data, key=lambda x: x["mse"])

            sector_color_map = _sector_color_map(sector_data, best_sector, worst_sector)

            if latlon is not None and lap_idx < len(latlon):
                st.plotly_chart(
                    _sector_track_map(latlon[lap_idx], sector_data, sector_color_map),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown(f"""
                <div style="font-size: 0.78rem; color: {COLORS['text_muted']}; text-align: center; margin: -4px 0 16px;">
                    Sector colours match this lap's reconstruction error: green = best sector, yellow = middle, red = worst sector.
                </div>
                """, unsafe_allow_html=True)

            for col, sd in zip(cols, sector_data):
                with col:
                    border_color = sector_color_map[sd["sector"]]
                    if sd["sector"] == best_sector["sector"]:
                        label = "BEST"
                    elif sd["sector"] == worst_sector["sector"]:
                        label = "WORST"
                    else:
                        label = ""

                    label_html = f'<span style="font-size: 0.65rem; background: {border_color}20; color: {border_color}; padding: 2px 8px; border-radius: 10px; margin-left: 8px;">{label}</span>' if label else ""

                    st.markdown(f"""
                    <div class="kpi-card" style="border-top: 3px solid {border_color};">
                        <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; text-transform: uppercase;">
                            Sector {sd['sector']} {label_html}
                        </div>
                        <div style="font-size: 0.75rem; color: {COLORS['text_secondary']}; margin-bottom: 8px;">
                            {sd['start_pct']:.0f}% – {sd['end_pct']:.0f}%
                        </div>
                        <div class="kpi-value" style="font-size: 1.4rem;">{sd['mse']:.6f}</div>
                        <div class="kpi-label">MSE</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Per-channel sector breakdown
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.85rem; font-weight: 600; color: {COLORS['text_secondary']};'>PER-CHANNEL SECTOR BREAKDOWN</div>", unsafe_allow_html=True)

            channels_to_show = ["Speed", "Throttle", "Brake", "SteeringWheelAngle"]
            sector_labels = [f"S{sd['sector']}" for sd in sector_data]

            fig = go.Figure()
            bar_colors = [COLORS["expert"], COLORS["amateur"], COLORS["delta"]]

            for ch_name in channels_to_show:
                ch_idx = FEATURE_COLS.index(ch_name)
                values = []
                for sd in sector_data:
                    s_start = int(sd["start_pct"] / 100 * N_POINTS)
                    s_end = int(sd["end_pct"] / 100 * N_POINTS)
                    values.append(float(lap_error[s_start:s_end, ch_idx].mean()))

                fig.add_trace(go.Bar(
                    name=CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name),
                    x=sector_labels, y=values,
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                barmode="group",
                xaxis_title="Sector",
                yaxis_title="MSE",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


    # ── Consistency Deep Dive ────────────────────────────────────────────────
    with tab_consistency:
        section_header("Lap-to-Lap Consistency", "📈")

        if n_laps < 2:
            st.info("Need at least 2 laps for consistency analysis.")
        else:
            mse_array = np.array([l.get("mse_normalised", l.get("mse", 0)) for l in laps_data])
            time_strs = [l.get("lap_time", "") for l in laps_data]

            fig = lap_consistency_chart(mse_array, time_strs, expert_mse_mean, expert_mse_std)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Consistency stats
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            lap_times_s = [l.get("lap_time_s", 0) for l in laps_data]
            time_std = np.std(lap_times_s)
            mse_std = np.std(mse_array)
            improvement = mse_array[0] - mse_array[-1] if len(mse_array) > 1 else 0

            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size: 1.3rem;">±{time_std:.3f}s</div>
                    <div class="kpi-label">Lap Time Std Dev</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size: 1.3rem;">±{mse_std:.6f}</div>
                    <div class="kpi-label">MSE Std Dev</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                trend_color = COLORS["good"] if improvement > 0 else COLORS["bad"]
                trend_icon = "📉" if improvement > 0 else "📈"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size: 1.3rem; color: {trend_color};">
                        {trend_icon} {improvement:+.6f}
                    </div>
                    <div class="kpi-label">MSE Trend (first → last)</div>
                </div>
                """, unsafe_allow_html=True)

            # Per-lap error evolution per channel
            st.markdown(f"<div style='font-size: 0.85rem; font-weight: 600; color: {COLORS['text_secondary']}; margin: 24px 0 8px;'>ERROR EVOLUTION BY CHANNEL</div>", unsafe_allow_html=True)

            fig = go.Figure()
            channels_to_track = ["Speed", "Throttle", "Brake", "SteeringWheelAngle"]

            for ch_name in channels_to_track:
                ch_idx = FEATURE_COLS.index(ch_name)
                ch_errors = [float(amateur_result["error"][l, :, ch_idx].mean())
                             for l in range(min(n_laps, len(amateur_result["error"])))]
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(ch_errors) + 1)),
                    y=ch_errors, mode="markers+lines",
                    name=CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name),
                    line=dict(color=CHANNEL_COLORS.get(ch_name, {}).get("amateur", COLORS["amateur"]), width=2),
                    marker=dict(size=8),
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT, height=300,
                xaxis_title="Lap Number", yaxis_title="Channel MSE",
            )
            fig.update_xaxes(tickmode="linear", dtick=1, gridcolor=COLORS["grid"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _sector_color_map(sector_data: list, best_sector: dict, worst_sector: dict) -> dict:
    colors = {}
    for sector in sector_data:
        sector_id = sector["sector"]
        if sector_id == best_sector["sector"]:
            colors[sector_id] = COLORS["good"]
        elif sector_id == worst_sector["sector"]:
            colors[sector_id] = COLORS["bad"]
        else:
            colors[sector_id] = COLORS["medium"]
    return colors


def _sector_track_map(latlon: np.ndarray, sector_data: list, sector_colors: dict) -> go.Figure:
    lat = latlon[:, 0]
    lon = latlon[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lon,
        y=lat,
        mode="lines",
        line=dict(color=COLORS["text_muted"], width=3),
        name="Track",
        hoverinfo="skip",
        showlegend=False,
    ))

    for sector in sector_data:
        sector_id = sector["sector"]
        idx_s = int(sector["start_pct"] / 100 * (N_POINTS - 1))
        idx_e = int(sector["end_pct"] / 100 * (N_POINTS - 1))
        idx_e = max(idx_s + 1, min(idx_e, N_POINTS - 1))
        color = sector_colors[sector_id]

        fig.add_trace(go.Scatter(
            x=lon[idx_s:idx_e + 1],
            y=lat[idx_s:idx_e + 1],
            mode="lines",
            line=dict(color=color, width=7),
            name=f"Sector {sector_id}",
            customdata=[[sector_id, sector["start_pct"], sector["end_pct"], sector["mse"]]],
            hovertemplate=(
                "<b>Sector %{customdata[0]}</b><br>"
                "%{customdata[1]:.0f}% - %{customdata[2]:.0f}%<br>"
                "MSE: %{customdata[3]:.6f}<extra></extra>"
            ),
        ))

        mid = (idx_s + idx_e) // 2
        fig.add_trace(go.Scatter(
            x=[lon[mid]],
            y=[lat[mid]],
            mode="markers+text",
            marker=dict(color=color, size=18, symbol="circle", line=dict(color=COLORS["bg_primary"], width=2)),
            text=[f"S{sector_id}"],
            textposition="middle center",
            textfont=dict(size=10, color="#FFFFFF", family="Inter"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[lon[0]],
        y=[lat[0]],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=12, symbol="diamond", line=dict(color="white", width=2)),
        name="Start/Finish",
        hovertemplate="<b>Start/Finish</b><extra></extra>",
    ))

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")},
        height=360,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    return fig
