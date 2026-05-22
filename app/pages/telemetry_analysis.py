"""
Page 3: Telemetry Analysis

Expert vs amateur overlays with synchronized charts and zone highlighting.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components.ui import section_header
from app.components.charts import (
    telemetry_overlay, telemetry_stacked, delta_time_chart, zone_focus_timeline,
    LAP_DIST,
)
from app.theme import COLORS, PLOTLY_LAYOUT

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURE_COLS, CHANNEL_UNITS, CHANNEL_DISPLAY_SCALE
from app.theme import CHANNEL_DISPLAY_NAMES


def render(session: dict):
    """Render the Telemetry Analysis page."""
    reports = session.get("reports", {})
    amateur_result = session.get("amateur_result", {})
    selected_lap = session.get("selected_lap", 1)
    summary = session.get("summary", {})

    section_header("Telemetry Analysis", "📡")

    if not amateur_result:
        st.info("No telemetry data loaded. Please load a session first.")
        return

    lap_idx = selected_lap - 1
    amateur_denorm = amateur_result["denorm"]
    recon_denorm = amateur_result["recon_denorm"]

    if lap_idx >= len(amateur_denorm):
        st.warning(f"Lap {selected_lap} not available.")
        return

    amateur_lap = amateur_denorm[lap_idx]
    expert_lap = recon_denorm[lap_idx]

    # Get zones for highlighting
    report = reports.get(selected_lap, {})
    zones = report.get("zones", [])

    # Get lap time
    laps_data = summary.get("laps", [])
    lap_time_s = 100.0
    for l in laps_data:
        if l.get("lap_number", l.get("lap", 0)) == selected_lap:
            lap_time_s = l.get("lap_time_s", 100.0)
            break

    # ── View Mode Toggle ─────────────────────────────────────────────────────
    view_mode = st.radio(
        "View mode",
        ["Individual Charts", "Stacked View"],
        horizontal=True,
        key="telem_view_mode",
    )

    # ── Channel Selection ────────────────────────────────────────────────────
    all_channels = FEATURE_COLS.copy()
    default_channels = ["Speed", "Throttle", "Brake", "SteeringWheelAngle"]

    selected_channels = st.multiselect(
        "Telemetry channels",
        all_channels,
        default=default_channels,
        format_func=lambda c: f"{CHANNEL_DISPLAY_NAMES.get(c, c)} ({CHANNEL_UNITS.get(c, '')})",
        key="telem_channels",
    )

    if not selected_channels:
        st.warning("Select at least one channel.")
        return

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    if view_mode == "Individual Charts":
        figures = telemetry_overlay(amateur_lap, expert_lap,
                                    channels=selected_channels, zones=zones, height=220)

        # Render in 2-column grid
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(figures):
                    ch_name, fig = figures[idx]
                    with col:
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displayModeBar": False})
    else:
        fig = telemetry_stacked(amateur_lap, expert_lap,
                                channels=selected_channels, zones=zones)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Delta Time Chart ─────────────────────────────────────────────────────
    section_header("Delta Time", "⏱️")
    delta_fig = delta_time_chart(amateur_lap, expert_lap, lap_time_s)
    st.plotly_chart(delta_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Driver Focus Timeline ────────────────────────────────────────────────
    section_header("Coaching Priority by Lap Position", "🎯")
    focus_fig = zone_focus_timeline(zones)
    st.plotly_chart(focus_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div style="font-size: 0.85rem; color: {COLORS['text_secondary']}; text-align: center; margin-top: 8px;">
        The horizontal line is your lap from 0% to 100%. Colored blocks are the zones where the AI found the largest coaching opportunity. Red blocks should be reviewed first.
    </div>
    """, unsafe_allow_html=True)

    # ── Zone Focus ───────────────────────────────────────────────────────────
    selected_zone_data = st.session_state.get("selected_zone_data")
    if selected_zone_data and zones:
        section_header("Zone Focus", "🔍")

        z = selected_zone_data
        idx_s = z.get("idx_start", 0)
        idx_e = z.get("idx_end", len(amateur_lap) - 1)
        margin = max(20, (idx_e - idx_s) // 2)
        view_s = max(0, idx_s - margin)
        view_e = min(len(amateur_lap) - 1, idx_e + margin)

        for ch_name in selected_channels[:4]:
            ch_idx = FEATURE_COLS.index(ch_name)
            scale = CHANNEL_DISPLAY_SCALE.get(ch_name, 1.0)
            unit = CHANNEL_UNITS.get(ch_name, "")
            display_name = CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name)

            fig = go.Figure()

            # Zone highlight
            fig.add_vrect(
                x0=LAP_DIST[idx_s], x1=LAP_DIST[idx_e],
                fillcolor="rgba(255,23,68,0.1)", line_width=0, layer="below",
            )

            fig.add_trace(go.Scatter(
                x=LAP_DIST[view_s:view_e],
                y=expert_lap[view_s:view_e, ch_idx] * scale,
                mode="lines", name="Expert",
                line=dict(color=COLORS["expert"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=LAP_DIST[view_s:view_e],
                y=amateur_lap[view_s:view_e, ch_idx] * scale,
                mode="lines", name="You",
                line=dict(color=COLORS["amateur"], width=2),
            ))

            fig.update_layout(
                **PLOTLY_LAYOUT, height=200,
                title=dict(text=f"{display_name} — Zone {z['zone_id']} Detail", font=dict(size=12)),
                xaxis_title="Lap Distance (%)",
                yaxis_title=f"{unit}" if unit else display_name,
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
