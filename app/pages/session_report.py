"""
Page 5: Session Report

Strengths, weaknesses, radar chart, corner ranking, and PDF export.
"""

import streamlit as st
import numpy as np
import io
import json

from app.components.ui import section_header, kpi_card
from app.components.charts import radar_chart, zone_ranking_chart
from app.theme import COLORS
from app.data_loader import get_severity_color

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURE_COLS, CHANNEL_DISPLAY_SCALE
from app.theme import CHANNEL_DISPLAY_NAMES


def render(session: dict):
    """Render the Session Report page."""
    reports = session.get("reports", {})
    amateur_result = session.get("amateur_result", {})
    selected_lap = session.get("selected_lap", 1)
    summary = session.get("summary", {})
    expert_mse_mean = session.get("expert_mse", 0.0)

    section_header("Session Report", "📊")

    report = reports.get(selected_lap, {})
    zones = report.get("zones", [])

    if not report or not amateur_result:
        st.info("No data available for report generation.")
        return

    lap_idx = selected_lap - 1
    error = amateur_result.get("error")
    if error is None or lap_idx >= len(error):
        st.warning("Error data not available.")
        return

    lap_error = error[lap_idx]  # (N_POINTS, N_FEATURES)

    # ── Performance Profile (Radar) ──────────────────────────────────────────
    col_radar, col_strengths = st.columns([1, 1])

    with col_radar:
        st.markdown(f"<div style='font-size: 0.9rem; font-weight: 600; color: {COLORS['text_secondary']}; margin-bottom: 8px;'>PERFORMANCE PROFILE</div>", unsafe_allow_html=True)

        # Compute per-channel scores (1 - normalized MSE relative to expert)
        channel_scores = {}
        for ch_idx, ch_name in enumerate(FEATURE_COLS):
            ch_mse = float(lap_error[:, ch_idx].mean())
            # Normalize: score = 1 - (driver_mse / (expert_mse + driver_mse))
            # Higher score = closer to expert
            score = max(0, 1 - ch_mse / (expert_mse_mean + ch_mse + 1e-8))
            display_name = CHANNEL_DISPLAY_NAMES.get(ch_name, ch_name)
            channel_scores[display_name] = round(score, 3)

        fig = radar_chart(channel_scores)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_strengths:
        # Categorize channels into strengths and weaknesses
        sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
        strengths = [(name, score) for name, score in sorted_channels if score > 0.7]
        weaknesses = [(name, score) for name, score in sorted_channels if score <= 0.5]
        moderate = [(name, score) for name, score in sorted_channels if 0.5 < score <= 0.7]

        # Strengths
        st.markdown(f"<div style='font-size: 0.9rem; font-weight: 600; color: {COLORS['good']}; margin-bottom: 12px;'>✅ STRENGTHS</div>", unsafe_allow_html=True)
        if strengths:
            for name, score in strengths:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px;
                            background: rgba(0,230,118,0.05); border-radius: 6px; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: {COLORS['text_primary']}; flex: 1;">{name}</span>
                    <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: {COLORS['good']};">{score:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: {COLORS['text_muted']}; font-size: 0.85rem;'>No strong areas identified yet — keep practicing!</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Weaknesses
        st.markdown(f"<div style='font-size: 0.9rem; font-weight: 600; color: {COLORS['bad']}; margin-bottom: 12px;'>⚠️ AREAS TO IMPROVE</div>", unsafe_allow_html=True)
        if weaknesses:
            for name, score in weaknesses:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px;
                            background: rgba(255,23,68,0.05); border-radius: 6px; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: {COLORS['text_primary']}; flex: 1;">{name}</span>
                    <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: {COLORS['bad']};">{score:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: {COLORS['text_muted']}; font-size: 0.85rem;'>No critical weaknesses detected.</div>", unsafe_allow_html=True)

        if moderate:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.9rem; font-weight: 600; color: {COLORS['medium']}; margin-bottom: 12px;'>🔶 MODERATE</div>", unsafe_allow_html=True)
            for name, score in moderate:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px;
                            background: rgba(255,214,0,0.05); border-radius: 6px; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: {COLORS['text_primary']}; flex: 1;">{name}</span>
                    <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: {COLORS['medium']};">{score:.0%}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Zone Ranking ─────────────────────────────────────────────────────────
    section_header("Zone Ranking", "🏆")

    if zones:
        fig = zone_ranking_chart(zones)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No zones to rank.")

    # ── Export ────────────────────────────────────────────────────────────────
    section_header("Export Report", "💾")

    col_json, col_csv = st.columns(2)

    with col_json:
        report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="📥 Download JSON Report",
            data=report_json,
            file_name=f"coaching_report_lap{selected_lap}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_csv:
        # Build a CSV summary
        csv_lines = ["zone_id,lap_pct_start,lap_pct_end,severity,time_loss_s,top_channel,top_deviation"]
        for z in zones:
            dom = z.get("dominant_channels", [])
            top_ch = dom[0]["channel"] if dom else ""
            top_dev = f"{dom[0]['signed_mean']:.2f}" if dom else ""
            csv_lines.append(
                f"{z.get('zone_id','')},{z.get('lap_pct_start','')},{z.get('lap_pct_end','')},"
                f"{z.get('severity_score','')},{z.get('estimated_time_loss_s','')},{top_ch},{top_dev}"
            )
        csv_text = "\n".join(csv_lines)
        st.download_button(
            label="📥 Download CSV Summary",
            data=csv_text,
            file_name=f"coaching_zones_lap{selected_lap}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Full JSON Preview ────────────────────────────────────────────────────
    with st.expander("📄 Raw JSON Report Preview"):
        st.json(report)
