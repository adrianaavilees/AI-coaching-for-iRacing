"""
Page 2: Track Map

Interactive circuit map with color-coded coaching zones.
Clicking a zone updates the telemetry view.
"""

import streamlit as st
import numpy as np
import html

from app.components.ui import section_header, coaching_card, corner_badge
from app.components.charts import track_map
from app.theme import COLORS
from app.data_loader import get_severity_color


def render(session: dict):
    """Render the Track Map page."""
    reports = session.get("reports", {})
    lap_latlon = session.get("latlon")
    if lap_latlon is None:
        lap_latlon = session.get("train_latlon")
    selected_lap = session.get("selected_lap", 1)

    section_header("Circuit Map", "🗺️")

    if lap_latlon is None:
        st.warning("No track GPS data available.")
        return

    # Get zones for selected lap
    report = reports.get(selected_lap, {})
    zones = report.get("zones", [])

    # ── Track Map ────────────────────────────────────────────────────────────
    col_map, col_zones = st.columns([3, 2])

    with col_map:
        selected_zone_id = st.session_state.get("selected_zone", None)
        fig = track_map(lap_latlon, zones=zones, selected_zone=selected_zone_id)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode="points",
        )
        clicked_zone = _clicked_zone_id(event)
        if clicked_zone is not None:
            st.session_state["selected_zone"] = clicked_zone
            for z in zones:
                if int(z.get("zone_id", -1)) == clicked_zone:
                    st.session_state["selected_zone_data"] = z
                    break

    with col_zones:
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: {COLORS['text_secondary']}; margin-bottom: 16px;">
            <strong>Lap {selected_lap}</strong> — {len(zones)} coaching zones detected
        </div>
        """, unsafe_allow_html=True)

        # Zone badges
        if zones:
            badge_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">'
            for z in zones:
                badge_html += corner_badge(z.get("zone_id", 0), z.get("severity_score", 0))
            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)

            # Zone selector
            zone_options = [f"Zone {z['zone_id']} ({z['lap_pct_start']:.1f}% – {z['lap_pct_end']:.1f}%)"
                           for z in zones]
            selected_idx = st.selectbox(
                "Select a zone to inspect",
                range(len(zone_options)),
                format_func=lambda i: zone_options[i],
                index=_selected_zone_index(zones),
                key="zone_selector",
            )

            if selected_idx is not None:
                z = zones[selected_idx]
                st.session_state["selected_zone"] = z.get("zone_id")
                st.session_state["selected_zone_data"] = z

                sev = z.get("severity_score", 0)
                sev_color = get_severity_color(sev)
                time_loss = z.get("estimated_time_loss_s", 0)

                st.markdown(f"""
                <div class="coaching-card" style="border-left-color: {sev_color};">
                    <div class="zone-header">
                        <span class="zone-title">Zone {z['zone_id']}</span>
                        <span class="zone-severity" style="background: {sev_color}20; color: {sev_color};">
                            {sev:.0%}
                        </span>
                    </div>
                    <div class="metric-row">
                        <div class="metric-item">
                            <span class="metric-value" style="color: {COLORS['bad']};">~{time_loss:.3f}s</span>
                            <span class="metric-label">Time Lost</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-value">{z['lap_pct_start']:.1f}% – {z['lap_pct_end']:.1f}%</span>
                            <span class="metric-label">Lap Position</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                feedback = _zone_feedback(z)
                st.markdown(f"""
                <div class="ai-zone-panel">
                    <div class="ai-zone-kicker">AI race engineer explanation</div>
                    <div class="ai-zone-copy">{feedback}</div>
                </div>
                """, unsafe_allow_html=True)

                # Dominant channels
                dom_channels = z.get("dominant_channels", [])
                sig = [d for d in dom_channels if d.get("severity", 0) > 0.1][:4]
                if sig:
                    st.markdown(f"<div style='margin-top: 12px; font-size: 0.85rem; color: {COLORS['text_secondary']}; font-weight: 600;'>KEY DEVIATIONS</div>", unsafe_allow_html=True)
                    for d in sig:
                        arrow = "▲" if d["direction"] == "over" else "▼"
                        color = COLORS["bad"] if d["direction"] == "over" else COLORS["medium"]
                        st.markdown(f"""
                        <div class="deviation-row">
                            <span class="deviation-channel">{arrow} {d['channel']}</span>
                            <span class="deviation-value" style="color: {color};">
                                {abs(d['signed_mean']):.1f} {d.get('unit', '')}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 40px; color: {COLORS['text_muted']};">
                <div style="font-size: 2rem; margin-bottom: 8px;">✅</div>
                <div>No significant coaching zones detected for this lap.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Legend ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; gap: 24px; justify-content: center; margin-top: 16px;
                padding: 12px; background: {COLORS['bg_card']}; border-radius: 8px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 16px; height: 4px; background: {COLORS['good']}; border-radius: 2px;"></div>
            <span style="font-size: 0.75rem; color: {COLORS['text_secondary']};">Good (&lt;33%)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 16px; height: 4px; background: {COLORS['medium']}; border-radius: 2px;"></div>
            <span style="font-size: 0.75rem; color: {COLORS['text_secondary']};">Medium (33-66%)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 16px; height: 4px; background: {COLORS['bad']}; border-radius: 2px;"></div>
            <span style="font-size: 0.75rem; color: {COLORS['text_secondary']};">Critical (&gt;66%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _clicked_zone_id(event):
    try:
        points = event.selection.points
    except AttributeError:
        return None
    if not points:
        return None
    customdata = points[0].get("customdata")
    if not customdata:
        return None
    try:
        return int(customdata[0])
    except (TypeError, ValueError):
        return None


def _selected_zone_index(zones):
    selected_zone = st.session_state.get("selected_zone")
    if selected_zone is None:
        return 0
    for idx, zone in enumerate(zones):
        if int(zone.get("zone_id", -1)) == int(selected_zone):
            return idx
    return 0


def _zone_feedback(zone: dict) -> str:
    feedback = zone.get("llm_feedback") or zone.get("template_feedback") or "Select this zone after the AI report finishes to see detailed coaching."
    feedback = " ".join(str(feedback).replace("\n", " ").split())
    return html.escape(feedback)
