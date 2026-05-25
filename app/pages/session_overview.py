"""
Page 1: Session Overview

Driver info, track info, KPI cards, and session summary.
"""

import streamlit as st
import numpy as np
import html

from app.components.ui import section_header, kpi_card
from app.components.charts import track_map
from app.data_loader import TRACK_INFO, CAR_INFO, get_severity_color
from app.theme import COLORS


def render(session: dict):
    """
    Render the Session Overview page.
    
    session keys:
        summary, reports, amateur_result, expert_mse, expert_mse_std,
        amateur_meta, train_meta, selected_lap
    """
    summary = session.get("summary", {})
    reports = session.get("reports", {})
    amateur_result = session.get("amateur_result", {})
    expert_mse_mean = session.get("expert_mse", 0.0)
    expert_mse_std = session.get("expert_mse_std", 0.0)
    amateur_meta = session.get("amateur_meta")
    train_meta = session.get("train_meta")
    selected_lap = session.get("selected_lap", 1)

    driver_name = summary.get("driver", "Unknown Driver")
    n_laps = summary.get("n_laps", 0)
    laps_data = summary.get("laps", [])

    # ── Driver & Session Info ────────────────────────────────────────────────
    section_header("Session Overview", "🏁")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card overview-info-card">
            <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; text-transform: uppercase;
                        letter-spacing: 1.5px; margin-bottom: 8px;">DRIVER</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['text_primary']};">{driver_name}</div>
            <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-top: 4px;">
                Uploaded telemetry profile</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card overview-info-card">
            <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; text-transform: uppercase;
                        letter-spacing: 1.5px; margin-bottom: 8px;">CIRCUIT</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['text_primary']};">
                {TRACK_INFO['short_name']}</div>
            <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-top: 4px;">
                {TRACK_INFO['length_km']} km • {TRACK_INFO['corners']} corners</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card overview-info-card">
            <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; text-transform: uppercase;
                        letter-spacing: 1.5px; margin-bottom: 8px;">CAR</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {COLORS['text_primary']};">
                {CAR_INFO['name']}</div>
            <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-top: 4px;">
                {CAR_INFO['class']} • {CAR_INFO['power']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    if laps_data:
        lap_times = [l.get("lap_time_s", 0) for l in laps_data]
        mse_values = [l.get("mse_normalised", l.get("mse", 0)) for l in laps_data]
        severities = [l.get("overall_severity", 0) for l in laps_data]
        time_losses = [l.get("estimated_time_loss_s", 0) for l in laps_data]

        best_lap_time = min(lap_times)
        best_lap_idx = lap_times.index(best_lap_time)
        best_lap_str = laps_data[best_lap_idx].get("lap_time", f"{best_lap_time:.3f}")

        # Expert best for delta
        if train_meta is not None and "lap_time_s" in train_meta.columns:
            expert_best = train_meta["lap_time_s"].min()
            delta = best_lap_time - expert_best
            delta_str = f"+{delta:.3f}s" if delta > 0 else f"{delta:.3f}s"
        else:
            delta_str = "N/A"
            delta = 0

        # Consistency = std of lap times
        consistency = np.std(lap_times)
        consistency_str = f"±{consistency:.3f}s"

        # Average severity
        avg_severity = np.mean(severities) if severities else 0
        total_time_loss = sum(time_losses)

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            kpi_card("Current Lap", f"Lap {selected_lap}", icon="🏁")
        with k2:
            kpi_card("Best Lap", best_lap_str, icon="⚡")
        with k3:
            kpi_card("Delta vs Expert", delta_str,
                     delta_type="negative" if delta > 0 else "positive", icon="⏱️")
        with k4:
            kpi_card("Avg Severity", f"{avg_severity:.0%}",
                     delta_type="negative" if avg_severity > 0.5 else "positive", icon="⚠️")
        with k5:
            kpi_card("Total Time Lost", f"~{total_time_loss:.3f}s",
                     delta_type="negative", icon="⏳")

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # ── Main Circuit Insight Map ─────────────────────────────────────────
        section_header("Circuit Coaching Map", "🗺️")
        report = reports.get(selected_lap, {})
        zones = report.get("zones", [])
        lap_latlon = session.get("latlon")
        if lap_latlon is None:
            lap_latlon = session.get("train_latlon")

        if lap_latlon is not None and zones:
            selected_zone = _current_zone(zones)
            col_map, col_detail = st.columns([1.55, 0.95])
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
                    _select_zone(zones, clicked_zone)
            with col_detail:
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: {COLORS['text_secondary']}; margin-bottom: 14px;">
                    <strong>Lap {selected_lap}</strong> — {len(zones)} coaching zones detected
                </div>
                """, unsafe_allow_html=True)
                _render_zone_buttons(zones)
                _render_zone_detail(selected_zone)
        else:
            st.info("No coaching zones are available yet for the circuit map.")

        # ── Per-Lap Summary Table ────────────────────────────────────────────
        section_header("Lap Breakdown", "📋")

        for lap_data in laps_data:
            lap_num = lap_data.get("lap_number", lap_data.get("lap", 0))
            lap_time = lap_data.get("lap_time", "")
            mse = lap_data.get("mse_normalised", lap_data.get("mse", 0))
            sev = lap_data.get("overall_severity", 0)
            tloss = lap_data.get("estimated_time_loss_s", 0)
            n_zones = lap_data.get("n_coaching_zones", 0)

            sev_color = get_severity_color(sev)

            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 24px; padding: 12px 16px;
                        background: {COLORS['bg_card']}; border-radius: 8px; margin-bottom: 8px;
                        border-left: 3px solid {sev_color};">
                <div style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 1.1rem;
                            color: {COLORS['text_primary']}; min-width: 60px;">Lap {lap_num}</div>
                <div style="font-family: 'JetBrains Mono'; color: {COLORS['text_primary']};
                            min-width: 80px;">{lap_time}</div>
                <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; min-width: 100px;">
                    MSE: {mse:.6f}</div>
                <div style="font-size: 0.8rem; color: {sev_color}; min-width: 80px;">
                    Severity: {sev:.0%}</div>
                <div style="font-size: 0.8rem; color: {COLORS['bad']}; min-width: 100px;">
                    Lost: ~{tloss:.3f}s</div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">
                    {n_zones} zones</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No lap data available. Run the coaching pipeline first.")


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


def _select_zone(zones: list, zone_id: int):
    for zone in zones:
        if int(zone.get("zone_id", -1)) == int(zone_id):
            previous_zone = st.session_state.get("selected_zone")
            st.session_state["selected_zone"] = int(zone_id)
            st.session_state["selected_zone_data"] = zone
            if previous_zone is None or int(previous_zone) != int(zone_id):
                st.rerun()
            return


def _current_zone(zones: list):
    selected_zone = st.session_state.get("selected_zone")
    if selected_zone is not None:
        for zone in zones:
            if int(zone.get("zone_id", -1)) == int(selected_zone):
                return zone
    priority = max(zones, key=lambda z: z.get("estimated_time_loss_s", z.get("severity_score", 0)))
    st.session_state["selected_zone"] = priority.get("zone_id")
    st.session_state["selected_zone_data"] = priority
    return priority


def _render_zone_buttons(zones: list):
    selected_zone = st.session_state.get("selected_zone")
    cols = st.columns(min(len(zones), 5))
    for idx, zone in enumerate(zones):
        zone_id = int(zone.get("zone_id", idx + 1))
        with cols[idx % len(cols)]:
            is_active = selected_zone is not None and int(selected_zone) == zone_id
            if st.button(
                f"Zone {zone_id}",
                key=f"overview_zone_button_{zone_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _select_zone(zones, zone_id)


def _render_zone_detail(zone: dict):
    feedback = zone.get("llm_feedback") or zone.get("template_feedback") or "Open this zone for detailed coaching."
    feedback = html.escape(_clean_feedback_text(feedback))
    severity = zone.get("severity_score", 0)
    time_loss = zone.get("estimated_time_loss_s", 0)
    color = get_severity_color(severity)
    st.markdown(f"""
    <div class="overview-zone-card" style="border-top-color: {color};">
        <div class="overview-zone-kicker">Selected coaching zone</div>
        <div class="overview-zone-title">Zone {zone.get('zone_id', 0)} · {zone.get('lap_pct_start', 0):.1f}% → {zone.get('lap_pct_end', 0):.1f}%</div>
        <div class="overview-zone-metrics">
            <div class="overview-zone-metric"><span class="overview-zone-metric-value" style="color: {COLORS['bad']};">~{time_loss:.3f}s</span><span class="overview-zone-metric-label">Time Lost</span></div>
            <div class="overview-zone-metric"><span class="overview-zone-metric-value">{severity:.0%}</span><span class="overview-zone-metric-label">Priority</span></div>
        </div>
    </div>
    <div class="ai-zone-panel">
        <div class="ai-zone-kicker">AI race engineer explanation</div>
        <div class="ai-zone-copy">{feedback}</div>
    </div>
    """, unsafe_allow_html=True)

    dom_channels = zone.get("dominant_channels", [])
    sig = [d for d in dom_channels if d.get("severity", 0) > 0.1][:4]
    if sig:
        st.markdown(f"<div style='margin-top: 12px; font-size: 0.85rem; color: {COLORS['text_secondary']}; font-weight: 600;'>KEY DEVIATIONS</div>", unsafe_allow_html=True)
        for d in sig:
            arrow = "▲" if d["direction"] == "over" else "▼"
            value_color = COLORS["bad"] if d["direction"] == "over" else COLORS["medium"]
            st.markdown(f"""
            <div class="deviation-row">
                <span class="deviation-channel">{arrow} {d['channel']}</span>
                <span class="deviation-value" style="color: {value_color};">
                    {abs(d['signed_mean']):.1f} {d.get('unit', '')}
                </span>
            </div>
            """, unsafe_allow_html=True)


def _clean_feedback_text(feedback) -> str:
    text = " ".join(str(feedback).replace("\n", " ").split())
    quote_pairs = [('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’')]
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in quote_pairs:
            if text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
                break
    return text
