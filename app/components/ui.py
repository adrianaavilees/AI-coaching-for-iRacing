"""
Reusable UI components for the coaching platform.

KPI cards, severity badges, section headers, coaching cards, etc.
"""

import streamlit as st
from app.theme import COLORS


def render_header():
    """Render the application header bar."""
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">🏎️ AI Race Engineer</div>
            <div class="app-subtitle">Intelligent Telemetry Coaching • iRacing GT3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, icon: str = ""):
    """Render a styled section header."""
    st.markdown(f"""
    <div class="section-header">
        <div class="accent-bar"></div>
        <h2>{icon} {title}</h2>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral",
             icon: str = ""):
    """
    Render a single KPI metric card.
    
    delta_type: "positive", "negative", or "neutral"
    """
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{icon} {value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def severity_badge(severity: float, size: str = "normal"):
    """Return HTML for a colored severity badge."""
    if severity < 0.33:
        cls, label = "severity-badge-low", "LOW"
    elif severity < 0.66:
        cls, label = "severity-badge-medium", "MEDIUM"
    else:
        cls, label = "severity-badge-high", "HIGH"
    
    return f'<span class="zone-severity {cls}">{label} ({severity:.0%})</span>'


def severity_bar(severity: float):
    """Render a horizontal severity bar."""
    if severity < 0.33:
        color = COLORS["severity_low"]
    elif severity < 0.66:
        color = COLORS["severity_mid"]
    else:
        color = COLORS["severity_high"]
    
    pct = min(severity * 100, 100)
    st.markdown(f"""
    <div class="severity-bar">
        <div class="severity-fill" style="width: {pct}%; background: {color};"></div>
    </div>
    """, unsafe_allow_html=True)


def coaching_card(zone_id: int, lap_pct_start: float, lap_pct_end: float,
                  severity: float, time_loss: float, feedback_text: str,
                  dominant_channels: list = None):
    """Render a styled coaching feedback card for a zone."""
    sev_class = "severity-low" if severity < 0.33 else ("severity-medium" if severity < 0.66 else "severity-high")
    badge = severity_badge(severity)
    
    # Channel deviations
    channels_html = ""
    if dominant_channels:
        sig = [d for d in dominant_channels if d.get("severity", 0) > 0.1][:3]
        if sig:
            rows = []
            for d in sig:
                arrow = "▲" if d["direction"] == "over" else "▼"
                color = COLORS["bad"] if d["direction"] == "over" else COLORS["medium"]
                rows.append(
                    f'<div class="deviation-row">'
                    f'<span class="deviation-channel">{arrow} {d["channel"]}</span>'
                    f'<span class="deviation-value" style="color: {color};">'
                    f'{abs(d["signed_mean"]):.1f} {d.get("unit", "")}'
                    f'</span></div>'
                )
            channels_html = "".join(rows)

    card_html = (
        f'<div class="coaching-card {sev_class}">'
        f'<div class="zone-header">'
        f'<span class="zone-title">Zone {zone_id} — {lap_pct_start:.1f}% → {lap_pct_end:.1f}%</span>'
        f'{badge}'
        f'</div>'
        f'{channels_html}'
        f'<div class="feedback-text">{feedback_text}</div>'
        f'<div class="metric-row">'
        f'<div class="metric-item"><span class="metric-value" style="color: {COLORS["bad"]};">~{time_loss:.3f}s</span><span class="metric-label">Time Lost</span></div>'
        f'<div class="metric-item"><span class="metric-value">{severity:.0%}</span><span class="metric-label">Severity</span></div>'
        f'<div class="metric-item"><span class="metric-value">{lap_pct_end - lap_pct_start:.1f}%</span><span class="metric-label">Lap Span</span></div>'
        f'</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def corner_badge(zone_id: int, severity: float):
    """Return HTML for a clickable corner badge."""
    if severity < 0.33:
        cls = "good"
    elif severity < 0.66:
        cls = "medium"
    else:
        cls = "bad"
    return f'<span class="corner-badge {cls}">Zone {zone_id}</span>'


def metric_row(items: list):
    """
    Render a row of small metric items.
    items = [{"label": "...", "value": "...", "color": "#..."}, ...]
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            color = item.get("color", COLORS["text_primary"])
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
                            font-weight: 700; color: {color};">{item['value']}</div>
                <div style="font-size: 0.7rem; color: {COLORS['text_secondary']};
                            text-transform: uppercase; letter-spacing: 1px;
                            margin-top: 4px;">{item['label']}</div>
            </div>
            """, unsafe_allow_html=True)


def empty_state(message: str, icon: str = "📊"):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div style="text-align: center; padding: 60px 20px; color: {COLORS['text_muted']};">
        <div style="font-size: 3rem; margin-bottom: 16px;">{icon}</div>
        <div style="font-size: 1.1rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
