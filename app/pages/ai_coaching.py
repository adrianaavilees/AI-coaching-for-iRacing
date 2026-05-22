"""
Page 4: AI Coaching Panel

Renders Groq/LLM feedback as styled coaching cards.
"""

import streamlit as st
import numpy as np

from app.components.ui import section_header, coaching_card, severity_bar
from app.theme import COLORS
from app.data_loader import get_severity_color


def render(session: dict):
    """Render the AI Coaching Panel page."""
    reports = session.get("reports", {})
    selected_lap = session.get("selected_lap", 1)
    summary = session.get("summary", {})

    section_header("AI Race Engineer", "🤖")

    report = reports.get(selected_lap, {})
    zones = report.get("zones", [])

    if not report:
        st.info("No coaching feedback available for this lap.")
        return

    # ── Overall Summary ──────────────────────────────────────────────────────
    overall_severity = report.get("overall_severity", 0)
    total_time_loss = report.get("estimated_time_loss_s", 0)
    n_zones = report.get("n_zones", len(zones))

    # Summary card
    summary_llm = report.get("summary_llm", "")
    summary_template = report.get("summary_template", "")
    summary_text = summary_llm or summary_template

    sev_color = get_severity_color(overall_severity)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_card_hover']} 100%);
                border: 1px solid {COLORS['border']}; border-radius: 16px; padding: 28px;
                border-top: 3px solid {sev_color}; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['text_primary']};">
                    📋 Lap {selected_lap} — Coaching Summary
                </div>
                <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-top: 4px;">
                    {n_zones} zones analysed • Overall severity: {overall_severity:.0%}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; color: {COLORS['bad']};">
                    ~{total_time_loss:.3f}s
                </div>
                <div style="font-size: 0.7rem; color: {COLORS['text_muted']}; text-transform: uppercase;">
                    Est. Time Lost
                </div>
            </div>
        </div>
        <div style="color: {COLORS['text_primary']}; line-height: 1.7; font-size: 0.95rem;
                    padding: 16px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            {_format_summary(summary_text)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Zone-by-Zone Feedback ────────────────────────────────────────────────
    section_header("Zone-by-Zone Analysis", "🔎")

    # Filter / sort options
    sort_by = st.selectbox("Sort zones by", ["Track Position", "Priority", "Time Lost"],
                           key="coaching_sort")

    # Apply sorting
    filtered_zones = list(zones)

    if sort_by == "Priority":
        filtered_zones.sort(key=lambda z: z.get("severity_score", 0), reverse=True)
    elif sort_by == "Time Lost":
        filtered_zones.sort(key=lambda z: z.get("estimated_time_loss_s", 0), reverse=True)
    # else: Track Position (default from pipeline, already sorted)

    if not filtered_zones:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; color: {COLORS['text_muted']};">
            <div style="font-size: 2rem;">✅</div>
            <div style="margin-top: 8px;">No zones match the current filter.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for z in filtered_zones:
        # Prefer LLM feedback, fallback to template
        feedback = z.get("llm_feedback") or z.get("template_feedback", "")

        coaching_card(
            zone_id=z.get("zone_id", 0),
            lap_pct_start=z.get("lap_pct_start", 0),
            lap_pct_end=z.get("lap_pct_end", 0),
            severity=z.get("severity_score", 0),
            time_loss=_zone_time_loss(z),
            feedback_text=_format_feedback(feedback),
            dominant_channels=z.get("dominant_channels", []),
        )

        # Causal chains expander
        chains = z.get("causal_chains", [])
        if chains:
            with st.expander(f"🔗 Root Cause Analysis — Zone {z.get('zone_id', 0)}", expanded=False):
                for chain in chains:
                    conf = chain.get("confidence", 0)
                    conf_color = COLORS["good"] if conf > 0.7 else (
                                 COLORS["medium"] if conf > 0.4 else COLORS["text_muted"])
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 12px; padding: 8px 12px;
                                background: {COLORS['bg_card']}; border-radius: 8px; margin-bottom: 6px;">
                        <div style="font-size: 0.85rem; color: {COLORS['text_primary']}; flex: 1;">
                            → {chain.get('description', '')}
                        </div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: {conf_color};
                                    min-width: 60px; text-align: right;">
                            {conf:.0%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Feedback Source ──────────────────────────────────────────────────────
    has_llm = any(z.get("llm_feedback") for z in zones)
    source = "🤖 Groq LLM (Llama 3.3 70B)" if has_llm else "📝 Template Engine (deterministic)"
    st.markdown(f"""
    <div style="text-align: center; margin-top: 24px; padding: 8px;
                font-size: 0.75rem; color: {COLORS['text_muted']};">
        Feedback source: {source}
    </div>
    """, unsafe_allow_html=True)


def _format_feedback(text: str) -> str:
    """Clean up feedback text for HTML rendering."""
    if not text:
        return "<em>No feedback available.</em>"
    # Replace newlines with <br>, escape basic HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    text = text.replace("━", "—")
    return text


def _zone_time_loss(zone: dict) -> float:
    """Return per-zone time loss even after Phase 3 replaces the Phase 2 report."""
    direct = zone.get("estimated_time_loss_s")
    if direct is not None:
        try:
            return float(direct)
        except (TypeError, ValueError):
            pass

    text = zone.get("template_feedback", "") or ""
    marker = "Estimated time impact: ~"
    if marker in text:
        try:
            return float(text.split(marker, 1)[1].split("s", 1)[0])
        except (IndexError, ValueError):
            return 0.0
    return 0.0


def _format_summary(text: str) -> str:
    """Format summary text for display."""
    if not text:
        return "<em>No summary available.</em>"
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    text = text.replace("═", "=").replace("━", "—")
    return text
