"""
🏎️ AI Race Engineer — Intelligent Telemetry Coaching for iRacing GT3

Main Streamlit application entry point.
Run with: streamlit run app/main.py

Architecture: Progressive Loading
    Phase 1 (instant):     Parse CSV → show metadata + raw telemetry
    Phase 2 (2-5s):        Autoencoder → show anomaly analysis
    Phase 3 (background):  LLM feedback → enrich selected-zone coaching
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np

from app.theme import CUSTOM_CSS, COLORS
from app.components.ui import render_header
from app.pipeline import (
    run_phase1, run_phase2, run_phase3, start_phase3_background,
    preload_resources, parse_filename,
)
from app.data_loader import TRACK_INFO, CAR_INFO
from app.pages import session_overview, telemetry_analysis, session_report, advanced


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Race Engineer • iRacing GT3",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_header()

# ─── Preload heavy resources at startup ──────────────────────────────────────
if "_resources_preloaded" not in st.session_state:
    preload_resources()
    st.session_state["_resources_preloaded"] = True


# ─── Navigation ──────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("📊", "Overview",   "session_overview"),
    ("📡", "Telemetry",  "telemetry_analysis"),
    ("📋", "Report",     "session_report"),
    ("⚙️", "Advanced",   "advanced"),
]

PAGE_MAP = {
    "session_overview":    session_overview,
    "telemetry_analysis":  telemetry_analysis,
    "session_report":      session_report,
    "advanced":            advanced,
}


# ─── Check Phase 3 completion (every rerun) ─────────────────────────────────

def _check_and_apply_phase3():
    """If background Phase 3 has finished, merge LLM results into session_data."""
    sd = st.session_state.get("session_data")
    if sd is None:
        return

    if sd.get("_phase3_done") and sd.get("_phase3_result"):
        report = sd["_phase3_result"]
        sd["reports"] = {1: report}

        # Update summary with LLM-derived data
        if sd["summary"]["laps"]:
            sd["summary"]["laps"][0]["n_coaching_zones"] = report.get("n_zones", 0)
            sd["summary"]["laps"][0]["overall_severity"] = report.get("overall_severity", 0)
            sd["summary"]["laps"][0]["estimated_time_loss_s"] = report.get("estimated_time_loss_s", 0)

        sd["_phase3_applied"] = True
        sd["_phase3_result"] = None  # free memory


_check_and_apply_phase3()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🏎️</div>
        <div class="sidebar-brand-title">AI Race Engineer</div>
        <div class="sidebar-brand-sub">Telemetry Coaching Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown("##### 📂 Upload Telemetry")
    st.markdown(
        f"<div style='font-size: 0.75rem; color: {COLORS['text_muted']}; margin-bottom: 8px;'>"
        "Upload one or more raw Garage61 CSV laps to analyse.</div>",
        unsafe_allow_html=True,
    )

    uploaded_files_sidebar = st.file_uploader(
        "Upload your lap(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="lap_upload_sidebar",
        label_visibility="collapsed",
    )

    if uploaded_files_sidebar:
        st.markdown(
            f"<div style='font-size: 0.8rem; color: {COLORS['text_secondary']};'>"
            f"📄 {len(uploaded_files_sidebar)} file(s) selected</div>",
            unsafe_allow_html=True,
        )

    analyse_btn_sidebar = st.button(
        "🚀 Analyse Laps",
        use_container_width=True,
        disabled=not uploaded_files_sidebar,
        type="primary",
        key="analyse_sidebar",
    )

    st.divider()

    # ── Session controls (navigation lives above the main page) ──────────────
    sd = st.session_state.get("session_data")
    has_analysis = sd is not None and "amateur_result" in sd

    if has_analysis:
        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "session_overview"

        # ── Lap Selector ─────────────────────────────────────────────────────
        n_laps = sd["summary"].get("n_laps", 1)
        laps_info = sd["summary"].get("laps", [])

        if n_laps > 1:
            st.markdown("##### 🏁 Lap Selection")
            selected_lap = st.selectbox(
                "Active lap",
                range(1, n_laps + 1),
                format_func=lambda l: (
                    f"Lap {l} — {laps_info[l-1].get('lap_time', '')}"
                    if l <= len(laps_info) else f"Lap {l}"
                ),
                key="lap_select",
            )
        else:
            selected_lap = 1

        st.divider()

        st.markdown(f"""
        <div class="sidebar-panel-note">
            <strong>Session ready</strong><br>
            Use the tabs above the dashboard to move between views. The sidebar is only for uploads and session context.
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_lap = 1

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sidebar-footer">
        <div>🏎️ {CAR_INFO['name']} • {CAR_INFO['class']}</div>
        <div>🏁 {TRACK_INFO['short_name']} • {TRACK_INFO['config']}</div>
        <div style="margin-top: 8px;">LSTM Autoencoder + Groq LLM</div>
        <div>TFG — AI Coaching for iRacing</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Handle Upload → Progressive Pipeline ───────────────────────────────────

def _run_progressive_pipeline(uploaded_files):
    """Execute Phases 1→2 synchronously, then launch Phase 3 in background."""
    if len(uploaded_files) != 1:
        st.warning("Multi-lap progressive loading: processing first file.")

    uf = uploaded_files[0]

    # ── Phase 1: Instant parse ───────────────────────────────────────────
    phase1 = run_phase1(uf)

    # ── Phase 2: Autoencoder inference ───────────────────────────────────
    session_data = run_phase2(phase1)

    # Store in session state immediately — UI can render
    st.session_state["session_data"] = session_data
    st.session_state["active_page"] = "session_overview"

    # ── Phase 3: LLM feedback in background ──────────────────────────────
    start_phase3_background(session_data)


def _show_processing_overlay(placeholder, title="Reading telemetry and running analysis..."):
    placeholder.markdown(f"""
    <div class="processing-overlay">
        <div class="processing-content">
            <div class="processing-spinner"></div>
            <div class="processing-title">{title}</div>
            <div class="processing-subtitle">
                Parsing the Garage61 CSV, rebuilding the lap trace, and detecting coaching zones.
            </div>
            <div class="processing-steps">
                <div class="processing-step">Reading telemetry channels</div>
                <div class="processing-step">Running autoencoder analysis</div>
                <div class="processing-step">Preparing circuit coaching map</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_top_navigation(sd: dict):
    """Render primary navigation across the top of the app."""
    cols = st.columns(len(NAV_ITEMS))
    for col, (icon, label, key) in zip(cols, NAV_ITEMS):
        is_active = st.session_state.get("active_page", "session_overview") == key
        with col:
            if st.button(f"{icon} {label}", key=f"top_nav_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["active_page"] = key
                st.rerun()


if analyse_btn_sidebar and uploaded_files_sidebar:
    # Show a brief progress bar for phases 1-2
    progress_placeholder = st.empty()
    _show_processing_overlay(progress_placeholder)

    try:
        _run_progressive_pipeline(uploaded_files_sidebar)
        progress_placeholder.empty()
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ Error processing telemetry: {e}")
        st.stop()
    st.rerun()


# ─── Main Content ────────────────────────────────────────────────────────────

sd = st.session_state.get("session_data")

if sd is None or "amateur_result" not in sd:
    # ── Landing Page ─────────────────────────────────────────────────────
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown(f"""
        <div class="landing-hero">
            <div class="landing-icon">🏎️</div>
            <div class="landing-title">AI Race Engineer</div>
            <div class="landing-subtitle">
                Intelligent telemetry coaching powered by Deep Learning
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="landing-upload-zone">
            <div class="upload-zone-icon">📂</div>
            <div class="upload-zone-title">Upload your lap to get started</div>
            <div class="upload-zone-desc">
                Drop your Garage61 CSV telemetry file below or click to browse.
                <br>The AI will analyse your driving and provide personalised coaching.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files_main = st.file_uploader(
            "Upload lap CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="lap_upload_main",
            label_visibility="collapsed",
        )

        if uploaded_files_main:
            st.markdown(
                f"<div style='text-align:center; font-size: 0.9rem; color: {COLORS['text_secondary']}; "
                f"margin: 8px 0;'>📄 {len(uploaded_files_main)} file(s) ready</div>",
                unsafe_allow_html=True,
            )
            if st.button("🚀 Analyse Laps", use_container_width=True, type="primary",
                         key="analyse_main"):
                progress_placeholder = st.empty()
                _show_processing_overlay(progress_placeholder)
                try:
                    _run_progressive_pipeline(uploaded_files_main)
                    progress_placeholder.empty()
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"❌ Error processing telemetry: {e}")
                    st.stop()
                st.rerun()

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="how-it-works-label">HOW IT WORKS</div>
        <div class="how-it-works-grid">
            <div class="how-step">
                <div class="how-step-number">1</div>
                <div class="how-step-title">Upload</div>
                <div class="how-step-desc">Raw CSV from Garage61</div>
            </div>
            <div class="how-step-arrow">→</div>
            <div class="how-step">
                <div class="how-step-number">2</div>
                <div class="how-step-title">Analyse</div>
                <div class="how-step-desc">LSTM Autoencoder detects anomalies</div>
            </div>
            <div class="how-step-arrow">→</div>
            <div class="how-step">
                <div class="how-step-number">3</div>
                <div class="how-step-title">Coach</div>
                <div class="how-step-desc">AI generates actionable feedback</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Dashboard with progressive rendering ─────────────────────────────
    sd["selected_lap"] = selected_lap

    _render_top_navigation(sd)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Phase 3 status
    phase3_applied = sd.get("_phase3_applied", False)
    phase3_finished = sd.get("_phase3_done", False)
    phase3_running = not phase3_finished and not phase3_applied

    # ── Status banner (non-blocking) ─────────────────────────────────────
    if phase3_running:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 16px;
                    background: linear-gradient(90deg, rgba(255,214,0,0.08), rgba(255,214,0,0.03));
                    border: 1px solid rgba(255,214,0,0.2); border-radius: 8px; margin-bottom: 16px;">
            <div class="processing-spinner" style="width:20px; height:20px; border-width:2px;"></div>
            <div style="font-size: 0.85rem; color: {COLORS['text_secondary']};">
                <strong style="color: #FFD600;">AI feedback generating...</strong>
                All other pages are fully available. This banner disappears when ready.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Render active page FIRST (never block) ───────────────────────────
    active_key = st.session_state.get("active_page", "session_overview")
    if active_key not in PAGE_MAP:
        active_key = "session_overview"
        st.session_state["active_page"] = active_key
    active_page = PAGE_MAP.get(active_key)
    if active_page:
        active_page.render(sd)
    else:
        st.error("Page not found.")

    # ── After rendering: check if Phase 3 completed during this run ──────
    if phase3_finished and not phase3_applied:
        _check_and_apply_phase3()
        st.rerun()

    # ── If still running: schedule a delayed rerun to poll ────────────────
    if phase3_running:
        time.sleep(3)
        if sd.get("_phase3_done"):
            _check_and_apply_phase3()
        st.rerun()
