"""
Motorsport dark theme — colors, Plotly templates, and custom CSS.
"""

# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    # Backgrounds
    "bg_primary":     "#0E1117",
    "bg_card":        "#161B22",
    "bg_card_hover":  "#1C2333",
    "bg_sidebar":     "#0D1117",

    # Accents
    "accent":         "#FF1E1E",   # Racing red
    "accent_light":   "#FF4444",
    "accent_glow":    "rgba(255,30,30,0.15)",

    # Performance
    "good":           "#00E676",
    "medium":         "#FFD600",
    "bad":            "#FF1744",

    # Telemetry channels
    "expert":         "#00B0FF",
    "amateur":        "#FF9100",
    "delta":          "#E040FB",

    # Severity
    "severity_low":   "#00E676",
    "severity_mid":   "#FFD600",
    "severity_high":  "#FF1744",

    # Text
    "text_primary":   "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted":     "#484F58",
    "text_accent":    "#FF6B6B",

    # Borders
    "border":         "#21262D",
    "border_accent":  "#FF1E1E",

    # Chart
    "grid":           "#21262D",
    "chart_bg":       "#0D1117",
}

# ─── Channel Colors ──────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "Speed":              {"expert": "#00B0FF", "amateur": "#FF9100"},
    "Throttle":           {"expert": "#00E676", "amateur": "#FF6D00"},
    "Brake":              {"expert": "#FF1744", "amateur": "#D500F9"},
    "RPM":                {"expert": "#448AFF", "amateur": "#FF9100"},
    "SteeringWheelAngle": {"expert": "#00BFA5", "amateur": "#FFD600"},
    "Gear":               {"expert": "#7C4DFF", "amateur": "#FF6E40"},
    "LatAccel":           {"expert": "#18FFFF", "amateur": "#FF4081"},
    "LongAccel":          {"expert": "#69F0AE", "amateur": "#FF6E40"},
    "YawRate":            {"expert": "#B388FF", "amateur": "#FF8A65"},
}

CHANNEL_DISPLAY_NAMES = {
    "Speed":              "Speed",
    "Throttle":           "Throttle",
    "Brake":              "Brake",
    "RPM":                "RPM",
    "SteeringWheelAngle": "Steering",
    "Gear":               "Gear",
    "LatAccel":           "Lat G",
    "LongAccel":          "Long G",
    "YawRate":            "Yaw Rate",
}


# ─── Plotly Layout Template ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["chart_bg"],
    plot_bgcolor=COLORS["chart_bg"],
    font=dict(family="Inter, SF Pro Display, -apple-system, sans-serif",
              color=COLORS["text_primary"], size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        showgrid=True,
        gridwidth=1,
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        showgrid=True,
        gridwidth=1,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color=COLORS["text_secondary"]),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor=COLORS["bg_card"],
        font_size=12,
        font_family="Inter, monospace",
        bordercolor=COLORS["border"],
    ),
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #0D1117 100%);
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0F15 0%, #111821 100%);
    border-right: 1px solid #21262D;
}

section[data-testid="stSidebar"] [data-testid="stButton"] button {
    border-radius: 8px;
}

.sidebar-panel-note {
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    background: rgba(255,255,255,0.035);
    color: #8B949E;
    font-size: 0.78rem;
    line-height: 1.5;
}

section[data-testid="stSidebar"] .stRadio > label {
    color: #8B949E !important;
    font-weight: 500;
}

/* ── KPI Cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    border-color: #FF1E1E;
    box-shadow: 0 0 20px rgba(255, 30, 30, 0.1);
    transform: translateY(-2px);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF1E1E, #FF4444);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.overview-info-card {
    min-height: 126px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #E6EDF3;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-label {
    font-size: 0.75rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
    font-weight: 600;
}

.kpi-delta {
    font-size: 0.85rem;
    margin-top: 4px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-delta.positive { color: #00E676; }
.kpi-delta.negative { color: #FF1744; }
.kpi-delta.neutral  { color: #FFD600; }

/* ── Section Headers ───────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid #21262D;
}

.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0;
}

.section-header .accent-bar {
    width: 4px;
    height: 28px;
    background: linear-gradient(180deg, #FF1E1E, #FF4444);
    border-radius: 2px;
}

/* ── Coaching Cards ────────────────────────────────────────────────────── */
.coaching-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    border-left: 4px solid #FF1E1E;
    transition: all 0.3s ease;
}

.coaching-card:hover {
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    transform: translateY(-1px);
}

.coaching-card.severity-low    { border-left-color: #00E676; }
.coaching-card.severity-medium { border-left-color: #FFD600; }
.coaching-card.severity-high   { border-left-color: #FF1744; }

.coaching-card .zone-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.coaching-card .zone-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E6EDF3;
}

.coaching-card .zone-severity {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.severity-badge-low    { background: rgba(0,230,118,0.15); color: #00E676; }
.severity-badge-medium { background: rgba(255,214,0,0.15);  color: #FFD600; }
.severity-badge-high   { background: rgba(255,23,68,0.15);  color: #FF1744; }

.coaching-card .feedback-text {
    color: #C9D1D9;
    line-height: 1.6;
    font-size: 0.95rem;
}

.coaching-card .metric-row {
    display: flex;
    gap: 24px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #21262D;
}

.coaching-card .metric-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.coaching-card .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    color: #E6EDF3;
}

.coaching-card .metric-label {
    font-size: 0.7rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Severity Bar ──────────────────────────────────────────────────────── */
.severity-bar {
    width: 100%;
    height: 6px;
    background: #21262D;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 8px;
}

.severity-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* ── Tab Styling ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #21262D;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: rgba(255, 30, 30, 0.1) !important;
    border-bottom: 2px solid #FF1E1E !important;
}

/* ── Top Navigation Buttons ───────────────────────────────────────────── */
div[data-testid="column"] [data-testid="stButton"] button {
    min-height: 42px;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.01em;
}

div[data-testid="column"] [data-testid="stButton"] button[kind="primary"] {
    box-shadow: 0 0 24px rgba(255, 30, 30, 0.16);
}

/* ── Overview Circuit Coach Card ──────────────────────────────────────── */
.overview-zone-card {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #21262D;
    border-top: 4px solid #FF1744;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.22);
}

.overview-zone-kicker {
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-size: 0.7rem;
    font-weight: 800;
}

.overview-zone-title {
    color: #E6EDF3;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 8px;
}

.overview-zone-copy {
    color: #C9D1D9;
    line-height: 1.65;
    font-size: 0.92rem;
    margin-top: 12px;
}

.overview-zone-hint {
    color: #8B949E;
    border-top: 1px solid #21262D;
    margin-top: 16px;
    padding-top: 12px;
    font-size: 0.78rem;
}

.ai-zone-panel {
    margin-top: 14px;
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid rgba(0,176,255,0.22);
    background: linear-gradient(135deg, rgba(0,176,255,0.08), rgba(22,27,34,0.95));
}

.ai-zone-kicker {
    color: #00B0FF;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.68rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.ai-zone-copy {
    color: #C9D1D9;
    line-height: 1.62;
    font-size: 0.9rem;
}

/* ── Corner Badge ──────────────────────────────────────────────────────── */
.corner-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.corner-badge:hover {
    transform: scale(1.05);
}

.corner-badge.good   { background: rgba(0,230,118,0.15);  color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
.corner-badge.medium { background: rgba(255,214,0,0.15);  color: #FFD600; border: 1px solid rgba(255,214,0,0.3);  }
.corner-badge.bad    { background: rgba(255,23,68,0.15);   color: #FF1744; border: 1px solid rgba(255,23,68,0.3);  }

/* ── Header Bar ────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 2px solid #21262D;
    margin-bottom: 24px;
}

.app-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF1E1E 0%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.app-subtitle {
    font-size: 0.85rem;
    color: #8B949E;
    font-weight: 400;
}

/* ── Channel Deviation Rows ────────────────────────────────────────────── */
.deviation-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
    background: rgba(22, 27, 34, 0.5);
}

.deviation-channel {
    font-weight: 600;
    font-size: 0.85rem;
    color: #E6EDF3;
    min-width: 80px;
}

.deviation-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Expander Styling ──────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #E6EDF3 !important;
    background: #161B22 !important;
    border-radius: 8px !important;
}

/* ── Hide Streamlit Branding ───────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Keep the header collapse button visible for sidebar toggle */
header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}
header[data-testid="stHeader"] .stDeployButton,
header[data-testid="stHeader"] .stToolbar {
    visibility: hidden;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: #21262D; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #30363D; }

/* ── Sidebar Brand ─────────────────────────────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 20px 0;
}
.sidebar-brand-icon {
    font-size: 2.2rem;
    margin-bottom: 4px;
}
.sidebar-brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF1E1E, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: #8B949E;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.sidebar-footer {
    font-size: 0.7rem;
    color: #484F58;
    text-align: center;
    padding: 8px;
}

/* ── Sidebar Nav Buttons ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #21262D !important;
    color: #8B949E !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255, 30, 30, 0.06) !important;
    border-color: #FF1E1E !important;
    color: #E6EDF3 !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, rgba(255,30,30,0.15), rgba(255,68,68,0.1)) !important;
    border: 1px solid rgba(255,30,30,0.4) !important;
    color: #FF6B6B !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    text-align: left !important;
    box-shadow: 0 0 12px rgba(255,30,30,0.08) !important;
}

/* ── Landing Page ──────────────────────────────────────────────────────── */
.landing-hero {
    text-align: center;
    padding: 24px 0 8px 0;
}
.landing-icon {
    font-size: 4.5rem;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 20px rgba(255,30,30,0.3));
}
.landing-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #FF1E1E 0%, #FF6B6B 50%, #FF1E1E 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.landing-subtitle {
    font-size: 1.15rem;
    color: #8B949E;
    margin-top: 12px;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Upload Zone ───────────────────────────────────────────────────────── */
.landing-upload-zone {
    margin-top: 28px;
    padding: 28px 32px 12px 32px;
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 2px dashed #30363D;
    border-radius: 16px;
    text-align: center;
    transition: all 0.3s ease;
}
.landing-upload-zone:hover {
    border-color: #FF1E1E;
    box-shadow: 0 0 30px rgba(255,30,30,0.08);
}
.upload-zone-icon {
    font-size: 2.5rem;
    margin-bottom: 8px;
}
.upload-zone-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 8px;
}
.upload-zone-desc {
    font-size: 0.85rem;
    color: #8B949E;
    line-height: 1.6;
    margin-bottom: 4px;
}

/* ── How It Works ──────────────────────────────────────────────────────── */
.how-it-works-label {
    text-align: center;
    font-size: 0.75rem;
    color: #484F58;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 20px;
}
.how-it-works-grid {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.how-step {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 24px 28px;
    min-width: 160px;
    text-align: center;
    transition: all 0.3s ease;
}
.how-step:hover {
    border-color: #FF1E1E;
    box-shadow: 0 4px 20px rgba(255,30,30,0.1);
    transform: translateY(-3px);
}
.how-step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #FF1E1E, #FF4444);
    color: #fff;
    font-weight: 800;
    font-size: 0.9rem;
    border-radius: 8px;
    margin-bottom: 10px;
}
.how-step-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 4px;
}
.how-step-desc {
    font-size: 0.75rem;
    color: #8B949E;
}
.how-step-arrow {
    font-size: 1.5rem;
    color: #30363D;
    font-weight: 300;
}

/* ── Processing Overlay ────────────────────────────────────────────────── */
.processing-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(14, 17, 23, 0.92);
    backdrop-filter: blur(8px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
}
.processing-content {
    text-align: center;
    max-width: 440px;
    padding: 48px;
}
.processing-spinner {
    width: 56px;
    height: 56px;
    border: 3px solid #21262D;
    border-top: 3px solid #FF1E1E;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 28px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.processing-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #E6EDF3;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}
.processing-subtitle {
    font-size: 0.95rem;
    color: #8B949E;
    line-height: 1.6;
    margin-bottom: 28px;
}
.processing-steps {
    text-align: left;
    display: inline-block;
}
.processing-step {
    font-size: 0.85rem;
    color: #8B949E;
    padding: 6px 0;
    border-bottom: 1px solid #161B22;
}
.processing-step:last-child {
    border-bottom: none;
}
</style>
"""
