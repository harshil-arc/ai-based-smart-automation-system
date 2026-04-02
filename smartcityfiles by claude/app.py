"""
╔══════════════════════════════════════════════════════════════════╗
║          SMART CITY MONITORING SYSTEM — app.py                  ║
║          Main Streamlit Application                             ║
║                                                                  ║
║  Run:  streamlit run app.py                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# ── Local modules ──────────────────────────────────────────────────
from weather_module import get_weather_by_city, get_weather_by_coords, CITY_DATABASE
from traffic_predictor import predict_resource_need, get_feature_importance
from dustbin_monitor import get_dustbin_status, get_all_bins, refresh_bin_levels

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Monitor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(""""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; }

/* ── Custom page title ── */
.city-header {
    background: linear-gradient(135deg, #0f1520 0%, #1a2540 50%, #0f1520 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.city-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,179,237,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.city-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #e8f4fd;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.city-subtitle {
    font-size: 13px;
    color: #6b8cae;
    font-weight: 400;
    margin: 0;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(72,199,116,0.12);
    border: 1px solid rgba(72,199,116,0.3);
    color: #48c774;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.live-dot {
    width: 6px; height: 6px;
    background: #48c774;
    border-radius: 50%;
    animation: blink 1.5s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1421;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 12px 12px 0 0;
}
.metric-card.blue::after  { background: #3b82f6; }
.metric-card.teal::after  { background: #14b8a6; }
.metric-card.amber::after { background: #f59e0b; }
.metric-card.red::after   { background: #ef4444; }
.metric-card.green::after { background: #22c55e; }
.metric-card.purple::after{ background: #a855f7; }

.metric-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #4a6a8a;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 30px;
    font-weight: 500;
    color: #e2eaf5;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-unit {
    font-size: 14px;
    color: #6b8cae;
    font-family: 'DM Sans', sans-serif;
}
.metric-sub {
    font-size: 12px;
    color: #4a6a8a;
    margin-top: 6px;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.section-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #c8dff0;
    margin: 0;
}
.section-desc {
    font-size: 12px;
    color: #4a6a8a;
    margin: 0;
}

/* ── Decision badge ── */
.decision-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
    text-align: center;
    width: 100%;
}
.decision-HIGH    { background: rgba(239,68,68,0.15);  border: 2px solid #ef4444; color: #fca5a5; }
.decision-MEDIUM  { background: rgba(245,158,11,0.15); border: 2px solid #f59e0b; color: #fcd34d; }
.decision-LOW     { background: rgba(34,197,94,0.15);  border: 2px solid #22c55e; color: #86efac; }

/* ── Dustbin status ── */
.bin-card {
    background: #0d1421;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.bin-id {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a6a8a;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.bin-location {
    font-size: 11px;
    color: #6b8cae;
    margin-bottom: 6px;
}
.bin-level-bar {
    height: 60px;
    width: 28px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px;
    margin: 0 auto 8px;
    position: relative;
    overflow: hidden;
}
.bin-fill {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    border-radius: 3px;
    transition: height 0.5s;
}
.bin-status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-FULL  { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
.status-HALF  { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
.status-EMPTY { background: rgba(34,197,94,0.2);  color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
.status-QUARTER { background: rgba(34,197,94,0.15); color: #6ee7b7; border: 1px solid rgba(34,197,94,0.3); }
.status-CRITICAL { background: rgba(239,68,68,0.3); color: #f87171; border: 2px solid #ef4444; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.07);
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    color: #6b8cae;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.1) !important;
    border-color: rgba(59,130,246,0.3) !important;
    color: #93c5fd !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0d1421 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #c8dff0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSlider > div { color: #6b8cae; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #3b82f6);
    border: none;
    border-radius: 8px;
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 24px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    transform: translateY(-1px);
}

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #93c5fd;
    margin: 12px 0;
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #fcd34d;
    margin: 12px 0;
}
.success-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #86efac;
    margin: 12px 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080d17;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown { color: #6b8cae; }

/* ── Tables / dataframes ── */
.dataframe { background: #0d1421 !important; color: #c8dff0 !important; }

/* ── Feature bar ── */
.feat-bar-wrap { margin: 4px 0; }
.feat-bar-label { font-size: 11px; color: #6b8cae; margin-bottom: 2px; }
.feat-bar-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
.feat-bar-fill  { height: 100%; border-radius: 3px; background: #3b82f6; }

/* ── Coords display ── */
.coords-box {
    background: rgba(20,184,166,0.08);
    border: 1px solid rgba(20,184,166,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #5eead4;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (defined before UI so they can be called anywhere)
# ─────────────────────────────────────────────────────────────────
def _uv_label(uv):
    if uv < 3:   return "Low"
    if uv < 6:   return "Moderate"
    if uv < 8:   return "High"
    if uv < 11:  return "Very High"
    return "Extreme"

def _heat_index(T, H):
    if T < 27: return T
    hi = (-8.784 + 1.611*T + 2.338*H - 0.146*T*H
          - 0.0123*T*T - 0.0164*H*H
          + 0.00221*T*T*H + 0.000726*T*H*H
          - 0.000003582*T*T*H*H)
    return round(hi, 1)

def _heat_index_label(hi):
    if hi < 27:  return "Comfortable",     "#22c55e"
    if hi < 32:  return "Caution",         "#84cc16"
    if hi < 38:  return "Extreme Caution", "#f59e0b"
    if hi < 45:  return "Danger",          "#ef4444"
    return "Extreme Danger", "#dc2626"

def _heat_island_risk(w):
    score = 0
    if w['temperature'] > 38: score += 3
    elif w['temperature'] > 33: score += 1
    if w['humidity'] > 70: score += 2
    if w.get('wind_speed', 10) < 5: score += 2
    if score >= 5: return "CRITICAL", "#fca5a5"
    if score >= 3: return "HIGH",     "#fcd34d"
    if score >= 1: return "MODERATE", "#86efac"
    return "LOW", "#6ee7b7"

def _flood_risk(w):
    score = 0
    if w.get('rain_mm', 0) > 20: score += 4
    elif w.get('rain_mm', 0) > 10: score += 2
    elif w.get('rain_mm', 0) > 2: score += 1
    if w['humidity'] > 85: score += 1
    if w.get('cloud_cover', 0) > 80: score += 1
    if score >= 5: return "HIGH",     "#fca5a5"
    if score >= 3: return "MODERATE", "#fcd34d"
    return "LOW",  "#86efac"

def _decision_icon(decision):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(decision, "⚪")

def _get_recommendations(decision, traffic, temp):
    recs = {
        "HIGH": [
            "Deploy additional traffic control personnel at major junctions",
            "Activate all traffic signal optimization systems",
            "Alert emergency services for potential congestion incidents",
            "Issue public advisory for alternate routes",
            f"Temperature {temp:.0f}°C — ensure road crew hydration" if temp > 35 else "Monitor road surface conditions",
        ],
        "MEDIUM": [
            "Monitor key traffic corridors via CCTV",
            "Keep standby crew on alert",
            "Adjust signal timing on secondary roads",
            "Update traffic information boards",
        ],
        "LOW": [
            "Standard monitoring protocols in effect",
            "Routine maintenance can proceed",
            "No additional resources required",
        ],
    }
    return recs.get(decision, ["Monitor situation"])


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "bins" not in st.session_state:
    st.session_state.bins = get_all_bins()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 20px;'>
        <div style='font-family: Syne, sans-serif; font-size: 20px; font-weight:800; color:#e2eaf5;'>
            🏙️ SmartCity AI
        </div>
        <div style='font-size:11px; color:#4a6a8a; margin-top:4px;'>Urban Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📍 Location Setup")

    input_mode = st.radio(
        "Input Mode",
        ["Search by City Name", "Enter Coordinates Manually"],
        label_visibility="collapsed",
    )

    city_input, lat_input, lon_input = None, None, None
    selected_state = None

    if input_mode == "Search by City Name":
        # Group cities by state for the dropdown
        state_list = sorted(set(v["state"] for v in CITY_DATABASE.values()))
        selected_state = st.selectbox("State / UT", ["All States"] + state_list)

        if selected_state == "All States":
            city_options = sorted(CITY_DATABASE.keys())
        else:
            city_options = sorted(
                k for k, v in CITY_DATABASE.items() if v["state"] == selected_state
            )

        city_input = st.selectbox("City", city_options)

        if city_input and city_input in CITY_DATABASE:
            info = CITY_DATABASE[city_input]
            st.markdown(f"""
            <div class='coords-box'>
                📌 {city_input}, {info['state']}<br>
                Lat: {info['lat']:.4f}° N &nbsp;|&nbsp; Lon: {info['lon']:.4f}° E<br>
                Elevation: {info.get('elevation', '—')} m
            </div>
            """, unsafe_allow_html=True)
    else:
        lat_input = st.number_input("Latitude",  value=24.5854, format="%.4f")
        lon_input = st.number_input("Longitude", value=73.7125, format="%.4f")
        city_input = st.text_input("Location Name (label)", value="Custom Location")

    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch Live Weather", use_container_width=True)

    if fetch_btn:
        with st.spinner("Fetching weather data…"):
            if input_mode == "Search by City Name" and city_input:
                data = get_weather_by_city(city_input)
            else:
                data = get_weather_by_coords(lat_input, lon_input, city_input)
            st.session_state.weather_data = data

    st.markdown("---")
    st.markdown("#### ⚙️ App Settings")
    auto_refresh = st.toggle("Auto-refresh bins (30s)", value=False)
    show_raw = st.toggle("Show raw data tables", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#2d4a6a; text-align:center;'>
        Smart City Monitor v2.0<br>
        Modules: Weather · Traffic · Dustbin
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown("""
    <div class='city-header'>
        <div class='city-title'>Smart City Monitoring System</div>
        <div class='city-subtitle'>Real-time intelligence across traffic, weather &amp; waste management</div>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    now_str = datetime.now().strftime("%d %b %Y<br>%H:%M:%S")
    st.markdown(f"""
    <div style='text-align:right; color:#4a6a8a; font-size:12px; font-family: DM Mono, monospace; padding-top:8px;'>
        {now_str}
    </div>
    <div style='text-align:right; margin-top:6px;'>
        <span class='live-pill'><span class='live-dot'></span>Live</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SUMMARY METRIC ROW
# ─────────────────────────────────────────────────────────────────
wd = st.session_state.weather_data
bins = st.session_state.bins
full_count = sum(1 for b in bins if b["status"] == "FULL")
crit_count = sum(1 for b in bins if b["status"] == "CRITICAL")
alert_count = full_count + crit_count

m1, m2, m3, m4, m5, m6 = st.columns(6)

def metric_card(col, label, value, unit, sub, color_class):
    with col:
        st.markdown(f"""
        <div class='metric-card {color_class}'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}<span class='metric-unit'>{unit}</span></div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

temp_val  = f"{wd['temperature']:.1f}" if wd else "--"
humid_val = f"{wd['humidity']:.0f}"    if wd else "--"
loc_val   = wd['city'][:10]            if wd else "—"
lat_val   = f"{wd['lat']:.2f}"         if wd else "--"

metric_card(m1, "Temperature",  temp_val,  "°C",  f"{wd['condition'] if wd else 'No data'}", "blue")
metric_card(m2, "Humidity",     humid_val, "%",   f"Dew pt {wd.get('dew_point','—')}°C" if wd else "—", "teal")
metric_card(m3, "Location",     loc_val,   "",    f"Lat {lat_val}", "purple")
metric_card(m4, "Active Alerts",str(alert_count), "", f"{full_count} bins full", "red" if alert_count > 0 else "green")
metric_card(m5, "Total Bins",   str(len(bins)), "", f"{crit_count} critical", "amber")
pred = st.session_state.prediction_result
pred_val = pred["decision"][:3] if pred else "N/A"
metric_card(m6, "Resource Need", pred_val, "", "Last prediction", "green" if pred_val=="LOW" else "red" if pred_val=="HIG" else "amber")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌡️  Weather & Climate",
    "🚦  Traffic Prediction",
    "🗑️  Dustbin Monitor",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — WEATHER
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon' style='background:rgba(59,130,246,0.15)'>🌡️</div>
        <div>
            <div class='section-title'>Temperature & Humidity Monitor</div>
            <div class='section-desc'>Live data from Open-Meteo API · No API key required</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not wd:
        st.markdown("""
        <div class='info-box'>
            👈 Select a city in the sidebar and click <strong>Fetch Live Weather</strong> to load real-time data.
        </div>
        """, unsafe_allow_html=True)
        # Show demo with Udaipur defaults
        with st.expander("📋 Demo with Udaipur defaults", expanded=True):
            demo_data = get_weather_by_city("Udaipur")
            st.session_state.weather_data = demo_data
            wd = demo_data
            st.markdown("""
            <div class='success-box'>Demo data loaded for Udaipur, Rajasthan</div>
            """, unsafe_allow_html=True)

    if wd:
        # Row 1: primary weather cards
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "🌡️ Temperature", f"{wd['temperature']:.1f}", "°C", wd.get("condition","—"), "blue"),
            (c2, "💧 Humidity",    f"{wd['humidity']:.0f}",    "%",  f"Dew point: {wd.get('dew_point','—')}°C", "teal"),
            (c3, "💨 Wind Speed",  f"{wd.get('wind_speed',0):.1f}", " km/h", f"Dir: {wd.get('wind_dir','—')}°", "amber"),
            (c4, "🔆 UV Index",    f"{wd.get('uv_index',0):.1f}", "", _uv_label(wd.get('uv_index',0)), "purple"),
        ]
        for col, lbl, val, unit, sub, cls in cards:
            with col:
                st.markdown(f"""
                <div class='metric-card {cls}' style='margin-bottom:0'>
                    <div class='metric-label'>{lbl}</div>
                    <div class='metric-value'>{val}<span class='metric-unit'>{unit}</span></div>
                    <div class='metric-sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: location + additional metrics
        left, right = st.columns([1, 1])

        with left:
            st.markdown("""
            <div class='section-header' style='margin-top:0'>
                <div class='section-title' style='font-size:14px'>📍 Location Details</div>
            </div>
            """, unsafe_allow_html=True)
            loc_rows = [
                ("City",       wd['city']),
                ("State",      wd.get('state', '—')),
                ("Country",    wd.get('country', 'India')),
                ("Latitude",   f"{wd['lat']:.4f}° N"),
                ("Longitude",  f"{wd['lon']:.4f}° E"),
                ("Elevation",  f"{wd.get('elevation','—')} m"),
                ("Timezone",   wd.get('timezone', 'Asia/Kolkata')),
            ]
            for k, v in loc_rows:
                c_k, c_v = st.columns([2, 3])
                c_k.markdown(f"<span style='font-size:12px;color:#4a6a8a'>{k}</span>", unsafe_allow_html=True)
                c_v.markdown(f"<span style='font-size:12px;color:#c8dff0;font-family:DM Mono,monospace'>{v}</span>", unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class='section-header' style='margin-top:0'>
                <div class='section-title' style='font-size:14px'>🌤 Atmospheric Conditions</div>
            </div>
            """, unsafe_allow_html=True)
            atm_rows = [
                ("Feels Like",    f"{wd.get('feels_like', wd['temperature']):.1f}°C"),
                ("Cloud Cover",   f"{wd.get('cloud_cover',0):.0f}%"),
                ("Pressure",      f"{wd.get('pressure',1010):.1f} hPa"),
                ("Visibility",    f"{wd.get('visibility',10):.1f} km"),
                ("Rain (1hr)",    f"{wd.get('rain_mm',0):.2f} mm"),
                ("Condition",     wd.get('condition','—')),
                ("Last Updated",  wd.get('fetched_at', datetime.now().strftime('%H:%M:%S'))),
            ]
            for k, v in atm_rows:
                c_k, c_v = st.columns([2, 3])
                c_k.markdown(f"<span style='font-size:12px;color:#4a6a8a'>{k}</span>", unsafe_allow_html=True)
                c_v.markdown(f"<span style='font-size:12px;color:#c8dff0;font-family:DM Mono,monospace'>{v}</span>", unsafe_allow_html=True)

        # Heat index + risk bands
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        hi = _heat_index(wd['temperature'], wd['humidity'])
        hi_label, hi_color = _heat_index_label(hi)
        with r1:
            st.markdown(f"""
            <div class='metric-card amber'>
                <div class='metric-label'>🔥 Heat Index</div>
                <div class='metric-value'>{hi:.1f}<span class='metric-unit'>°C</span></div>
                <div class='metric-sub' style='color:{hi_color}'>{hi_label}</div>
            </div>
            """, unsafe_allow_html=True)

        hi_risk, hi_risk_c = _heat_island_risk(wd)
        with r2:
            st.markdown(f"""
            <div class='metric-card red'>
                <div class='metric-label'>🌆 Heat Island Risk</div>
                <div class='metric-value' style='font-size:22px;color:{hi_risk_c}'>{hi_risk}</div>
                <div class='metric-sub'>Urban heat assessment</div>
            </div>
            """, unsafe_allow_html=True)

        fl_risk, fl_risk_c = _flood_risk(wd)
        with r3:
            st.markdown(f"""
            <div class='metric-card blue'>
                <div class='metric-label'>🌧 Flood Risk</div>
                <div class='metric-value' style='font-size:22px;color:{fl_risk_c}'>{fl_risk}</div>
                <div class='metric-sub'>Based on rain & humidity</div>
            </div>
            """, unsafe_allow_html=True)

        if show_raw:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Raw weather payload:")
            st.json(wd)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — TRAFFIC PREDICTION
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon' style='background:rgba(245,158,11,0.15)'>🚦</div>
        <div>
            <div class='section-title'>Traffic Resource Prediction</div>
            <div class='section-desc'>ML model predicts resource need: HIGH / MEDIUM / LOW</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("##### 📥 Input Features")

        # Auto-fill from weather if available
        auto_temp  = wd['temperature'] if wd else 30.0
        auto_humid = wd['humidity']    if wd else 55.0

        traffic_vol = st.slider(
            "🚗 Traffic Volume (vehicles/hr)",
            min_value=0, max_value=5000, value=2500, step=50,
        )
        temp_inp = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-5.0, max_value=55.0,
            value=float(round(auto_temp, 1)),
            step=0.5,
            help="Auto-filled from weather module if loaded",
        )
        humid_inp = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0, max_value=100.0,
            value=float(round(auto_humid, 1)),
            step=1.0,
            help="Auto-filled from weather module if loaded",
        )
        population = st.slider(
            "👥 Population Density (people/km²)",
            min_value=100, max_value=50000, value=8000, step=100,
        )
        hour_of_day = st.slider(
            "🕐 Hour of Day (0–23)",
            min_value=0, max_value=23,
            value=datetime.now().hour,
        )
        is_weekend = st.toggle("📅 Weekend / Holiday", value=False)
        weather_code = st.selectbox(
            "🌤 Weather Condition",
            options=["Clear", "Partly Cloudy", "Overcast", "Rain", "Heavy Rain", "Fog"],
        )
        road_quality = st.select_slider(
            "🛣️ Road Quality Index",
            options=["Poor", "Below Average", "Average", "Good", "Excellent"],
            value="Good",
        )

        if wd:
            st.markdown("""
            <div class='success-box'>
                ✅ Temperature &amp; Humidity auto-filled from live weather data
            </div>
            """, unsafe_allow_html=True)

        predict_btn = st.button("⚡ Predict Resource Need", use_container_width=True)

    with col_result:
        st.markdown("##### 📊 Prediction Output")

        if predict_btn:
            with st.spinner("Running ML model…"):
                time.sleep(0.6)  # visual feedback
                result = predict_resource_need(
                    traffic_volume=traffic_vol,
                    temperature=temp_inp,
                    humidity=humid_inp,
                    population_density=population,
                    hour_of_day=hour_of_day,
                    is_weekend=int(is_weekend),
                    weather_code=weather_code,
                    road_quality=road_quality,
                )
                st.session_state.prediction_result = result

        pr = st.session_state.prediction_result

        if pr:
            decision = pr["decision"]
            st.markdown(f"""
            <div class='decision-badge decision-{decision}'>
                {_decision_icon(decision)}  {decision} RESOURCE NEED
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence bars
            st.markdown("**Model Confidence**")
            for cls, prob in pr["probabilities"].items():
                bar_color = {"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}.get(cls,"#3b82f6")
                pct = round(prob * 100, 1)
                st.markdown(f"""
                <div class='feat-bar-wrap'>
                    <div class='feat-bar-label'>{cls}  {pct}%</div>
                    <div class='feat-bar-track'>
                        <div class='feat-bar-fill' style='width:{pct}%;background:{bar_color}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Recommendations
            st.markdown("**💡 Recommendations**")
            for rec in _get_recommendations(decision, traffic_vol, temp_inp):
                st.markdown(f"- {rec}")

            # Feature importance
            fi = get_feature_importance()
            if fi:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📈 Feature Importance**")
                for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:6]:
                    pct = round(imp * 100, 1)
                    st.markdown(f"""
                    <div class='feat-bar-wrap'>
                        <div class='feat-bar-label'>{feat}  {pct}%</div>
                        <div class='feat-bar-track'>
                            <div class='feat-bar-fill' style='width:{pct}%;background:#3b82f6'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if show_raw:
                st.json(pr)
        else:
            st.markdown("""
            <div class='info-box'>
                Fill in the input features on the left and click <strong>Predict Resource Need</strong>.
            </div>
            """, unsafe_allow_html=True)
            # Show example scenarios
            st.markdown("##### 📋 Example Scenarios")
            scenarios_df = pd.DataFrame([
                {"Scenario": "Peak Morning Rush", "Traffic": "4500", "Temp": "32°C", "Result": "HIGH"},
                {"Scenario": "Weekend Afternoon", "Traffic": "1800", "Temp": "28°C", "Result": "MEDIUM"},
                {"Scenario": "Late Night",         "Traffic": "400",  "Temp": "22°C", "Result": "LOW"},
                {"Scenario": "Rainy Evening",      "Traffic": "3200", "Temp": "25°C", "Result": "HIGH"},
            ])
            st.dataframe(
                scenarios_df,
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DUSTBIN MONITOR
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon' style='background:rgba(34,197,94,0.15)'>🗑️</div>
        <div>
            <div class='section-title'>Dustbin Fill Level Monitor</div>
            <div class='section-desc'>Real-time bin status across all city zones · Simulated IoT sensors</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary row
    total    = len(bins)
    n_full   = sum(1 for b in bins if b["status"] in ("FULL", "CRITICAL"))
    n_half   = sum(1 for b in bins if b["status"] == "HALF")
    n_empty  = sum(1 for b in bins if b["status"] in ("EMPTY", "QUARTER"))

    sb1, sb2, sb3, sb4 = st.columns(4)
    with sb1:
        st.markdown(f"""
        <div class='metric-card blue'>
            <div class='metric-label'>Total Bins</div>
            <div class='metric-value'>{total}</div>
            <div class='metric-sub'>Across {len(set(b['zone'] for b in bins))} zones</div>
        </div>
        """, unsafe_allow_html=True)
    with sb2:
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>Full / Critical</div>
            <div class='metric-value' style='color:#fca5a5'>{n_full}</div>
            <div class='metric-sub'>Needs collection</div>
        </div>
        """, unsafe_allow_html=True)
    with sb3:
        st.markdown(f"""
        <div class='metric-card amber'>
            <div class='metric-label'>Half Full</div>
            <div class='metric-value' style='color:#fcd34d'>{n_half}</div>
            <div class='metric-sub'>Monitor closely</div>
        </div>
        """, unsafe_allow_html=True)
    with sb4:
        st.markdown(f"""
        <div class='metric-card green'>
            <div class='metric-label'>Empty / Low</div>
            <div class='metric-value' style='color:#86efac'>{n_empty}</div>
            <div class='metric-sub'>No action needed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Refresh + filter controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        zone_filter = st.selectbox(
            "Filter by Zone",
            ["All Zones"] + sorted(set(b["zone"] for b in bins)),
        )
    with ctrl2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All Statuses", "CRITICAL", "FULL", "HALF", "QUARTER", "EMPTY"],
        )
    with ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Bins"):
            st.session_state.bins = refresh_bin_levels(st.session_state.bins)
            bins = st.session_state.bins
            st.rerun()

    # Filter bins
    filtered = bins
    if zone_filter != "All Zones":
        filtered = [b for b in filtered if b["zone"] == zone_filter]
    if status_filter != "All Statuses":
        filtered = [b for b in filtered if b["status"] == status_filter]

    if n_full > 0:
        st.markdown(f"""
        <div class='warn-box'>
            ⚠️ <strong>{n_full} bin(s)</strong> are full or critical and require immediate collection.
            Zones affected: {', '.join(set(b['zone'] for b in bins if b['status'] in ('FULL','CRITICAL')))}
        </div>
        """, unsafe_allow_html=True)

    # Bin grid display
    st.markdown(f"**Showing {len(filtered)} of {len(bins)} bins**")
    st.markdown("<br>", unsafe_allow_html=True)

    COLS = 5
    rows = [filtered[i:i+COLS] for i in range(0, len(filtered), COLS)]

    for row in rows:
        cols = st.columns(COLS)
        for i, (col, b) in enumerate(zip(cols, row)):
            with col:
                level_pct = b["fill_level"]
                fill_color = (
                    "#ef4444" if b["status"] in ("FULL","CRITICAL") else
                    "#f59e0b" if b["status"] == "HALF" else
                    "#22c55e"
                )
                fill_h = int(58 * level_pct / 100)
                st.markdown(f"""
                <div class='bin-card'>
                    <div class='bin-id'>{b['bin_id']}</div>
                    <div class='bin-location'>{b['location'][:18]}</div>
                    <div style='display:flex;justify-content:center;'>
                        <div class='bin-level-bar'>
                            <div class='bin-fill' style='height:{fill_h}px;background:{fill_color};'></div>
                        </div>
                    </div>
                    <div style='font-size:11px;color:#4a6a8a;margin-bottom:5px;'>{level_pct:.0f}%</div>
                    <span class='bin-status-pill status-{b["status"]}'>{b["status"]}</span>
                    <div class='bin-location' style='margin-top:5px;'>{b['zone']}</div>
                    <div style='font-size:9px;color:#2d4a6a;margin-top:3px;'>Last: {b['last_updated']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table view
    with st.expander("📋 Detailed Bin Table", expanded=False):
        df_bins = pd.DataFrame([{
            "Bin ID":      b["bin_id"],
            "Zone":        b["zone"],
            "Location":    b["location"],
            "Fill %":      f"{b['fill_level']:.0f}%",
            "Status":      b["status"],
            "Capacity (L)": b.get("capacity_litres", 120),
            "Last Updated": b["last_updated"],
            "Sensor ID":   b.get("sensor_id","—"),
        } for b in filtered])

        st.dataframe(df_bins, use_container_width=True, hide_index=True)

    # Collection route suggestion
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🚛 Collection Route Suggestion", expanded=n_full > 0):
        urgent = [b for b in bins if b["status"] in ("FULL","CRITICAL")]
        if urgent:
            st.markdown("**Priority Collection Order:**")
            for i, b in enumerate(sorted(urgent, key=lambda x: -x["fill_level"]), 1):
                icon = "🔴" if b["status"] == "CRITICAL" else "🟠"
                st.markdown(f"{i}. {icon} **{b['bin_id']}** — {b['location']} ({b['zone']}) — {b['fill_level']:.0f}% full")
        else:
            st.markdown("""
            <div class='success-box'>✅ No urgent collections needed right now.</div>
            """, unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed > 30:
            st.session_state.bins = refresh_bin_levels(st.session_state.bins)
            st.session_state.last_refresh = time.time()
            st.rerun()



