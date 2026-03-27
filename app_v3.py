"""
╔══════════════════════════════════════════════════════════════════╗
║   SMART CITY AI v4.0  —  app.py                                 ║
║   Run:  streamlit run app.py                                    ║
║                                                                  ║
║   pip install streamlit ultralytics opencv-python-headless      ║
║               numpy pandas requests pillow matplotlib           ║
║               scikit-learn                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, time, random, tempfile, math, pickle
from datetime import datetime
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

try:
    from ultralytics import YOLO as _YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    SK_OK = True
except ImportError:
    SK_OK = False


# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartCity AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
#MainMenu,footer        { visibility:hidden; }
.block-container        { padding:1rem 1.8rem 2rem; }

.hero{ background:linear-gradient(135deg,#07101f 0%,#0d1c38 55%,#07101f 100%);
  border:1px solid rgba(56,189,248,.15); border-radius:14px;
  padding:20px 28px; margin-bottom:18px; }
.hero-title{ font-family:'Syne',sans-serif; font-size:24px; font-weight:800;
  color:#e0f2fe; letter-spacing:-.4px; margin:0 0 3px; }
.hero-sub{ font-size:12px; color:#3d6080; margin:0; }
.live-badge{ display:inline-flex; align-items:center; gap:5px;
  background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.25);
  color:#4ade80; padding:3px 11px; border-radius:20px;
  font-size:11px; font-weight:600; letter-spacing:.5px; text-transform:uppercase; }
.live-dot{ width:6px; height:6px; background:#4ade80; border-radius:50%;
  animation:blink 1.6s infinite; }
@keyframes blink{ 0%,100%{opacity:1} 50%{opacity:.2} }

.city-panel{ background:#080f1c; border:1.5px solid rgba(56,189,248,.2);
  border-radius:12px; padding:16px 22px; margin-bottom:16px; }
.city-panel-title{ font-family:'Syne',sans-serif; font-size:13px; font-weight:700;
  color:#7dd3fc; margin:0 0 12px; }

.mc{ background:#080f1c; border:1px solid rgba(255,255,255,.07);
  border-radius:11px; padding:14px 16px; overflow:hidden; position:relative; }
.mc::after{ content:''; position:absolute; top:0; left:0; right:0;
  height:2px; border-radius:11px 11px 0 0; }
.mc.blue::after{background:#3b82f6;} .mc.teal::after{background:#14b8a6;}
.mc.amber::after{background:#f59e0b;} .mc.red::after{background:#ef4444;}
.mc.green::after{background:#22c55e;} .mc.purple::after{background:#a855f7;}
.mc.sky::after{background:#38bdf8;}
.mc-lbl{ font-size:10px; font-weight:600; text-transform:uppercase;
  letter-spacing:.7px; color:#1e3a5f; margin-bottom:6px; }
.mc-val{ font-family:'DM Mono',monospace; font-size:24px; font-weight:500;
  color:#dbeafe; line-height:1; margin-bottom:3px; }
.mc-unit{ font-size:11px; color:#3d6080; }
.mc-sub{ font-size:10px; color:#1e3a5f; margin-top:4px; }

.sh{ display:flex; align-items:center; gap:9px;
  border-bottom:1px solid rgba(255,255,255,.05);
  padding-bottom:9px; margin-bottom:12px; }
.sh-icon{ width:28px; height:28px; border-radius:7px;
  display:flex; align-items:center; justify-content:center; font-size:14px; }
.sh-title{ font-family:'Syne',sans-serif; font-size:14px;
  font-weight:700; color:#bfdbfe; margin:0; }
.sh-desc{ font-size:11px; color:#1e3a5f; margin:0; }

.db{ display:block; padding:10px 18px; border-radius:9px;
  font-family:'Syne',sans-serif; font-size:18px; font-weight:800;
  letter-spacing:1px; text-align:center; width:100%; }
.db-HIGH  { background:rgba(239,68,68,.12); border:2px solid #ef4444; color:#fca5a5; }
.db-MEDIUM{ background:rgba(245,158,11,.12); border:2px solid #f59e0b; color:#fcd34d; }
.db-LOW   { background:rgba(34,197,94,.12);  border:2px solid #22c55e; color:#86efac; }

.ib{ background:rgba(59,130,246,.07); border-left:3px solid #3b82f6;
  border-radius:0 7px 7px 0; padding:9px 13px; font-size:12px; color:#93c5fd; margin:7px 0; }
.wb{ background:rgba(245,158,11,.07); border-left:3px solid #f59e0b;
  border-radius:0 7px 7px 0; padding:9px 13px; font-size:12px; color:#fcd34d; margin:7px 0; }
.sb{ background:rgba(34,197,94,.07);  border-left:3px solid #22c55e;
  border-radius:0 7px 7px 0; padding:9px 13px; font-size:12px; color:#86efac; margin:7px 0; }
.eb{ background:rgba(239,68,68,.07);  border-left:3px solid #ef4444;
  border-radius:0 7px 7px 0; padding:9px 13px; font-size:12px; color:#fca5a5; margin:7px 0; }

.sug{ background:#080f1c; border-left:3px solid #38bdf8;
  border-radius:0 7px 7px 0; padding:8px 12px;
  margin:4px 0; font-size:12px; color:#bae6fd; }

.bw{ background:#080f1c; border:1px solid rgba(255,255,255,.06);
  border-radius:9px; padding:11px; text-align:center; }
.bid{ font-family:'DM Mono',monospace; font-size:9px; color:#1e3a5f;
  text-transform:uppercase; margin-bottom:3px; }
.bbo{ width:22px; height:52px; background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08); border-radius:3px;
  margin:0 auto 5px; position:relative; overflow:hidden; }
.bfi{ position:absolute; bottom:0; left:0; right:0; border-radius:2px; }
.bp{ display:inline-block; padding:1px 7px; border-radius:8px;
  font-size:8px; font-weight:600; text-transform:uppercase; letter-spacing:.3px; }
.bp-FULL    { background:rgba(239,68,68,.2);  color:#fca5a5; border:1px solid rgba(239,68,68,.35); }
.bp-CRITICAL{ background:rgba(239,68,68,.3);  color:#f87171; border:1.5px solid #ef4444; }
.bp-HALF    { background:rgba(245,158,11,.2); color:#fcd34d; border:1px solid rgba(245,158,11,.35); }
.bp-QUARTER { background:rgba(34,197,94,.15); color:#6ee7b7; border:1px solid rgba(34,197,94,.3); }
.bp-EMPTY   { background:rgba(34,197,94,.2);  color:#86efac; border:1px solid rgba(34,197,94,.35); }

.fb{ margin:3px 0; }
.fbl{ font-size:10px; color:#3d6080; margin-bottom:1px; }
.fbt{ height:5px; background:rgba(255,255,255,.05); border-radius:3px; overflow:hidden; }
.fbf{ height:100%; border-radius:3px; }

.cb{ background:rgba(20,184,166,.07); border:1px solid rgba(20,184,166,.16);
  border-radius:7px; padding:7px 11px;
  font-family:'DM Mono',monospace; font-size:11px; color:#5eead4; margin:5px 0; }

.stTabs [data-baseweb="tab-list"]{
  gap:4px; background:transparent; border-bottom:1px solid rgba(255,255,255,.05); }
.stTabs [data-baseweb="tab"]{
  background:transparent; border:1px solid rgba(255,255,255,.06);
  border-bottom:none; border-radius:7px 7px 0 0;
  color:#3d6080; font-size:13px; font-weight:500; padding:7px 15px; }
.stTabs [aria-selected="true"]{
  background:rgba(56,189,248,.07) !important;
  border-color:rgba(56,189,248,.22) !important; color:#7dd3fc !important; }

div[data-baseweb="select"]>div,
div[data-baseweb="input"] input{
  background:#080f1c !important; border:1px solid rgba(255,255,255,.1) !important;
  border-radius:7px !important; color:#c8dff0 !important; }
.stButton>button{
  background:linear-gradient(135deg,#1e3a6e,#2563eb);
  border:none; border-radius:7px; color:#fff;
  font-weight:600; font-size:13px; padding:9px 18px; width:100%; }
.stButton>button:hover{ background:linear-gradient(135deg,#1d4ed8,#60a5fa); }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# COMPLETE INDIA CITY DATABASE  —  800+ cities across all 28 states
# + 8 UTs
# ════════════════════════════════════════════════════════════════════
CITIES = {
    # ── Andhra Pradesh ────────────────────────────────────────────
    "Visakhapatnam":{"lat":17.6868,"lon":83.2185,"state":"Andhra Pradesh","elev":45},
    "Vijayawada":{"lat":16.5062,"lon":80.6480,"state":"Andhra Pradesh","elev":26},
    "Guntur":{"lat":16.3067,"lon":80.4365,"state":"Andhra Pradesh","elev":33},
    "Nellore":{"lat":14.4426,"lon":79.9865,"state":"Andhra Pradesh","elev":17},
    "Tirupati":{"lat":13.6288,"lon":79.4192,"state":"Andhra Pradesh","elev":152},
    "Kurnool":{"lat":15.8281,"lon":78.0373,"state":"Andhra Pradesh","elev":273},
    "Kadapa":{"lat":14.4674,"lon":78.8241,"state":"Andhra Pradesh","elev":138},
    "Rajahmundry":{"lat":17.0005,"lon":81.8040,"state":"Andhra Pradesh","elev":20},
    "Kakinada":{"lat":16.9891,"lon":82.2475,"state":"Andhra Pradesh","elev":8},
    "Anantapur":{"lat":14.6819,"lon":77.6006,"state":"Andhra Pradesh","elev":352},
    "Vizianagaram":{"lat":18.1066,"lon":83.3956,"state":"Andhra Pradesh","elev":66},
    "Eluru":{"lat":16.7107,"lon":81.0952,"state":"Andhra Pradesh","elev":20},
    "Ongole":{"lat":15.5057,"lon":80.0499,"state":"Andhra Pradesh","elev":11},
    "Srikakulam":{"lat":18.2951,"lon":83.8970,"state":"Andhra Pradesh","elev":14},
    "Chittoor":{"lat":13.2172,"lon":79.1003,"state":"Andhra Pradesh","elev":274},
    "Machilipatnam":{"lat":16.1875,"lon":81.1389,"state":"Andhra Pradesh","elev":5},
    "Bhimavaram":{"lat":16.5449,"lon":81.5212,"state":"Andhra Pradesh","elev":10},
    "Adoni":{"lat":15.6277,"lon":77.2740,"state":"Andhra Pradesh","elev":447},
    "Hindupur":{"lat":13.8281,"lon":77.4919,"state":"Andhra Pradesh","elev":645},
    "Proddatur":{"lat":14.7500,"lon":78.5500,"state":"Andhra Pradesh","elev":149},
    "Amaravati":{"lat":16.5736,"lon":80.3573,"state":"Andhra Pradesh","elev":28},

    # ── Arunachal Pradesh ──────────────────────────────────────────
    "Itanagar":{"lat":27.0844,"lon":93.6053,"state":"Arunachal Pradesh","elev":250},
    "Naharlagun":{"lat":27.1041,"lon":93.6934,"state":"Arunachal Pradesh","elev":213},
    "Pasighat":{"lat":28.0660,"lon":95.3290,"state":"Arunachal Pradesh","elev":155},
    "Tezpur":{"lat":26.6338,"lon":92.7936,"state":"Arunachal Pradesh","elev":48},
    "Ziro":{"lat":27.5484,"lon":93.8279,"state":"Arunachal Pradesh","elev":1524},
    "Bomdila":{"lat":27.2645,"lon":92.4156,"state":"Arunachal Pradesh","elev":2415},

    # ── Assam ──────────────────────────────────────────────────────
    "Guwahati":{"lat":26.1445,"lon":91.7362,"state":"Assam","elev":55},
    "Silchar":{"lat":24.8333,"lon":92.7789,"state":"Assam","elev":22},
    "Dibrugarh":{"lat":27.4728,"lon":94.9120,"state":"Assam","elev":107},
    "Jorhat":{"lat":26.7465,"lon":94.2026,"state":"Assam","elev":116},
    "Tinsukia":{"lat":27.4925,"lon":95.3552,"state":"Assam","elev":111},
    "Nagaon":{"lat":26.3460,"lon":92.6923,"state":"Assam","elev":55},
    "Tezpur":{"lat":26.6338,"lon":92.7936,"state":"Assam","elev":48},
    "Bongaigaon":{"lat":26.4780,"lon":90.5583,"state":"Assam","elev":40},
    "Dhubri":{"lat":26.0181,"lon":89.9771,"state":"Assam","elev":32},
    "Diphu":{"lat":25.8447,"lon":93.4333,"state":"Assam","elev":270},
    "North Lakhimpur":{"lat":27.2344,"lon":94.1017,"state":"Assam","elev":100},
    "Goalpara":{"lat":26.1669,"lon":90.6169,"state":"Assam","elev":42},
    "Karimganj":{"lat":24.8667,"lon":92.3667,"state":"Assam","elev":22},
    "Sivasagar":{"lat":26.9819,"lon":94.6381,"state":"Assam","elev":90},

    # ── Bihar ──────────────────────────────────────────────────────
    "Patna":{"lat":25.5941,"lon":85.1376,"state":"Bihar","elev":53},
    "Gaya":{"lat":24.7955,"lon":85.0077,"state":"Bihar","elev":113},
    "Bhagalpur":{"lat":25.2425,"lon":86.9842,"state":"Bihar","elev":44},
    "Muzaffarpur":{"lat":26.1209,"lon":85.3647,"state":"Bihar","elev":60},
    "Purnia":{"lat":25.7771,"lon":87.4753,"state":"Bihar","elev":36},
    "Darbhanga":{"lat":26.1542,"lon":85.8918,"state":"Bihar","elev":52},
    "Bihar Sharif":{"lat":25.1975,"lon":85.5236,"state":"Bihar","elev":74},
    "Arrah":{"lat":25.5563,"lon":84.6631,"state":"Bihar","elev":60},
    "Begusarai":{"lat":25.4182,"lon":86.1272,"state":"Bihar","elev":42},
    "Chapra":{"lat":25.7776,"lon":84.7474,"state":"Bihar","elev":57},
    "Katihar":{"lat":25.5393,"lon":87.5799,"state":"Bihar","elev":31},
    "Munger":{"lat":25.3723,"lon":86.4734,"state":"Bihar","elev":43},
    "Saharsa":{"lat":25.8820,"lon":86.6006,"state":"Bihar","elev":35},
    "Hajipur":{"lat":25.6906,"lon":85.2091,"state":"Bihar","elev":55},
    "Sasaram":{"lat":24.9469,"lon":84.0322,"state":"Bihar","elev":103},
    "Dehri":{"lat":24.9018,"lon":84.1835,"state":"Bihar","elev":98},
    "Sitamarhi":{"lat":26.5922,"lon":85.4876,"state":"Bihar","elev":63},
    "Bettiah":{"lat":26.8000,"lon":84.5000,"state":"Bihar","elev":88},
    "Motihari":{"lat":26.6499,"lon":84.9167,"state":"Bihar","elev":75},
    "Supaul":{"lat":26.1245,"lon":86.6014,"state":"Bihar","elev":44},

    # ── Chhattisgarh ───────────────────────────────────────────────
    "Raipur":{"lat":21.2514,"lon":81.6296,"state":"Chhattisgarh","elev":298},
    "Bhilai":{"lat":21.2090,"lon":81.4285,"state":"Chhattisgarh","elev":299},
    "Bilaspur":{"lat":22.0797,"lon":82.1391,"state":"Chhattisgarh","elev":264},
    "Korba":{"lat":22.3460,"lon":82.7015,"state":"Chhattisgarh","elev":299},
    "Durg":{"lat":21.1904,"lon":81.2849,"state":"Chhattisgarh","elev":304},
    "Rajnandgaon":{"lat":21.0972,"lon":81.0296,"state":"Chhattisgarh","elev":317},
    "Raigarh":{"lat":21.8974,"lon":83.3950,"state":"Chhattisgarh","elev":208},
    "Jagdalpur":{"lat":19.0709,"lon":82.0144,"state":"Chhattisgarh","elev":553},
    "Ambikapur":{"lat":23.1188,"lon":83.1948,"state":"Chhattisgarh","elev":625},
    "Dhamtari":{"lat":20.7082,"lon":81.5495,"state":"Chhattisgarh","elev":295},
    "Chirmiri":{"lat":23.2333,"lon":82.3500,"state":"Chhattisgarh","elev":600},
    "Koriya":{"lat":23.3300,"lon":82.7200,"state":"Chhattisgarh","elev":574},

    # ── Goa ────────────────────────────────────────────────────────
    "Panaji":{"lat":15.4909,"lon":73.8278,"state":"Goa","elev":7},
    "Margao":{"lat":15.2832,"lon":73.9862,"state":"Goa","elev":28},
    "Vasco da Gama":{"lat":15.3982,"lon":73.8138,"state":"Goa","elev":4},
    "Mapusa":{"lat":15.5930,"lon":73.8146,"state":"Goa","elev":44},
    "Ponda":{"lat":15.4024,"lon":74.0121,"state":"Goa","elev":21},
    "Bicholim":{"lat":15.5959,"lon":74.0016,"state":"Goa","elev":40},

    # ── Gujarat ────────────────────────────────────────────────────
    "Ahmedabad":{"lat":23.0225,"lon":72.5714,"state":"Gujarat","elev":53},
    "Surat":{"lat":21.1702,"lon":72.8311,"state":"Gujarat","elev":13},
    "Vadodara":{"lat":22.3072,"lon":73.1812,"state":"Gujarat","elev":37},
    "Rajkot":{"lat":22.3039,"lon":70.8022,"state":"Gujarat","elev":138},
    "Bhavnagar":{"lat":21.7645,"lon":72.1519,"state":"Gujarat","elev":26},
    "Jamnagar":{"lat":22.4707,"lon":70.0577,"state":"Gujarat","elev":26},
    "Junagadh":{"lat":21.5222,"lon":70.4579,"state":"Gujarat","elev":107},
    "Gandhinagar":{"lat":23.2156,"lon":72.6369,"state":"Gujarat","elev":81},
    "Anand":{"lat":22.5645,"lon":72.9289,"state":"Gujarat","elev":42},
    "Navsari":{"lat":20.9467,"lon":72.9520,"state":"Gujarat","elev":12},
    "Morbi":{"lat":22.8210,"lon":70.8374,"state":"Gujarat","elev":22},
    "Mehsana":{"lat":23.5880,"lon":72.3693,"state":"Gujarat","elev":93},
    "Bhuj":{"lat":23.2420,"lon":69.6669,"state":"Gujarat","elev":82},
    "Surendranagar":{"lat":22.7269,"lon":71.6481,"state":"Gujarat","elev":116},
    "Amreli":{"lat":21.6047,"lon":71.2208,"state":"Gujarat","elev":136},
    "Valsad":{"lat":20.5992,"lon":72.9342,"state":"Gujarat","elev":6},
    "Bharuch":{"lat":21.7051,"lon":72.9959,"state":"Gujarat","elev":9},
    "Veraval":{"lat":20.9070,"lon":70.3670,"state":"Gujarat","elev":11},
    "Godhra":{"lat":22.7780,"lon":73.6185,"state":"Gujarat","elev":115},
    "Palanpur":{"lat":24.1722,"lon":72.4382,"state":"Gujarat","elev":214},
    "Porbandar":{"lat":21.6417,"lon":69.6293,"state":"Gujarat","elev":4},

    # ── Haryana ────────────────────────────────────────────────────
    "Faridabad":{"lat":28.4089,"lon":77.3178,"state":"Haryana","elev":198},
    "Gurgaon":{"lat":28.4595,"lon":77.0266,"state":"Haryana","elev":217},
    "Panipat":{"lat":29.3909,"lon":76.9635,"state":"Haryana","elev":219},
    "Ambala":{"lat":30.3752,"lon":76.7821,"state":"Haryana","elev":270},
    "Yamunanagar":{"lat":30.1290,"lon":77.2674,"state":"Haryana","elev":277},
    "Rohtak":{"lat":28.8955,"lon":76.6066,"state":"Haryana","elev":220},
    "Hisar":{"lat":29.1492,"lon":75.7217,"state":"Haryana","elev":215},
    "Karnal":{"lat":29.6857,"lon":76.9905,"state":"Haryana","elev":245},
    "Sonipat":{"lat":28.9931,"lon":77.0151,"state":"Haryana","elev":239},
    "Panchkula":{"lat":30.6942,"lon":76.8606,"state":"Haryana","elev":360},
    "Bhiwani":{"lat":28.7975,"lon":76.1322,"state":"Haryana","elev":219},
    "Sirsa":{"lat":29.5330,"lon":75.0140,"state":"Haryana","elev":205},
    "Bahadurgarh":{"lat":28.6814,"lon":76.9287,"state":"Haryana","elev":218},
    "Kurukshetra":{"lat":29.9695,"lon":76.8783,"state":"Haryana","elev":249},
    "Rewari":{"lat":28.1951,"lon":76.6178,"state":"Haryana","elev":240},

    # ── Himachal Pradesh ──────────────────────────────────────────
    "Shimla":{"lat":31.1048,"lon":77.1734,"state":"Himachal Pradesh","elev":2200},
    "Dharamsala":{"lat":32.2190,"lon":76.3234,"state":"Himachal Pradesh","elev":1457},
    "Solan":{"lat":30.9045,"lon":77.0967,"state":"Himachal Pradesh","elev":1337},
    "Mandi":{"lat":31.7082,"lon":76.9318,"state":"Himachal Pradesh","elev":850},
    "Hamirpur":{"lat":31.6846,"lon":76.5218,"state":"Himachal Pradesh","elev":786},
    "Nahan":{"lat":30.5594,"lon":77.2947,"state":"Himachal Pradesh","elev":932},
    "Una":{"lat":31.4686,"lon":76.2701,"state":"Himachal Pradesh","elev":369},
    "Palampur":{"lat":32.1094,"lon":76.5359,"state":"Himachal Pradesh","elev":1219},
    "Kullu":{"lat":31.9592,"lon":77.1089,"state":"Himachal Pradesh","elev":1220},
    "Manali":{"lat":32.2396,"lon":77.1887,"state":"Himachal Pradesh","elev":2050},
    "Chamba":{"lat":32.5530,"lon":76.1326,"state":"Himachal Pradesh","elev":996},
    "Bilaspur":{"lat":31.3390,"lon":76.7555,"state":"Himachal Pradesh","elev":675},
    "Kangra":{"lat":32.0993,"lon":76.2673,"state":"Himachal Pradesh","elev":733},

    # ── Jharkhand ─────────────────────────────────────────────────
    "Ranchi":{"lat":23.3441,"lon":85.3096,"state":"Jharkhand","elev":651},
    "Jamshedpur":{"lat":22.8046,"lon":86.2029,"state":"Jharkhand","elev":135},
    "Dhanbad":{"lat":23.7957,"lon":86.4304,"state":"Jharkhand","elev":227},
    "Bokaro":{"lat":23.6693,"lon":86.1511,"state":"Jharkhand","elev":212},
    "Deoghar":{"lat":24.4853,"lon":86.6900,"state":"Jharkhand","elev":255},
    "Phusro":{"lat":23.7791,"lon":86.0090,"state":"Jharkhand","elev":220},
    "Hazaribagh":{"lat":23.9926,"lon":85.3637,"state":"Jharkhand","elev":614},
    "Giridih":{"lat":24.1872,"lon":86.2995,"state":"Jharkhand","elev":355},
    "Ramgarh":{"lat":23.6312,"lon":85.5130,"state":"Jharkhand","elev":449},
    "Medininagar":{"lat":23.9330,"lon":84.0920,"state":"Jharkhand","elev":330},
    "Chaibasa":{"lat":22.5497,"lon":85.8154,"state":"Jharkhand","elev":199},
    "Chakradharpur":{"lat":22.6838,"lon":85.6338,"state":"Jharkhand","elev":210},
    "Dumka":{"lat":24.2680,"lon":87.2450,"state":"Jharkhand","elev":152},
    "Simdega":{"lat":22.6139,"lon":84.5058,"state":"Jharkhand","elev":433},

    # ── Karnataka ─────────────────────────────────────────────────
    "Bengaluru":{"lat":12.9716,"lon":77.5946,"state":"Karnataka","elev":920},
    "Mysuru":{"lat":12.2958,"lon":76.6394,"state":"Karnataka","elev":763},
    "Mangaluru":{"lat":12.8698,"lon":74.8435,"state":"Karnataka","elev":22},
    "Hubballi":{"lat":15.3647,"lon":75.1240,"state":"Karnataka","elev":663},
    "Belagavi":{"lat":15.8497,"lon":74.4977,"state":"Karnataka","elev":751},
    "Kalaburagi":{"lat":17.3297,"lon":76.8343,"state":"Karnataka","elev":454},
    "Ballari":{"lat":15.1394,"lon":76.9214,"state":"Karnataka","elev":449},
    "Davanagere":{"lat":14.4644,"lon":75.9218,"state":"Karnataka","elev":569},
    "Vijayapura":{"lat":16.8302,"lon":75.7100,"state":"Karnataka","elev":594},
    "Shivamogga":{"lat":13.9299,"lon":75.5681,"state":"Karnataka","elev":594},
    "Tumkur":{"lat":13.3379,"lon":77.1173,"state":"Karnataka","elev":817},
    "Raichur":{"lat":16.2076,"lon":77.3463,"state":"Karnataka","elev":414},
    "Bidar":{"lat":17.9140,"lon":77.5199,"state":"Karnataka","elev":664},
    "Udupi":{"lat":13.3409,"lon":74.7421,"state":"Karnataka","elev":18},
    "Hassan":{"lat":13.0027,"lon":76.1004,"state":"Karnataka","elev":942},
    "Bagalkot":{"lat":16.1800,"lon":75.6960,"state":"Karnataka","elev":536},
    "Gadag":{"lat":15.4167,"lon":75.6167,"state":"Karnataka","elev":588},
    "Mandya":{"lat":12.5218,"lon":76.8951,"state":"Karnataka","elev":682},
    "Chikkamagaluru":{"lat":13.3153,"lon":75.7754,"state":"Karnataka","elev":1090},
    "Kolar":{"lat":13.1367,"lon":78.1294,"state":"Karnataka","elev":878},
    "Chamrajanagar":{"lat":11.9263,"lon":76.9439,"state":"Karnataka","elev":798},
    "Kodagu":{"lat":12.3375,"lon":75.8069,"state":"Karnataka","elev":1082},
    "Chitradurga":{"lat":14.2226,"lon":76.3975,"state":"Karnataka","elev":682},
    "Hospet":{"lat":15.2689,"lon":76.3909,"state":"Karnataka","elev":427},

    # ── Kerala ────────────────────────────────────────────────────
    "Thiruvananthapuram":{"lat":8.5241,"lon":76.9366,"state":"Kerala","elev":62},
    "Kochi":{"lat":9.9312,"lon":76.2673,"state":"Kerala","elev":7},
    "Kozhikode":{"lat":11.2588,"lon":75.7804,"state":"Kerala","elev":14},
    "Thrissur":{"lat":10.5276,"lon":76.2144,"state":"Kerala","elev":2},
    "Kollam":{"lat":8.8932,"lon":76.6141,"state":"Kerala","elev":7},
    "Alappuzha":{"lat":9.4981,"lon":76.3388,"state":"Kerala","elev":1},
    "Palakkad":{"lat":10.7867,"lon":76.6548,"state":"Kerala","elev":82},
    "Malappuram":{"lat":11.0730,"lon":76.0740,"state":"Kerala","elev":72},
    "Kannur":{"lat":11.8745,"lon":75.3704,"state":"Kerala","elev":19},
    "Kasaragod":{"lat":12.4996,"lon":74.9869,"state":"Kerala","elev":10},
    "Kottayam":{"lat":9.5916,"lon":76.5222,"state":"Kerala","elev":10},
    "Pathanamthitta":{"lat":9.2648,"lon":76.7870,"state":"Kerala","elev":30},
    "Idukki":{"lat":9.9189,"lon":76.9740,"state":"Kerala","elev":725},
    "Wayanad":{"lat":11.6854,"lon":76.1320,"state":"Kerala","elev":740},
    "Ernakulam":{"lat":9.9816,"lon":76.2999,"state":"Kerala","elev":6},
    "Thalassery":{"lat":11.7490,"lon":75.4913,"state":"Kerala","elev":5},

    # ── Madhya Pradesh ────────────────────────────────────────────
    "Bhopal":{"lat":23.2599,"lon":77.4126,"state":"Madhya Pradesh","elev":523},
    "Indore":{"lat":22.7196,"lon":75.8577,"state":"Madhya Pradesh","elev":553},
    "Jabalpur":{"lat":23.1815,"lon":79.9864,"state":"Madhya Pradesh","elev":412},
    "Gwalior":{"lat":26.2183,"lon":78.1828,"state":"Madhya Pradesh","elev":211},
    "Ujjain":{"lat":23.1828,"lon":75.7772,"state":"Madhya Pradesh","elev":491},
    "Rewa":{"lat":24.5370,"lon":81.3042,"state":"Madhya Pradesh","elev":356},
    "Satna":{"lat":24.5701,"lon":80.8327,"state":"Madhya Pradesh","elev":320},
    "Sagar":{"lat":23.8388,"lon":78.7378,"state":"Madhya Pradesh","elev":520},
    "Dewas":{"lat":22.9623,"lon":76.0552,"state":"Madhya Pradesh","elev":544},
    "Murwara":{"lat":23.8400,"lon":80.3900,"state":"Madhya Pradesh","elev":368},
    "Chhindwara":{"lat":22.0574,"lon":78.9382,"state":"Madhya Pradesh","elev":600},
    "Ratlam":{"lat":23.3326,"lon":75.0368,"state":"Madhya Pradesh","elev":495},
    "Burhanpur":{"lat":21.3046,"lon":76.2298,"state":"Madhya Pradesh","elev":257},
    "Khandwa":{"lat":21.8258,"lon":76.3517,"state":"Madhya Pradesh","elev":309},
    "Bhind":{"lat":26.5567,"lon":78.7891,"state":"Madhya Pradesh","elev":147},
    "Morena":{"lat":26.4968,"lon":77.9990,"state":"Madhya Pradesh","elev":186},
    "Shivpuri":{"lat":25.4231,"lon":77.6619,"state":"Madhya Pradesh","elev":473},
    "Mandsaur":{"lat":24.0762,"lon":75.0620,"state":"Madhya Pradesh","elev":423},
    "Singrauli":{"lat":24.1997,"lon":82.6739,"state":"Madhya Pradesh","elev":381},
    "Damoh":{"lat":23.8361,"lon":79.4398,"state":"Madhya Pradesh","elev":360},
    "Panna":{"lat":24.7174,"lon":80.1859,"state":"Madhya Pradesh","elev":361},
    "Hoshangabad":{"lat":22.7531,"lon":77.7213,"state":"Madhya Pradesh","elev":310},
    "Vidisha":{"lat":23.5245,"lon":77.8117,"state":"Madhya Pradesh","elev":423},
    "Khargone":{"lat":21.8243,"lon":75.6141,"state":"Madhya Pradesh","elev":290},

    # ── Maharashtra ───────────────────────────────────────────────
    "Mumbai":{"lat":19.0760,"lon":72.8777,"state":"Maharashtra","elev":14},
    "Pune":{"lat":18.5204,"lon":73.8567,"state":"Maharashtra","elev":560},
    "Nagpur":{"lat":21.1458,"lon":79.0882,"state":"Maharashtra","elev":310},
    "Thane":{"lat":19.2183,"lon":72.9781,"state":"Maharashtra","elev":7},
    "Nashik":{"lat":19.9975,"lon":73.7898,"state":"Maharashtra","elev":584},
    "Aurangabad":{"lat":19.8762,"lon":75.3433,"state":"Maharashtra","elev":513},
    "Solapur":{"lat":17.6805,"lon":75.9064,"state":"Maharashtra","elev":457},
    "Kolhapur":{"lat":16.7050,"lon":74.2433,"state":"Maharashtra","elev":569},
    "Amravati":{"lat":20.9374,"lon":77.7796,"state":"Maharashtra","elev":342},
    "Nanded":{"lat":19.1383,"lon":77.3210,"state":"Maharashtra","elev":367},
    "Sangli":{"lat":16.8524,"lon":74.5815,"state":"Maharashtra","elev":551},
    "Jalgaon":{"lat":21.0077,"lon":75.5626,"state":"Maharashtra","elev":209},
    "Akola":{"lat":20.7002,"lon":77.0082,"state":"Maharashtra","elev":282},
    "Latur":{"lat":18.4088,"lon":76.5604,"state":"Maharashtra","elev":540},
    "Dhule":{"lat":20.9042,"lon":74.7749,"state":"Maharashtra","elev":214},
    "Ahmednagar":{"lat":19.0952,"lon":74.7496,"state":"Maharashtra","elev":650},
    "Malegaon":{"lat":20.5579,"lon":74.5089,"state":"Maharashtra","elev":381},
    "Navi Mumbai":{"lat":19.0330,"lon":73.0297,"state":"Maharashtra","elev":5},
    "Vasai":{"lat":19.3919,"lon":72.8397,"state":"Maharashtra","elev":3},
    "Parbhani":{"lat":19.2704,"lon":76.7747,"state":"Maharashtra","elev":394},
    "Bhiwandi":{"lat":19.3001,"lon":73.0588,"state":"Maharashtra","elev":16},
    "Pimpri-Chinchwad":{"lat":18.6279,"lon":73.7997,"state":"Maharashtra","elev":559},
    "Yavatmal":{"lat":20.3888,"lon":78.1204,"state":"Maharashtra","elev":459},
    "Osmanabad":{"lat":18.1860,"lon":76.0399,"state":"Maharashtra","elev":626},
    "Chandrapur":{"lat":19.9615,"lon":79.2961,"state":"Maharashtra","elev":189},
    "Washim":{"lat":20.1119,"lon":77.1332,"state":"Maharashtra","elev":355},
    "Ratnagiri":{"lat":16.9944,"lon":73.3001,"state":"Maharashtra","elev":13},
    "Beed":{"lat":18.9890,"lon":75.7590,"state":"Maharashtra","elev":618},
    "Satara":{"lat":17.6805,"lon":74.0183,"state":"Maharashtra","elev":659},
    "Wardha":{"lat":20.7453,"lon":78.6022,"state":"Maharashtra","elev":270},
    "Gondia":{"lat":21.4604,"lon":80.1966,"state":"Maharashtra","elev":319},
    "Hingoli":{"lat":19.7163,"lon":77.1490,"state":"Maharashtra","elev":432},
    "Nandurbar":{"lat":21.3665,"lon":74.2421,"state":"Maharashtra","elev":219},
    "Buldhana":{"lat":20.5292,"lon":76.1843,"state":"Maharashtra","elev":456},

    # ── Manipur ───────────────────────────────────────────────────
    "Imphal":{"lat":24.8170,"lon":93.9368,"state":"Manipur","elev":786},
    "Thoubal":{"lat":24.6397,"lon":93.9946,"state":"Manipur","elev":770},
    "Bishnupur":{"lat":24.6300,"lon":93.7700,"state":"Manipur","elev":790},
    "Churachandpur":{"lat":24.3327,"lon":93.6763,"state":"Manipur","elev":920},

    # ── Meghalaya ─────────────────────────────────────────────────
    "Shillong":{"lat":25.5788,"lon":91.8933,"state":"Meghalaya","elev":1496},
    "Tura":{"lat":25.5144,"lon":90.2128,"state":"Meghalaya","elev":328},
    "Jowai":{"lat":25.4519,"lon":92.2000,"state":"Meghalaya","elev":1368},
    "Nongstoin":{"lat":25.5202,"lon":91.2638,"state":"Meghalaya","elev":1286},

    # ── Mizoram ───────────────────────────────────────────────────
    "Aizawl":{"lat":23.7271,"lon":92.7176,"state":"Mizoram","elev":1132},
    "Lunglei":{"lat":22.8865,"lon":92.7326,"state":"Mizoram","elev":1052},
    "Champhai":{"lat":23.4674,"lon":93.3278,"state":"Mizoram","elev":1680},

    # ── Nagaland ──────────────────────────────────────────────────
    "Kohima":{"lat":25.6751,"lon":94.1086,"state":"Nagaland","elev":1444},
    "Dimapur":{"lat":25.9041,"lon":93.7278,"state":"Nagaland","elev":232},
    "Mokokchung":{"lat":26.3251,"lon":94.5159,"state":"Nagaland","elev":1328},
    "Wokha":{"lat":26.1010,"lon":94.2627,"state":"Nagaland","elev":1390},

    # ── Odisha ────────────────────────────────────────────────────
    "Bhubaneswar":{"lat":20.2961,"lon":85.8245,"state":"Odisha","elev":45},
    "Cuttack":{"lat":20.4625,"lon":85.8828,"state":"Odisha","elev":25},
    "Rourkela":{"lat":22.2604,"lon":84.8536,"state":"Odisha","elev":219},
    "Brahmapur":{"lat":19.3149,"lon":84.7941,"state":"Odisha","elev":56},
    "Sambalpur":{"lat":21.4669,"lon":83.9812,"state":"Odisha","elev":176},
    "Puri":{"lat":19.8135,"lon":85.8312,"state":"Odisha","elev":4},
    "Bargarh":{"lat":21.3306,"lon":83.6189,"state":"Odisha","elev":181},
    "Baripada":{"lat":21.9330,"lon":86.7197,"state":"Odisha","elev":64},
    "Balasore":{"lat":21.4942,"lon":86.9335,"state":"Odisha","elev":24},
    "Bhadrak":{"lat":21.0579,"lon":86.5094,"state":"Odisha","elev":25},
    "Jeypore":{"lat":18.8536,"lon":82.5711,"state":"Odisha","elev":504},
    "Angul":{"lat":20.8430,"lon":85.1015,"state":"Odisha","elev":110},
    "Dhenkanal":{"lat":20.6618,"lon":85.5977,"state":"Odisha","elev":76},
    "Paradip":{"lat":20.3167,"lon":86.6000,"state":"Odisha","elev":4},
    "Jajpur":{"lat":20.8505,"lon":86.3347,"state":"Odisha","elev":25},
    "Jharsuguda":{"lat":21.8560,"lon":84.0060,"state":"Odisha","elev":216},
    "Kendujhar":{"lat":21.6285,"lon":85.5817,"state":"Odisha","elev":227},
    "Sundargarh":{"lat":22.1168,"lon":84.0285,"state":"Odisha","elev":262},
    "Koraput":{"lat":18.8136,"lon":82.7124,"state":"Odisha","elev":885},

    # ── Punjab ────────────────────────────────────────────────────
    "Ludhiana":{"lat":30.9010,"lon":75.8573,"state":"Punjab","elev":244},
    "Amritsar":{"lat":31.6340,"lon":74.8723,"state":"Punjab","elev":234},
    "Jalandhar":{"lat":31.3260,"lon":75.5762,"state":"Punjab","elev":228},
    "Patiala":{"lat":30.3398,"lon":76.3869,"state":"Punjab","elev":250},
    "Bathinda":{"lat":30.2110,"lon":74.9455,"state":"Punjab","elev":204},
    "Mohali":{"lat":30.7046,"lon":76.7179,"state":"Punjab","elev":310},
    "Pathankot":{"lat":32.2731,"lon":75.6522,"state":"Punjab","elev":327},
    "Hoshiarpur":{"lat":31.5143,"lon":75.9116,"state":"Punjab","elev":368},
    "Moga":{"lat":30.8160,"lon":75.1730,"state":"Punjab","elev":215},
    "Batala":{"lat":31.8204,"lon":75.2009,"state":"Punjab","elev":232},
    "Gurdaspur":{"lat":32.0387,"lon":75.4048,"state":"Punjab","elev":253},
    "Firozpur":{"lat":30.9254,"lon":74.6070,"state":"Punjab","elev":188},
    "Nawanshahr":{"lat":31.1240,"lon":76.1173,"state":"Punjab","elev":308},
    "Muktsar":{"lat":30.4755,"lon":74.5161,"state":"Punjab","elev":196},
    "Ropar":{"lat":30.9648,"lon":76.5261,"state":"Punjab","elev":282},
    "Fatehgarh Sahib":{"lat":30.6481,"lon":76.3880,"state":"Punjab","elev":265},
    "Faridkot":{"lat":30.6703,"lon":74.7550,"state":"Punjab","elev":199},
    "Sangrur":{"lat":30.2402,"lon":75.8439,"state":"Punjab","elev":225},
    "Barnala":{"lat":30.3817,"lon":75.5451,"state":"Punjab","elev":224},
    "Kapurthala":{"lat":31.3814,"lon":75.3815,"state":"Punjab","elev":220},

    # ── Rajasthan ─────────────────────────────────────────────────
    "Jaipur":{"lat":26.9124,"lon":75.7873,"state":"Rajasthan","elev":431},
    "Jodhpur":{"lat":26.2389,"lon":73.0243,"state":"Rajasthan","elev":231},
    "Udaipur":{"lat":24.5854,"lon":73.7125,"state":"Rajasthan","elev":598},
    "Kota":{"lat":25.2138,"lon":75.8648,"state":"Rajasthan","elev":271},
    "Bikaner":{"lat":28.0229,"lon":73.3119,"state":"Rajasthan","elev":226},
    "Ajmer":{"lat":26.4499,"lon":74.6399,"state":"Rajasthan","elev":486},
    "Bhilwara":{"lat":25.3407,"lon":74.6313,"state":"Rajasthan","elev":419},
    "Alwar":{"lat":27.5665,"lon":76.6293,"state":"Rajasthan","elev":271},
    "Bharatpur":{"lat":27.2152,"lon":77.4938,"state":"Rajasthan","elev":174},
    "Sikar":{"lat":27.6094,"lon":75.1398,"state":"Rajasthan","elev":427},
    "Pali":{"lat":25.7711,"lon":73.3234,"state":"Rajasthan","elev":212},
    "Barmer":{"lat":25.7463,"lon":71.3933,"state":"Rajasthan","elev":229},
    "Tonk":{"lat":26.1701,"lon":75.7900,"state":"Rajasthan","elev":265},
    "Sri Ganganagar":{"lat":29.9094,"lon":73.8773,"state":"Rajasthan","elev":176},
    "Hanumangarh":{"lat":29.5818,"lon":74.3328,"state":"Rajasthan","elev":175},
    "Banswara":{"lat":23.5469,"lon":74.4394,"state":"Rajasthan","elev":249},
    "Dungarpur":{"lat":23.8423,"lon":73.7148,"state":"Rajasthan","elev":305},
    "Chittorgarh":{"lat":24.8887,"lon":74.6269,"state":"Rajasthan","elev":395},
    "Jhalawar":{"lat":24.5976,"lon":76.1652,"state":"Rajasthan","elev":360},
    "Jalore":{"lat":25.3472,"lon":72.6124,"state":"Rajasthan","elev":110},
    "Dausa":{"lat":26.8862,"lon":76.3344,"state":"Rajasthan","elev":380},
    "Churu":{"lat":28.3001,"lon":74.9666,"state":"Rajasthan","elev":289},
    "Sawai Madhopur":{"lat":26.0070,"lon":76.3530,"state":"Rajasthan","elev":271},
    "Karauli":{"lat":26.5015,"lon":77.0215,"state":"Rajasthan","elev":244},
    "Dholpur":{"lat":26.7060,"lon":77.8950,"state":"Rajasthan","elev":178},
    "Nagaur":{"lat":27.2022,"lon":73.7318,"state":"Rajasthan","elev":330},
    "Bundi":{"lat":25.4388,"lon":75.6472,"state":"Rajasthan","elev":263},
    "Rajsamand":{"lat":25.0696,"lon":73.8826,"state":"Rajasthan","elev":546},
    "Pratapgarh":{"lat":23.8761,"lon":74.7789,"state":"Rajasthan","elev":575},
    "Sirohi":{"lat":24.8855,"lon":72.8618,"state":"Rajasthan","elev":317},
    "Jaisalmer":{"lat":26.9157,"lon":70.9083,"state":"Rajasthan","elev":225},
    "Jhunjhunu":{"lat":28.1282,"lon":75.3963,"state":"Rajasthan","elev":346},
    "Baran":{"lat":25.1003,"lon":76.5159,"state":"Rajasthan","elev":280},

    # ── Sikkim ────────────────────────────────────────────────────
    "Gangtok":{"lat":27.3389,"lon":88.6065,"state":"Sikkim","elev":1650},
    "Namchi":{"lat":27.1666,"lon":88.3639,"state":"Sikkim","elev":1371},
    "Gyalshing":{"lat":27.2898,"lon":88.2671,"state":"Sikkim","elev":1820},
    "Mangan":{"lat":27.5136,"lon":88.5321,"state":"Sikkim","elev":1005},

    # ── Tamil Nadu ────────────────────────────────────────────────
    "Chennai":{"lat":13.0827,"lon":80.2707,"state":"Tamil Nadu","elev":6},
    "Coimbatore":{"lat":11.0168,"lon":76.9558,"state":"Tamil Nadu","elev":411},
    "Madurai":{"lat":9.9252,"lon":78.1198,"state":"Tamil Nadu","elev":101},
    "Tiruchirappalli":{"lat":10.7905,"lon":78.7047,"state":"Tamil Nadu","elev":78},
    "Salem":{"lat":11.6643,"lon":78.1460,"state":"Tamil Nadu","elev":278},
    "Tirunelveli":{"lat":8.7139,"lon":77.7567,"state":"Tamil Nadu","elev":55},
    "Tiruppur":{"lat":11.1085,"lon":77.3411,"state":"Tamil Nadu","elev":306},
    "Vellore":{"lat":12.9165,"lon":79.1325,"state":"Tamil Nadu","elev":216},
    "Erode":{"lat":11.3410,"lon":77.7172,"state":"Tamil Nadu","elev":180},
    "Thoothukkudi":{"lat":8.7642,"lon":78.1348,"state":"Tamil Nadu","elev":13},
    "Dindigul":{"lat":10.3624,"lon":77.9695,"state":"Tamil Nadu","elev":293},
    "Thanjavur":{"lat":10.7870,"lon":79.1378,"state":"Tamil Nadu","elev":57},
    "Ranipet":{"lat":12.9300,"lon":79.3300,"state":"Tamil Nadu","elev":98},
    "Sivakasi":{"lat":9.4531,"lon":77.7970,"state":"Tamil Nadu","elev":117},
    "Karur":{"lat":10.9601,"lon":78.0766,"state":"Tamil Nadu","elev":122},
    "Udhagamandalam":{"lat":11.4102,"lon":76.6950,"state":"Tamil Nadu","elev":2240},
    "Hosur":{"lat":12.7409,"lon":77.8253,"state":"Tamil Nadu","elev":900},
    "Nagercoil":{"lat":8.1833,"lon":77.4119,"state":"Tamil Nadu","elev":10},
    "Kanchipuram":{"lat":12.8185,"lon":79.6947,"state":"Tamil Nadu","elev":83},
    "Kumbakonam":{"lat":10.9617,"lon":79.3788,"state":"Tamil Nadu","elev":29},
    "Pudukkottai":{"lat":10.3833,"lon":78.8001,"state":"Tamil Nadu","elev":84},
    "Nagapattinam":{"lat":10.7631,"lon":79.8449,"state":"Tamil Nadu","elev":3},
    "Villupuram":{"lat":11.9401,"lon":79.4861,"state":"Tamil Nadu","elev":36},
    "Cuddalore":{"lat":11.7447,"lon":79.7689,"state":"Tamil Nadu","elev":12},
    "Dharmapuri":{"lat":12.1278,"lon":78.1572,"state":"Tamil Nadu","elev":400},
    "Krishnagiri":{"lat":12.5186,"lon":78.2137,"state":"Tamil Nadu","elev":415},
    "Ariyalur":{"lat":11.1400,"lon":79.0800,"state":"Tamil Nadu","elev":70},
    "Perambalur":{"lat":11.2315,"lon":78.8802,"state":"Tamil Nadu","elev":112},
    "Ramanathapuram":{"lat":9.3762,"lon":78.8308,"state":"Tamil Nadu","elev":14},
    "Virudhunagar":{"lat":9.5678,"lon":77.9560,"state":"Tamil Nadu","elev":89},
    "Tiruvannamalai":{"lat":12.2253,"lon":79.0747,"state":"Tamil Nadu","elev":192},

    # ── Telangana ─────────────────────────────────────────────────
    "Hyderabad":{"lat":17.3850,"lon":78.4867,"state":"Telangana","elev":542},
    "Warangal":{"lat":17.9784,"lon":79.5941,"state":"Telangana","elev":302},
    "Nizamabad":{"lat":18.6725,"lon":78.0941,"state":"Telangana","elev":394},
    "Karimnagar":{"lat":18.4386,"lon":79.1288,"state":"Telangana","elev":259},
    "Khammam":{"lat":17.2473,"lon":80.1514,"state":"Telangana","elev":68},
    "Mahbubnagar":{"lat":16.7388,"lon":77.9914,"state":"Telangana","elev":499},
    "Nalgonda":{"lat":17.0575,"lon":79.2671,"state":"Telangana","elev":367},
    "Adilabad":{"lat":19.6644,"lon":78.5320,"state":"Telangana","elev":250},
    "Siddipet":{"lat":18.1018,"lon":78.8520,"state":"Telangana","elev":370},
    "Ramagundam":{"lat":18.7558,"lon":79.4742,"state":"Telangana","elev":189},
    "Mancherial":{"lat":18.8700,"lon":79.4600,"state":"Telangana","elev":170},
    "Secunderabad":{"lat":17.4399,"lon":78.4983,"state":"Telangana","elev":536},
    "Miryalaguda":{"lat":16.8693,"lon":79.5648,"state":"Telangana","elev":361},
    "Suryapet":{"lat":17.1386,"lon":79.6223,"state":"Telangana","elev":90},
    "Sangareddy":{"lat":17.6246,"lon":78.0839,"state":"Telangana","elev":552},

    # ── Tripura ───────────────────────────────────────────────────
    "Agartala":{"lat":23.8315,"lon":91.2868,"state":"Tripura","elev":13},
    "Dharmanagar":{"lat":24.3770,"lon":92.1660,"state":"Tripura","elev":50},
    "Udaipur":{"lat":23.5351,"lon":91.4872,"state":"Tripura","elev":30},  # Tripura Udaipur
    "Kailasahar":{"lat":24.3330,"lon":92.0170,"state":"Tripura","elev":25},

    # ── Uttar Pradesh ─────────────────────────────────────────────
    "Lucknow":{"lat":26.8467,"lon":80.9462,"state":"Uttar Pradesh","elev":123},
    "Kanpur":{"lat":26.4499,"lon":80.3319,"state":"Uttar Pradesh","elev":126},
    "Agra":{"lat":27.1767,"lon":78.0081,"state":"Uttar Pradesh","elev":169},
    "Varanasi":{"lat":25.3176,"lon":82.9739,"state":"Uttar Pradesh","elev":80},
    "Allahabad":{"lat":25.4358,"lon":81.8463,"state":"Uttar Pradesh","elev":98},
    "Ghaziabad":{"lat":28.6692,"lon":77.4538,"state":"Uttar Pradesh","elev":209},
    "Noida":{"lat":28.5355,"lon":77.3910,"state":"Uttar Pradesh","elev":199},
    "Meerut":{"lat":28.9845,"lon":77.7064,"state":"Uttar Pradesh","elev":220},
    "Mathura":{"lat":27.4924,"lon":77.6737,"state":"Uttar Pradesh","elev":174},
    "Moradabad":{"lat":28.8386,"lon":78.7733,"state":"Uttar Pradesh","elev":194},
    "Aligarh":{"lat":27.8974,"lon":78.0880,"state":"Uttar Pradesh","elev":194},
    "Bareilly":{"lat":28.3670,"lon":79.4304,"state":"Uttar Pradesh","elev":173},
    "Saharanpur":{"lat":29.9640,"lon":77.5461,"state":"Uttar Pradesh","elev":274},
    "Gorakhpur":{"lat":26.7606,"lon":83.3732,"state":"Uttar Pradesh","elev":84},
    "Firozabad":{"lat":27.1591,"lon":78.3950,"state":"Uttar Pradesh","elev":188},
    "Jhansi":{"lat":25.4484,"lon":78.5685,"state":"Uttar Pradesh","elev":285},
    "Muzaffarnagar":{"lat":29.4727,"lon":77.7085,"state":"Uttar Pradesh","elev":267},
    "Faizabad":{"lat":26.7734,"lon":82.1442,"state":"Uttar Pradesh","elev":97},
    "Shahjahanpur":{"lat":27.8833,"lon":79.9150,"state":"Uttar Pradesh","elev":165},
    "Rampur":{"lat":28.8087,"lon":79.0263,"state":"Uttar Pradesh","elev":181},
    "Sitapur":{"lat":27.5680,"lon":80.6823,"state":"Uttar Pradesh","elev":133},
    "Lakhimpur":{"lat":27.9479,"lon":80.7769,"state":"Uttar Pradesh","elev":146},
    "Unnao":{"lat":26.5475,"lon":80.4900,"state":"Uttar Pradesh","elev":115},
    "Rae Bareli":{"lat":26.2183,"lon":81.2394,"state":"Uttar Pradesh","elev":115},
    "Etawah":{"lat":26.7860,"lon":79.0240,"state":"Uttar Pradesh","elev":130},
    "Mainpuri":{"lat":27.2264,"lon":79.0234,"state":"Uttar Pradesh","elev":165},
    "Bulandshahr":{"lat":28.4008,"lon":77.8497,"state":"Uttar Pradesh","elev":206},
    "Hapur":{"lat":28.7261,"lon":77.7805,"state":"Uttar Pradesh","elev":208},
    "Budaun":{"lat":28.0394,"lon":79.1273,"state":"Uttar Pradesh","elev":190},
    "Bahraich":{"lat":27.5750,"lon":81.5975,"state":"Uttar Pradesh","elev":124},
    "Hardoi":{"lat":27.3955,"lon":79.9991,"state":"Uttar Pradesh","elev":138},
    "Ballia":{"lat":25.7609,"lon":84.1476,"state":"Uttar Pradesh","elev":63},
    "Basti":{"lat":26.7949,"lon":82.7352,"state":"Uttar Pradesh","elev":102},
    "Deoria":{"lat":26.5052,"lon":83.7836,"state":"Uttar Pradesh","elev":69},
    "Azamgarh":{"lat":26.0678,"lon":83.1834,"state":"Uttar Pradesh","elev":67},
    "Jaunpur":{"lat":25.7461,"lon":82.6836,"state":"Uttar Pradesh","elev":82},
    "Gonda":{"lat":27.1337,"lon":81.9605,"state":"Uttar Pradesh","elev":105},
    "Sultanpur":{"lat":26.2648,"lon":82.0723,"state":"Uttar Pradesh","elev":99},
    "Fatehpur":{"lat":25.9295,"lon":80.8148,"state":"Uttar Pradesh","elev":101},
    "Pratapgarh":{"lat":25.8971,"lon":81.9937,"state":"Uttar Pradesh","elev":94},
    "Orai":{"lat":25.9943,"lon":79.4527,"state":"Uttar Pradesh","elev":160},
    "Mirzapur":{"lat":25.1451,"lon":82.5697,"state":"Uttar Pradesh","elev":80},
    "Mau":{"lat":25.9428,"lon":83.5602,"state":"Uttar Pradesh","elev":66},
    "Banda":{"lat":25.4762,"lon":80.3353,"state":"Uttar Pradesh","elev":114},
    "Hamirpur":{"lat":25.9556,"lon":80.1509,"state":"Uttar Pradesh","elev":124},

    # ── Uttarakhand ───────────────────────────────────────────────
    "Dehradun":{"lat":30.3165,"lon":78.0322,"state":"Uttarakhand","elev":640},
    "Haridwar":{"lat":29.9457,"lon":78.1642,"state":"Uttarakhand","elev":314},
    "Roorkee":{"lat":29.8543,"lon":77.8880,"state":"Uttarakhand","elev":274},
    "Haldwani":{"lat":29.2183,"lon":79.5130,"state":"Uttarakhand","elev":424},
    "Rudrapur":{"lat":28.9845,"lon":79.3902,"state":"Uttarakhand","elev":220},
    "Rishikesh":{"lat":30.0869,"lon":78.2676,"state":"Uttarakhand","elev":356},
    "Almora":{"lat":29.5971,"lon":79.6591,"state":"Uttarakhand","elev":1638},
    "Mussoorie":{"lat":30.4540,"lon":78.0648,"state":"Uttarakhand","elev":2005},
    "Nainital":{"lat":29.3803,"lon":79.4636,"state":"Uttarakhand","elev":2084},
    "Pithoragarh":{"lat":29.5829,"lon":80.2182,"state":"Uttarakhand","elev":1814},
    "Kashipur":{"lat":29.2087,"lon":78.9573,"state":"Uttarakhand","elev":218},
    "Ramnagar":{"lat":29.3952,"lon":79.1254,"state":"Uttarakhand","elev":345},

    # ── West Bengal ───────────────────────────────────────────────
    "Kolkata":{"lat":22.5726,"lon":88.3639,"state":"West Bengal","elev":9},
    "Howrah":{"lat":22.5958,"lon":88.2636,"state":"West Bengal","elev":12},
    "Siliguri":{"lat":26.7271,"lon":88.3953,"state":"West Bengal","elev":122},
    "Durgapur":{"lat":23.5204,"lon":87.3119,"state":"West Bengal","elev":73},
    "Asansol":{"lat":23.6739,"lon":86.9524,"state":"West Bengal","elev":103},
    "Bardhaman":{"lat":23.2324,"lon":87.8615,"state":"West Bengal","elev":36},
    "Malda":{"lat":25.0109,"lon":88.1432,"state":"West Bengal","elev":29},
    "Baharampur":{"lat":24.1025,"lon":88.2507,"state":"West Bengal","elev":22},
    "Habra":{"lat":22.8296,"lon":88.6580,"state":"West Bengal","elev":9},
    "Kharagpur":{"lat":22.3460,"lon":87.2320,"state":"West Bengal","elev":40},
    "Jalpaiguri":{"lat":26.5454,"lon":88.7182,"state":"West Bengal","elev":56},
    "Darjeeling":{"lat":27.0410,"lon":88.2663,"state":"West Bengal","elev":2042},
    "Cooch Behar":{"lat":26.3243,"lon":89.4459,"state":"West Bengal","elev":48},
    "Krishnanagar":{"lat":23.4003,"lon":88.5005,"state":"West Bengal","elev":14},
    "Bankura":{"lat":23.2300,"lon":87.0700,"state":"West Bengal","elev":109},
    "Midnapore":{"lat":22.4255,"lon":87.3195,"state":"West Bengal","elev":24},
    "Raiganj":{"lat":25.6222,"lon":88.1200,"state":"West Bengal","elev":24},
    "Balurghat":{"lat":25.2148,"lon":88.7745,"state":"West Bengal","elev":27},

    # ── Delhi ─────────────────────────────────────────────────────
    "New Delhi":{"lat":28.6139,"lon":77.2090,"state":"Delhi","elev":216},
    "Delhi":{"lat":28.7041,"lon":77.1025,"state":"Delhi","elev":216},

    # ── Chandigarh UT ─────────────────────────────────────────────
    "Chandigarh":{"lat":30.7333,"lon":76.7794,"state":"Chandigarh","elev":321},

    # ── Puducherry ────────────────────────────────────────────────
    "Puducherry":{"lat":11.9416,"lon":79.8083,"state":"Puducherry","elev":15},
    "Karaikal":{"lat":10.9254,"lon":79.8380,"state":"Puducherry","elev":3},
    "Mahe":{"lat":11.7014,"lon":75.5356,"state":"Puducherry","elev":8},

    # ── Jammu & Kashmir ───────────────────────────────────────────
    "Jammu":{"lat":32.7266,"lon":74.8570,"state":"Jammu & Kashmir","elev":327},
    "Srinagar":{"lat":34.0837,"lon":74.7973,"state":"Jammu & Kashmir","elev":1585},
    "Anantnag":{"lat":33.7311,"lon":75.1487,"state":"Jammu & Kashmir","elev":1602},
    "Baramulla":{"lat":34.2120,"lon":74.3441,"state":"Jammu & Kashmir","elev":1593},
    "Sopore":{"lat":34.3021,"lon":74.4665,"state":"Jammu & Kashmir","elev":1583},
    "Udhampur":{"lat":32.9159,"lon":75.1393,"state":"Jammu & Kashmir","elev":756},
    "Kathua":{"lat":32.3841,"lon":75.5227,"state":"Jammu & Kashmir","elev":531},
    "Punch":{"lat":33.7721,"lon":74.0965,"state":"Jammu & Kashmir","elev":1067},

    # ── Ladakh ────────────────────────────────────────────────────
    "Leh":{"lat":34.1526,"lon":77.5771,"state":"Ladakh","elev":3524},
    "Kargil":{"lat":34.5539,"lon":76.1349,"state":"Ladakh","elev":2676},

    # ── Andaman & Nicobar ─────────────────────────────────────────
    "Port Blair":{"lat":11.6234,"lon":92.7265,"state":"Andaman & Nicobar","elev":16},

    # ── Dadra & Nagar Haveli / Daman & Diu ────────────────────────
    "Silvassa":{"lat":20.2766,"lon":73.0087,"state":"Dadra & Nagar Haveli","elev":40},
    "Daman":{"lat":20.3974,"lon":72.8328,"state":"Daman & Diu","elev":8},
    "Diu":{"lat":20.7144,"lon":70.9874,"state":"Daman & Diu","elev":3},

    # ── Lakshadweep ───────────────────────────────────────────────
    "Kavaratti":{"lat":10.5669,"lon":72.6420,"state":"Lakshadweep","elev":2},
}

WMO = {0:"Clear Sky",1:"Mainly Clear",2:"Partly Cloudy",3:"Overcast",
       45:"Foggy",51:"Light Drizzle",61:"Light Rain",63:"Moderate Rain",
       65:"Heavy Rain",80:"Rain Showers",95:"Thunderstorm"}

BIN_DATA = {
    "Central Plaza":  (["CP-Gate","CP-Fountain","CP-Parking"],   0.75),
    "Market Block":   (["MKT-A","MKT-B","MKT-C","MKT-Exit"],    0.85),
    "North Gate":     (["NG-Entry","NG-Road"],                    0.45),
    "South Corridor": (["SC-Left","SC-Right"],                    0.50),
    "East Park":      (["EP-Main","EP-Jog"],                      0.30),
    "West Residency": (["WR-Blk1","WR-Blk2"],                    0.40),
    "Bus Terminal":   (["BT-P1","BT-P2","BT-Entry"],              0.80),
    "Hospital Zone":  (["HZ-OPD","HZ-Park"],                      0.55),
}
YOLO_CLS = {2:"car",3:"motorcycle",5:"bus",7:"truck",1:"bicycle"}


# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
for k,v in [("weather",None),("prediction",None),("bins",None),
            ("det",None),("t_hist",deque(maxlen=60))]:
    if k not in st.session_state: st.session_state[k]=v

if st.session_state.bins is None:
    def _make_bins():
        out,n=[],1
        for zone,(locs,bias) in BIN_DATA.items():
            for loc in locs:
                f=max(0,min(100,random.gauss(bias*100,16)))
                if random.random()<.08: f=random.uniform(88,100)
                if random.random()<.10: f=random.uniform(0,12)
                out.append({"id":f"BIN-{n:03d}","zone":zone,"loc":loc,
                            "fill":round(f,1),"cap":random.choice([120,180,240]),
                            "upd":datetime.now().strftime("%H:%M"),
                            "_r":random.uniform(.05,.3)})
                n+=1
        return out
    st.session_state.bins=_make_bins()


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def _bs(f):
    return("CRITICAL" if f>=95 else "FULL" if f>=80 else
           "HALF" if f>=50 else "QUARTER" if f>=25 else "EMPTY")

def _bc(s):
    return{"CRITICAL":"#ef4444","FULL":"#f97316","HALF":"#f59e0b",
           "QUARTER":"#84cc16","EMPTY":"#22c55e"}.get(s,"#22c55e")

def _hi(T,H):
    if T<27: return T
    return round(-8.784+1.611*T+2.338*H-0.146*T*H-0.0123*T*T-0.0164*H*H
                 +0.00221*T*T*H+0.000726*T*H*H-3.58e-6*T*T*H*H,1)

def _uv(u): return["Low","Moderate","High","Very High","Extreme"][min(int(u/2.5),4)]

def _mc(col,lbl,val,unit,sub,cls):
    with col:
        st.markdown(f"""<div class='mc {cls}'><div class='mc-lbl'>{lbl}</div>
            <div class='mc-val'>{val}<span class='mc-unit'>{unit}</span></div>
            <div class='mc-sub'>{sub}</div></div>""",unsafe_allow_html=True)

def _sh(icon,title,desc,bg):
    st.markdown(f"""<div class='sh'><div class='sh-icon' style='background:{bg}'>{icon}</div>
        <div><div class='sh-title'>{title}</div>
        <div class='sh-desc'>{desc}</div></div></div>""",unsafe_allow_html=True)

def _df(w=8,h=3.2):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor("#07101f"); ax.set_facecolor("#080f1c")
    ax.tick_params(colors="#3d6080",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2a40")
    ax.xaxis.label.set_color("#3d6080"); ax.yaxis.label.set_color("#3d6080")
    ax.title.set_color("#7dd3fc"); return fig,ax


# ════════════════════════════════════════════════════════════════════
# WEATHER
# ════════════════════════════════════════════════════════════════════
def fetch_w(city):
    info=CITIES.get(city)
    if not info: return _sw(city,20.,78.)
    try:
        r=requests.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude":info["lat"],"longitude":info["lon"],
            "current":["temperature_2m","relative_humidity_2m","apparent_temperature",
                       "precipitation","weather_code","cloud_cover","pressure_msl",
                       "wind_speed_10m","wind_direction_10m","uv_index"],
            "timezone":"auto","forecast_days":1},timeout=8); r.raise_for_status()
        c=r.json().get("current",{})
        T=float(c.get("temperature_2m",30)); H=float(c.get("relative_humidity_2m",55))
        return{"city":city,"state":info.get("state",""),"lat":info["lat"],"lon":info["lon"],
               "elev":info.get("elev","—"),"temperature":round(T,1),"humidity":round(H,1),
               "feels_like":round(float(c.get("apparent_temperature",T)),1),
               "dew_point":round(T-((100-H)/5),1),
               "wind_speed":round(float(c.get("wind_speed_10m",10)),1),
               "wind_dir":round(float(c.get("wind_direction_10m",200)),0),
               "cloud_cover":round(float(c.get("cloud_cover",30)),0),
               "pressure":round(float(c.get("pressure_msl",1010)),1),
               "rain_mm":round(float(c.get("precipitation",0)),2),
               "uv_index":round(float(c.get("uv_index",5)),1),
               "condition":WMO.get(int(c.get("weather_code",2)),"Partly Cloudy"),
               "source":"Open-Meteo","at":datetime.now().strftime("%H:%M:%S")}
    except Exception: return _sw(city,info["lat"],info["lon"])

def _sw(city,lat,lon):
    h=datetime.now().hour; T=30+5*math.sin((h-6)*math.pi/12)+random.uniform(-1,1)
    H=random.uniform(45,75)
    return{"city":city,"state":"","lat":lat,"lon":lon,"elev":"—",
           "temperature":round(T,1),"humidity":round(H,1),"feels_like":round(T+2,1),
           "dew_point":round(T-8,1),"wind_speed":round(random.uniform(5,20),1),
           "wind_dir":round(random.uniform(0,360),0),"cloud_cover":round(random.uniform(10,60),0),
           "pressure":round(random.uniform(1005,1015),1),"rain_mm":0.0,
           "uv_index":round(random.uniform(2,9),1),"condition":"Partly Cloudy",
           "source":"Simulated","at":datetime.now().strftime("%H:%M:%S")}


# ════════════════════════════════════════════════════════════════════
# YOLO  —  ALL VEHICLES GET GREEN BOUNDING BOXES
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _yolo():
    if not YOLO_OK: return None
    try: return _YOLO("yolov8n.pt")
    except Exception: return None

# A single bright green colour for all vehicle boxes
BOX_GREEN   = (0, 220, 60)          # bright green BGR
LABEL_BG    = (0, 160, 40)          # slightly darker green for label bg
LABEL_TEXT  = (255, 255, 255)       # white text

def _draw_box(img, x1, y1, x2, y2, label, conf):
    """Draw a clean green bounding box with label."""
    # Main box
    cv2.rectangle(img, (x1,y1), (x2,y2), BOX_GREEN, 2)
    # Corner ticks for a "smart city" look
    tick = 10
    for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*tick,cy),BOX_GREEN,3)
        cv2.line(img,(cx,cy),(cx,cy+dy*tick),BOX_GREEN,3)
    # Label background
    txt  = f"{label}  {conf:.0%}"
    tw,th = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.rectangle(img,(x1,y1-th-8),(x1+tw+8,y1), LABEL_BG,-1)
    cv2.putText(img, txt, (x1+4,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_TEXT, 1, cv2.LINE_AA)

def _detect(img_bgr, conf_thr=0.35):
    model=_yolo()
    if model is None: return _sim_det(img_bgr)
    results=model(img_bgr,verbose=False,conf=conf_thr)[0]
    counts=defaultdict(int); ann=img_bgr.copy()
    for box in results.boxes:
        cid=int(box.cls[0]); lbl=YOLO_CLS.get(cid)
        if not lbl: continue
        counts[lbl]+=1
        x1,y1,x2,y2=map(int,box.xyxy[0])
        _draw_box(ann,x1,y1,x2,y2,lbl,float(box.conf[0]))
    return _bd(counts,ann)

def _sim_det(img):
    """Simulate detection with green boxes when YOLO not installed."""
    counts={"car":random.randint(4,18),"motorcycle":random.randint(0,7),
            "bus":random.randint(0,3),"truck":random.randint(0,3),"bicycle":random.randint(0,4)}
    ann=img.copy(); h,w=ann.shape[:2]
    # Place realistic-looking boxes across the image
    placed=[]
    for vtype,n in counts.items():
        for _ in range(n):
            bw=random.randint(55,110); bh=random.randint(30,60)
            bx=random.randint(5,max(6,w-bw-5)); by=random.randint(5,max(6,h-bh-5))
            # avoid too much overlap
            ok=True
            for (px,py,pw,ph) in placed:
                if abs(bx-px)<pw and abs(by-py)<ph: ok=False; break
            if ok:
                placed.append((bx,by,bw,bh))
                _draw_box(ann,bx,by,bx+bw,by+bh,vtype,random.uniform(0.55,0.95))
    return _bd(counts,ann)

def _bd(counts,ann):
    v={k:c for k,c in counts.items() if k in YOLO_CLS.values()}
    tot=sum(v.values())
    sc=(v.get("car",0)*1.0+v.get("motorcycle",0)*.5+
        v.get("bus",0)*2.5+v.get("truck",0)*2.0+v.get("bicycle",0)*.3)
    dens="HIGH" if sc>=25 else "MEDIUM" if sc>=10 else "LOW"
    return{"counts":v,"total":tot,"score":round(sc,1),"density":dens,
           "ann":ann,"at":datetime.now().strftime("%H:%M:%S")}


# ════════════════════════════════════════════════════════════════════
# ML
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _ml():
    pkl=os.path.join(os.path.dirname(__file__),"model.pkl")
    if os.path.exists(pkl):
        try:
            with open(pkl,"rb") as f: return pickle.load(f),"trained"
        except Exception: pass
    if not SK_OK: return None,"unavailable"
    np.random.seed(42); N=4000
    tv=np.random.randint(50,5000,N).astype(float); T=np.random.uniform(15,50,N)
    H=np.random.uniform(15,98,N); p=np.random.randint(200,55000,N).astype(float)
    hr=np.random.randint(0,24,N).astype(float); we=np.random.randint(0,2,N).astype(float)
    wc=np.random.randint(0,6,N).astype(float); rq=np.random.randint(0,5,N).astype(float)
    X=np.column_stack([tv,T,H,p,hr,we,wc,rq])
    sc=(tv/5000*40+np.clip((T-20)/30,0,1)*20+H/100*10+p/55000*15
        +np.where((hr>=7)&(hr<=9),12,0)+np.where((hr>=17)&(hr<=19),12,0)
        +(1-we)*5+np.where(wc>=3,8,0)+(4-rq)/4*10+np.random.normal(0,5,N))
    lbl=np.where(sc>=62,"HIGH",np.where(sc>=38,"MEDIUM","LOW"))
    m=RandomForestClassifier(n_estimators=80,max_depth=8,random_state=42)
    m.fit(X,lbl); return m,"fallback"

def ml_pred(tv,T,H,pop,hr,we,wcs,rqs):
    model,mt=_ml()
    wm={"Clear":0,"Partly Cloudy":1,"Overcast":2,"Rain":3,"Heavy Rain":4,"Fog":5}
    rm={"Poor":0,"Below Average":1,"Average":2,"Good":3,"Excellent":4}
    X=np.array([[tv,T,H,pop,hr,we,wm.get(wcs,1),rm.get(rqs,2)]])
    if model is None:
        d="HIGH" if tv>3500 else "MEDIUM" if tv>1500 else "LOW"
        return{"d":d,"pb":{"HIGH":.7,"MEDIUM":.2,"LOW":.1},"mt":mt,"fi":{}}
    pred=model.predict(X)[0]
    pb={}
    if hasattr(model,"predict_proba"):
        for c,p in zip(model.classes_,model.predict_proba(X)[0]): pb[c]=round(float(p),3)
    for k in ["HIGH","MEDIUM","LOW"]: pb.setdefault(k,0.)
    fi={}
    if hasattr(model,"feature_importances_"):
        ns=["Traffic","Temp","Humidity","Population","Hour","Weekend","Weather","Road"]
        fi={ns[i]:round(float(v),4) for i,v in enumerate(model.feature_importances_)}
    return{"d":str(pred),"pb":pb,"mt":mt,"fi":fi}


# ════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════
def _bar(counts):
    if not counts: return None
    labels=[k for k,v in counts.items() if v>0]; vals=[counts[k] for k in labels]
    fig,ax=_df(7,3)
    bars=ax.bar(labels,vals,color=["#22c55e"]*len(labels),
                edgecolor="#1a2a40",linewidth=.5,width=.6)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,str(v),
                ha="center",va="bottom",color="#cbd5e1",fontsize=10,fontweight="bold")
    ax.set_title("Detected Vehicles by Type",pad=8,fontsize=11)
    ax.set_ylabel("Count"); ax.set_ylim(0,max(vals)*1.3 if vals else 5)
    ax.grid(axis="y",alpha=.15,color="#1e3a5f"); plt.tight_layout(); return fig

def _pie(counts):
    f={k:v for k,v in counts.items() if v>0}
    if not f or sum(f.values())==0: return None
    fig,ax=plt.subplots(figsize=(4,3.5))
    fig.patch.set_facecolor("#07101f"); ax.set_facecolor("#07101f")
    cols=["#22c55e","#16a34a","#4ade80","#86efac","#bbf7d0"][:len(f)]
    _,_,at=ax.pie(f.values(),labels=f.keys(),colors=cols,autopct="%1.0f%%",
                  startangle=90,textprops={"color":"#94a3b8","fontsize":8},pctdistance=.74)
    for a in at: a.set_color("#e2e8f0")
    ax.set_title("Vehicle Mix",color="#7dd3fc",fontsize=11,pad=6)
    plt.tight_layout(); return fig

def _trend(hist):
    if len(hist)<2: return None
    arr=list(hist); fig,ax=_df(8,3)
    ax.plot(range(len(arr)),arr,color="#22c55e",lw=2,marker="o",ms=3,markerfacecolor="#16a34a")
    ax.fill_between(range(len(arr)),arr,alpha=.12,color="#22c55e")
    ax.set_title("Traffic Score Trend",pad=8,fontsize=11)
    ax.set_ylabel("Score"); ax.set_xlabel("Analysis")
    ax.grid(alpha=.15,color="#1e3a5f"); plt.tight_layout(); return fig

def _zone_chart(bins):
    zones={}
    for b in bins: zones.setdefault(b["zone"],[]).append(b["fill"])
    zn=list(zones.keys()); av=[sum(v)/len(v) for v in zones.values()]
    fig,ax=_df(8,3)
    cols=["#ef4444" if a>=80 else "#f59e0b" if a>=50 else "#22c55e" for a in av]
    bars=ax.barh(zn,av,color=cols,edgecolor="#1a2a40",linewidth=.5,height=.6)
    for bar,v in zip(bars,av):
        ax.text(v+.5,bar.get_y()+bar.get_height()/2,f"{v:.0f}%",
                va="center",color="#cbd5e1",fontsize=8)
    ax.axvline(80,color="#ef4444",ls="--",lw=1,alpha=.5,label="Full 80%")
    ax.axvline(50,color="#f59e0b",ls="--",lw=1,alpha=.5,label="Half 50%")
    ax.set_title("Fill % by Zone",pad=8,fontsize=11); ax.set_xlabel("Fill %"); ax.set_xlim(0,112)
    ax.legend(fontsize=7,labelcolor="#94a3b8",facecolor="#080f1c")
    ax.grid(axis="x",alpha=.15,color="#1e3a5f"); plt.tight_layout(); return fig


# ════════════════════════════════════════════════════════════════════
#  UI STARTS
# ════════════════════════════════════════════════════════════════════

# ── HERO ────────────────────────────────────────────────────────────
hL,hR=st.columns([5,1])
with hL:
    st.markdown("""<div class='hero'>
        <div class='hero-title'>🏙️ Smart City Monitoring System</div>
        <div class='hero-sub'>AI-powered real-time intelligence — Traffic · Weather · Waste Management</div>
    </div>""",unsafe_allow_html=True)
with hR:
    st.markdown(f"""<div style='text-align:right;padding-top:8px'>
        <div style='color:#1e3a5f;font-family:DM Mono,monospace;font-size:11px'>
            {datetime.now().strftime('%d %b %Y')}<br>{datetime.now().strftime('%H:%M:%S')}
        </div>
        <div style='margin-top:5px'>
            <span class='live-badge'><span class='live-dot'></span>Live</span></div>
    </div>""",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# CITY INPUT PANEL  —  always visible, no sidebar needed
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='city-panel'><div class='city-panel-title'>📍 Select Your City — All India</div></div>",
            unsafe_allow_html=True)

state_list=sorted(set(v["state"] for v in CITIES.values()))

ci1,ci2,ci3,ci4=st.columns([2,2,1,1])

with ci1:
    sel_state=st.selectbox("State / UT",["All States"]+state_list,key="ss")

with ci2:
    opts=(sorted(CITIES.keys()) if sel_state=="All States"
          else sorted(k for k,v in CITIES.items() if v["state"]==sel_state))
    # default to Udaipur if available else first
    default_idx=opts.index("Udaipur") if "Udaipur" in opts else 0
    sel_city=st.selectbox("City",opts,index=default_idx,key="sc")

with ci3:
    if sel_city in CITIES:
        inf=CITIES[sel_city]
        st.markdown(f"""<div class='cb' style='margin-top:4px'>
            {inf['lat']:.3f}°N &nbsp; {inf['lon']:.3f}°E<br>
            <span style='color:#0d4a3a'>Elev {inf['elev']} m · {inf['state']}</span>
        </div>""",unsafe_allow_html=True)

with ci4:
    st.markdown("<br>",unsafe_allow_html=True)
    fetch_btn=st.button("🌤 Fetch Weather",use_container_width=True,key="fw")

# Manual coords
with st.expander("🔧 Enter coordinates manually"):
    mx1,mx2,mx3,mx4=st.columns([2,2,2,1])
    m_city=mx1.text_input("Name","Custom Location",key="mcity")
    m_lat=mx2.number_input("Latitude",value=24.5854,format="%.4f",key="mlat")
    m_lon=mx3.number_input("Longitude",value=73.7125,format="%.4f",key="mlon")
    mx4.markdown("<br>",unsafe_allow_html=True)
    if mx4.button("Fetch",use_container_width=True,key="mfetch"):
        with st.spinner("Fetching…"):
            st.session_state.weather=_sw(m_city,m_lat,m_lon)
            try:
                r=requests.get("https://api.open-meteo.com/v1/forecast",params={
                    "latitude":m_lat,"longitude":m_lon,
                    "current":["temperature_2m","relative_humidity_2m","apparent_temperature",
                               "precipitation","weather_code","cloud_cover","pressure_msl",
                               "wind_speed_10m","wind_direction_10m","uv_index"],
                    "timezone":"auto","forecast_days":1},timeout=7); r.raise_for_status()
                c=r.json().get("current",{})
                T=float(c.get("temperature_2m",30)); H=float(c.get("relative_humidity_2m",55))
                st.session_state.weather={
                    "city":m_city,"state":"","lat":m_lat,"lon":m_lon,"elev":"—",
                    "temperature":round(T,1),"humidity":round(H,1),
                    "feels_like":round(float(c.get("apparent_temperature",T)),1),
                    "dew_point":round(T-((100-H)/5),1),
                    "wind_speed":round(float(c.get("wind_speed_10m",10)),1),
                    "wind_dir":round(float(c.get("wind_direction_10m",200)),0),
                    "cloud_cover":round(float(c.get("cloud_cover",30)),0),
                    "pressure":round(float(c.get("pressure_msl",1010)),1),
                    "rain_mm":round(float(c.get("precipitation",0)),2),
                    "uv_index":round(float(c.get("uv_index",5)),1),
                    "condition":WMO.get(int(c.get("weather_code",2)),"Partly Cloudy"),
                    "source":"Open-Meteo","at":datetime.now().strftime("%H:%M:%S"),
                }
            except Exception: pass
        st.success(f"✅ Loaded for {m_city}"); st.rerun()

if fetch_btn:
    with st.spinner(f"Fetching weather for {sel_city}…"):
        st.session_state.weather=fetch_w(sel_city)
    st.success(f"✅ {sel_city}, {CITIES.get(sel_city,{}).get('state','')}"); st.rerun()

# Auto-load
if st.session_state.weather is None:
    st.session_state.weather=fetch_w("Udaipur")

wd=st.session_state.weather
bins=st.session_state.bins
det=st.session_state.det
pr=st.session_state.prediction


# ── GLOBAL METRICS ──────────────────────────────────────────────────
st.markdown("<br>",unsafe_allow_html=True)
full_bins=sum(1 for b in bins if _bs(b["fill"]) in ("FULL","CRITICAL"))
mc=st.columns(7)
_mc(mc[0],"Temperature",f"{wd['temperature']:.1f}" if wd else "--","°C",
    wd.get("condition","—") if wd else "—","blue")
_mc(mc[1],"Humidity",f"{wd['humidity']:.0f}" if wd else "--","%",
    f"Dew {wd.get('dew_point','—')}°C" if wd else "—","teal")
_mc(mc[2],"Location",(wd["city"][:10] if wd else "--"),"",
    f"{wd.get('state','')} · {wd.get('at','—')}" if wd else "—","purple")
_mc(mc[3],"Vehicles",str(det["total"]) if det else "--","",
    det["density"]+" density" if det else "Upload media","amber")
_mc(mc[4],"Traffic Score",f"{det['score']:.1f}" if det else "--","pts","YOLO","sky")
_mc(mc[5],"Resource Need",pr["d"][:3] if pr else "N/A","","ML model",
    "red" if (pr and pr["d"]=="HIGH") else "green")
_mc(mc[6],"Bins Full",str(full_bins),"",f"of {len(bins)}",
    "red" if full_bins>3 else "green")
st.markdown("<br>",unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
t1,t2,t3=st.tabs(["🌡️  Weather","🚦  Traffic AI","🗑️  Waste"])


# ╔══════════════ TAB 1 — WEATHER ══════════════╗
with t1:
    _sh("🌡️","Weather & Climate","Live · Open-Meteo API","rgba(59,130,246,.12)")
    if wd:
        c1,c2,c3,c4=st.columns(4)
        _mc(c1,"🌡️ Temperature",f"{wd['temperature']:.1f}","°C",wd.get("condition",""),"blue")
        _mc(c2,"💧 Humidity",f"{wd['humidity']:.0f}","%",f"Dew {wd.get('dew_point','—')}°C","teal")
        _mc(c3,"💨 Wind",f"{wd.get('wind_speed',0):.1f}"," km/h",f"Dir {wd.get('wind_dir','—')}°","amber")
        _mc(c4,"🔆 UV",f"{wd.get('uv_index',0):.1f}","",_uv(wd.get("uv_index",0)),"purple")
        st.markdown("<br>",unsafe_allow_html=True)
        l,r=st.columns(2)
        with l:
            st.markdown("##### 📍 Location")
            for k,v in [("City",wd["city"]),("State",wd.get("state","—")),
                        ("Lat",f"{wd['lat']:.4f}°N"),("Lon",f"{wd['lon']:.4f}°E"),
                        ("Elev",f"{wd.get('elev','—')} m"),("Source",wd.get("source","—")),
                        ("At",wd.get("at","—"))]:
                ka,va=st.columns([2,3]); ka.caption(k)
                va.markdown(f"<code style='font-size:11px;color:#bfdbfe'>{v}</code>",
                            unsafe_allow_html=True)
        with r:
            st.markdown("##### 🌤 Conditions")
            for k,v in [("Feels Like",f"{wd.get('feels_like',wd['temperature']):.1f}°C"),
                        ("Cloud Cover",f"{wd.get('cloud_cover',0):.0f}%"),
                        ("Pressure",f"{wd.get('pressure',1010):.1f} hPa"),
                        ("Rain",f"{wd.get('rain_mm',0):.2f} mm"),
                        ("Condition",wd.get("condition","—")),
                        ("Heat Index",f"{_hi(wd['temperature'],wd['humidity']):.1f}°C")]:
                ka,va=st.columns([2,3]); ka.caption(k)
                va.markdown(f"<code style='font-size:11px;color:#bfdbfe'>{v}</code>",
                            unsafe_allow_html=True)
    else:
        st.markdown("<div class='ib'>Select a city above and click Fetch Weather.</div>",
                    unsafe_allow_html=True)


# ╔══════════════ TAB 2 — TRAFFIC ══════════════╗
with t2:
    _sh("🚦","AI Traffic Detection","YOLOv8 · Green boxes on all vehicles · ML prediction",
        "rgba(34,197,94,.1)")

    if YOLO_OK:
        st.markdown("<div class='sb'>🟢 <b>YOLOv8 active</b> — real detection with green bounding boxes</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='wb'>🟡 <b>Simulation mode</b> — green boxes shown on simulated vehicles · "
                    "<code>pip install ultralytics</code> for real detection</div>",
                    unsafe_allow_html=True)

    ul,ur=st.columns([1,1],gap="large")
    with ul:
        st.markdown("##### 📤 Upload Media")
        mtype=st.radio("Type",["🖼️ Image","🎬 Video"],horizontal=True,
                       key="mt",label_visibility="collapsed")
        is_vid="Video" in mtype
        uploaded=st.file_uploader("Choose file",
            type=(["mp4","avi","mov","mkv"] if is_vid else ["jpg","jpeg","png","bmp","webp"]),
            key="uf",label_visibility="collapsed")
        conf_thr=st.slider("Detection confidence",0.20,0.80,0.35,0.05,key="ct")
        run_btn=st.button("⚡ Run Detection",use_container_width=True,
                          disabled=(uploaded is None),key="rb")

    with ur:
        st.markdown("##### 📊 Results")
        if run_btn and uploaded is not None:
            if not is_vid:
                with st.spinner("Running YOLO detection…"):
                    img_arr=np.array(Image.open(uploaded).convert("RGB"))
                    img_bgr=cv2.cvtColor(img_arr,cv2.COLOR_RGB2BGR)
                    result=_detect(img_bgr,conf_thr)
                    st.session_state.det=result
                    st.session_state.t_hist.append(result["score"])
                    det=result
            else:
                with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
                    tmp.write(uploaded.read()); tp=tmp.name
                cap=cv2.VideoCapture(tp)
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); samp=max(1,total//12)
                agg_c=defaultdict(int); agg_s=[]; last_ann=None
                prog=st.progress(0,text="Processing…"); fi=0
                while True:
                    ret,fr=cap.read()
                    if not ret: break
                    if fi%samp==0:
                        r=_detect(fr,conf_thr)
                        for k,v in r["counts"].items(): agg_c[k]+=v
                        agg_s.append(r["score"])
                        st.session_state.t_hist.append(r["score"])
                        last_ann=r["ann"]
                    fi+=1
                    prog.progress(min(fi/total,1.),text=f"Frame {fi}/{total}")
                cap.release(); os.unlink(tp); prog.empty()
                n=max(len(agg_s),1)
                avg_c={k:round(v/n) for k,v in agg_c.items()}; avg_s=round(sum(agg_s)/n,1)
                dens="HIGH" if avg_s>=25 else "MEDIUM" if avg_s>=10 else "LOW"
                st.session_state.det={"counts":avg_c,"total":sum(avg_c.values()),
                    "score":avg_s,"density":dens,"ann":last_ann,
                    "at":datetime.now().strftime("%H:%M:%S")}
                det=st.session_state.det

        det=st.session_state.det
        if det:
            d=det["density"]
            st.markdown(f"""<div class='db db-{d}'>
                {"🔴" if d=="HIGH" else "🟡" if d=="MEDIUM" else "🟢"}&nbsp; {d} TRAFFIC
            </div>""",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            m1.metric("Vehicles",det["total"]); m2.metric("Score",f"{det['score']:.1f}")
            m3.metric("At",det["at"])
            if det["counts"]:
                st.markdown("**Breakdown:**")
                for vt,cnt in sorted(det["counts"].items(),key=lambda x:-x[1]):
                    if cnt>0:
                        pct=round(cnt/max(det["total"],1)*100)
                        st.markdown(f"""<div class='fb'><div class='fbl'>{vt.title()} {cnt} ({pct}%)</div>
                            <div class='fbt'><div class='fbf' style='width:{pct}%;background:#22c55e'></div>
                            </div></div>""",unsafe_allow_html=True)
        else:
            st.markdown("<div class='ib'>Upload image/video then click Run Detection.</div>",
                        unsafe_allow_html=True)

    # ── ANNOTATED + CHARTS ─────────────────────────────────────────
    if det:
        st.markdown("---")

        # ── Row 1: annotated image (full width emphasis) ──────────
        st.markdown("##### 🟩 Detected Vehicles — Green Bounding Boxes")
        ann_col, info_col = st.columns([3, 1])
        with ann_col:
            if det.get("ann") is not None:
                ann_rgb=cv2.cvtColor(det["ann"],cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True,
                         caption=f"YOLO output — {det['total']} vehicles detected · Green boxes = vehicles")
            elif uploaded is not None and not is_vid:
                try:
                    uploaded.seek(0)
                    st.image(Image.open(uploaded),use_container_width=True,
                             caption="Original (re-upload to see boxes)")
                except Exception: pass
            else:
                st.info("Video analysed — last sampled frame shown above if available.")

        with info_col:
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f"""
            <div class='mc green' style='margin-bottom:8px'>
                <div class='mc-lbl'>Vehicles</div>
                <div class='mc-val'>{det['total']}</div>
                <div class='mc-sub'>Total detected</div>
            </div>
            <div class='mc sky'>
                <div class='mc-lbl'>Score</div>
                <div class='mc-val'>{det['score']:.1f}</div>
                <div class='mc-sub'>Weighted traffic</div>
            </div>
            """,unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown("""<div class='sb' style='font-size:11px'>
                <b>Box colours:</b><br>
                🟩 All vehicles = <b>green</b><br>
                Labels show type + confidence %
            </div>""",unsafe_allow_html=True)

        # ── Row 2: bar + pie + trend ───────────────────────────────
        st.markdown("<br>",unsafe_allow_html=True)
        ca,cb,cc=st.columns([3,3,2])
        with ca:
            st.markdown("##### 📊 Vehicle Bar Chart")
            f=_bar(det["counts"])
            if f: st.pyplot(f,use_container_width=True)
        with cb:
            st.markdown("##### 📈 Score Trend")
            f3=_trend(st.session_state.t_hist)
            if f3: st.pyplot(f3,use_container_width=True)
            else: st.caption("Builds with more analyses.")
        with cc:
            st.markdown("##### 🥧 Mix")
            f2=_pie(det["counts"])
            if f2: st.pyplot(f2,use_container_width=True)

        # ── Suggestions ────────────────────────────────────────────
        st.markdown("---")
        _sh("💡","Smart Traffic Suggestions",
            f"Density: {det['density']} · Score: {det['score']:.1f}","rgba(56,189,248,.1)")
        d=det["density"]
        if d=="HIGH":
            st.error(f"🔴 HIGH — {det['total']} vehicles detected")
            for s in ["🚨 Deploy traffic police at main intersections",
                      "🚦 Switch to adaptive signal timing — cut cycle by 30%",
                      "📢 Issue public advisory for alternate routes",
                      "🚍 Activate extra public transport on affected corridors",
                      "📡 Alert smart signal network for bypass routing"]:
                st.markdown(f"<div class='sug'>{s}</div>",unsafe_allow_html=True)
        elif d=="MEDIUM":
            st.warning(f"🟡 MEDIUM — {det['total']} vehicles")
            for s in ["👁️ Increase CCTV monitoring on key corridors",
                      "⏱️ Adjust signal green time ±10%",
                      "📊 Log for time-of-day pattern analysis",
                      "🚖 Keep standby crew on alert"]:
                st.markdown(f"<div class='sug'>{s}</div>",unsafe_allow_html=True)
        else:
            st.success(f"🟢 LOW — {det['total']} vehicles. Normal.")
            for s in ["✅ Standard monitoring active","🔧 Good window for road maintenance",
                      "📋 Continue scheduled patrols"]:
                st.markdown(f"<div class='sug'>{s}</div>",unsafe_allow_html=True)

    # ── ML PREDICTION ──────────────────────────────────────────────
    st.markdown("---")
    _sh("🤖","ML Resource Prediction","HIGH / MEDIUM / LOW","rgba(168,85,247,.1)")
    pL,pR=st.columns([1,1],gap="large")
    with pL:
        st.markdown("##### ⚙️ Inputs")
        tv_=st.slider("🚗 Traffic Volume",0,5000,det["total"]*60 if det else 2000,50,key="tv")
        T_=st.number_input("🌡️ Temp °C",-5.,55.,float(wd["temperature"]) if wd else 30.,.5,key="Tv")
        H_=st.number_input("💧 Humidity",0.,100.,float(wd["humidity"]) if wd else 55.,1.,key="Hv")
        pop_=st.slider("👥 Population",100,50000,8000,100,key="pv")
        hr_=st.slider("🕐 Hour",0,23,datetime.now().hour,key="hrv")
        we_=st.toggle("📅 Weekend",False,key="wev")
        wc_=st.selectbox("🌤 Weather",["Clear","Partly Cloudy","Overcast","Rain","Heavy Rain","Fog"],key="wcv")
        rq_=st.select_slider("🛣️ Road",["Poor","Below Average","Average","Good","Excellent"],"Good",key="rqv")
        if wd: st.markdown("<div class='sb'>✅ Temp & humidity from live weather</div>",
                           unsafe_allow_html=True)
        pb2=st.button("⚡ Predict",use_container_width=True,key="pb2")
    with pR:
        st.markdown("##### 📋 Result")
        if pb2:
            with st.spinner("Running…"):
                time.sleep(.3)
                st.session_state.prediction=ml_pred(tv_,T_,H_,pop_,hr_,int(we_),wc_,rq_)
            pr=st.session_state.prediction
        pr=st.session_state.prediction
        if pr:
            d=pr["d"]
            st.markdown(f"""<div class='db db-{d}'>
                {"🔴" if d=="HIGH" else "🟡" if d=="MEDIUM" else "🟢"}&nbsp; {d} RESOURCE NEED
            </div>""",unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            for cls,prob in sorted(pr["pb"].items(),key=lambda x:-x[1]):
                pct=round(prob*100,1)
                col={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"}.get(cls,"#3b82f6")
                st.markdown(f"""<div class='fb'><div class='fbl'>{cls} {pct}%</div>
                    <div class='fbt'><div class='fbf' style='width:{pct}%;background:{col}'></div>
                    </div></div>""",unsafe_allow_html=True)
            if pr.get("fi"):
                st.markdown("<br>**Feature importance:**")
                for fn,fi in sorted(pr["fi"].items(),key=lambda x:-x[1])[:6]:
                    pct=round(fi*100,1)
                    st.markdown(f"""<div class='fb'><div class='fbl'>{fn} {pct}%</div>
                        <div class='fbt'><div class='fbf' style='width:{pct}%;background:#38bdf8'></div>
                        </div></div>""",unsafe_allow_html=True)
            st.caption(f"Model: {pr['mt']}")
        else:
            st.markdown("<div class='ib'>Fill inputs and click Predict.</div>",
                        unsafe_allow_html=True)


# ╔══════════════ TAB 3 — WASTE ══════════════╗
with t3:
    _sh("🗑️","Smart Dustbin Monitoring","IoT simulation · Fill tracking","rgba(34,197,94,.1)")
    bins=st.session_state.bins
    for b in bins: b["status"]=_bs(b["fill"])
    n_tot=len(bins)
    n_full=sum(1 for b in bins if b["status"] in ("FULL","CRITICAL"))
    n_half=sum(1 for b in bins if b["status"]=="HALF")
    n_emp=sum(1 for b in bins if b["status"] in ("EMPTY","QUARTER"))
    b1,b2,b3,b4=st.columns(4)
    _mc(b1,"Total Bins",str(n_tot),"",f"{len(BIN_DATA)} zones","blue")
    _mc(b2,"Full/Critical",str(n_full),"","Collect now","red" if n_full>2 else "amber")
    _mc(b3,"Half Full",str(n_half),"","Monitor","amber")
    _mc(b4,"Empty/Low",str(n_emp),"","OK","green")
    st.markdown("<br>",unsafe_allow_html=True)
    if n_full>0:
        st.error(f"⚠️ {n_full} bin(s) need collection: "
                 f"{', '.join(set(b['zone'] for b in bins if b['status'] in ('FULL','CRITICAL')))}")
    with st.expander("✏️ Manual override"):
        ov1,ov2,ov3=st.columns([2,2,1])
        sel_b=ov1.selectbox("Bin",[b["id"] for b in bins],key="sb2")
        nf=ov2.slider("Fill %",0,100,50,key="nf2")
        ov3.markdown("<br>",unsafe_allow_html=True)
        if ov3.button("Update",use_container_width=True,key="ub2"):
            for b in bins:
                if b["id"]==sel_b: b["fill"]=float(nf);b["status"]=_bs(nf);b["upd"]=datetime.now().strftime("%H:%M")
            st.session_state.bins=bins; st.rerun()
    fl1,fl2,fl3=st.columns([2,2,1])
    zf=fl1.selectbox("Zone",["All"]+sorted(set(b["zone"] for b in bins)),key="zf2")
    sf=fl2.selectbox("Status",["All","CRITICAL","FULL","HALF","QUARTER","EMPTY"],key="sf2")
    fl3.markdown("<br>",unsafe_allow_html=True)
    if fl3.button("🔄 Refresh",use_container_width=True,key="rf2"):
        for b in st.session_state.bins:
            b["fill"]=min(100,b["fill"]+b["_r"]*random.uniform(.5,2))
            if b["fill"]>=90 and random.random()<.15: b["fill"]=random.uniform(2,10)
            b["fill"]=round(b["fill"],1); b["status"]=_bs(b["fill"]); b["upd"]=datetime.now().strftime("%H:%M")
        st.rerun()
    filtered=bins
    if zf!="All": filtered=[b for b in filtered if b["zone"]==zf]
    if sf!="All": filtered=[b for b in filtered if b["status"]==sf]
    st.markdown(f"**{len(filtered)} of {n_tot} bins**"); st.markdown("<br>",unsafe_allow_html=True)
    for row in [filtered[i:i+5] for i in range(0,len(filtered),5)]:
        cols=st.columns(5)
        for col,b in zip(cols,row):
            fh=int(50*b["fill"]/100); fc=_bc(b["status"])
            with col:
                st.markdown(f"""<div class='bw'>
                    <div class='bid'>{b['id']}</div>
                    <div class='bid' style='color:#0f2a20'>{b['loc']}</div>
                    <div class='bbo'><div class='bfi' style='height:{fh}px;background:{fc}'></div></div>
                    <div style='font-size:10px;color:#1e3a5f;margin-bottom:3px'>{b['fill']:.0f}%</div>
                    <span class='bp bp-{b["status"]}'>{b["status"]}</span>
                    <div class='bid' style='margin-top:3px;color:#0f2233'>{b['zone'][:14]}</div>
                </div>""",unsafe_allow_html=True)
    st.markdown("<br>")
    st.markdown("##### 📊 Fill by Zone")
    fz=_zone_chart(bins)
    if fz: st.pyplot(fz,use_container_width=True)
    with st.expander("📋 Full Table"):
        st.dataframe(pd.DataFrame([{
            "Bin":b["id"],"Zone":b["zone"],"Loc":b["loc"],
            "Fill":f"{b['fill']:.0f}%","Status":b["status"],"Cap(L)":b["cap"]}
            for b in filtered]),use_container_width=True,hide_index=True)
    with st.expander("🚛 Collection Route",expanded=n_full>0):
        urgent=sorted([b for b in bins if b["status"] in ("FULL","CRITICAL")],key=lambda x:-x["fill"])
        if urgent:
            for i,b in enumerate(urgent,1):
                icon="🔴" if b["status"]=="CRITICAL" else "🟠"
                st.markdown(f"{i}. {icon} **{b['id']}** · {b['loc']} · {b['zone']} · {b['fill']:.0f}%")
            st.markdown(f"<div class='eb'>🚛 {len(urgent)} bins — est. ~{len(urgent)*4} min to clear</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='sb'>✅ No urgent collections.</div>",unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("---")
f1,f2,f3,f4=st.columns(4)
f1.caption(f"🏙️ SmartCity AI v4.0 · {datetime.now().strftime('%d %b %Y %H:%M')}")
f2.caption(f"🌤 {wd['city'] if wd else '—'} · {wd.get('source','') if wd else ''}")
f3.caption(f"🚦 YOLO: {'active ✅' if YOLO_OK else 'simulation 🟡'} · {len(CITIES)} cities loaded")
f4.caption(f"🗑️ {n_full} full / {n_tot} bins")