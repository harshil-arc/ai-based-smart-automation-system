"""
╔═══════════════════════════════════════════════════════════════╗
║      SMART CITY — MODULE 2: REAL-TIME WEATHER MONITOR       ║
║           Streamlit Web App Version (with User Input)        ║
╚═══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from collections import deque
import random
import csv
import os

# ─────────────────────────────────────────────────────────────
# PRESET CITIES WITH COORDINATES
# ─────────────────────────────────────────────────────────────
PRESET_CITIES = {
    "Udaipur, Rajasthan": (24.5854, 73.7125),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore, Karnataka": (12.9716, 77.5946),
    "Kolkata, West Bengal": (22.5726, 88.3639),
    "Chennai, Tamil Nadu": (13.0827, 80.2707),
    "Hyderabad, Telangana": (17.3850, 78.4867),
    "Jaipur, Rajasthan": (26.9124, 75.7873),
    "Pune, Maharashtra": (18.5204, 73.8567),
    "Ahmedabad, Gujarat": (23.0225, 72.5714),
}

LOG_FILE = "weather_log.csv"

# ─────────────────────────────────────────────────────────────
# WEATHER DATA CLASS
# ─────────────────────────────────────────────────────────────
class WeatherReading:
    def __init__(self):
        self.temperature = 30.0
        self.feels_like = 32.0
        self.humidity = 55.0
        self.wind_speed = 12.0
        self.wind_dir = 225.0
        self.pressure = 1010.0
        self.uv_index = 6.0
        self.rain_mm = 0.0
        self.visibility = 8.0
        self.cloud_cover = 30.0
        self.dew_point = 18.0
        self.aqi_pm25 = 22.0
        self.condition = "Partly Cloudy"
        self.timestamp = datetime.now()

    def heat_index(self):
        T, H = self.temperature, self.humidity
        if T < 27:
            return T
        hi = (-8.78469475556 + 1.61139411*T + 2.33854883889*H
              - 0.14611605*T*H - 0.012308094*T*T
              - 0.016424828*H*H + 0.002211732*T*T*H
              + 0.00072546*T*H*H - 0.000003582*T*T*H*H)
        return round(hi, 1)

    def heat_island_risk(self):
        score = 0
        if self.temperature > 38: score += 3
        elif self.temperature > 33: score += 1
        if self.humidity > 70: score += 2
        if self.wind_speed < 5: score += 2
        if self.uv_index > 8: score += 1
        if score >= 6: return "🔴 CRITICAL"
        if score >= 4: return "🟠 HIGH"
        if score >= 2: return "🟡 MODERATE"
        return "🟢 LOW"

    def flood_risk(self):
        score = 0
        if self.rain_mm > 20: score += 4
        elif self.rain_mm > 10: score += 2
        elif self.rain_mm > 2: score += 1
        if self.humidity > 85: score += 1
        if self.cloud_cover > 80: score += 1
        if score >= 5: return "🔴 HIGH"
        if score >= 3: return "🟡 MODERATE"
        return "🟢 LOW"

# ─────────────────────────────────────────────────────────────
# WMO WEATHER CODES
# ─────────────────────────────────────────────────────────────
WMO_CODES = {
    0:'Clear Sky', 1:'Mainly Clear', 2:'Partly Cloudy', 3:'Overcast',
    45:'Foggy', 48:'Icy Fog', 51:'Light Drizzle', 53:'Moderate Drizzle',
    55:'Heavy Drizzle', 61:'Light Rain', 63:'Moderate Rain', 65:'Heavy Rain',
    71:'Light Snow', 73:'Moderate Snow', 75:'Heavy Snow', 80:'Rain Showers',
    81:'Heavy Showers', 82:'Violent Showers', 95:'Thunderstorm',
    96:'Thunderstorm + Hail', 99:'Heavy Thunderstorm',
}

# ─────────────────────────────────────────────────────────────
# GEOCODING - CONVERT CITY/STATE TO COORDINATES
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_coordinates(city, state):
    """Convert city/state to lat/lon using Open-Meteo Geocoding API"""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city,
            "state": state,
            "language": "en",
            "limit": 1,
            "format": "json"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.warning(f"⚠️ Could not find coordinates for {city}, {state}")
        return None, None

# ─────────────────────────────────────────────────────────────
# FETCH WEATHER DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_weather_live(lat, lon):
    """Fetch weather from Open-Meteo API (Free, No Key Required)"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "weather_code", "cloud_cover", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "uv_index",
        ],
        "hourly": [
            "temperature_2m", "precipitation_probability",
            "precipitation", "wind_speed_10m", "uv_index", "relative_humidity_2m"
        ],
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current", {})
        hourly = data.get("hourly", {})

        reading = WeatherReading()
        reading.temperature = float(cur.get("temperature_2m", 30))
        reading.humidity = float(cur.get("relative_humidity_2m", 55))
        reading.feels_like = float(cur.get("apparent_temperature", reading.temperature + 2))
        reading.rain_mm = float(cur.get("precipitation", 0))
        reading.cloud_cover = float(cur.get("cloud_cover", 30))
        reading.pressure = float(cur.get("pressure_msl", 1010))
        reading.wind_speed = float(cur.get("wind_speed_10m", 12))
        reading.wind_dir = float(cur.get("wind_direction_10m", 200))
        reading.uv_index = float(cur.get("uv_index", 5))
        wmo = int(cur.get("weather_code", 2))
        reading.condition = WMO_CODES.get(wmo, "Unknown")
        reading.timestamp = datetime.now()
        reading.dew_point = reading.temperature - ((100 - reading.humidity) / 5)

        forecast = {
            "hours": hourly.get("time", []),
            "temp": hourly.get("temperature_2m", []),
            "precip_p": hourly.get("precipitation_probability", []),
        }
        return reading, forecast, True
    except Exception as e:
        st.error(f"⚠️ API Fetch failed: {e}")
        return None, None, False

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartCity Weather Monitor", layout="wide")

# ─────────────────────────────────────────────────────────────
# SIDEBAR - LOCATION SELECTION
# ─────────────────────────────────────────────────────────────
st.sidebar.title("📍 Location Settings")
st.sidebar.markdown("---")

input_method = st.sidebar.radio(
    "Choose how to select location:",
    ["📌 Preset Cities", "✏️ Custom City Search"]
)

lat = None
lon = None
city_name = "Udaipur, Rajasthan"

if input_method == "📌 Preset Cities":
    selected_location = st.sidebar.selectbox(
        "Select a city:",
        list(PRESET_CITIES.keys())
    )
    city_name = selected_location
    lat, lon = PRESET_CITIES[selected_location]
    st.sidebar.success(f"✅ Selected: {city_name}")

else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        city_input = st.text_input("🏙️ City:", value="Udaipur", key="city_input")
    with col2:
        state_input = st.text_input("📍 State:", value="Rajasthan", key="state_input")
    
    if st.sidebar.button("🔍 Search", key="search_btn"):
        with st.spinner("🔍 Searching for location..."):
            lat, lon = get_coordinates(city_input, state_input)
            if lat is not None and lon is not None:
                city_name = f"{city_input}, {state_input}"
                st.sidebar.success(f"✅ Found: {city_name}")
            else:
                st.sidebar.error(f"❌ City not found. Try another.")
    
    # Set default coordinates if search not done
    if lat is None:
        lat, lon = 24.5854, 73.7125  # Udaipur default

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
st.title("🌦️ SmartCity Weather Intelligence Platform")
st.markdown(f"**Location:** {city_name} ({lat:.4f}°N, {lon:.4f}°E)")

# Fetch weather data
reading, forecast, success = fetch_weather_live(lat, lon)

if not success or reading is None:
    st.error("❌ Failed to fetch weather data. Please check your connection and try again.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# MAIN WEATHER METRICS
# ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Current Weather Conditions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{reading.temperature:.1f}°C",
        f"Feels like {reading.feels_like:.1f}°C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{reading.humidity:.0f}%",
        f"Dew Point: {reading.dew_point:.1f}°C"
    )

with col3:
    st.metric(
        "💨 Wind Speed",
        f"{reading.wind_speed:.1f} km/h",
        f"Direction: {reading.wind_dir:.0f}°"
    )

with col4:
    st.metric(
        "📊 Pressure",
        f"{reading.pressure:.0f} hPa",
        f"Cloud: {reading.cloud_cover:.0f}%"
    )

# ─────────────────────────────────────────────────────────────
# WEATHER STATUS & RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚠️ Weather Status & Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Condition:** {reading.condition}")

with col2:
    st.warning(f"🔥 **Heat Island Risk:** {reading.heat_island_risk()}")

with col3:
    st.error(f"🌊 **Flood Risk:** {reading.flood_risk()}")

# ─────────────────────────────────────────────────────────────
# DETAILED METRICS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Detailed Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Heat Index", f"{reading.heat_index():.1f}°C")
    st.metric("☀️ UV Index", f"{reading.uv_index:.1f}")

with col2:
    st.metric("🌧️ Rainfall", f"{reading.rain_mm:.1f} mm/hr")
    st.metric("👁️ Visibility", f"{reading.visibility:.1f} km")

with col3:
    st.metric("💨 Air Quality PM2.5", f"{reading.aqi_pm25:.1f} μg/m³")
    ts = reading.timestamp.strftime('%H:%M:%S')
    st.caption(f"Last updated: {ts}")

# ─────────────────────────────────────────────────────────────
# 24-HOUR FORECAST
# ─────────────────────────────────────────────────────────────
if forecast and forecast.get('hours'):
    st.markdown("---")
    st.markdown("### 📊 24-Hour Forecast")
    
    hours = forecast['hours'][:24]
    temps = forecast['temp'][:24]
    precip_p = forecast['precip_p'][:24]
    
    if temps and hours:
        forecast_df = pd.DataFrame({
            'Time': [h[11:16] if len(h) > 11 else str(i) for i, h in enumerate(hours)],
            'Temperature (°C)': temps,
            'Precipitation Prob (%)': precip_p
        })
        
        # Temperature Chart
        st.subheader("🌡️ Temperature Trend")
        st.line_chart(
            forecast_df.set_index('Time')[['Temperature (°C)']],
            use_container_width=True
        )
        
        # Precipitation Chart
        st.subheader("🌧️ Precipitation Probability")
        st.bar_chart(
            forecast_df.set_index('Time')[['Precipitation Prob (%)']],
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🌍 <b>SmartCity AI — Weather Monitor</b> | Module 2 of 3</p>
    <p><small>Data provided by Open-Meteo API (Free & No API Key Required)</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
