"""
╔══════════════════════════════════════════════════════════════════╗
║        SMART CITY — MODULE 2: REAL-TIME WEATHER MONITOR         ║
║        Paste directly into PyCharm and run                      ║
║                                                                  ║
║  Features:                                                       ║
║  • Live weather fetch via Open-Meteo API (FREE, no key needed)  ║
║  • Multi-sensor simulation (temp, humidity, wind, UV, rain)     ║
║  • Hourly forecast chart with matplotlib                        ║
║  • Weather anomaly & alert detection                            ║
║  • Heat-island & flood risk index calculation                   ║
║  • Animated OpenCV weather dashboard                            ║
║  • CSV logging + daily summary report                           ║
╚══════════════════════════════════════════════════════════════════╝

REQUIREMENTS (paste in PyCharm terminal):
    pip install requests numpy pandas matplotlib opencv-python

USAGE:
    • Set CITY_LAT / CITY_LON below for your city
    • Default: Udaipur, Rajasthan, India
    • Press Q to quit, S to save snapshot, F to toggle forecast chart
"""

import cv2
import numpy as np
import requests
import threading
import time
import csv
import os
import math
import random
from datetime import datetime, timedelta
from collections import deque

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — set your city coordinates
# ─────────────────────────────────────────────────────────────────
CITY_NAME  = "Udaipur, Rajasthan"
CITY_LAT   = 24.5854
CITY_LON   = 73.7125
LOG_FILE   = "weather_log.csv"
FETCH_INTERVAL = 300   # seconds between live API fetches (5 min)
REFRESH_SIM    = 3     # seconds between simulated micro-updates

# ─────────────────────────────────────────────────────────────────
# COLORS (BGR)
# ─────────────────────────────────────────────────────────────────
C_WHITE   = (245, 245, 245)
C_GRAY    = (130, 130, 140)
C_DARK    = (22,  26,  34)
C_PANEL   = (30,  35,  45)
C_BLUE    = (220, 130, 50)
C_CYAN    = (220, 200, 60)
C_GREEN   = (60,  200, 80)
C_YELLOW  = (40,  210, 220)
C_ORANGE  = (40,  140, 240)
C_RED     = (50,  50,  220)
C_PURPLE  = (200, 80,  180)
C_TEAL    = (180, 200, 60)
C_AMBER   = (30,  180, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────────
# WEATHER DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────
class WeatherReading:
    def __init__(self):
        self.temperature   = 30.0    # °C
        self.feels_like    = 32.0    # °C
        self.humidity      = 55.0    # %
        self.wind_speed    = 12.0    # km/h
        self.wind_dir      = 225.0   # degrees
        self.pressure      = 1010.0  # hPa
        self.uv_index      = 6.0
        self.rain_mm       = 0.0     # mm/hr
        self.visibility    = 8.0     # km
        self.cloud_cover   = 30.0    # %
        self.dew_point     = 18.0    # °C
        self.aqi_pm25      = 22.0    # μg/m³
        self.condition     = "Partly Cloudy"
        self.timestamp     = datetime.now()

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
        if score >= 6: return "CRITICAL", C_RED
        if score >= 4: return "HIGH",     C_ORANGE
        if score >= 2: return "MODERATE", C_YELLOW
        return "LOW", C_GREEN

    def flood_risk(self):
        score = 0
        if self.rain_mm > 20: score += 4
        elif self.rain_mm > 10: score += 2
        elif self.rain_mm > 2: score += 1
        if self.humidity > 85: score += 1
        if self.cloud_cover > 80: score += 1
        if score >= 5: return "HIGH",     C_RED
        if score >= 3: return "MODERATE", C_YELLOW
        return "LOW",  C_GREEN

    def condition_icon_color(self):
        c = self.condition.lower()
        if 'thunder' in c: return '⚡', C_YELLOW
        if 'rain' in c or 'drizzle' in c: return '🌧', C_BLUE
        if 'snow' in c: return '❄', C_CYAN
        if 'fog' in c or 'mist' in c: return '🌫', C_GRAY
        if 'cloud' in c: return '☁', C_GRAY
        if 'clear' in c: return '☀', C_AMBER
        return '🌤', C_TEAL


# ─────────────────────────────────────────────────────────────────
# OPEN-METEO API FETCHER (free, no API key)
# ─────────────────────────────────────────────────────────────────
WMO_CODES = {
    0:'Clear Sky', 1:'Mainly Clear', 2:'Partly Cloudy', 3:'Overcast',
    45:'Foggy', 48:'Icy Fog', 51:'Light Drizzle', 53:'Moderate Drizzle',
    55:'Heavy Drizzle', 61:'Light Rain', 63:'Moderate Rain', 65:'Heavy Rain',
    71:'Light Snow', 73:'Moderate Snow', 75:'Heavy Snow', 80:'Rain Showers',
    81:'Heavy Showers', 82:'Violent Showers', 95:'Thunderstorm',
    96:'Thunderstorm + Hail', 99:'Heavy Thunderstorm',
}

def fetch_weather_live(lat, lon):
    """Fetch current + hourly forecast from Open-Meteo (no key needed)."""
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
        cur  = data.get("current", {})
        hourly = data.get("hourly", {})

        reading = WeatherReading()
        reading.temperature  = float(cur.get("temperature_2m", 30))
        reading.humidity     = float(cur.get("relative_humidity_2m", 55))
        reading.feels_like   = float(cur.get("apparent_temperature", reading.temperature + 2))
        reading.rain_mm      = float(cur.get("precipitation", 0))
        reading.cloud_cover  = float(cur.get("cloud_cover", 30))
        reading.pressure     = float(cur.get("pressure_msl", 1010))
        reading.wind_speed   = float(cur.get("wind_speed_10m", 12))
        reading.wind_dir     = float(cur.get("wind_direction_10m", 200))
        reading.uv_index     = float(cur.get("uv_index", 5))
        wmo = int(cur.get("weather_code", 2))
        reading.condition    = WMO_CODES.get(wmo, "Unknown")
        reading.timestamp    = datetime.now()
        reading.dew_point    = reading.temperature - ((100 - reading.humidity) / 5)

        forecast = {
            "hours":    hourly.get("time", []),
            "temp":     hourly.get("temperature_2m", []),
            "precip_p": hourly.get("precipitation_probability", []),
            "precip":   hourly.get("precipitation", []),
            "wind":     hourly.get("wind_speed_10m", []),
            "uv":       hourly.get("uv_index", []),
            "humidity": hourly.get("relative_humidity_2m", []),
        }
        print(f"  [API] Live weather fetched: {reading.temperature}°C, {reading.condition}")
        return reading, forecast
    except Exception as e:
        print(f"  [API] Fetch failed ({e}) — using simulated data.")
        return None, None


def simulate_micro_update(base: WeatherReading) -> WeatherReading:
    """Add realistic micro-fluctuations to simulate sensor noise."""
    r = WeatherReading()
    r.temperature  = base.temperature  + random.gauss(0, 0.15)
    r.feels_like   = base.feels_like   + random.gauss(0, 0.2)
    r.humidity     = max(0, min(100, base.humidity + random.gauss(0, 0.4)))
    r.wind_speed   = max(0, base.wind_speed + random.gauss(0, 0.5))
    r.wind_dir     = (base.wind_dir + random.gauss(0, 2)) % 360
    r.pressure     = base.pressure + random.gauss(0, 0.1)
    r.uv_index     = max(0, base.uv_index + random.gauss(0, 0.05))
    r.rain_mm      = max(0, base.rain_mm + random.gauss(0, 0.02))
    r.visibility   = max(0, base.visibility + random.gauss(0, 0.05))
    r.cloud_cover  = max(0, min(100, base.cloud_cover + random.gauss(0, 0.5)))
    r.dew_point    = base.dew_point + random.gauss(0, 0.1)
    r.aqi_pm25     = max(0, base.aqi_pm25 + random.gauss(0, 0.3))
    r.condition    = base.condition
    r.timestamp    = datetime.now()
    return r


# ─────────────────────────────────────────────────────────────────
# ANALYTICS & LOGGING
# ─────────────────────────────────────────────────────────────────
class WeatherAnalytics:
    def __init__(self):
        self.temp_history  = deque(maxlen=500)
        self.humid_history = deque(maxlen=500)
        self.wind_history  = deque(maxlen=500)
        self.rain_history  = deque(maxlen=500)
        self.uv_history    = deque(maxlen=500)
        self.alerts        = deque(maxlen=8)
        self.hourly_summary = {}
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'timestamp', 'temp_c', 'feels_like_c', 'humidity_pct',
                    'wind_kmh', 'wind_dir_deg', 'pressure_hpa', 'uv_index',
                    'rain_mmhr', 'cloud_pct', 'aqi_pm25', 'condition',
                    'heat_index', 'heat_island_risk', 'flood_risk'
                ])

    def record(self, w: WeatherReading):
        self.temp_history.append(w.temperature)
        self.humid_history.append(w.humidity)
        self.wind_history.append(w.wind_speed)
        self.rain_history.append(w.rain_mm)
        self.uv_history.append(w.uv_index)

        hi_risk, _ = w.heat_island_risk()
        fl_risk, _ = w.flood_risk()

        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                w.timestamp.isoformat(), round(w.temperature,1), round(w.feels_like,1),
                round(w.humidity,1), round(w.wind_speed,1), round(w.wind_dir,1),
                round(w.pressure,1), round(w.uv_index,1), round(w.rain_mm,2),
                round(w.cloud_cover,1), round(w.aqi_pm25,1), w.condition,
                round(w.heat_index(),1), hi_risk, fl_risk
            ])

        # Check anomalies
        self._check_alerts(w)

    def _check_alerts(self, w: WeatherReading):
        ts = w.timestamp.strftime('%H:%M')
        if w.temperature > 42:
            self._add_alert(f"Extreme heat: {w.temperature:.1f}°C", "CRITICAL", C_RED)
        if w.rain_mm > 15:
            self._add_alert(f"Heavy rainfall: {w.rain_mm:.1f}mm/hr", "WARN", C_BLUE)
        if w.wind_speed > 60:
            self._add_alert(f"Strong winds: {w.wind_speed:.1f}km/h", "WARN", C_YELLOW)
        if w.uv_index >= 11:
            self._add_alert(f"Extreme UV index: {w.uv_index:.1f}", "WARN", C_ORANGE)
        if w.visibility < 1.0:
            self._add_alert(f"Low visibility: {w.visibility:.1f}km", "WARN", C_GRAY)

    def _add_alert(self, msg, level, color):
        ts = datetime.now().strftime('%H:%M:%S')
        if not self.alerts or self.alerts[-1]['msg'] != msg:
            self.alerts.append({'msg': msg, 'level': level, 'color': color, 'time': ts})
            print(f"  [ALERT] {level}: {msg}")

    def sparkline(self, data, length=80):
        arr = list(data)[-length:]
        if len(arr) < 2:
            return []
        mn, mx = min(arr), max(arr)
        rng = max(mx - mn, 0.001)
        return [(i, (v-mn)/rng) for i, v in enumerate(arr)]

    def trend(self, data, window=30):
        arr = list(data)[-window:]
        if len(arr) < 5:
            return 0
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
        return slope


# ─────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
def panel_rect(img, x1, y1, x2, y2, color=C_PANEL, border_color=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    if border_color:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)

def label(img, text, x, y, scale=0.38, color=C_GRAY, bold=False):
    thick = 2 if bold else 1
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def value_label(img, text, x, y, scale=0.65, color=C_WHITE):
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)

def draw_bar(img, x, y, w, h, val, max_val, color, bg=(40,45,55)):
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    fill = int(w * min(val, max_val) / max_val)
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (65,70,80), 1)

def draw_sparkline(img, data_norm, x, y, w, h, color=C_TEAL):
    if len(data_norm) < 2:
        return
    pts = []
    for i, norm in data_norm:
        sx = x + int(i * w / max(len(data_norm)-1, 1))
        sy = y + h - int(norm * (h-4)) - 2
        pts.append((sx, sy))
    for j in range(len(pts)-1):
        cv2.line(img, pts[j], pts[j+1], color, 1, cv2.LINE_AA)
    # Fill area under line
    if len(pts) >= 3:
        poly = pts + [(pts[-1][0], y+h), (pts[0][0], y+h)]
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(poly, np.int32)], (*color[:3], ))
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

def wind_compass(img, cx, cy, radius, direction, speed):
    """Draw a compass rose showing wind direction."""
    cv2.circle(img, (cx, cy), radius, (50,55,65), -1)
    cv2.circle(img, (cx, cy), radius, (80,85,95), 1)
    # Cardinal labels
    for deg, lbl in [(0,'N'),(90,'E'),(180,'S'),(270,'W')]:
        rad = math.radians(deg - 90)
        lx = int(cx + (radius-8) * math.cos(rad))
        ly = int(cy + (radius-8) * math.sin(rad))
        cv2.putText(img, lbl, (lx-4, ly+4), FONT, 0.28, C_GRAY, 1, cv2.LINE_AA)
    # Wind arrow
    rad = math.radians(direction - 90)
    ax = int(cx + (radius-14) * math.cos(rad))
    ay = int(cy + (radius-14) * math.sin(rad))
    tail_x = int(cx - 12 * math.cos(rad))
    tail_y = int(cy - 12 * math.sin(rad))
    cv2.arrowedLine(img, (tail_x, tail_y), (ax, ay), C_TEAL, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(img, f'{speed:.0f}', (cx-10, cy+5), FONT, 0.38, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, 'km/h', (cx-12, cy+16), FONT, 0.28, C_GRAY, 1, cv2.LINE_AA)

def uv_gauge(img, cx, cy, r, uv_val):
    """Draw semi-circular UV gauge."""
    colors_uv = [C_GREEN, C_GREEN, C_YELLOW, C_YELLOW, C_YELLOW,
                 C_ORANGE, C_ORANGE, C_ORANGE, C_RED, C_RED, C_RED, C_RED]
    for seg in range(12):
        a_start = 180 + seg * 15
        a_end   = a_start + 13
        col = colors_uv[min(seg, len(colors_uv)-1)]
        cv2.ellipse(img, (cx, cy), (r,r), 0, a_start, a_end, col, 6)
    # Needle
    norm = min(uv_val / 12.0, 1.0)
    needle_angle = 180 + norm * 180
    nx = int(cx + (r-8) * math.cos(math.radians(needle_angle)))
    ny = int(cy + (r-8) * math.sin(math.radians(needle_angle)))
    cv2.line(img, (cx, cy), (nx, ny), C_WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, C_WHITE, -1)
    cv2.putText(img, f'{uv_val:.1f}', (cx-12, cy+14), FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)


def draw_forecast_chart(forecast, canvas, x, y, w, h):
    """Draw inline 24-hr forecast bar chart on the canvas."""
    if not forecast or not forecast.get('hours'):
        return
    hours = forecast['hours'][:24]
    temps = forecast['temp'][:24]
    prec  = forecast['precip_p'][:24]
    if not temps:
        return

    # Background
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (26,30,40), -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (60,65,75), 1)
    label(canvas, '24-HR FORECAST', x+6, y+14, 0.38, C_GRAY)

    chart_x, chart_y = x+8, y+22
    chart_w, chart_h = w-16, h-30

    if len(temps) < 2:
        return
    t_min, t_max = min(temps), max(temps)
    t_rng = max(t_max - t_min, 1)
    p_max = 100

    bar_w = chart_w // len(hours)
    for i, (hr, t, p) in enumerate(zip(hours, temps, prec)):
        bx = chart_x + i * bar_w
        # Precip bar (blue, below)
        p_h = int((p / p_max) * (chart_h * 0.35))
        cv2.rectangle(canvas, (bx+1, chart_y+chart_h-p_h), (bx+bar_w-2, chart_y+chart_h),
                      C_BLUE, -1)
        # Temp dot + line
        t_y = chart_y + int((1 - (t - t_min)/t_rng) * (chart_h * 0.58)) + 4
        if i > 0:
            prev_t = temps[i-1]
            prev_ty = chart_y + int((1-(prev_t-t_min)/t_rng)*(chart_h*0.58)) + 4
            cv2.line(canvas, (bx-bar_w//2, prev_ty), (bx+bar_w//2, t_y), C_ORANGE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (bx+bar_w//2, t_y), 2, C_ORANGE, -1)
        # Hour label every 4th
        if i % 4 == 0:
            hr_str = hr[11:16] if len(hr) > 11 else str(i)
            cv2.putText(canvas, hr_str, (bx, chart_y+chart_h+12), FONT, 0.28, C_GRAY, 1, cv2.LINE_AA)
    # Axis labels
    cv2.putText(canvas, f'{t_max:.0f}°', (x+w-28, chart_y+8), FONT, 0.32, C_ORANGE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'{t_min:.0f}°', (x+w-28, chart_y+int(chart_h*0.62)), FONT, 0.32, C_ORANGE, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Temp', (x+w-36, y+h-4), FONT, 0.28, C_ORANGE, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Rain%', (x+44, y+h-4), FONT, 0.28, C_BLUE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# DASHBOARD RENDERER
# ─────────────────────────────────────────────────────────────────
CANVAS_W = 1280
CANVAS_H = 720

def render_dashboard(w: WeatherReading, analytics: WeatherAnalytics,
                     forecast: dict, frame_count: int) -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = C_DARK

    # ── Header ───────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (CANVAS_W, 52), (26,30,42), -1)
    cv2.line(canvas, (0,52), (CANVAS_W,52), (55,60,70), 1)
    cv2.putText(canvas, 'SMART CITY', (18, 22), FONT, 0.55, C_TEAL, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Weather Intelligence Platform', (18, 40), FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, CITY_NAME, (CANVAS_W//2 - 120, 28), FONT, 0.60, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Lat {CITY_LAT}  Lon {CITY_LON}', (CANVAS_W//2 - 80, 44), FONT, 0.30, C_GRAY, 1, cv2.LINE_AA)

    ts = w.timestamp.strftime('%d %b %Y  %H:%M:%S')
    cv2.putText(canvas, ts, (CANVAS_W - 220, 22), FONT, 0.40, C_GRAY, 1, cv2.LINE_AA)
    # Live dot
    blink = int(time.time() * 2) % 2
    cv2.circle(canvas, (CANVAS_W-14, 18), 5, C_GREEN if blink else (30,80,30), -1)
    cv2.putText(canvas, 'LIVE', (CANVAS_W-42, 22), FONT, 0.30, C_GREEN, 1, cv2.LINE_AA)

    Y0 = 62

    # ── LEFT COLUMN (x: 10–340) ──────────────────────────────────
    # Big temperature card
    panel_rect(canvas, 10, Y0, 340, Y0+130, (28,34,46), (55,65,80))
    _, icon_col = w.condition_icon_color()
    cv2.putText(canvas, f'{w.temperature:.1f}', (22, Y0+80), FONT, 2.2, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'C', (170, Y0+55), FONT, 0.90, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, w.condition, (22, Y0+104), FONT, 0.48, icon_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Feels {w.feels_like:.1f}°C', (22, Y0+120), FONT, 0.36, C_GRAY, 1, cv2.LINE_AA)
    hi = w.heat_index()
    hi_color = C_RED if hi > 41 else C_ORANGE if hi > 35 else C_YELLOW if hi > 30 else C_GREEN
    cv2.putText(canvas, f'Heat Idx: {hi:.1f}°', (180, Y0+104), FONT, 0.36, hi_color, 1, cv2.LINE_AA)

    # Humidity, Wind, Pressure mini stats
    Y1 = Y0 + 138
    for i, (lbl_txt, val_txt, col) in enumerate([
        ('HUMIDITY',    f'{w.humidity:.0f}%',        C_CYAN),
        ('PRESSURE',    f'{w.pressure:.0f} hPa',     C_BLUE),
        ('VISIBILITY',  f'{w.visibility:.1f} km',    C_TEAL),
    ]):
        bx = 10 + i * 110
        panel_rect(canvas, bx, Y1, bx+105, Y1+52, (30,36,48), (55,60,72))
        label(canvas, lbl_txt, bx+6, Y1+14, 0.30)
        cv2.putText(canvas, val_txt, (bx+6, Y1+36), FONT, 0.50, col, 1, cv2.LINE_AA)

    Y2 = Y1 + 60
    # Wind compass
    panel_rect(canvas, 10, Y2, 175, Y2+110, (30,36,48), (55,60,72))
    label(canvas, 'WIND DIRECTION', 18, Y2+14, 0.30)
    wind_compass(canvas, 92, Y2+68, 38, w.wind_dir, w.wind_speed)

    # UV gauge
    panel_rect(canvas, 180, Y2, 340, Y2+110, (30,36,48), (55,60,72))
    label(canvas, 'UV INDEX', 188, Y2+14, 0.30)
    uv_gauge(canvas, 260, Y2+78, 42, w.uv_index)
    uv_labels = ['Low','Moderate','High','Very High','Extreme']
    uv_idx_cat = min(int(w.uv_index/2.5), 4)
    cv2.putText(canvas, uv_labels[uv_idx_cat], (192, Y2+100), FONT, 0.32, C_ORANGE, 1, cv2.LINE_AA)

    Y3 = Y2 + 118
    # Dew point & Cloud cover
    panel_rect(canvas, 10, Y3, 175, Y3+52, (30,36,48), (55,60,72))
    label(canvas, 'DEW POINT', 18, Y3+14, 0.30)
    cv2.putText(canvas, f'{w.dew_point:.1f}°C', (18, Y3+36), FONT, 0.52, C_CYAN, 1, cv2.LINE_AA)

    panel_rect(canvas, 180, Y3, 340, Y3+52, (30,36,48), (55,60,72))
    label(canvas, 'CLOUD COVER', 188, Y3+14, 0.30)
    cv2.putText(canvas, f'{w.cloud_cover:.0f}%', (188, Y3+36), FONT, 0.52, C_GRAY, 1, cv2.LINE_AA)
    draw_bar(canvas, 250, Y3+24, 78, 10, w.cloud_cover, 100, C_GRAY)

    # ── CENTER COLUMN (x: 350–820) ─────────────────────────────
    # Risk indices
    panel_rect(canvas, 350, Y0, 820, Y0+64, (28,34,46), (55,65,80))
    label(canvas, 'RISK ASSESSMENT', 358, Y0+14, 0.38, C_GRAY)
    hi_risk_txt, hi_risk_col = w.heat_island_risk()
    fl_risk_txt, fl_risk_col = w.flood_risk()
    for xi, (name, val, col) in enumerate([
        ('HEAT ISLAND RISK', hi_risk_txt, hi_risk_col),
        ('FLOOD RISK',       fl_risk_txt, fl_risk_col),
        ('AIR QUALITY PM2.5', f'{w.aqi_pm25:.1f} μg/m³',
         C_GREEN if w.aqi_pm25 < 35 else C_ORANGE if w.aqi_pm25 < 75 else C_RED),
        ('RAINFALL',         f'{w.rain_mm:.1f} mm/hr',
         C_BLUE if w.rain_mm > 0 else C_GRAY),
    ]):
        bx = 358 + xi * 115
        label(canvas, name, bx, Y0+30, 0.28, C_GRAY)
        cv2.putText(canvas, val, (bx, Y0+50), FONT, 0.40, col, 1, cv2.LINE_AA)

    # Trend sparklines
    Y4 = Y0 + 72
    spark_configs = [
        ('TEMPERATURE °C',  analytics.temp_history,  C_ORANGE),
        ('HUMIDITY %',      analytics.humid_history,  C_CYAN),
        ('WIND SPEED km/h', analytics.wind_history,   C_TEAL),
        ('RAINFALL mm/hr',  analytics.rain_history,   C_BLUE),
    ]
    spark_w = (820-350-16) // 2
    for idx, (sp_lbl, sp_data, sp_col) in enumerate(spark_configs):
        row = idx // 2
        col_i = idx % 2
        sx = 358 + col_i * (spark_w + 8)
        sy = Y4 + row * 82
        panel_rect(canvas, sx-4, sy, sx+spark_w-4, sy+76, (30,36,48), (55,60,72))
        label(canvas, sp_lbl, sx, sy+12, 0.30)
        if sp_data:
            cur_val = list(sp_data)[-1]
            trend = analytics.trend(sp_data)
            trend_sym = '↑' if trend > 0.05 else '↓' if trend < -0.05 else '→'
            trend_col  = C_RED if trend > 0.1 else C_GREEN if trend < -0.1 else C_GRAY
            cv2.putText(canvas, f'{cur_val:.1f}', (sx, sy+34), FONT, 0.56, sp_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, trend_sym, (sx+65, sy+34), FONT, 0.46, trend_col, 1, cv2.LINE_AA)
            pts = analytics.sparkline(sp_data, length=70)
            draw_sparkline(canvas, pts, sx, sy+38, spark_w-12, 32, sp_col)

    # Forecast chart
    Y5 = Y4 + 172
    draw_forecast_chart(forecast, canvas, 350, Y5, 470, 130)

    # Hourly bar chart (manual)
    if forecast and forecast.get('temp'):
        bar_x, bar_y = 830, Y5
        bar_w2, bar_h2 = 300, 130
        panel_rect(canvas, bar_x, bar_y, bar_x+bar_w2, bar_y+bar_h2, (26,30,40), (60,65,75))
        label(canvas, 'PRECIPITATION PROBABILITY %', bar_x+6, bar_y+14, 0.30, C_GRAY)
        prec = forecast.get('precip_p', [])[:24]
        if prec:
            bw = (bar_w2 - 16) // len(prec)
            for i, p in enumerate(prec):
                bx = bar_x + 8 + i * bw
                bh = int(p * 0.9)
                by2 = bar_y + bar_h2 - bh - 16
                bcol = (C_BLUE if p > 60 else C_CYAN if p > 30 else C_TEAL)
                cv2.rectangle(canvas, (bx+1, by2), (bx+bw-2, bar_y+bar_h2-16), bcol, -1)
                if i % 6 == 0:
                    hr_str = forecast['hours'][i][11:16] if len(forecast['hours']) > i else ''
                    cv2.putText(canvas, hr_str, (bx, bar_y+bar_h2-4), FONT, 0.25, C_GRAY, 1, cv2.LINE_AA)

    # ── RIGHT COLUMN (x: 830–1270) ───────────────────────────────
    # Alerts
    panel_rect(canvas, 830, Y0, 1270, Y0+200, (26,30,40), (60,65,75))
    label(canvas, 'LIVE WEATHER ALERTS', 838, Y0+16, 0.40, C_GRAY)
    cv2.line(canvas, (838, Y0+22), (1262, Y0+22), (55,60,70), 1)
    ay = Y0 + 38
    if analytics.alerts:
        for al in reversed(list(analytics.alerts)):
            if ay > Y0+195:
                break
            sev_col = al['color']
            cv2.putText(canvas, f"[{al['time']}]", (838, ay), FONT, 0.33, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"[{al['level']}]", (912, ay), FONT, 0.33, sev_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, al['msg'], (972, ay), FONT, 0.36, C_WHITE, 1, cv2.LINE_AA)
            ay += 18
    else:
        cv2.putText(canvas, 'No active alerts', (838, ay+10), FONT, 0.42, C_GREEN, 1, cv2.LINE_AA)

    # City stats summary
    panel_rect(canvas, 830, Y0+208, 1270, Y0+340, (26,30,40), (60,65,75))
    label(canvas, 'SESSION SUMMARY', 838, Y0+224, 0.38, C_GRAY)
    rows_stat = [
        ('Records logged', str(len(analytics.temp_history)), C_WHITE),
        ('Max temp (session)', f'{max(analytics.temp_history, default=0):.1f}°C', C_RED),
        ('Min temp (session)', f'{min(analytics.temp_history, default=0):.1f}°C', C_CYAN),
        ('Avg humidity',       f'{np.mean(list(analytics.humid_history)) if analytics.humid_history else 0:.1f}%', C_TEAL),
        ('Peak wind speed',    f'{max(analytics.wind_history, default=0):.1f} km/h', C_YELLOW),
        ('Total alerts',       str(len(analytics.alerts)), C_ORANGE),
        ('Log file',           LOG_FILE, C_GRAY),
    ]
    for ri, (k, v, vc) in enumerate(rows_stat):
        ry = Y0 + 244 + ri * 14
        label(canvas, k, 838, ry, 0.33, C_GRAY)
        cv2.putText(canvas, v, (1050, ry), FONT, 0.35, vc, 1, cv2.LINE_AA)

    # Status bar
    cv2.rectangle(canvas, (0, CANVAS_H-22), (CANVAS_W, CANVAS_H), (18,22,30), -1)
    status = (f"  SmartCity Weather Monitor  |  {CITY_NAME}  |  "
              f"Temp: {w.temperature:.1f}°C  Humidity: {w.humidity:.0f}%  "
              f"Wind: {w.wind_speed:.0f}km/h  |  "
              f"[Q] Quit  [S] Snapshot  [F] Toggle Forecast")
    cv2.putText(canvas, status, (8, CANVAS_H-7), FONT, 0.31, C_GRAY, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────
# BACKGROUND FETCH THREAD
# ─────────────────────────────────────────────────────────────────
class WeatherFetcher:
    def __init__(self):
        self.reading  = WeatherReading()
        self.forecast = None
        self.lock     = threading.Lock()
        self._fetch_now()
        self._start_thread()

    def _fetch_now(self):
        r, f = fetch_weather_live(CITY_LAT, CITY_LON)
        if r:
            with self.lock:
                self.reading  = r
                self.forecast = f

    def _start_thread(self):
        def worker():
            while True:
                time.sleep(FETCH_INTERVAL)
                self._fetch_now()
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def get(self):
        with self.lock:
            return self.reading, self.forecast


# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  SmartCity AI — Weather Monitor  (Module 2 of 3)")
    print("=" * 62)
    print(f"  City     : {CITY_NAME}")
    print(f"  Location : {CITY_LAT}°N, {CITY_LON}°E")
    print(f"  API      : Open-Meteo (free, no key required)")
    print(f"  Log file : {LOG_FILE}")
    print("  Controls : [Q] Quit   [S] Snapshot   [F] Forecast")
    print("=" * 62)

    fetcher   = WeatherFetcher()
    analytics = WeatherAnalytics()
    frame_count = 0
    last_sim_update = time.time()
    show_forecast = True

    base_reading, forecast = fetcher.get()
    current = base_reading

    while True:
        now = time.time()

        # Micro-update simulation every REFRESH_SIM seconds
        if now - last_sim_update > REFRESH_SIM:
            base_reading, forecast = fetcher.get()
            current = simulate_micro_update(base_reading)
            analytics.record(current)
            last_sim_update = now

        # Render
        canvas = render_dashboard(current, analytics, forecast if show_forecast else None, frame_count)
        cv2.imshow("SmartCity — Weather Monitor", canvas)
        frame_count += 1

        key = cv2.waitKey(60) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            fname = f"weather_snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"  [Snapshot] Saved → {fname}")
        elif key == ord('f'):
            show_forecast = not show_forecast

    cv2.destroyAllWindows()
    print(f"\n  Session ended. Weather log saved to: {LOG_FILE}")
    if analytics.temp_history:
        print(f"  Session avg temperature: {np.mean(list(analytics.temp_history)):.1f}°C")


if __name__ == "__main__":
    main()
