"""
weather_module.py
─────────────────────────────────────────────────────────────────
Smart City Weather Module

• City database with lat/lon for 80+ Indian cities
• Fetches live data from Open-Meteo API (FREE, no key needed)
• Returns structured weather dict used by app.py
"""

import requests
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# CITY DATABASE — name → {lat, lon, state, elevation}
# Add more cities here as needed.
# ─────────────────────────────────────────────────────────────────
CITY_DATABASE = {
    # ── Rajasthan ──────────────────────────────────────────────
    "Udaipur":       {"lat": 24.5854, "lon": 73.7125, "state": "Rajasthan", "elevation": 598},
    "Jaipur":        {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "elevation": 431},
    "Jodhpur":       {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan", "elevation": 231},
    "Kota":          {"lat": 25.2138, "lon": 75.8648, "state": "Rajasthan", "elevation": 271},
    "Bikaner":       {"lat": 28.0229, "lon": 73.3119, "state": "Rajasthan", "elevation": 226},
    "Ajmer":         {"lat": 26.4499, "lon": 74.6399, "state": "Rajasthan", "elevation": 486},
    "Alwar":         {"lat": 27.5665, "lon": 76.6293, "state": "Rajasthan", "elevation": 271},
    "Bharatpur":     {"lat": 27.2152, "lon": 77.4938, "state": "Rajasthan", "elevation": 174},
    "Sikar":         {"lat": 27.6094, "lon": 75.1398, "state": "Rajasthan", "elevation": 427},
    "Pali":          {"lat": 25.7711, "lon": 73.3234, "state": "Rajasthan", "elevation": 212},

    # ── Maharashtra ────────────────────────────────────────────
    "Mumbai":        {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "elevation": 14},
    "Pune":          {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "elevation": 560},
    "Nagpur":        {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "elevation": 310},
    "Nashik":        {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra", "elevation": 584},
    "Aurangabad":    {"lat": 19.8762, "lon": 75.3433, "state": "Maharashtra", "elevation": 513},
    "Solapur":       {"lat": 17.6805, "lon": 75.9064, "state": "Maharashtra", "elevation": 457},
    "Kolhapur":      {"lat": 16.7050, "lon": 74.2433, "state": "Maharashtra", "elevation": 569},
    "Amravati":      {"lat": 20.9374, "lon": 77.7796, "state": "Maharashtra", "elevation": 342},

    # ── Karnataka ──────────────────────────────────────────────
    "Bengaluru":     {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "elevation": 920},
    "Mysuru":        {"lat": 12.2958, "lon": 76.6394, "state": "Karnataka", "elevation": 763},
    "Mangaluru":     {"lat": 12.8698, "lon": 74.8435, "state": "Karnataka", "elevation": 22},
    "Hubballi":      {"lat": 15.3647, "lon": 75.1240, "state": "Karnataka", "elevation": 663},
    "Belagavi":      {"lat": 15.8497, "lon": 74.4977, "state": "Karnataka", "elevation": 751},

    # ── Tamil Nadu ──────────────────────────────────────────────
    "Chennai":       {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu",  "elevation": 6},
    "Coimbatore":    {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu",  "elevation": 411},
    "Madurai":       {"lat":  9.9252, "lon": 78.1198, "state": "Tamil Nadu",  "elevation": 101},
    "Tiruchirappalli":{"lat":10.7905, "lon": 78.7047, "state": "Tamil Nadu",  "elevation": 78},
    "Salem":         {"lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu",  "elevation": 278},
    "Tirunelveli":   {"lat":  8.7139, "lon": 77.7567, "state": "Tamil Nadu",  "elevation": 55},

    # ── Uttar Pradesh ──────────────────────────────────────────
    "Lucknow":       {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "elevation": 123},
    "Kanpur":        {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh", "elevation": 126},
    "Agra":          {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh", "elevation": 169},
    "Varanasi":      {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh", "elevation": 80},
    "Allahabad":     {"lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh", "elevation": 98},
    "Noida":         {"lat": 28.5355, "lon": 77.3910, "state": "Uttar Pradesh", "elevation": 199},
    "Meerut":        {"lat": 28.9845, "lon": 77.7064, "state": "Uttar Pradesh", "elevation": 220},
    "Mathura":       {"lat": 27.4924, "lon": 77.6737, "state": "Uttar Pradesh", "elevation": 174},

    # ── Gujarat ────────────────────────────────────────────────
    "Ahmedabad":     {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "elevation": 53},
    "Surat":         {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "elevation": 13},
    "Vadodara":      {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat", "elevation": 37},
    "Rajkot":        {"lat": 22.3039, "lon": 70.8022, "state": "Gujarat", "elevation": 138},
    "Bhavnagar":     {"lat": 21.7645, "lon": 72.1519, "state": "Gujarat", "elevation": 26},

    # ── Delhi NCR ──────────────────────────────────────────────
    "New Delhi":     {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "elevation": 216},
    "Delhi":         {"lat": 28.7041, "lon": 77.1025, "state": "Delhi", "elevation": 216},
    "Gurgaon":       {"lat": 28.4595, "lon": 77.0266, "state": "Haryana", "elevation": 217},
    "Faridabad":     {"lat": 28.4089, "lon": 77.3178, "state": "Haryana", "elevation": 198},
    "Ghaziabad":     {"lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh", "elevation": 209},

    # ── West Bengal ────────────────────────────────────────────
    "Kolkata":       {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "elevation": 9},
    "Howrah":        {"lat": 22.5958, "lon": 88.2636, "state": "West Bengal", "elevation": 12},
    "Siliguri":      {"lat": 26.7271, "lon": 88.3953, "state": "West Bengal", "elevation": 122},
    "Durgapur":      {"lat": 23.5204, "lon": 87.3119, "state": "West Bengal", "elevation": 73},

    # ── Telangana / Andhra Pradesh ─────────────────────────────
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "elevation": 542},
    "Warangal":      {"lat": 17.9784, "lon": 79.5941, "state": "Telangana", "elevation": 302},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "elevation": 45},
    "Vijayawada":    {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "elevation": 26},
    "Guntur":        {"lat": 16.3067, "lon": 80.4365, "state": "Andhra Pradesh", "elevation": 33},

    # ── Madhya Pradesh ─────────────────────────────────────────
    "Bhopal":        {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "elevation": 523},
    "Indore":        {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "elevation": 553},
    "Gwalior":       {"lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh", "elevation": 211},
    "Jabalpur":      {"lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh", "elevation": 412},
    "Ujjain":        {"lat": 23.1828, "lon": 75.7772, "state": "Madhya Pradesh", "elevation": 491},

    # ── Punjab / Haryana ───────────────────────────────────────
    "Ludhiana":      {"lat": 30.9010, "lon": 75.8573, "state": "Punjab", "elevation": 244},
    "Amritsar":      {"lat": 31.6340, "lon": 74.8723, "state": "Punjab", "elevation": 234},
    "Chandigarh":    {"lat": 30.7333, "lon": 76.7794, "state": "Punjab",  "elevation": 321},
    "Jalandhar":     {"lat": 31.3260, "lon": 75.5762, "state": "Punjab", "elevation": 228},
    "Ambala":        {"lat": 30.3752, "lon": 76.7821, "state": "Haryana", "elevation": 270},

    # ── Kerala ─────────────────────────────────────────────────
    "Thiruvananthapuram": {"lat": 8.5241,  "lon": 76.9366, "state": "Kerala", "elevation": 62},
    "Kochi":         {"lat": 9.9312,  "lon": 76.2673, "state": "Kerala", "elevation": 7},
    "Kozhikode":     {"lat": 11.2588, "lon": 75.7804, "state": "Kerala", "elevation": 14},
    "Thrissur":      {"lat": 10.5276, "lon": 76.2144, "state": "Kerala", "elevation": 2},

    # ── Others ─────────────────────────────────────────────────
    "Patna":         {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "elevation": 53},
    "Ranchi":        {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "elevation": 651},
    "Bhubaneswar":   {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "elevation": 45},
    "Raipur":        {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh", "elevation": 298},
    "Guwahati":      {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "elevation": 55},
    "Dehradun":      {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "elevation": 640},
    "Shimla":        {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "elevation": 2200},
    "Jammu":         {"lat": 32.7266, "lon": 74.8570, "state": "J&K", "elevation": 327},
    "Srinagar":      {"lat": 34.0837, "lon": 74.7973, "state": "J&K", "elevation": 1585},
    "Puducherry":    {"lat": 11.9416, "lon": 79.8083, "state": "Puducherry", "elevation": 15},
    "Panaji":        {"lat": 15.4909, "lon": 73.8278, "state": "Goa", "elevation": 7},
}

# WMO weather code → human-readable
WMO_CODES = {
    0:"Clear Sky", 1:"Mainly Clear", 2:"Partly Cloudy", 3:"Overcast",
    45:"Foggy", 48:"Icy Fog", 51:"Light Drizzle", 53:"Moderate Drizzle",
    55:"Heavy Drizzle", 61:"Light Rain", 63:"Moderate Rain", 65:"Heavy Rain",
    71:"Light Snow", 73:"Moderate Snow", 75:"Heavy Snow",
    80:"Rain Showers", 81:"Heavy Showers", 82:"Violent Showers",
    95:"Thunderstorm", 96:"Thunderstorm + Hail", 99:"Heavy Thunderstorm",
}


# ─────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def get_weather_by_city(city_name: str) -> dict:
    """
    Fetch live weather for a named city.
    Falls back to simulated data if API is unavailable.
    """
    city_name = city_name.strip().title()

    # Try exact match, then partial match
    info = CITY_DATABASE.get(city_name)
    if not info:
        for k, v in CITY_DATABASE.items():
            if city_name.lower() in k.lower():
                info = v
                city_name = k
                break

    if not info:
        return _simulated_weather(city_name, 20.0, 78.0)

    result = get_weather_by_coords(info["lat"], info["lon"], city_name)
    result["state"]     = info.get("state", "")
    result["elevation"] = info.get("elevation", "—")
    return result


def get_weather_by_coords(lat: float, lon: float, city_name: str = "Custom") -> dict:
    """
    Fetch live weather from Open-Meteo for given coordinates.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "weather_code", "cloud_cover", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "uv_index",
        ],
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cur  = data.get("current", {})

        temp    = float(cur.get("temperature_2m", 30))
        humid   = float(cur.get("relative_humidity_2m", 55))
        wmo     = int(cur.get("weather_code", 2))
        wind    = float(cur.get("wind_speed_10m", 10))
        wind_d  = float(cur.get("wind_direction_10m", 200))
        feels   = float(cur.get("apparent_temperature", temp))
        cloud   = float(cur.get("cloud_cover", 30))
        press   = float(cur.get("pressure_msl", 1010))
        rain    = float(cur.get("precipitation", 0))
        uv      = float(cur.get("uv_index", 5))
        dew     = round(temp - ((100 - humid) / 5), 1)
        tz      = data.get("timezone", "Asia/Kolkata")

        return {
            "city":        city_name,
            "state":       "",
            "country":     "India",
            "lat":         lat,
            "lon":         lon,
            "elevation":   "—",
            "timezone":    tz,
            "temperature": round(temp, 1),
            "humidity":    round(humid, 1),
            "feels_like":  round(feels, 1),
            "dew_point":   dew,
            "wind_speed":  round(wind, 1),
            "wind_dir":    round(wind_d, 0),
            "cloud_cover": round(cloud, 0),
            "pressure":    round(press, 1),
            "rain_mm":     round(rain, 2),
            "uv_index":    round(uv, 1),
            "visibility":  10.0,
            "condition":   WMO_CODES.get(wmo, "Unknown"),
            "source":      "Open-Meteo API",
            "fetched_at":  datetime.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        print(f"[WeatherModule] API error: {e} — using simulated data")
        return _simulated_weather(city_name, lat, lon)


def _simulated_weather(city_name: str, lat: float, lon: float) -> dict:
    """Return realistic simulated weather when API is unavailable."""
    import random, math
    hour = datetime.now().hour
    base_temp = 30 + 5 * math.sin((hour - 6) * math.pi / 12)
    return {
        "city":        city_name,
        "state":       "",
        "country":     "India",
        "lat":         lat,
        "lon":         lon,
        "elevation":   "—",
        "timezone":    "Asia/Kolkata",
        "temperature": round(base_temp + random.uniform(-1, 1), 1),
        "humidity":    round(random.uniform(45, 75), 1),
        "feels_like":  round(base_temp + 2, 1),
        "dew_point":   round(base_temp - 8, 1),
        "wind_speed":  round(random.uniform(5, 20), 1),
        "wind_dir":    round(random.uniform(0, 360), 0),
        "cloud_cover": round(random.uniform(10, 60), 0),
        "pressure":    round(random.uniform(1005, 1015), 1),
        "rain_mm":     0.0,
        "uv_index":    round(random.uniform(2, 9), 1),
        "visibility":  10.0,
        "condition":   "Partly Cloudy",
        "source":      "Simulated (API unavailable)",
        "fetched_at":  datetime.now().strftime("%H:%M:%S"),
    }
