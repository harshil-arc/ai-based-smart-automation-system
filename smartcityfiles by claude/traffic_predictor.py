"""
traffic_predictor.py
─────────────────────────────────────────────────────────────────
Smart City Traffic Prediction Module

• Loads model.pkl if it exists (your trained model)
• Falls back to a built-in Random Forest if model.pkl is missing
• Exposes predict_resource_need() and get_feature_importance()
• Decision: HIGH / MEDIUM / LOW
"""

import os
import pickle
import numpy as np

# ─────────────────────────────────────────────────────────────────
# ENCODERS (keep in sync with training)
# ─────────────────────────────────────────────────────────────────
WEATHER_CODE_MAP = {
    "Clear": 0, "Partly Cloudy": 1, "Overcast": 2,
    "Rain": 3, "Heavy Rain": 4, "Fog": 5,
}

ROAD_QUALITY_MAP = {
    "Poor": 0, "Below Average": 1, "Average": 2,
    "Good": 3, "Excellent": 4,
}

FEATURE_NAMES = [
    "traffic_volume",
    "temperature",
    "humidity",
    "population_density",
    "hour_of_day",
    "is_weekend",
    "weather_code",
    "road_quality",
]

# ─────────────────────────────────────────────────────────────────
# LOAD OR BUILD MODEL
# ─────────────────────────────────────────────────────────────────
_model = None
_model_type = "none"


def _load_or_build_model():
    global _model, _model_type

    # 1. Try loading model.pkl from current directory
    pkl_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                _model = pickle.load(f)
            _model_type = "trained"
            print(f"[TrafficPredictor] Loaded model.pkl from {pkl_path}")
            return
        except Exception as e:
            print(f"[TrafficPredictor] Could not load model.pkl: {e}")

    # 2. Fall back: train a lightweight Random Forest on synthetic data
    print("[TrafficPredictor] model.pkl not found — training fallback RF model…")
    _model = _build_fallback_model()
    _model_type = "fallback"


def _build_fallback_model():
    """
    Train a simple Random Forest on synthetically generated data.
    This runs automatically if model.pkl is absent.
    Labels are logic-based (not arbitrary), so predictions are sensible.
    """
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    N = 3000

    traffic   = np.random.randint(100, 5000, N).astype(float)
    temp      = np.random.uniform(18, 48, N)
    humidity  = np.random.uniform(20, 95, N)
    pop       = np.random.randint(500, 50000, N).astype(float)
    hour      = np.random.randint(0, 24, N).astype(float)
    weekend   = np.random.randint(0, 2, N).astype(float)
    w_code    = np.random.randint(0, 6, N).astype(float)
    road_q    = np.random.randint(0, 5, N).astype(float)

    X = np.column_stack([traffic, temp, humidity, pop, hour, weekend, w_code, road_q])

    # Rule-based labels (realistic)
    score = (
        traffic / 5000 * 40
        + (temp - 18) / 30 * 20
        + humidity / 100 * 10
        + pop / 50000 * 15
        + ((hour >= 7) & (hour <= 9)).astype(float) * 10
        + ((hour >= 17) & (hour <= 19)).astype(float) * 10
        + (1 - weekend) * 5
        + (w_code >= 3).astype(float) * 8
        + (4 - road_q) / 4 * 10
    )

    labels = np.where(score >= 65, "HIGH", np.where(score >= 40, "MEDIUM", "LOW"))

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    clf.fit(X, labels)
    print(f"[TrafficPredictor] Fallback RF trained. Classes: {clf.classes_}")
    return clf


# Load on import
_load_or_build_model()


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────
def predict_resource_need(
    traffic_volume: float,
    temperature: float,
    humidity: float,
    population_density: float,
    hour_of_day: int,
    is_weekend: int,
    weather_code: str,
    road_quality: str,
) -> dict:
    """
    Returns:
        {
          "decision":      "HIGH" | "MEDIUM" | "LOW",
          "probabilities": {"HIGH": 0.7, "MEDIUM": 0.2, "LOW": 0.1},
          "model_type":    "trained" | "fallback",
          "features":      {feature_name: value, ...},
        }
    """
    wc = WEATHER_CODE_MAP.get(weather_code, 0)
    rq = ROAD_QUALITY_MAP.get(road_quality, 2)

    X = np.array([[
        float(traffic_volume),
        float(temperature),
        float(humidity),
        float(population_density),
        float(hour_of_day),
        float(is_weekend),
        float(wc),
        float(rq),
    ]])

    prediction = _model.predict(X)[0]

    # Get probabilities — handle models that may not have predict_proba
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        classes = list(_model.classes_)
        prob_dict = {c: round(float(p), 3) for c, p in zip(classes, proba)}
    else:
        prob_dict = {"HIGH": 0.0, "MEDIUM": 0.0, "LOW": 0.0}
        prob_dict[prediction] = 1.0

    # Ensure all three keys exist
    for k in ["HIGH", "MEDIUM", "LOW"]:
        prob_dict.setdefault(k, 0.0)

    return {
        "decision":      str(prediction),
        "probabilities": prob_dict,
        "model_type":    _model_type,
        "features": {
            "Traffic Volume":     traffic_volume,
            "Temperature (°C)":   temperature,
            "Humidity (%)":       humidity,
            "Population Density": population_density,
            "Hour of Day":        hour_of_day,
            "Is Weekend":         bool(is_weekend),
            "Weather":            weather_code,
            "Road Quality":       road_quality,
        },
    }


def get_feature_importance() -> dict:
    """
    Returns feature importance dict if the model supports it.
    Keys are human-readable feature names.
    """
    if not hasattr(_model, "feature_importances_"):
        return {}
    raw   = _model.feature_importances_
    total = raw.sum() or 1.0
    labels = [
        "Traffic Volume", "Temperature", "Humidity",
        "Population Density", "Hour of Day", "Is Weekend",
        "Weather Code", "Road Quality",
    ]
    return {labels[i]: round(float(raw[i] / total), 4) for i in range(len(raw))}


def get_model_info() -> dict:
    """Return metadata about the current model."""
    info = {
        "model_type": _model_type,
        "model_class": type(_model).__name__ if _model else "None",
        "features": FEATURE_NAMES,
    }
    if hasattr(_model, "n_estimators"):
        info["n_estimators"] = _model.n_estimators
    if hasattr(_model, "classes_"):
        info["classes"] = list(_model.classes_)
    return info
