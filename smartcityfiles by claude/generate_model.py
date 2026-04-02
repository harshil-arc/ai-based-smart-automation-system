"""
generate_model.py
─────────────────────────────────────────────────────────────────
Run this ONCE to create model.pkl in your project folder.

    python generate_model.py

After running, model.pkl will be loaded automatically by app.py.

If you already have your own trained model.pkl, skip this file.
Just make sure your model:
  • Has .predict() and .predict_proba() methods
  • Was trained on features in this order:
      [traffic_volume, temperature, humidity, population_density,
       hour_of_day, is_weekend, weather_code_int, road_quality_int]
  • Outputs labels: "HIGH", "MEDIUM", or "LOW"
"""

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("=" * 60)
print("  Smart City — Model Training Script")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# GENERATE SYNTHETIC TRAINING DATA
# ─────────────────────────────────────────────────────────────────
np.random.seed(2024)
N = 6000
print(f"\n  Generating {N} synthetic training samples…")

traffic   = np.random.randint(50, 5000, N).astype(float)
temp      = np.random.uniform(15, 50, N)
humidity  = np.random.uniform(15, 98, N)
pop       = np.random.randint(200, 55000, N).astype(float)
hour      = np.random.randint(0, 24, N).astype(float)
weekend   = np.random.randint(0, 2, N).astype(float)
w_code    = np.random.randint(0, 6, N).astype(float)   # 0=Clear…5=Fog
road_q    = np.random.randint(0, 5, N).astype(float)   # 0=Poor…4=Excellent

X = np.column_stack([traffic, temp, humidity, pop, hour, weekend, w_code, road_q])

# ── Rule-based scoring (realistic urban logic) ──────────────────
score = (
    # Traffic weight (40 pts max)
    (traffic / 5000) * 40
    # Temperature stress (20 pts max)
    + np.clip((temp - 20) / 30, 0, 1) * 20
    # Humidity discomfort (10 pts max)
    + (humidity / 100) * 10
    # Population pressure (15 pts max)
    + (pop / 55000) * 15
    # Rush hour boost
    + np.where((hour >= 7) & (hour <= 9),   12, 0)
    + np.where((hour >= 17) & (hour <= 19), 12, 0)
    + np.where((hour >= 12) & (hour <= 13),  5, 0)
    # Weekday premium
    + (1 - weekend) * 5
    # Bad weather penalty
    + np.where(w_code >= 3, 8, 0)
    + np.where(w_code >= 4, 5, 0)
    # Poor road quality surcharge
    + ((4 - road_q) / 4) * 10
    # Night reduction
    + np.where((hour <= 4) | (hour >= 23), -15, 0)
)

# Add noise
score += np.random.normal(0, 5, N)

labels = np.where(score >= 62, "HIGH", np.where(score >= 38, "MEDIUM", "LOW"))

print(f"  Label distribution:")
for lbl in ["HIGH", "MEDIUM", "LOW"]:
    count = (labels == lbl).sum()
    print(f"    {lbl:<8} {count:4d}  ({count/N*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\n  Training Gradient Boosting Classifier…")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Test accuracy: {acc*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["HIGH","LOW","MEDIUM"]))

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
feat_names = [
    "Traffic Volume", "Temperature", "Humidity",
    "Population Density", "Hour of Day", "Is Weekend",
    "Weather Code", "Road Quality"
]
fi = model.feature_importances_
print("  Feature Importance:")
for name, imp in sorted(zip(feat_names, fi), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"    {name:<22} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n  ✅ model.pkl saved successfully!")
print("  Now run:  streamlit run app.py")
print("=" * 60)
