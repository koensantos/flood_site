"""
Step 3 — Model Training Script (Jersey City)
Flood Risk Modeling — Jersey City, NJ
Koen Mitchel Santos
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FEATURE_CSV     = "jc_feature_grid.csv"
OUTPUT_CSV      = "jc_model_predictions.csv"
METRICS_CSV     = "jc_model_metrics.csv"
RANDOM_STATE    = 42
FEATURES        = ["elevation_m", "rainfall_inches"]
TARGET          = "label"

# ─────────────────────────────────────────────
# STEP 1 - LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading feature matrix...")
print("=" * 60)

df = pd.read_csv(FEATURE_CSV)
df["label"] = df["in_fema_flood_zone"]
print(f"  Loaded {len(df)} grid cells")
print(f"\n  Class distribution:")
print(f"    Flooded (1):     {df['label'].sum()}")
print(f"    Not flooded (0): {(df['label'] == 0).sum()}")

# ─────────────────────────────────────────────
# STEP 2 - PREPARE FEATURES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 - Preparing features...")
print("=" * 60)

df_clean = df.dropna(subset=FEATURES + [TARGET])
print(f"  Clean rows: {len(df_clean)}")

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Training set: {len(X_train)} cells")
print(f"  Test set:     {len(X_test)} cells")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 3 - TRAIN RANDOM FOREST
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 - Training Random Forest...")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

rf_preds = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
rf_acc   = accuracy_score(y_test, rf_preds)
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_cv    = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

print(f"  Accuracy:      {rf_acc:.4f}")
print(f"  ROC-AUC:       {rf_auc:.4f}")
print(f"  Cross-val AUC: {rf_cv:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["Not Flooded", "Flooded"]))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# ─────────────────────────────────────────────
# STEP 4 - TRAIN XGBOOST
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Training XGBoost...")
print("=" * 60)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos = neg / pos if pos > 0 else 1

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
    eval_metric="logloss", verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_acc   = accuracy_score(y_test, xgb_preds)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_cv    = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

print(f"  Accuracy:      {xgb_acc:.4f}")
print(f"  ROC-AUC:       {xgb_auc:.4f}")
print(f"  Cross-val AUC: {xgb_cv:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, xgb_preds, target_names=["Not Flooded", "Flooded"]))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

# ─────────────────────────────────────────────
# STEP 5 - COMPARE MODELS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 - Model Comparison")
print("=" * 60)

metrics = pd.DataFrame({
    "Model":        ["Random Forest", "XGBoost"],
    "Accuracy":     [rf_acc, xgb_acc],
    "ROC_AUC":      [rf_auc, xgb_auc],
    "CrossVal_AUC": [rf_cv, xgb_cv]
})
print(metrics.to_string(index=False))
metrics.to_csv(METRICS_CSV, index=False)
winner = "Random Forest" if rf_auc >= xgb_auc else "XGBoost"
print(f"\n  Best model by ROC-AUC: {winner}")

# ─────────────────────────────────────────────
# STEP 6 - SAVE PREDICTIONS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 - Saving predictions...")
print("=" * 60)

X_all        = df_clean[FEATURES].values
X_all_scaled = scaler.transform(X_all)

df_clean = df_clean.copy()
df_clean["rf_prediction"]   = rf.predict(X_all_scaled)
df_clean["rf_probability"]  = rf.predict_proba(X_all_scaled)[:, 1]
df_clean["xgb_prediction"]  = xgb_model.predict(X_all_scaled)
df_clean["xgb_probability"] = xgb_model.predict_proba(X_all_scaled)[:, 1]
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"  Predictions saved to: {OUTPUT_CSV}")

# ─────────────────────────────────────────────
# STEP 7 - CROSS CITY COMPARISON
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 - Cross-city comparison (Hoboken vs Jersey City)...")
print("=" * 60)

try:
    hoboken_metrics = pd.read_csv("model_metrics.csv")
    hoboken_metrics.insert(0, "City", "Hoboken")
    jc_metrics = metrics.copy()
    jc_metrics.insert(0, "City", "Jersey City")
    combined = pd.concat([hoboken_metrics, jc_metrics], ignore_index=True)
    combined.to_csv("cross_city_metrics.csv", index=False)
    print(combined.to_string(index=False))
    print(f"\n  Cross-city metrics saved to: cross_city_metrics.csv")
except Exception as e:
    print(f"  Hoboken metrics not found: {e}")
    print(f"  Run train_model.py first to generate Hoboken metrics")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Predictions: {OUTPUT_CSV}")
print(f"  Metrics:     {METRICS_CSV}")
print("  Next step: Run jc_visualize_results.py")
print("=" * 60)