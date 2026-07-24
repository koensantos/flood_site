"""
Step 3 — Model Training Script
Flood Risk Modeling — Hoboken, NJ
Koen Mitchel Santos

This script:
1. Loads the feature matrix
2. Trains Random Forest and XGBoost models
3. Evaluates and compares both models
4. Saves predictions and performance metrics
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

FEATURE_CSV     = "feature_grid.csv"
OUTPUT_CSV      = "model_predictions.csv"
METRICS_CSV     = "model_metrics.csv"
RANDOM_STATE    = 42

# ─────────────────────────────────────────────
# STEP 1 - LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading feature matrix...")
print("=" * 60)

df = pd.read_csv(FEATURE_CSV)
print(f"  Loaded {len(df)} grid cells")
print(f"  Columns: {list(df.columns)}")

# Use in_fema_flood_zone as label
df["label"] = df["in_fema_flood_zone"]

print(f"\n  Class distribution:")
print(f"    Flooded (1):     {df['label'].sum()}")
print(f"    Not flooded (0): {(df['label'] == 0).sum()}")

# ─────────────────────────────────────────────
# STEP 2 - PREPARE FEATURES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 - Preparing features...")
print("=" * 60)

FEATURES = ["elevation_m", "rainfall_inches"]
TARGET   = "label"

# Drop rows with missing values
df_clean = df.dropna(subset=FEATURES + [TARGET])
print(f"  Clean rows: {len(df_clean)} (dropped {len(df) - len(df_clean)} with NaN)")

X = df_clean[FEATURES].values
y = df_clean[TARGET].values

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Training set: {len(X_train)} cells")
print(f"  Test set:     {len(X_test)} cells")

# Scale features
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
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

rf_preds  = rf.predict(X_test_scaled)
rf_proba  = rf.predict_proba(X_test_scaled)[:, 1]
rf_acc    = accuracy_score(y_test, rf_preds)
rf_auc    = roc_auc_score(y_test, rf_proba)
rf_cv     = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

print(f"  Accuracy:        {rf_acc:.4f}")
print(f"  ROC-AUC:         {rf_auc:.4f}")
print(f"  Cross-val AUC:   {rf_cv:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["Not Flooded", "Flooded"]))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print(f"\n  Feature Importances:")
for feat, imp in zip(FEATURES, rf.feature_importances_):
    print(f"    {feat}: {imp:.4f}")

# ─────────────────────────────────────────────
# STEP 4 - TRAIN XGBOOST
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Training XGBoost...")
print("=" * 60)

# Calculate class weight for imbalanced data
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos = neg / pos if pos > 0 else 1

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_acc   = accuracy_score(y_test, xgb_preds)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_cv    = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

print(f"  Accuracy:        {xgb_acc:.4f}")
print(f"  ROC-AUC:         {xgb_auc:.4f}")
print(f"  Cross-val AUC:   {xgb_cv:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, xgb_preds, target_names=["Not Flooded", "Flooded"]))
print(f"  Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

print(f"\n  Feature Importances:")
xgb_importances = xgb_model.feature_importances_
for feat, imp in zip(FEATURES, xgb_importances):
    print(f"    {feat}: {imp:.4f}")

# ─────────────────────────────────────────────
# STEP 5 - COMPARE MODELS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 - Model Comparison")
print("=" * 60)

metrics = pd.DataFrame({
    "Model":         ["Random Forest", "XGBoost"],
    "Accuracy":      [rf_acc, xgb_acc],
    "ROC_AUC":       [rf_auc, xgb_auc],
    "CrossVal_AUC":  [rf_cv, xgb_cv]
})
print(metrics.to_string(index=False))
metrics.to_csv(METRICS_CSV, index=False)
print(f"\n  Metrics saved to: {METRICS_CSV}")

winner = "Random Forest" if rf_auc >= xgb_auc else "XGBoost"
print(f"\n  Best model by ROC-AUC: {winner}")

# ─────────────────────────────────────────────
# STEP 6 - SAVE FULL PREDICTIONS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 - Saving predictions for all grid cells...")
print("=" * 60)

X_all        = df_clean[FEATURES].values
X_all_scaled = scaler.transform(X_all)

df_clean = df_clean.copy()
df_clean["rf_prediction"]    = rf.predict(X_all_scaled)
df_clean["rf_probability"]   = rf.predict_proba(X_all_scaled)[:, 1]
df_clean["xgb_prediction"]   = xgb_model.predict(X_all_scaled)
df_clean["xgb_probability"]  = xgb_model.predict_proba(X_all_scaled)[:, 1]

df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"  Predictions saved to: {OUTPUT_CSV}")

# ─────────────────────────────────────────────
# STEP 7 - PLOT FEATURE IMPORTANCES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 - Plotting feature importances...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
axes[0].barh(FEATURES, rf.feature_importances_, color="steelblue")
axes[0].set_title("Random Forest — Feature Importances")
axes[0].set_xlabel("Importance")

# XGBoost
axes[1].barh(FEATURES, xgb_importances, color="darkorange")
axes[1].set_title("XGBoost — Feature Importances")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.close()
print(f"  Plot saved to: feature_importances.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  Predictions: {OUTPUT_CSV}")
print(f"  Metrics:     {METRICS_CSV}")
print(f"  Plot:        feature_importances.png")
print("  Next step: Run visualize_results.py to map flood predictions")
print("=" * 60)