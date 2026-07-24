"""
Step 5 — Advanced Analysis Script
Flood Risk Modeling — Hoboken & Jersey City, NJ
Koen Mitchel Santos

This script generates:
1. ROC curves for both models and both cities
2. Confusion matrix heatmaps
3. Precision-Recall curves
4. Cross-city performance comparison bar chart
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    accuracy_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURES     = ["elevation_m", "rainfall_inches"]
TARGET       = "label"

# ─────────────────────────────────────────────
# HELPER: TRAIN AND RETURN MODELS + PREDICTIONS
# ─────────────────────────────────────────────

def train_models(csv_path):
    df = pd.read_csv(csv_path)
    df[TARGET] = df["in_fema_flood_zone"]
    df_clean = df.dropna(subset=FEATURES + [TARGET])

    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    # XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg/pos if pos > 0 else 1,
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
    )
    xgb_model.fit(X_train_s, y_train)

    return {
        "rf": rf, "xgb": xgb_model,
        "scaler": scaler,
        "X_test": X_test_s, "y_test": y_test,
        "X_train": X_train_s, "y_train": y_train,
        "n_cells": len(df_clean)
    }

# ─────────────────────────────────────────────
# LOAD BOTH CITIES
# ─────────────────────────────────────────────

print("Training models for Hoboken...")
hoboken = train_models("feature_grid.csv")
print("Training models for Jersey City...")
jc      = train_models("jc_feature_grid.csv")

cities = {
    "Hoboken":     hoboken,
    "Jersey City": jc
}

# ─────────────────────────────────────────────
# FIGURE 1: ROC CURVES — BOTH CITIES SIDE BY SIDE
# ─────────────────────────────────────────────

print("\nGenerating ROC curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {"Random Forest": "#2166ac", "XGBoost": "#d73027"}

for ax, (city_name, data) in zip(axes, cities.items()):
    for model_name, model in [("Random Forest", data["rf"]), ("XGBoost", data["xgb"])]:
        y_prob = model.predict_proba(data["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(data["y_test"], y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[model_name], linewidth=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {city_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

plt.suptitle("ROC Curves: Random Forest vs XGBoost\nHoboken and Jersey City, NJ",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_curves.png")

# ─────────────────────────────────────────────
# FIGURE 2: CONFUSION MATRICES — 2x2 GRID
# ─────────────────────────────────────────────

print("Generating confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

model_pairs = [
    ("Hoboken", "Random Forest", hoboken["rf"]),
    ("Hoboken", "XGBoost",       hoboken["xgb"]),
    ("Jersey City", "Random Forest", jc["rf"]),
    ("Jersey City", "XGBoost",       jc["xgb"]),
]

for ax, (city, model_name, model) in zip(axes.flatten(), model_pairs):
    data = hoboken if city == "Hoboken" else jc
    y_pred = model.predict(data["X_test"])
    cm = confusion_matrix(data["y_test"], y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                cbar=True, linewidths=0.5)

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.35, str(cm[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
            ax.text(j + 0.5, i + 0.65, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    acc = accuracy_score(data["y_test"], y_pred)
    ax.set_title(f"{city} — {model_name}\nAccuracy: {acc:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xticklabels(["Not Flooded", "Flooded"], fontsize=9)
    ax.set_yticklabels(["Not Flooded", "Flooded"], fontsize=9, rotation=0)

plt.suptitle("Confusion Matrices: All Models and Cities",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: confusion_matrices.png")

# ─────────────────────────────────────────────
# FIGURE 3: PRECISION-RECALL CURVES
# ─────────────────────────────────────────────

print("Generating precision-recall curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (city_name, data) in zip(axes, cities.items()):
    for model_name, model in [("Random Forest", data["rf"]), ("XGBoost", data["xgb"])]:
        y_prob = model.predict_proba(data["X_test"])[:, 1]
        precision, recall, _ = precision_recall_curve(data["y_test"], y_prob)
        ap = average_precision_score(data["y_test"], y_prob)
        ax.plot(recall, precision, color=colors[model_name], linewidth=2,
                label=f"{model_name} (AP = {ap:.3f})")

    baseline = data["y_test"].mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (AP = {baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {city_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

plt.suptitle("Precision-Recall Curves: Random Forest vs XGBoost\nHoboken and Jersey City, NJ",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: precision_recall_curves.png")

# ─────────────────────────────────────────────
# FIGURE 4: CROSS-CITY PERFORMANCE BAR CHART
# ─────────────────────────────────────────────

print("Generating cross-city comparison chart...")

metrics_data = []
for city_name, data in cities.items():
    for model_name, model in [("Random Forest", data["rf"]), ("XGBoost", data["xgb"])]:
        y_pred = model.predict(data["X_test"])
        y_prob = model.predict_proba(data["X_test"])[:, 1]
        cv_auc = cross_val_score(model, data["X_train"], data["y_train"],
                                 cv=5, scoring="roc_auc").mean()
        metrics_data.append({
            "City": city_name,
            "Model": model_name,
            "Accuracy": accuracy_score(data["y_test"], y_pred),
            "ROC-AUC": roc_auc_score(data["y_test"], y_prob),
            "CV-AUC": cv_auc
        })

metrics_df = pd.DataFrame(metrics_data)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metric_names = ["Accuracy", "ROC-AUC", "CV-AUC"]
bar_colors   = ["#2166ac", "#d73027", "#4dac26", "#f1a340"]
labels       = [f"{r['City']}\n{r['Model']}" for _, r in metrics_df.iterrows()]

for ax, metric in zip(axes, metric_names):
    bars = ax.bar(range(len(metrics_df)), metrics_df[metric],
                  color=bar_colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(len(metrics_df)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.suptitle("Cross-City Model Performance Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("cross_city_comparison_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cross_city_comparison_chart.png")

# ─────────────────────────────────────────────
# FIGURE 5: ELEVATION DISTRIBUTION BY FLOOD ZONE
# ─────────────────────────────────────────────

print("Generating elevation distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (city_name, csv_path) in zip(axes, [
    ("Hoboken", "feature_grid.csv"),
    ("Jersey City", "jc_feature_grid.csv")
]):
    df = pd.read_csv(csv_path).dropna(subset=["elevation_m"])
    flooded     = df[df["in_fema_flood_zone"] == 1]["elevation_m"]
    not_flooded = df[df["in_fema_flood_zone"] == 0]["elevation_m"]

    ax.hist(not_flooded, bins=30, alpha=0.6, color="#2166ac",
            label=f"Not Flooded (n={len(not_flooded)})", density=True)
    ax.hist(flooded, bins=30, alpha=0.6, color="#d73027",
            label=f"Flooded (n={len(flooded)})", density=True)

    ax.axvline(flooded.mean(), color="#d73027", linestyle="--", linewidth=1.5,
               label=f"Flooded mean: {flooded.mean():.2f}m")
    ax.axvline(not_flooded.mean(), color="#2166ac", linestyle="--", linewidth=1.5,
               label=f"Not flooded mean: {not_flooded.mean():.2f}m")

    ax.set_xlabel("Elevation (meters, NAVD88)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Elevation Distribution by Flood Zone\n{city_name}", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Elevation Distributions: Flooded vs Non-Flooded Grid Cells",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("elevation_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: elevation_distribution.png")

# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ADVANCED ANALYSIS COMPLETE")
print("=" * 60)
print(metrics_df.to_string(index=False))
print("\nOutput files:")
print("  roc_curves.png")
print("  confusion_matrices.png")
print("  precision_recall_curves.png")
print("  cross_city_comparison_chart.png")
print("  elevation_distribution.png")
print("\nUpload all .png files to Overleaf and add to paper.")
print("=" * 60)