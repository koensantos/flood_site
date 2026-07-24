"""
Step 4 — Visualization Script
Flood Risk Modeling — Hoboken, NJ
Koen Mitchel Santos

This script:
1. Loads model predictions
2. Creates a flood susceptibility map of Hoboken
3. Compares RF vs XGBoost predictions side by side
4. Overlays FEMA flood zones for comparison
5. Saves all maps as PNG files
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PREDICTIONS_CSV = "output_csvs/model_predictions.csv"
BOUNDARY_SHP    = "hoboken_boundary/NJ_Municipalities_3857.shp"
FEMA_SHP        = "fema_flood_zones_dataset/S_FLD_HAZ_AR.shp"
GRID_SIZE_M     = 50

# ─────────────────────────────────────────────
# STEP 1 - LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading data...")
print("=" * 60)

df = pd.read_csv(PREDICTIONS_CSV)
hoboken = gpd.read_file(BOUNDARY_SHP).to_crs("EPSG:32618")
fema = gpd.read_file(FEMA_SHP).to_crs("EPSG:32618")
high_risk = fema[fema["FLD_ZONE"].isin(["A", "AE", "AH", "VE"])]

print(f"  Predictions loaded: {len(df)} grid cells")
print(f"  RF flooded predictions:  {df['rf_prediction'].sum()}")
print(f"  XGB flooded predictions: {df['xgb_prediction'].sum()}")

# ─────────────────────────────────────────────
# STEP 2 - REBUILD GRID GEOMETRIES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 - Rebuilding grid geometries...")
print("=" * 60)

grid_cells = []
for _, row in df.iterrows():
    cx, cy = row["centroid_x"], row["centroid_y"]
    cell = box(cx - GRID_SIZE_M/2, cy - GRID_SIZE_M/2,
               cx + GRID_SIZE_M/2, cy + GRID_SIZE_M/2)
    grid_cells.append(cell)

grid = gpd.GeoDataFrame(df.copy(), geometry=grid_cells, crs="EPSG:32618")
print(f"  Grid rebuilt with {len(grid)} cells")

# ─────────────────────────────────────────────
# STEP 3 - MAP 1: FLOOD PROBABILITY MAPS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 - Creating flood probability maps...")
print("=" * 60)

# Custom colormap: blue (low risk) to red (high risk)
colors = ["#2166ac", "#74add1", "#ffffbf", "#f46d43", "#d73027"]
cmap = LinearSegmentedColormap.from_list("flood_risk", colors)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, prob_col, title in zip(
    axes,
    ["rf_probability", "xgb_probability"],
    ["Random Forest — Flood Probability", "XGBoost — Flood Probability"]
):
    grid.plot(
        column=prob_col,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        legend=True,
        legend_kwds={"label": "Flood Probability", "shrink": 0.7}
    )
    hoboken.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

plt.suptitle("Flood Susceptibility Map — Hoboken, NJ\nHurricane Ida (September 2021)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("map_flood_probability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: map_flood_probability.png")

# ─────────────────────────────────────────────
# STEP 4 - MAP 2: FLOOD AMOUNT / INTENSITY MAP
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Creating flood amount intensity map...")
print("=" * 60)

grid["avg_probability"] = grid[["rf_probability", "xgb_probability"]].mean(axis=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors = ["#2166ac", "#74add1", "#ffffbf", "#f46d43", "#d73027"]
cmap = LinearSegmentedColormap.from_list("flood_risk", colors)

grid.plot(
    column="avg_probability",
    ax=ax,
    cmap=cmap,
    vmin=0,
    vmax=1,
    legend=True,
    legend_kwds={"label": "Flood Amount / Probability", "shrink": 0.7}
)
hoboken.boundary.plot(ax=ax, color="black", linewidth=1.5)
high_risk.boundary.plot(ax=ax, color="#b30000", linewidth=1.0)
ax.set_title("Flood Amount Map — Hoboken, NJ\nAverage RF + XGB Flood Probability",
             fontsize=13, fontweight="bold")
ax.set_axis_off()

plt.tight_layout()
plt.savefig("map_flood_amount.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: map_flood_amount.png")

# ─────────────────────────────────────────────
# STEP 5 - MAP 3: BINARY PREDICTIONS VS FEMA
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Creating binary prediction vs FEMA comparison map...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Color maps for binary predictions
flood_colors = {0: "#d1e5f0", 1: "#d73027"}

# Plot 1 - Random Forest
for val, color in flood_colors.items():
    subset = grid[grid["rf_prediction"] == val]
    subset.plot(ax=axes[0], color=color, linewidth=0)
hoboken.boundary.plot(ax=axes[0], color="black", linewidth=1.5)
axes[0].set_title("Random Forest\nPrediction", fontsize=12, fontweight="bold")
axes[0].set_axis_off()

# Plot 2 - XGBoost
for val, color in flood_colors.items():
    subset = grid[grid["xgb_prediction"] == val]
    subset.plot(ax=axes[1], color=color, linewidth=0)
hoboken.boundary.plot(ax=axes[1], color="black", linewidth=1.5)
axes[1].set_title("XGBoost\nPrediction", fontsize=12, fontweight="bold")
axes[1].set_axis_off()

# Plot 3 - FEMA Flood Zones (baseline)
hoboken.plot(ax=axes[2], color="#d1e5f0", linewidth=0)
high_risk.plot(ax=axes[2], color="#d73027", linewidth=0)
hoboken.boundary.plot(ax=axes[2], color="black", linewidth=1.5)
axes[2].set_title("FEMA Flood Zones\n(Baseline)", fontsize=12, fontweight="bold")
axes[2].set_axis_off()

# Shared legend
flood_patch   = mpatches.Patch(color="#d73027", label="Flood Risk")
no_flood_patch = mpatches.Patch(color="#d1e5f0", label="No Flood Risk")
fig.legend(handles=[flood_patch, no_flood_patch],
           loc="lower center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Model Predictions vs FEMA Flood Zones — Hoboken, NJ",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("map_predictions_vs_fema.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: map_predictions_vs_fema.png")

# ─────────────────────────────────────────────
# STEP 6 - MAP 4: ELEVATION MAP
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 - Creating elevation map...")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

grid.plot(
    column="elevation_m",
    ax=ax,
    cmap="terrain",
    legend=True,
    legend_kwds={"label": "Elevation (meters)", "shrink": 0.7}
)
hoboken.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title("Digital Elevation Model — Hoboken, NJ\n(USGS 10m DEM)",
             fontsize=13, fontweight="bold")
ax.set_axis_off()

plt.tight_layout()
plt.savefig("map_elevation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: map_elevation.png")

# ─────────────────────────────────────────────
# STEP 7 - AGREEMENT MAP (where RF and XGB agree)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 - Creating model agreement map...")
print("=" * 60)

grid["agreement"] = (grid["rf_prediction"] == grid["xgb_prediction"]).astype(int)
agreement_pct = grid["agreement"].mean() * 100
print(f"  Model agreement: {agreement_pct:.1f}%")

# Agreement categories
def get_agreement_label(row):
    if row["rf_prediction"] == 1 and row["xgb_prediction"] == 1:
        return "Both predict flood"
    elif row["rf_prediction"] == 0 and row["xgb_prediction"] == 0:
        return "Both predict no flood"
    elif row["rf_prediction"] == 1 and row["xgb_prediction"] == 0:
        return "RF only predicts flood"
    else:
        return "XGB only predicts flood"

grid["agreement_label"] = grid.apply(get_agreement_label, axis=1)

agreement_colors = {
    "Both predict flood":       "#d73027",
    "Both predict no flood":    "#d1e5f0",
    "RF only predicts flood":   "#fc8d59",
    "XGB only predicts flood":  "#fee090"
}

fig, ax = plt.subplots(1, 1, figsize=(9, 8))

for label, color in agreement_colors.items():
    subset = grid[grid["agreement_label"] == label]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, linewidth=0, label=label)

hoboken.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title(f"Model Agreement Map — Hoboken, NJ\nOverall Agreement: {agreement_pct:.1f}%",
             fontsize=13, fontweight="bold")
ax.set_axis_off()
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("map_model_agreement.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: map_model_agreement.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("  Output files:")
print("    map_flood_probability.png   — RF vs XGB probability maps")
print("    map_predictions_vs_fema.png — Binary predictions vs FEMA")
print("    map_elevation.png           — DEM elevation map")
print("    map_model_agreement.png     — Where models agree/disagree")
print("=" * 60)