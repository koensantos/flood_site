"""
Step 4 — Visualization Script (Jersey City)
Flood Risk Modeling — Jersey City, NJ
Koen Mitchel Santos
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

PREDICTIONS_CSV = "jc_model_predictions.csv"
BOUNDARY_SHP    = "jerseycity_boundary/NJ_Municipalities_3857.shp"
FEMA_SHP        = "fema_flood_zones_dataset/S_FLD_HAZ_AR.shp"
GRID_SIZE_M     = 50

# ─────────────────────────────────────────────
# STEP 1 - LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading data...")
print("=" * 60)

df = pd.read_csv(PREDICTIONS_CSV)

muni = gpd.read_file(BOUNDARY_SHP)
for col in muni.columns:
    if muni[col].dtype == object:
        matches = muni[muni[col].str.contains("JERSEY CITY", case=False, na=False)]
        if len(matches) > 0:
            jerseycity = matches.to_crs("EPSG:32618")
            break

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

colors = ["#2166ac", "#74add1", "#ffffbf", "#f46d43", "#d73027"]
cmap = LinearSegmentedColormap.from_list("flood_risk", colors)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, prob_col, title in zip(
    axes,
    ["rf_probability", "xgb_probability"],
    ["Random Forest — Flood Probability", "XGBoost — Flood Probability"]
):
    grid.plot(column=prob_col, ax=ax, cmap=cmap, vmin=0, vmax=1,
              legend=True, legend_kwds={"label": "Flood Probability", "shrink": 0.7})
    jerseycity.boundary.plot(ax=ax, color="black", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

plt.suptitle("Flood Susceptibility Map — Jersey City, NJ\nHurricane Ida (September 2021)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("jc_map_flood_probability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: jc_map_flood_probability.png")

# ─────────────────────────────────────────────
# STEP 4 - MAP 2: BINARY PREDICTIONS VS FEMA
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Creating binary prediction vs FEMA map...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
flood_colors = {0: "#d1e5f0", 1: "#d73027"}

for val, color in flood_colors.items():
    grid[grid["rf_prediction"] == val].plot(ax=axes[0], color=color, linewidth=0)
jerseycity.boundary.plot(ax=axes[0], color="black", linewidth=1.5)
axes[0].set_title("Random Forest\nPrediction", fontsize=12, fontweight="bold")
axes[0].set_axis_off()

for val, color in flood_colors.items():
    grid[grid["xgb_prediction"] == val].plot(ax=axes[1], color=color, linewidth=0)
jerseycity.boundary.plot(ax=axes[1], color="black", linewidth=1.5)
axes[1].set_title("XGBoost\nPrediction", fontsize=12, fontweight="bold")
axes[1].set_axis_off()

jerseycity.plot(ax=axes[2], color="#d1e5f0", linewidth=0)
high_risk.plot(ax=axes[2], color="#d73027", linewidth=0)
jerseycity.boundary.plot(ax=axes[2], color="black", linewidth=1.5)
axes[2].set_title("FEMA Flood Zones\n(Baseline)", fontsize=12, fontweight="bold")
axes[2].set_axis_off()

flood_patch    = mpatches.Patch(color="#d73027", label="Flood Risk")
no_flood_patch = mpatches.Patch(color="#d1e5f0", label="No Flood Risk")
fig.legend(handles=[flood_patch, no_flood_patch],
           loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Model Predictions vs FEMA Flood Zones — Jersey City, NJ",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("jc_map_predictions_vs_fema.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: jc_map_predictions_vs_fema.png")

# ─────────────────────────────────────────────
# STEP 5 - MAP 3: ELEVATION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 - Creating elevation map...")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
grid.plot(column="elevation_m", ax=ax, cmap="terrain", legend=True,
          legend_kwds={"label": "Elevation (meters)", "shrink": 0.7})
jerseycity.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title("Digital Elevation Model — Jersey City, NJ\n(USGS 10m DEM)",
             fontsize=13, fontweight="bold")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("jc_map_elevation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: jc_map_elevation.png")

# ─────────────────────────────────────────────
# STEP 6 - MAP 4: MODEL AGREEMENT
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 - Creating model agreement map...")
print("=" * 60)

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
agreement_pct = (grid["rf_prediction"] == grid["xgb_prediction"]).mean() * 100

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

jerseycity.boundary.plot(ax=ax, color="black", linewidth=1.5)
ax.set_title(f"Model Agreement Map — Jersey City, NJ\nOverall Agreement: {agreement_pct:.1f}%",
             fontsize=13, fontweight="bold")
ax.set_axis_off()
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig("jc_map_model_agreement.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: jc_map_model_agreement.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("  Output files:")
print("    jc_map_flood_probability.png")
print("    jc_map_predictions_vs_fema.png")
print("    jc_map_elevation.png")
print("    jc_map_model_agreement.png")
print("=" * 60)