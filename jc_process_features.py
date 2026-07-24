"""
Step 2 — Feature Grid & Extraction Script (Jersey City)
Flood Risk Modeling — Jersey City, NJ
Koen Mitchel Santos
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import box, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEM_PATH        = "jerseycity_dem/jerseycity_dem.tif"
BOUNDARY_SHP    = "jerseycity_boundary/NJ_Municipalities_3857.shp"
RAINFALL_CSV    = "rainfall_dataset/lcd_newark_sept2021.csv"
FEMA_SHP        = "fema_flood_zones_dataset/S_FLD_HAZ_AR.shp"
OUTPUT_CSV      = "jc_feature_grid.csv"
GRID_SIZE_M     = 50

# ─────────────────────────────────────────────
# STEP 1 - LOAD ALL DATASETS
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading datasets...")
print("=" * 60)

# Load Jersey City boundary
muni = gpd.read_file(BOUNDARY_SHP)
for col in muni.columns:
    if muni[col].dtype == object:
        matches = muni[muni[col].str.contains("JERSEY CITY", case=False, na=False)]
        if len(matches) > 0:
            jerseycity = matches.to_crs("EPSG:32618")
            break

jerseycity_union = unary_union(jerseycity.geometry)
print(f"  Jersey City boundary loaded - CRS: {jerseycity.crs}")

# Load DEM
dem_src = rasterio.open(DEM_PATH)
dem_crs = dem_src.crs
print(f"  DEM loaded - CRS: {dem_crs}, Shape: {dem_src.shape}")

# Load Rainfall
rain = pd.read_csv(RAINFALL_CSV, low_memory=False)
rain["DATE"] = pd.to_datetime(rain["DATE"], errors="coerce")
ida_rain = rain[(rain["DATE"] >= "2021-09-01") & (rain["DATE"] <= "2021-09-02")].copy()
ida_rain["HourlyPrecipitation"] = pd.to_numeric(
    ida_rain["HourlyPrecipitation"].astype(str).str.replace("s", "").str.replace("T", "0"),
    errors="coerce"
)
total_rainfall_inches = ida_rain["HourlyPrecipitation"].sum(skipna=True)
print(f"  Rainfall loaded - Total Ida precipitation: {total_rainfall_inches:.2f} inches")

# Load FEMA Flood Zones
fema = gpd.read_file(FEMA_SHP).to_crs("EPSG:32618")
high_risk = fema[fema["FLD_ZONE"].isin(["A", "AE", "AH", "VE"])]
fema_union = unary_union(high_risk.geometry)
print(f"  FEMA flood zones loaded - {len(high_risk)} high-risk features")

# ─────────────────────────────────────────────
# STEP 2 - CREATE 50M GRID OVER JERSEY CITY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 - Creating 50m grid over Jersey City...")
print("=" * 60)

minx, miny, maxx, maxy = jerseycity_union.bounds

grid_cells = []
x = minx
while x < maxx:
    y = miny
    while y < maxy:
        cell = box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)
        if cell.intersects(jerseycity_union):
            grid_cells.append(cell)
        y += GRID_SIZE_M
    x += GRID_SIZE_M

grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs="EPSG:32618")
grid["cell_id"] = range(len(grid))
grid["centroid_x"] = grid.geometry.centroid.x
grid["centroid_y"] = grid.geometry.centroid.y
print(f"  Created {len(grid)} grid cells")

# ─────────────────────────────────────────────
# STEP 3 - EXTRACT ELEVATION PER CELL
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 - Extracting elevation per grid cell...")
print("=" * 60)

centroids_dem_crs = grid.copy().to_crs(dem_crs)
centroids_dem_crs["centroid_lon"] = centroids_dem_crs.geometry.centroid.x
centroids_dem_crs["centroid_lat"] = centroids_dem_crs.geometry.centroid.y

dem_band = dem_src.read(1)
elevations = []

for _, row in centroids_dem_crs.iterrows():
    try:
        r, c = rowcol(dem_src.transform, row["centroid_lon"], row["centroid_lat"])
        if 0 <= r < dem_band.shape[0] and 0 <= c < dem_band.shape[1]:
            val = dem_band[r, c]
            if val == dem_src.nodata or val == -9999:
                elevations.append(np.nan)
            else:
                elevations.append(float(val))
        else:
            elevations.append(np.nan)
    except Exception:
        elevations.append(np.nan)

dem_src.close()
grid["elevation_m"] = elevations

valid_elev = sum(1 for e in elevations if not np.isnan(e))
print(f"  Elevation extracted for {valid_elev}/{len(grid)} cells")
print(f"  Elevation range: {np.nanmin(elevations):.2f}m to {np.nanmax(elevations):.2f}m")

# ─────────────────────────────────────────────
# STEP 4 - ASSIGN RAINFALL TO EACH CELL
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 - Assigning rainfall to grid cells...")
print("=" * 60)

grid["rainfall_inches"] = total_rainfall_inches
print(f"  Assigned {total_rainfall_inches:.2f} inches to all {len(grid)} cells")

# ─────────────────────────────────────────────
# STEP 5 - CALCULATE DISTANCE TO FEMA FLOOD ZONES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 - Calculating distance to FEMA flood zones...")
print("=" * 60)

distances = []
for _, row in grid.iterrows():
    centroid = Point(row["centroid_x"], row["centroid_y"])
    dist = centroid.distance(fema_union)
    distances.append(dist)

grid["dist_to_flood_zone_m"] = distances
grid["in_fema_flood_zone"] = grid.apply(
    lambda row: 1 if Point(row["centroid_x"], row["centroid_y"]).within(fema_union) else 0,
    axis=1
)

print(f"  Distance calculated for all {len(grid)} cells")
print(f"  Cells inside FEMA flood zone: {grid['in_fema_flood_zone'].sum()}")

# ─────────────────────────────────────────────
# STEP 6 - LABEL CELLS USING FEMA FLOOD ZONES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 - Labeling flooded cells using FEMA flood zones...")
print("=" * 60)

grid["flooded"] = grid["in_fema_flood_zone"]
flooded_count = int(grid["flooded"].sum())
print(f"  Flooded cells:     {flooded_count}")
print(f"  Non-flooded cells: {len(grid) - flooded_count}")

# ─────────────────────────────────────────────
# STEP 7 - SAVE FEATURE MATRIX
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 - Saving feature matrix...")
print("=" * 60)

feature_cols = [
    "cell_id", "centroid_x", "centroid_y",
    "elevation_m", "rainfall_inches",
    "dist_to_flood_zone_m", "in_fema_flood_zone", "flooded"
]

output_df = grid[feature_cols].copy()
output_df = output_df.dropna(subset=["elevation_m"])
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"  Feature matrix saved to: {OUTPUT_CSV}")
print(f"  Total cells: {len(output_df)}")
print(f"\n  Class distribution:")
print(f"    Flooded (1):     {output_df['flooded'].sum()}")
print(f"    Not flooded (0): {(output_df['flooded'] == 0).sum()}")

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print(f"  Output: {OUTPUT_CSV}")
print("  Next step: Run jc_train_model.py")
print("=" * 60)