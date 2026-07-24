"""
Step 1 — LiDAR Processing Script (Jersey City)
Flood Risk Modeling — Jersey City, NJ
Koen Mitchel Santos

This script:
1. Loads the pre-made DEM GeoTIFF
2. Filters and clips it to Jersey City's boundary
3. Saves the final clipped DEM
"""

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEM_INPUT      = "lidar_tif/output_USGS10m.tif"
BOUNDARY_SHP   = "jerseycity_boundary/NJ_Municipalities_3857.shp"
DEM_OUTPUT     = "jerseycity_dem/jerseycity_dem.tif"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

os.makedirs("jerseycity_dem", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 - LOAD AND FILTER JERSEY CITY BOUNDARY
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading Jersey City boundary...")
print("=" * 60)

muni = gpd.read_file(BOUNDARY_SHP)
print(f"  Loaded {len(muni)} municipalities")
print(f"  Columns: {list(muni.columns)}")

# Find Jersey City — check available name columns
for col in muni.columns:
    if muni[col].dtype == object:
        matches = muni[muni[col].str.contains("JERSEY CITY", case=False, na=False)]
        if len(matches) > 0:
            print(f"  Found Jersey City in column: {col}")
            jerseycity = matches
            break

if len(jerseycity) == 0:
    print("  ERROR: Could not find Jersey City in shapefile.")
    print("  Available values:", muni.iloc[:, 1].unique()[:20])
    exit()

jerseycity = jerseycity.to_crs("EPSG:32618")
print(f"  Jersey City boundary loaded - CRS: {jerseycity.crs}")
print(f"  Bounds: {jerseycity.total_bounds}")

# ─────────────────────────────────────────────
# STEP 2 - LOAD AND INSPECT DEM
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 - Loading DEM...")
print("=" * 60)

try:
    with rasterio.open(DEM_INPUT) as src:
        print(f"  DEM loaded successfully")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Shape: {src.shape}")
        print(f"  Bounds: {src.bounds}")
except Exception as e:
    print(f"  ERROR loading DEM: {e}")
    exit()

# ─────────────────────────────────────────────
# STEP 3 - CLIP DEM TO JERSEY CITY BOUNDARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 - Clipping DEM to Jersey City boundary...")
print("=" * 60)

try:
    with rasterio.open(DEM_INPUT) as src:
        jerseycity_reproj = jerseycity.to_crs(src.crs)
        shapes = [mapping(geom) for geom in jerseycity_reproj.geometry]

        out_image, out_transform = mask(src, shapes, crop=True, nodata=-9999)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999,
            "compress": "lzw"
        })

        with rasterio.open(DEM_OUTPUT, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"  Clipped DEM saved to: {DEM_OUTPUT}")
    print(f"  Shape: {out_image.shape}")
    print(f"  Transform: {out_transform}")

except Exception as e:
    print(f"  ERROR clipping DEM: {e}")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("LIDAR PROCESSING COMPLETE")
print(f"  Final DEM: {DEM_OUTPUT}")
print("  Next step: Run jc_process_features.py")
print("=" * 60)