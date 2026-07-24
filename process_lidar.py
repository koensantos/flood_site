"""
Step 1 — LiDAR Processing Script (Simplified)
Flood Risk Modeling — Hoboken, NJ
Koen Mitchel Santos

This script:
1. Loads the pre-made DEM GeoTIFF
2. Clips it to Hoboken's boundary
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
BOUNDARY_SHP   = "hoboken_boundary/NJ_Municipalities_3857.shp"
DEM_OUTPUT     = "lidar_hoboken/hoboken_dem.tif"

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

os.makedirs("lidar_hoboken", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 - LOAD HOBOKEN BOUNDARY
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 - Loading Hoboken boundary...")
print("=" * 60)

hoboken = gpd.read_file(BOUNDARY_SHP)
print(f"  Boundary loaded - CRS: {hoboken.crs}")
print(f"  Bounds: {hoboken.total_bounds}")

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
# STEP 3 - CLIP DEM TO HOBOKEN BOUNDARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 - Clipping DEM to Hoboken boundary...")
print("=" * 60)

try:
    with rasterio.open(DEM_INPUT) as src:
        # Reproject Hoboken boundary to match DEM CRS
        hoboken_reproj = hoboken.to_crs(src.crs)
        shapes = [mapping(geom) for geom in hoboken_reproj.geometry]

        # Clip
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
print("  Next step: Run process_features.py to build the model grid")
print("=" * 60)