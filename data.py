"""
Dataset Validation Script
Flood Risk Modeling — Hoboken, NJ
Koen Mitchel Santos

Run this script to verify all 4 datasets are readable and properly formatted.
Edit the file paths in the CONFIG section below to match where your files are saved.
"""

import os
import sys

# ─────────────────────────────────────────────
# CONFIG — Update these paths to your files
# ─────────────────────────────────────────────

LIDAR_FOLDER        = "lidar_dataset"
HWM_CSV             = "flood_ground_truth_dataset/FilteredHWMs_NewJersey.csv"
RAINFALL_CSV        = "rainfall_dataset/lcd_newark_sept2021.csv"
FEMA_GDB = "fema_flood_zones_dataset/S_FLD_HAZ_AR.shp"
# ─────────────────────────────────────────────
# CHECK DEPENDENCIES
# ─────────────────────────────────────────────

print("=" * 60)
print("Checking required Python libraries...")
print("=" * 60)

missing = []
for lib in ["pandas", "geopandas", "rasterio", "fiona"]:
    try:
        __import__(lib)
        print(f"  ✅ {lib} is installed")
    except ImportError:
        print(f"  ❌ {lib} is NOT installed — run: pip install {lib}")
        missing.append(lib)

if missing:
    print(f"\n⚠️  Missing libraries: {missing}")
    print("Install them first with: pip install " + " ".join(missing))
    sys.exit(1)

import pandas as pd
import geopandas as gpd
import rasterio
import fiona

print("\nAll libraries found. Proceeding with dataset checks...\n")

# ─────────────────────────────────────────────
# 1. LIDAR TILES
# ─────────────────────────────────────────────

print("=" * 60)
print("1. LIDAR TILES")
print("=" * 60)

try:
    tif_files = [f for f in os.listdir(LIDAR_FOLDER) if f.endswith(".laz")]
    print(f"  📁 Folder found: {LIDAR_FOLDER}")
    print(f"  📄 Number of .tif files: {len(tif_files)}")

    if len(tif_files) == 0:
        print("  ❌ No .tif files found. Check your folder path.")
    else:
        # Open the first tile to check it's readable
        sample = os.path.join(LIDAR_FOLDER, tif_files[0])
        with rasterio.open(sample) as src:
            print(f"  ✅ Sample tile readable: {tif_files[0]}")
            print(f"     CRS: {src.crs}")
            print(f"     Resolution: {src.res}")
            print(f"     Bounds: {src.bounds}")
except Exception as e:
    print(f"  ❌ Error reading LiDAR folder: {e}")

# ─────────────────────────────────────────────
# 2. USGS HIGH-WATER MARKS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. USGS IDA HIGH-WATER MARKS")
print("=" * 60)

try:
    hwm = pd.read_csv(HWM_CSV)
    print(f"  ✅ CSV loaded successfully")
    print(f"  📄 Rows: {len(hwm)}")
    print(f"  📄 Columns: {list(hwm.columns[:6])} ...")  # Show first 6 columns

    # Check for key columns
    key_cols = ["latitude", "longitude", "elev_ft"]
    for col in key_cols:
        match = [c for c in hwm.columns if col.lower() in c.lower()]
        if match:
            print(f"  ✅ Found column '{match[0]}'")
        else:
            print(f"  ⚠️  Could not find expected column '{col}' — check data dictionary")

except Exception as e:
    print(f"  ❌ Error reading HWM CSV: {e}")

# ─────────────────────────────────────────────
# 3. LCD RAINFALL — NEWARK AIRPORT
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. LCD RAINFALL — NEWARK AIRPORT")
print("=" * 60)

try:
    rain = pd.read_csv(RAINFALL_CSV)
    print(f"  ✅ CSV loaded successfully")
    print(f"  📄 Rows: {len(rain)}")

    # Check for HourlyPrecipitation column
    if "HourlyPrecipitation" in rain.columns:
        print(f"  ✅ HourlyPrecipitation column found")
    else:
        print(f"  ❌ HourlyPrecipitation column not found — check file")

    # Check date range
    if "DATE" in rain.columns:
        rain["DATE"] = pd.to_datetime(rain["DATE"], errors="coerce")
        print(f"  📅 Date range: {rain['DATE'].min()} → {rain['DATE'].max()}")

        # Check Ida dates are present
        ida = rain[(rain["DATE"] >= "2021-09-01") & (rain["DATE"] <= "2021-09-02")]
        if len(ida) > 0:
            print(f"  ✅ Hurricane Ida dates (Sept 1–2 2021) found: {len(ida)} hourly records")
        else:
            print(f"  ❌ No records found for Sept 1–2 2021 — check your date range when downloading")

except Exception as e:
    print(f"  ❌ Error reading rainfall CSV: {e}")

# ─────────────────────────────────────────────
# 4. FEMA NFHL FLOOD ZONES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. FEMA NFHL FLOOD ZONES")
print("=" * 60)

try:
    layers = fiona.listlayers(FEMA_GDB)
    print(f"  ✅ FEMA geodatabase readable")
    print(f"  📄 Available layers: {layers}")

    # Load the flood zone layer (S_FLD_HAZ_AR is the standard FEMA flood zone polygon layer)
    target = "S_FLD_HAZ_AR"
    if target in layers:
        fema = gpd.read_file(FEMA_GDB)
        print(f"  ✅ Flood zone layer '{target}' loaded")
        print(f"  📄 Features: {len(fema)}")
        print(f"  📄 CRS: {fema.crs}")
        print(f"  📄 Flood zones present: {fema['FLD_ZONE'].unique() if 'FLD_ZONE' in fema.columns else 'column not found'}")
    else:
        print(f"  ⚠️  Layer '{target}' not found. Available layers listed above — look for flood zone polygons.")

except Exception as e:
    print(f"  ❌ Error reading FEMA geodatabase: {e}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("Fix any ❌ errors above before proceeding to processing.")
print("=" * 60)