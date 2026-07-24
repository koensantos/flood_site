"""
Diagnostic — Check HWM locations vs Hoboken grid
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

BOUNDARY_SHP = "hoboken_boundary/NJ_Municipalities_3857.shp"
HWM_CSV      = "flood_ground_truth_dataset/FilteredHWMs_NewJersey.csv"

# Load Hoboken boundary
hoboken = gpd.read_file(BOUNDARY_SHP).to_crs("EPSG:32618")
hoboken_union = unary_union(hoboken.geometry)
minx, miny, maxx, maxy = hoboken_union.bounds
print(f"Hoboken bounds (UTM meters):")
print(f"  X: {minx:.0f} to {maxx:.0f}")
print(f"  Y: {miny:.0f} to {maxy:.0f}")

# Load HWMs
hwm = pd.read_csv(HWM_CSV)
hwm = hwm.dropna(subset=["latitude_dd", "longitude_dd"])
hwm_gdf = gpd.GeoDataFrame(
    hwm,
    geometry=gpd.points_from_xy(hwm["longitude_dd"], hwm["latitude_dd"]),
    crs="EPSG:4326"
).to_crs("EPSG:32618")

print(f"\nHWM bounds (UTM meters):")
print(f"  X: {hwm_gdf.geometry.x.min():.0f} to {hwm_gdf.geometry.x.max():.0f}")
print(f"  Y: {hwm_gdf.geometry.y.min():.0f} to {hwm_gdf.geometry.y.max():.0f}")

# Find closest HWM to Hoboken center
hoboken_center = hoboken_union.centroid
distances = hwm_gdf.geometry.distance(hoboken_center)
hwm_gdf["dist_to_hoboken_center_m"] = distances
closest = hwm_gdf.nsmallest(10, "dist_to_hoboken_center_m")[
    ["latitude_dd", "longitude_dd", "dist_to_hoboken_center_m", "eventName"]
]

print(f"\n10 closest HWMs to Hoboken center:")
print(closest.to_string(index=False))
print(f"\nClosest HWM distance: {distances.min():.0f} meters")