import rasterio
from rasterio.transform import from_origin
import numpy as np
from pyproj import CRS
import os

print("🧪 Rasterio version:", rasterio.__version__)
print("🧪 GDAL version:", rasterio.__gdal_version__)
print("🧪 PROJ version:", rasterio.__proj_version__)
print("🧪 Rasterio default CRS test:", CRS.from_epsg(4326))

# Create dummy data
data = np.random.rand(100, 100).astype(np.float32)
transform = from_origin(0, 100, 1, 1)  # top-left x, top-left y, x pixel size, y pixel size

# Define output file
outfile = "test_raster.tif"
if os.path.exists(outfile):
    os.remove(outfile)

# Write GeoTIFF
with rasterio.open(
    outfile,
    "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(data, 1)

print("✅ Successfully wrote GeoTIFF")

# Read GeoTIFF
with rasterio.open(outfile) as src:
    read_data = src.read(1)
    print("✅ Successfully read GeoTIFF, shape:", read_data.shape)
    print("📌 CRS:", src.crs)

# Clean up
os.remove(outfile)
print("🧹 Cleaned up test file")
