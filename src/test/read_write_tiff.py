# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:50:46 2025

@author: jetschny
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin

data = np.ones((100, 100), dtype=np.float32)

transform = from_origin(0, 100, 1, 1)  # top-left x, top-left y, x pixel size, y pixel size
with rasterio.open(
    "test_output.tif", "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform
) as dst:
    dst.write(data, 1)

print("GeoTIFF written successfully.")
