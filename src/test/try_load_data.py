# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:49:51 2022

@author: RekanS
"""


import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# import rasterio
# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')

# out_grid_file="ES002L2_BARCELONA_UA2018_v013.npy"
out_grid_file="S002L2_BARCELONA_UA2012_revised_v021_crop.npy"
plot_switch=False
write_switch=True

print("#### Loading file")

#2018 urban atlas area classifcation functional urban areas
filename = 'ES002L2_BARCELONA_UA2018_v013.gpkg'
#2012 urban atlas area classifcation functional urban areas
filename = './ES002L2_BARCELONA_UA2012_revised_v021/Data/ES002L2_BARCELONA_UA2012_revised_v021.gpkg'

gdf = gpd.read_file(filename)

print("#### Loading file done\n")



print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)
# Create a custom polygon
polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=gdf.crs)

# code12220_clipped = code12220.clip(polygon)
# code12220_clean=code12220_clipped[["code_2018", "geometry"]]
# # .loc["code_2018"]=pd.to_numeric(code12220_clean["code_2018"])

# code12220_clean = code12220_clean.astype({"code_2018": int})
# code12220_clean = code12220_clean.drop(code12220_clean.index[7])

gdf_clipped = gdf.clip(polygon)
gdf_clean=gdf_clipped[["code_2018", "geometry"]]

gdf_clean = gdf_clean.astype({"code_2018": int})
# gdf_clean = code12220_clean.drop(code12220_clean.index[7])


print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    gdf.plot( ax=ax1, legend=True)
    poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    gdf_clean.plot( ax=ax2, legend=True)
    plt.show()

    print("#### Plotting file done \n")


# lose a bit of resolution, but this is a fairly large file, and this is only an example.
# shape = 1000, 1000

# transform = rasterio.transform.from_bounds(*code12220['geometry'].total_bounds, *shape)
# rasterize_rivernet = rasterize(
#     [(shape, 1) for shape in code12220['geometry']],
#     out_shape=shape,
#     transform=transform,
#     fill=0,
#     all_touched=True,
#     dtype=rasterio.uint8)

print("#### Gridding vector file")

out_grid = make_geocube(
    vector_data=gdf_clean,
    resolution=(-10, 10),
)

print("#### Gridding vector file done \n")

if plot_switch:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    out_grid.code_2018.plot(ax=ax1)
    gdf_clean.plot( ax=ax2, legend=True)
    ax1.set_aspect('equal', 'box')
    plt.show()
 
if write_switch:
    np.save(out_grid_file,np.array(out_grid.code_2018))
