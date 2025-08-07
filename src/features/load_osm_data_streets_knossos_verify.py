#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:55:04 2022

@author: sjet
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# import requests
import sys
import matplotlib.pyplot as plt
import numpy as np 
import osmnx as ox
from rasterio import warp
# import pandas as pd
# from statistics import mean
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
from shapely.geometry import box
from pyproj import Transformer
import geopandas as gpd
import json
import rasterio

plt.close('all')
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "TRUE", "True")

if len(sys.argv) >2:
    print("Total number of arguments passed:", len(sys.argv))
    print("\nArguments passed:", end = " ")
    for i in range(0,len(sys.argv) ):
        print(sys.argv[i],"\t", end = " ")
    plot_switch=str2bool(sys.argv[3])
    write_switch=str2bool(sys.argv[4])
else:
    plot_switch=True
    write_switch=True

# plot_switch=False
# write_switch=False
clip_switch=True
interp_switch=True
convcrs_switch=True

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city_string_in="Oslo"
city_string_out="OSL" 

# city_string_in=sys.argv[1]
# city_string_out=sys.argv[2]
 
print("\n######################## \n")
print("Downloading OSM data for road information feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder_pic ="Z:/NoiseML/2024/city_data_pics/"

in_file_target='_MRoadsLden.tif'
#output file
out_grid_file="_raw_osm_roads"


img_target = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r')

print("#### Loading file")

#define corner points for crop window
# corner_point1=np.round(np.array((3.6560 , 2.0600 ))*1e6)-1000
# corner_point2=np.round(np.array((3.6690 , 2.0740 ))*1e6)+1000
img_target_bounds=img_target.bounds
corner_point1=np.array((img_target_bounds[0] , img_target_bounds[1] ))-1000
corner_point2=np.array((img_target_bounds[2] , img_target_bounds[3] ))+1000


#transform corner points from source (3035) to destination (4326) reference system
# coords_transformed2 = warp.transform({'init': 'epsg:3035'},{'init': 'epsg:4326'},[corner_point1[0], corner_point2[0]], [corner_point1[1], corner_point2[1]])
transformer = Transformer.from_crs(3035, 4326, always_xy=True)
coords_transformed = transformer.transform([corner_point1[0], corner_point2[0]], [corner_point1[1], corner_point2[1]])


# Original bounds (in EPSG:3035, for example)
bounds_3035 = img_target.bounds  # (minx, miny, maxx, maxy)
box_3035 = box(*bounds_3035)
gdf_3035 = gpd.GeoDataFrame(geometry=[box_3035], crs="EPSG:3035")

# Transform bounds corner-by-corner
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
xmin, ymin = transformer.transform(bounds_3035[0], bounds_3035[1])
xmax, ymax = transformer.transform(bounds_3035[2], bounds_3035[3])

box_4326 = box(xmin, ymin, xmax, ymax)
gdf_4326 = gpd.GeoDataFrame(geometry=[box_4326], crs="EPSG:4326")


fig, ax = plt.subplots(figsize=(10, 6))

# Reproject to same CRS for plotting
gdf_3035_wgs = gdf_3035.to_crs("EPSG:4326")
gdf_3035_wgs.plot(ax=ax, facecolor='none', edgecolor='red', label='Original in 3035 (transformed)')
gdf_4326.plot(ax=ax, facecolor='none', edgecolor='blue', linestyle='--', label='Transformed Bounding Box')

ax.legend()
ax.set_title("Bounding Boxes Comparison (EPSG:4326)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()