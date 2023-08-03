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
import matplotlib.pyplot as plt
import numpy as np 
import osmnx as ox
from rasterio import warp
import pandas as pd
# from statistics import mean
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
from shapely.geometry import box
import geopandas as gpd
import json

plt.close('all')

write_switch=True
plot_switch=True
convcrs_switch=True

base_out_folder="/home/sjet/data/323_end_noise/HAN_data/"

#output file
out_grid_file="OSM_roads_han"


print("#### Loading file")

#define corner points for crop window
# corner_point1=np.array((4.295 , 3.244 ))*1e6-1000
# corner_point2=np.array((4.314 , 3.259 ))*1e6+1000

corner_point1=np.array((4.295 , 3.244 ))*1e6
corner_point2=np.array((4.314 , 3.259 ))*1e6


#transform corner points from source (3035) to destination (4326) reference system
coords_transformed = warp.transform({'init': 'epsg:3035'},{'init': 'epsg:4326'},[corner_point1[0], corner_point2[0]], [corner_point1[1], corner_point2[1]])

#download OSM street data based on transformed corner points
G = ox.graph_from_bbox(coords_transformed[1][0],coords_transformed[1][1], coords_transformed[0][0], coords_transformed[0][1], network_type='drive')
G_projected = ox.project_graph(G)

print("#### Loading file done\n")

#display OSM data
# ox.plot_graph(G_projected)

# Retrieve nodes and edges
nodes, edges = ox.graph_to_gdfs(G_projected)

print("#### Converting CRS ")
if convcrs_switch:
    edges_conv = edges.to_crs("EPSG:3035")

print("#### Converting CRS done\n")


polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                   (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=edges_conv.crs)

edges_clipped = edges_conv.clip(polygon)
# gdf_clipped=gdf_clipped[["Rang", "geometry"]]


# creating max_speed data layer

#select only 2 columns from dataframe
df_maxspeed=edges_clipped[["maxspeed", "geometry"]]
#remove index columns
df_maxspeed.reset_index(drop=True, inplace=True)
#flagg strings and nested arrays as nan
df_maxspeed.maxspeed=pd.to_numeric(df_maxspeed.maxspeed, "coerce")
# df_maxspeed=df_maxspeed[df_maxspeed['maxspeed'] != "none"]
# df_maxspeed=df_maxspeed[df_maxspeed['maxspeed'] != "signals"]
# df_maxspeed=df_maxspeed[df_maxspeed['maxspeed'] != "walk"]

#remove nan values
df_maxspeed=df_maxspeed.dropna()
# remove nested array and replace with mean of array
# df_maxspeed["maxspeed"] = df_maxspeed["maxspeed"].map(lambda x: np.mean(np.int_(x))) 
#format column as int16 type
df_maxspeed = df_maxspeed.astype({"maxspeed": np.int16})


# Make a GeoJSON string of the bounding box feature
bbox = gpd.GeoSeries(box(*polygon.bounds))
geom = bbox.__geo_interface__["features"][0]["geometry"]
# Add CRS
geom["crs"] = {"properties": {"name": f"EPSG:3035"}}

out_grid_maxspeed = make_geocube(vector_data=df_maxspeed,resolution=(-10, 10),geom=json.dumps(geom))

if write_switch:
    print("#### Saving to npy file")
    # out_grid_file=out_grid_file+"_maxspeed_clip.npy"
    # a=np.array(out_grid_maxspeed.maxspeed, dtype=np.uint16)
    np.save(base_out_folder+out_grid_file+"_maxspeed_clip.npy",np.array(out_grid_maxspeed.maxspeed, dtype=np.uint16))
    print("#### Saving to npy file done")

if plot_switch:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    edges_clipped.plot( ax=ax1, legend=True)
    out_grid_maxspeed.maxspeed.plot(ax=ax2)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    plt.show()
    plt.savefig(base_out_folder+out_grid_file+"_maxspeed_clip_grid.png")
    
    
    
    
# creating StreetClass layer
#select only 2 columns from dataframe
df_hway=edges_clipped[["highway", "geometry"]]
#remove index columns
df_hway.reset_index(drop=True, inplace=True)

#remove nan values
df_hway=df_hway.dropna()

# expldoe nested list of strings to multiple rows
df_hway=df_hway.explode("highway")
# remove duplicated indices, only pick the first 
df_hway = df_hway[~df_hway.index.duplicated(keep='first')]
#remove rows with highway==unclassified
df_hway = df_hway.drop(df_hway[df_hway.highway == "unclassified"].index)
df_hway = df_hway.drop(df_hway[df_hway.highway == "rest_area"].index)
df_hway = df_hway.drop(df_hway[df_hway.highway == "road"].index)

#get list of unique highway classes
# road_classes_old=sorted(df_hway["highway"].unique())


# road_classes_new=np.zeros(len(road_classes_old))
# df_hway_explod=edges.explode("highway")
# road_classes_new=np.array(range(0, len(road_classes_old), 1))
# road_classes_new=np.array([7,1,1,3,3,6,4,4,5,5,2,2])

# road_classes = np.empty((len(road_classes_old), 0))
# road_classes=np.append(road_classes, np.array([road_classes_old]).transpose(), axis=1)
# road_classes=np.append(road_classes, np.array([road_classes_new]).transpose(), axis=1)

df_road_classes=pd.DataFrame({'Name':['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 
                           'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'residential', 'living_street'],
                   'Class':[1,1,2,2,3,3,4,4,5,5,6,7]})

# counter=0
# for a in road_classes_old:
#     road_classes_new[counter]=round(np.mean(np.array(df_hway_explod[df_hway_explod["highway"]==a]["lanes"].dropna().explode("lanes"), dtype=np.uint8)))
#     counter=counter+1
    
    
# counter=0
for a in df_road_classes.Name.tolist():
    df_hway.loc[df_hway["highway"]==a,"highway_class"]=int(df_road_classes[df_road_classes.Name==a].Class)
    # counter=counter+1
  
df_hway=df_hway[["highway_class","geometry"]]
  
out_grid_hway = make_geocube(vector_data=df_hway,resolution=(-10, 10),geom=json.dumps(geom))

if write_switch:
    print("#### Saving to npy file")
    # out_grid_file=out_grid_file+"_maxspeed_clip.npy"
    np.save(base_out_folder+out_grid_file+"_streetclass_clip.npy",np.array(out_grid_hway.highway_class, dtype=np.uint8))
    print("#### Saving to npy file done")
    print("#### Saving to class file")
    df_road_classes.to_csv(base_out_folder+out_grid_file+"_streetclass_def.csv")
    # np.savetxt(out_grid_file+"_streetclass_def.csv", road_classes, '%s', ',')
    print("#### Saving to class file done")
    
if plot_switch:
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
     edges.plot( ax=ax1, legend=True)
     out_grid_hway.highway_class.plot(ax=ax2)
     
     ax1.set_aspect('equal', 'box')
     ax2.set_aspect('equal', 'box')
     plt.show()
     plt.savefig(base_out_folder+out_grid_file+"_highway_clip_grid.png")
     
# creating StreetLane layer
#select only 2 columns from dataframe
df_lanes=edges_clipped[["lanes", "geometry"]]
#remove index columns
df_lanes.reset_index(drop=True, inplace=True)
#flagg strings and nested arrays as nan
df_lanes.lanes=pd.to_numeric(df_lanes.lanes, "coerce")
#remove nan values
df_lanes=df_lanes.dropna()
# expldoe nested list of strings to multiple rows
# df_lanes=df_lanes.explode("lanes")
# remove duplicated indices, only pick the first 
# df_lanes = df_lanes[~df_lanes.index.duplicated(keep='first')]
df_lanes = df_lanes.astype({"lanes": np.int8})

out_grid_lanes = make_geocube(vector_data=df_lanes,resolution=(-10, 10),geom=json.dumps(geom))

if write_switch:
    print("#### Saving to npy file")
    # out_grid_file=out_grid_file+"_maxspeed_clip.npy"
    np.save(base_out_folder+out_grid_file+"_nlanes_clip.npy",np.array(out_grid_lanes.lanes, dtype=np.uint8))
    print("#### Saving to npy file done")
    
if plot_switch:
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
     edges.plot( ax=ax1, legend=True)
     out_grid_lanes.lanes.plot(ax=ax2)
     
     ax1.set_aspect('equal', 'box')
     ax2.set_aspect('equal', 'box')
     plt.show()
     plt.savefig(base_out_folder+out_grid_file+"_nlanes_clip_grid.png")