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
# city_string_in="Madrid"
city_string_in=sys.argv[1]
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
# city_string_out="MAD" 
city_string_out=sys.argv[2]

print("\n######################## \n")
print("Downloading OSM data for road information feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder:  str ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder: str ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder_pic: str ="Z:/NoiseML/2024/city_data_pics/"

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

# corner_point1=np.round(np.array((3.6560 , 2.0600 ))*1e6)
# corner_point2=np.round(np.array((3.6690 , 2.0740 ))*1e6)
corner_point1=np.array((img_target_bounds[0] , img_target_bounds[1] ))
corner_point2=np.array((img_target_bounds[2] , img_target_bounds[3] ))

polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                   (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=edges_conv.crs)

edges_clipped = edges_conv.clip(polygon)
# gdf_clipped=gdf_clipped[["Rang", "geometry"]]

##########################################
# creating max_speed data layer
##########################################

#select only 2 columns from dataframe
df_maxspeed=edges_clipped[["maxspeed", "geometry"]]
#remove index columns
df_maxspeed.reset_index(drop=True, inplace=True)
#remove nan values
df_maxspeed=df_maxspeed.dropna()

if city_string_out=="CLF":
    df_maxspeed=df_maxspeed.replace("FR:urban",30)
    # df_maxspeed=df_maxspeed.replace("['FR:urban', '30']",30)
    df_maxspeed.loc[27030].maxspeed[0]=30
    df_maxspeed.loc[27031].maxspeed[0]=30
    
if city_string_out=="VIE":
    df_maxspeed=df_maxspeed.replace("walk",10)
    df_maxspeed=df_maxspeed.replace("AT:walk",10)
    df_maxspeed=df_maxspeed.replace("AT:zone:30",30)
    df_maxspeed.loc[32879].maxspeed[0]=10
    df_maxspeed.loc[32879].maxspeed[1]=10

if city_string_out=="NIC":
    df_maxspeed=df_maxspeed.replace("25 mph",40)

if city_string_out=="MAD":
        df_maxspeed=df_maxspeed.replace("50|30",40)

# remove nested array and replace with mean of array
df_maxspeed["maxspeed"] = df_maxspeed["maxspeed"].map(lambda x: np.mean(np.int_(x))) 
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
   
    grid1=np.array(out_grid_maxspeed.maxspeed,dtype=np.float32)
    grid1 = grid1[0:img_target.shape[0],0:img_target.shape[1]]
    grid1= np.nan_to_num(grid1, nan=-999.25)
    np.save(base_out_folder+city_string_in+"/" + city_string_out+out_grid_file+"_maxspeed.npy",grid1)
        
    print("#### Saving to npy file done")

if plot_switch:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    edges_clipped.plot( ax=ax1, legend=True)
    out_grid_maxspeed.maxspeed.plot(ax=ax2)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    plt.savefig(base_out_folder_pic+"/" + city_string_out+out_grid_file+"_maxspeed.png")
    plt.show()

##########################################    
# creating StreetClass layer
##########################################

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
#get list of unique highway classes
road_classes_old=sorted(df_hway["highway"].unique())


# road_classes_new=np.zeros(len(road_classes_old))
# df_hway_explod=edges.explode("highway")
road_classes_new=np.array(range(0, len(road_classes_old), 1))
# road_classes_new=np.array([7,1,1,3,3,6,4,4,5,5,2,2])

road_classes = np.empty((len(road_classes_old), 0))
road_classes=np.append(road_classes, np.array([road_classes_old]).transpose(), axis=1)
road_classes=np.append(road_classes, np.array([road_classes_new]).transpose(), axis=1)


counter=0
for a in road_classes_old:
    df_hway.loc[df_hway["highway"]==a,"highway_class"]=road_classes_new[counter]
    counter=counter+1
  
df_hway=df_hway[["highway_class","geometry"]]  
out_grid_hway = make_geocube(vector_data=df_hway,resolution=(-10, 10),geom=json.dumps(geom))

if write_switch:
    print("#### Saving to npy file")
    
    grid1=np.array(out_grid_hway.highway_class,dtype=np.float32)
    grid1 = grid1[0:img_target.shape[0],0:img_target.shape[1]]
    grid1= np.nan_to_num(grid1, nan=-999.25)
    # index0 = np.where(grid1 == 0)    
    # grid1[index0]=-999.25
    
    np.save(base_out_folder+city_string_in+"/" + city_string_out+out_grid_file+"_streetclass.npy",grid1)
    
    print("#### Saving to npy file done")
    print("#### Saving to class file")
    np.savetxt(base_out_folder+city_string_in+"/" + city_string_out+out_grid_file+"_streetclass_def.csv", road_classes, '%s', ',')
    print("#### Saving to class file done")
    
if plot_switch:
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
     edges_clipped.plot( ax=ax1, legend=True)
     out_grid_hway.highway_class.plot(ax=ax2)
     
     ax1.set_aspect('equal', 'box')
     ax2.set_aspect('equal', 'box')
     plt.savefig(base_out_folder_pic+"/" + city_string_out+out_grid_file+"_streetclass.png")
     plt.show()
     
##########################################
# creating StreetLane layer
##########################################

#select only 2 columns from dataframe
df_lanes=edges_clipped[["lanes", "geometry"]]
#remove index columns
df_lanes.reset_index(drop=True, inplace=True)
#remove nan values
df_lanes=df_lanes.dropna()
# expldoe nested list of strings to multiple rows
df_lanes=df_lanes.explode("lanes")
# remove duplicated indices, only pick the first 
df_lanes = df_lanes[~df_lanes.index.duplicated(keep='first')]

df_lanes = df_lanes.astype({"lanes": np.int8})

out_grid_lanes = make_geocube(vector_data=df_lanes,resolution=(-10, 10),geom=json.dumps(geom))

if write_switch:
    print("#### Saving to npy file")
    grid1=np.array(out_grid_lanes.lanes,dtype=np.float32)
    grid1 = grid1[0:img_target.shape[0],0:img_target.shape[1]]
    grid1= np.nan_to_num(grid1, nan=-999.25)
    
    np.save(base_out_folder+city_string_in+"/" + city_string_out+out_grid_file+"_nlanes.npy",grid1)
    print("#### Saving to npy file done")
    
if plot_switch:
     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
     edges_clipped.plot( ax=ax1, legend=True)
     out_grid_lanes.lanes.plot(ax=ax2)
     
     ax1.set_aspect('equal', 'box')
     ax2.set_aspect('equal', 'box')
     plt.savefig(base_out_folder_pic+"/" + city_string_out+out_grid_file+"_nlanes.png")
     plt.show()