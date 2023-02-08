# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:49:51 2022

@author: RekanS
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

# import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box
import json

# import rasterio
# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')

# out_grid_file="ES002L2_BARCELONA_UA2018_v013.npy"
out_grid_file="2017_isofones_total_dia_mapa_estrategic_soroll_bcn"

plot_switch=False
write_switch=True
clip_switch=True
convcrs_switch=True

print("#### Loading file")

#2018 urban atlas area classifcation functional urban areas
filename = '2017_isofones_total_dia_mapa_estrategic_soroll_bcn.gpkg'
#2012 urban atlas area classifcation functional urban areas
# filename = './ES002L2_BARCELONA_UA2012_revised_v021/Data/ES002L2_BARCELONA_UA2012_revised_v021.gpkg'

gdf = gpd.read_file(filename)
# corner_point1=np.array((3.655 , 2.06))*1e6
# corner_point2=np.array((3.670 , 2.08))*1e6
# # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
# polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
#                        (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
                       
# # clip_box=(3.655*1e6 , 3.670*1e6 , 2.06*1e6, 2.08*1e6)

# gdf = gpd.read_file(filename, mask=polygon)


print("#### Loading file done\n")

print("#### Converting CRS ")
if convcrs_switch:
    gdf = gdf.to_crs("EPSG:3035")

print("#### Converting CRS done\n")

print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)


# code12220_clipped = code12220.clip(polygon)
# code12220_clean=code12220_clipped[["code_2018", "geometry"]]
# # .loc["code_2018"]=pd.to_numeric(code12220_clean["code_2018"])

# code12220_clean = code12220_clean.astype({"code_2018": int})
# code12220_clean = code12220_clean.drop(code12220_clean.index[7])

if clip_switch:
    # Create a custom polygon
    corner_point1=np.array((3.656 , 2.06 ))*1e6
    corner_point2=np.array((3.669 , 2.074 ))*1e6
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=gdf.crs)

    gdf_clipped = gdf.clip(polygon)
    gdf_clipped=gdf_clipped[["Rang", "geometry"]]
else:
    gdf_clipped=gdf[["Rang", "geometry"]]


# gdf_clipped = gdf_clipped.astype({"code_2018": int})
# gdf_clean = code12220_clean.drop(code12220_clean.index[7])


print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf.plot( ax=ax1, legend=True)
    if clip_switch:
        poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    gdf_clipped.plot( ax=ax2, legend=True)
    plt.show()
    plt.savefig(out_grid_file+"_clip.png")

    print("#### Plotting file done \n")


noise_classes_old=sorted(gdf_clipped["Rang"].unique())
noise_classes_new=np.array(range(42, 80, 5))
noise_classes_new=np.append(noise_classes_new,[ 32, 87])
#if needed, classes convert into Pa instead of dB
noise_classes_new_pa=10**(noise_classes_new/20)*(2e-5)

counter=0
for a in noise_classes_old:
    gdf_clipped.loc[gdf_clipped["Rang"]==a,"noise_class"]=noise_classes_new[counter]
    counter=counter+1
    

gdf_clipped=gdf_clipped[["noise_class", "geometry"]]

print("#### Gridding vector file")


# Make a GeoJSON string of the bounding box feature
bbox = gpd.GeoSeries(box(*polygon.bounds))
geom = bbox.__geo_interface__["features"][0]["geometry"]
# Add CRS
geom["crs"] = {"properties": {"name": f"EPSG:3035"}}

out_grid = make_geocube(vector_data=gdf_clipped,resolution=(-10, 10),geom=json.dumps(geom))


print("#### Gridding vector file done \n")

if plot_switch:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf_clipped.plot( ax=ax1, legend=True)
    out_grid.noise_class.plot(ax=ax2)
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    plt.show()
    plt.savefig(out_grid_file+"_grid.png")
 
if write_switch:
    print("#### Saving to npy file")
    if clip_switch:
        out_grid_file=out_grid_file+"_clip.npy"
    else:
        out_grid_file=out_grid_file+".npy"
    np.save(out_grid_file,np.array(out_grid.noise_class, dtype=np.uint8))
    print("#### Saving to npy file done")
