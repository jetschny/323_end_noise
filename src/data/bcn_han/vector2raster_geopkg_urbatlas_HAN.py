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
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# import rasterio
# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')

base_in_folder="/home/sjet/data/323_end_noise/HAN_data/"
base_out_folder="/home/sjet/data/323_end_noise/HAN_data/"

in_file  = 'DE013L1_HANNOVER_UA2018_v013/Data/DE013L1_HANNOVER_UA2018_v013.gpkg'
out_file = "DE013L1_HANNOVER_UA2018_v013"

plot_switch=True
write_switch=True
clip_switch=True

print("#### Loading file")

#2018 urban atlas area classifcation functional urban areas

#2012 urban atlas area classifcation functional urban areas
# filename = './ES002L2_BARCELONA_UA2012_revised_v021/Data/ES002L2_BARCELONA_UA2012_revised_v021.gpkg'

corner_point1=np.array((4.295 , 3.244 ))*1e6
corner_point2=np.array((4.314 , 3.259 ))*1e6
# polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
                       
# clip_box=(3.655*1e6 , 3.670*1e6 , 2.06*1e6, 2.08*1e6)

gdf = gpd.read_file(base_in_folder+in_file, mask=polygon)

print("#### Loading file done\n")



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
    # corner_point1=np.array((3.656 , 2.06 ))*1e6
    # corner_point2=np.array((3.669 , 2.074 ))*1e6
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=gdf.crs)

    gdf_clipped = gdf.clip(polygon)
    gdf_clipped=gdf_clipped[["code_2018", "geometry"]]
else:
    gdf_clipped=gdf[["code_2018", "geometry"]]


gdf_clipped = gdf_clipped.astype({"code_2018": int})
# gdf_clean = code12220_clean.drop(code12220_clean.index[7])


print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf.plot( ax=ax1, legend=True)
    poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    gdf_clipped.plot( ax=ax2, legend=True)
    plt.show()
    plt.savefig(base_out_folder+out_file+"_clip.png")
    print("#### Plotting file done \n")



print("#### Gridding vector file")

out_grid = make_geocube(
    vector_data=gdf_clipped,
    resolution=(-10, 10),
)

print("#### Gridding vector file done \n")

if plot_switch:
    # Get the colormap and set the under and bad colors
    colMap = plt.cm.get_cmap("gist_rainbow").copy()
    colMap.set_under(color='white')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    gdf_clipped.plot( ax=ax1, legend=True)
    out_grid.code_2018.plot(ax=ax2)
    
    im2=ax2.imshow(np.array(out_grid.code_2018, dtype=np.uint16), cmap=colMap, vmin = 0.2)
    
       
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    plt.show()
    plt.savefig(base_out_folder+out_file+"_grid.png")
    
if write_switch:
    print("#### Saving to npy file")
    if clip_switch:
        out_grid_filename=base_out_folder+out_file+"_clip.npy"
    else:
        out_grid_filename=base_out_folder+out_file+".npy"
    np.save(out_grid_filename,np.array(out_grid.code_2018, dtype=np.uint16))
    print("#### Saving to npy file done")
