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
from PIL import Image
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from skimage.transform import resize

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')

plot_switch=True
write_switch=True
clip_switch=True
interp_switch=True

print("#### Loading file")

city_string_in="Clermont_Ferrand" #"Riga"
city_string_out="CLF" #"RIG"

base_in_folder="/home/sjet/data/323_end_noise/"
base_out_folder="/home/sjet/data/323_end_noise/"
in_file  = '_ua2018_10m.tif'
in_file_target='_MRoadsLden.tif'
out_file = "_raw_ua_road"



img = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file, 'r') 
img_target = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r') 
  
print("#### Loading file done\n")


print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)

if clip_switch:
    # Create a custom polygon
    
    # corner_point1=np.array((3.656 , 2.06 ))*1e6
    # corner_point2=np.array((3.669 , 2.074 ))*1e6
    img_target_bounds=img_target.bounds
    corner_point1=np.array((img_target_bounds[0] , img_target_bounds[1] ))
    corner_point2=np.array((img_target_bounds[2] , img_target_bounds[3] ))
     
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
       
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
    grid1, out_transform = mask(img, shapes=[polygon], crop=True)
else:
    grid1=img.read()
    # img_clipped=np.array(img)

grid1=np.squeeze(grid1)

if interp_switch:
    # grid2 = grid1[img_target.shape]
    grid1 = grid1[0:img_target.shape[0],0:img_target.shape[1]]
    # grid1 = rescale(grid1,2.5)
    # 45

index0 = np.where(grid1 == img.nodata)
grid1[index0]=0


print("#### Cropping file done \n")

print("#### Processing file")

grid1_distance=grid1.astype(np.float32)
grid1_distance[index0]=-999.25

print("#### Processing file done \n")


if write_switch:
    print("#### Saving to npy file")
    # if clip_switch:
    #     out_grid_file=out_file+"_clip.npy"
    # else:
    #     out_grid_file=out_file+".npy"
    np.save(base_out_folder+city_string_in+"/" + city_string_out+out_file+".npy",grid1_distance)
    print("#### Saving to npy file done")
    
if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
  
    # retted=show(img, ax=ax1, vmin=0, vmax=0.75*np.max(grid1_distance))
    retted=show(img, ax=ax1, vmin=12210, vmax=12220)
    im = retted.get_images()[0]
    fig.colorbar(im, ax=ax1)
    
    if clip_switch:
        poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    plt.imshow(grid1_distance,cmap="jet")
    # plt.clim(0, 0.75*np.max(grid1_distance))
    plt.clim(12210, 12220)
    plt.colorbar()
    plt.savefig(base_out_folder+city_string_in+"/" + city_string_out+out_file+".png")
    plt.show()

    print("#### Plotting file done \n")
