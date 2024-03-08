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
# from skimage.transform import resize

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')

plot_switch=True
write_switch=True
clip_switch=False
interp_switch=False

print("#### Loading file")

# city_string_in="Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga"
# city_string_out="VIE" #"PIL" #"CLF" #"RIG"
city_string_in="Riga"
city_string_out="RIG"

base_in_folder_tif="/home/sjet/data/323_end_noise/"
base_in_folder_npy="/home/sjet/repos/323_end_noise/data/output/"
base_out_folder="/home/sjet/repos/323_end_noise/data/output/"

in_file_tif  = '_MRoadsLden.tif'
in_file_npy = "_full_dnn_map.npy"
out_file_tif  = '_full_dnn_map.tif'

img = rasterio.open(base_in_folder_tif+city_string_in+"/" + city_string_in + in_file_tif, 'r') 
grid1=np.load(base_in_folder_npy+city_string_in+in_file_npy)
  
print("#### Loading file done\n")


print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)

if clip_switch:
    # Create a custom polygon
    corner_point1=np.array((3.656 , 2.06 ))*1e6
    corner_point2=np.array((3.669 , 2.074 ))*1e6
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
    img_clipped, out_transform = mask(img, shapes=[polygon], crop=True)
    out_meta = img.meta
    out_meta.update({"driver": "GTiff",
                 "height": img_clipped.shape[1],
                 "width": img_clipped.shape[2],
                 "transform": out_transform,
                 "nodata" : 0})
else:
    img_clipped =img.read()
    # img_clipped=np.array(img)
    out_meta = img.meta
    out_meta.update({"driver": "GTiff",
                 "height": img_clipped.shape[1],
                 "width": img_clipped.shape[2],
                 "nodata" : 0})

# if interp_switch:
#     img_clipped = np.resize(np.squeeze(img_clipped),(1400,1300))

noise_classes_old=sorted(np.unique(grid1))
noise_classes_new=sorted(np.unique(img_clipped))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid1 ==a)
    grid1[indexxy]=noise_classes_new[counter]
    counter=counter+1
    

grid1=grid1.astype(np.float32)
index0 = np.where(grid1 == img.nodata)
grid1[index0]=0

print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # show(img, ax=ax1)
       
    # if clip_switch:
    #     poly_gdf.boundary.plot(ax=ax1, color="red")
    ax1.set_title("TIFF input", fontsize=14)
    ax2.set_title("NPY predictions", fontsize=14)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    im1=ax1.imshow(np.squeeze(img_clipped),cmap="jet")
    im1.set_clim(50, 76)
    fig.colorbar(im1, orientation='vertical', ax=ax1)
    # plt.colorbar()
    # plt.savefig(base_in_folder+out_grid_file+"_clip.png")
    im2=ax2.imshow(np.squeeze(grid1),cmap="jet")
    # plt.clim(0, 76)
    im2.set_clim(50, 76)
    fig.colorbar(im2, orientation='vertical', ax=ax2)
    plt.show()

print("#### Plotting file done \n")


if write_switch:
    print("#### Saving to tif file")
    profile = img.profile

    with rasterio.open(base_in_folder_npy+city_string_in+out_file_tif, 'w', **out_meta) as dst:
        dst.write(grid1.astype(rasterio.uint8), 1)
    print("#### Saving to tif file done")
    

