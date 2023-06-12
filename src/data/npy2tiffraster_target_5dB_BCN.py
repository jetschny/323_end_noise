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
clip_switch=True
interp_switch=False

print("#### Loading file")

#2018 imperviousness Density
base_in_folder="/home/sjet/data/323_end_noise/BCN_data/"
base_out_folder="/home/sjet/data/323_end_noise/BCN_data/"
in_file_tif  = 'ES002_BARCELONA_UA2012_DHM_V010/Dataset/ES002_BARCELONA_UA2012_DHM_V010.tif'
in_file_npy = "2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"
out_file_tif  = '2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.tif'

img = rasterio.open(base_in_folder+in_file_tif, 'r') 
grid1=np.load(base_in_folder+in_file_npy)
  
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
                 "transform": out_transform})
else:
    img_clipped=img.read()
    # img_clipped=np.array(img)

if interp_switch:
    img_clipped = np.resize(np.squeeze(img_clipped),(1400,1300))

img_clipped[img_clipped==65535]=0

print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # show(img, ax=ax1)
       
    # if clip_switch:
    #     poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    im1=ax1.imshow(np.squeeze(img_clipped),cmap="jet")
    im1.set_clim(0, 50)
    fig.colorbar(im1, orientation='vertical', ax=ax1)
    # plt.colorbar()
    # plt.savefig(base_in_folder+out_grid_file+"_clip.png")
    im2=ax2.imshow(np.squeeze(grid1),cmap="jet")
    # plt.clim(0, 300)
    fig.colorbar(im2, orientation='vertical', ax=ax2)
    plt.show()

print("#### Plotting file done \n")


if write_switch:
    print("#### Saving to tif file")
    profile = img.profile

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    # profile.update(
    #     dtype=rasterio.uint8,
    #     count=1,
    #     compress='lzw')


    with rasterio.open(base_out_folder+out_file_tif, 'w', **out_meta) as dst:
        dst.write(grid1.astype(rasterio.uint16), 1)
    print("#### Saving to npy file done")
    

