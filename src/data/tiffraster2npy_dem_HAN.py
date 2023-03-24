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

#2018 imperviousness Density
base_in_folder="/home/sjet/data/323_end_noise/HAN_data/"
base_out_folder="/home/sjet/data/323_end_noise/HAN_data/"
in_file = 'Hannover_eu_dem_v11_E40N30.TIF'
out_file = 'Hannover_eu_dem_v11_E40N30'

img = rasterio.open(base_in_folder+in_file, 'r') 
  
print("#### Loading file done\n")


print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)

if clip_switch:
    # Create a custom polygon
    corner_point1=np.array((4.295 , 3.244 ))*1e6
    corner_point2=np.array((4.314 , 3.259 ))*1e6
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
    img_clipped, out_transform = mask(img, shapes=[polygon], crop=True)
else:
    img_clipped=img.read()
    # img_clipped=np.array(img)

if interp_switch:
    img_clipped = resize(np.squeeze(img_clipped),(1500,1900))


print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # show(img, ax=ax1)
       
    if clip_switch:
        poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    plt.imshow(np.squeeze(img_clipped),cmap="jet")
    plt.clim(0, 300)
    plt.colorbar()
    plt.savefig(base_in_folder+out_file+"_clip.png")
    plt.show()

print("#### Plotting file done \n")


if write_switch:
    print("#### Saving to npy file")
    if clip_switch:
        out_grid_file=base_out_folder+out_file+"_clip.npy"
    else:
        out_grid_file=base_out_folder+out_file+".npy"
    np.save(out_grid_file,np.squeeze(img_clipped))
    print("#### Saving to npy file done")
    

