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

out_grid_file="ES_Major_Roads_Lden_line"
plot_switch=True
write_switch=True
clip_switch=True
interp_switch=False

print("#### Loading file")

#2018 imperviousness Density
filename = './END/ES_Major_Roads_Lden_line.tif'

img = rasterio.open(filename, 'r') 
  
print("#### Loading file done\n")


print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)

new_dataset = rasterio.open(filename,'r', driver='GTiff',
    height=100, width=100,
    count=1, crs='+proj=latlong')

if clip_switch:
    # Create a custom polygon
    corner_point1=np.array((3.656 , 2.06 ))*1e6
    corner_point2=np.array((3.669 , 2.074 ))*1e6
    # corner_point1=np.array((3.5 , 2.01 ))*1e6
    # corner_point2=np.array((3.7 , 2.08 ))*1e6
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
    img_clipped = resize(np.squeeze(img_clipped),(1400,1300))

with rasterio.open("./END/"+out_grid_file+"_clip.tif", "w", **out_meta) as dest:
    dest.write(img_clipped)

img_clipped2 = rasterio.open("./END/"+out_grid_file+"_clip.tif")
 
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
    # plt.imshow(np.squeeze(img_clipped),cmap="jet")
    show(img_clipped2)
    # plt.clim(0, 100)
    # plt.colorbar()
    plt.savefig(out_grid_file+"_clip.png")
    plt.show()

print("#### Plotting file done \n")


if write_switch:
    print("#### Saving to npy file")
    if clip_switch:
        out_grid_file=out_grid_file+"_clip.npy"
    else:
        out_grid_file=out_grid_file+".npy"
    np.save(out_grid_file,np.squeeze(img_clipped))
    print("#### Saving to npy file done")
    

