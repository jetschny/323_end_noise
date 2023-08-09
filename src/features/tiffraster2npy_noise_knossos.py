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
from rasterio import warp
from skimage.transform import resize

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

plt.close('all')


plot_switch=True
write_switch=True
clip_switch=False
interp_switch=False

print("#### Loading file")

city_string_in="Vienna" #"Innsbruck" #"Bordeaux" #"Grenoble" #"Salzburg" #"Pilsen" #"Clermont_Ferrand" #"Riga"
city_string_out="VIE" #"INN" #"BOR" #"GRE" #"SAL" #"PIL" #"CLF" #"RIG"


base_in_folder="/home/sjet/data/323_end_noise/"
base_out_folder="/home/sjet/data/323_end_noise/"

in_file = '_MRoadsLden.tif'
out_file="_target_noise_Aggroad_Lden"

img = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file, 'r') 
  
print("#### Loading file done\n")


print("#### Cropping file")



if clip_switch:
    # Create a custom polygon
    # corner_point1=np.array((4.296 , 3.243 ))*1e6
    # corner_point2=np.array((4.311 , 3.258 ))*1e6
    
    # polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
    #                    (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])

    # # coords_transformed = warp.transform({'init': 'epsg:3035'},{'init': 'epsg:25832'},[corner_point1[0], corner_point2[0]], [corner_point1[1], corner_point2[1]])
    
    # # corner_point1=np.array(coords_transformed[0])
    # # corner_point2=np.array(coords_transformed[1])
    # # 
    # polygon_transformed = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
    #                    (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
    # poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon_transformed], crs=img.crs)

    # img_clipped, out_transform = mask(img, shapes=[polygon_transformed], crop=True)
    
    # Create a custom polygon
    corner_point1=np.array((4.295 , 3.244 ))*1e6
    corner_point2=np.array((4.314 , 3.259 ))*1e6
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    polygon = Polygon([(corner_point1[0], corner_point1[1] ), (corner_point2[0], corner_point1[1]), 
                       (corner_point2[0], corner_point2[1]),(corner_point1[0], corner_point2[1]), (corner_point1[0],corner_point1[1])])
    
    # polygon = Polygon([(3.645*1e06, 2.05*1e06 ), (3.67*1e06, 2.05*1e06), (3.67*1e06, 2.07*1e06), (3.645*1e06, 2.07*1e06), (3.645*1e06,2.05*1e06)])
    poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=img.crs)
    grid1, out_transform = mask(img, shapes=[polygon], crop=True)
    
else:
    grid1=img.read()
    # img_clipped=np.array(img)

grid1=np.squeeze(grid1)

if interp_switch:
    # img_clipped = resize(np.squeeze(img_clipped),(1400,1300))
    # img_clipped = rescale(np.squeeze(img_clipped),2.5)    
    # grid1 = rescale(grid1,0.5)    
    45

print("#### Cropping file done \n")

print("#### Processing file")

grid1=grid1.astype(np.float32)
index0 = np.where(grid1 == img.nodata)
grid1[index0]=-999.25

print("#### Processing file done \n")


if write_switch:
    print("#### Saving to npy file")
    # if clip_switch:
    #     out_grid_filename=out_file+"_clip.npy"
    # else:
    #     out_grid_filename=out_file+".npy"
    # np.save(out_grid_filename,grid1)
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid1)
    print("#### Saving to npy file done")
    
if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    retted=show(img, ax=ax1, vmin=0, vmax=0.75*np.max(grid1))
    # add colorbar using the now hidden image
    im = retted.get_images()[0]
    fig.colorbar(im, ax=ax1)
    
       
    if clip_switch:
        poly_gdf.boundary.plot(ax=ax1, color="red")
    
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(grid1, ax=ax2)
    plt.imshow(grid1,cmap="jet")
    # plt.clim(0, 300)
    plt.colorbar()
    plt.savefig(base_in_folder+out_file+"_clip.png")
    plt.show()

print("#### Plotting file done \n")



    

