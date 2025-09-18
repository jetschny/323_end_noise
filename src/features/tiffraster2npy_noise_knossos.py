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
import sys
import geopandas as gpd
# from geocube.api.core import make_geocube
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
# from rasterio import warp
# from skimage.transform import resize

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds
import builtins
globals()["__builtins__"] = builtins


plt.close('all')
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1", "TRUE", "True")

if len(sys.argv) >2:
    print("Total number of arguments passed:", len(sys.argv))
    print("\nArguments passed:", end = " ")
    for i in range(0,len(sys.argv) ):
        print(sys.argv[i],"\t", end = " # ")
    plot_switch=str2bool(sys.argv[3])
    write_switch=str2bool(sys.argv[4])
else:
    plot_switch=True
    write_switch=True

# plot_switch=False
write_switch=True
clip_switch=False
interp_switch=False

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
city_string_in="Budapest"
city_string_out="BUD" 

# city_string_in="Vienna"
# city_string_out="VIE" 

# city_string_in=sys.argv[1]
# city_string_out=sys.argv[2]


print("\n######################## \n")
print("Noise target creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder          ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder         ="Z:/NoiseML/2024/city_data_features/"
base_out_folder_pic     ="Z:/NoiseML/2024/city_data_pics/"

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

# reclassifying noise (if different from 5 dB bands)
grid1[(grid1 >= 0) & (grid1 <= 57)] = 55
grid1[(grid1 >= 57) & (grid1 <= 62)] = 60
grid1[(grid1 >= 62) & (grid1 <= 67)] = 65
grid1[(grid1 >= 67) & (grid1 <= 72)] = 70
grid1[(grid1 >= 72)] = 75


print("#### Processing file done \n")


if write_switch:
    print("#### Saving to output files")
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid1)

    print("... Saving to npy file done")
    
    out_meta = img.meta.copy()
    # epsg_code = int(img.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "dtype" : 'float32',
                     "nodata" : -999.25,
                     "crs": img.crs})
    with rasterio.open(base_out_folder+city_string_in+"/"+city_string_out+out_file+".tif", "w", **out_meta) as dest:
        dest.write(grid1[np.newaxis,:,:])
        
    print("... Saving to tiff file done")
    
if plot_switch:
    print("#### Plotting file")
    color_axis_crop_min=50;
    color_axis_crop_max=80;
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    retted=show(img, ax=ax1, vmin=color_axis_crop_min, vmax=color_axis_crop_max)
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
    plt.clim(color_axis_crop_min, color_axis_crop_max)
    plt.colorbar()
    plt.savefig(base_out_folder_pic+city_string_out+out_file+"_clip.png")
    plt.show()

print("#### Plotting file done \n")



    

