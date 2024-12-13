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

import sys
# import pandas as pd
import geopandas as gpd
# from geocube.api.core import make_geocube
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from skimage.transform import resize
# from numpy import newaxis
# from skimage.transform import rescale

# from rasterio.features import rasterize
# from rasterio.transform import from_bounds

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

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
# city_string_in="Budapest"
# city_string_out="BUD" 

city_string_in=sys.argv[1]
city_string_out=sys.argv[2]

print("\n######################## \n")
print("DEM topography feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder:  str ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder: str ="Z:/NoiseML/2024/city_data_features/"
base_out_folder_pic: str ="Z:/NoiseML/2024/city_data_pics/"

in_file = '_eu_dem_v11.tif'
in_file_target='_MRoadsLden.tif'
out_file = "_feat_eu_dem_v11"

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
grid1=grid1.astype(np.float32)
# grid1=np.nan_to_num(grid1, nan=img.nodata)

index0 = np.where(grid1 == img.nodata)
# no data is set to very low random value to avoid edge artifacts
# not working perfectly though...
# grid1[index0]=0.000123
grid1[index0]=np.nan
# grid1[index0]=0.0

if interp_switch:
    grid1 = resize(grid1,img_target.shape, mode='edge')
    # grid1 = rescale(grid1,2.5)
    # 45

# index0 = np.where(grid1 == np.nan)
# grid1[index0]=0.000123
grid1=np.nan_to_num(grid1, copy=False, nan=-999.25)
index0 = np.where(grid1 == -999.25)

print("#### Cropping file done \n")

print("#### Processing file")

side_length=25
radius=divmod(side_length,2)[0]

grid1_pad=np.pad(grid1, [radius,radius], "symmetric")
dim_grid1_pad=grid1_pad.shape
grid1_distance=np.zeros(dim_grid1_pad)
distance_matrix=np.zeros([radius,radius])

# for indexxy, item in np.ndenumerate(distance_matrix):
#     distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius)**2 + (indexxy[1]-radius)**2))
    

def calc_distance(indexxy):
    grid1_pad_window=grid1_pad[indexxy[0]-radius:indexxy[0]+radius,indexxy[1]-radius:indexxy[1]+radius]
    index0_grid1_pad_window = np.where(grid1_pad_window > -999.25)
    if len( index0_grid1_pad_window[0]) >0 :
        grid1_pad_window_mean=np.mean(grid1_pad_window[index0_grid1_pad_window])
    else:
        grid1_pad_window_mean=-999.25
    return grid1_pad[indexxy]-grid1_pad_window_mean
  
def check_frame(indexxy):
    if (min(indexxy)>=radius) and (indexxy[0]<(dim_grid1_pad[0]-radius)) and (indexxy[1]<(dim_grid1_pad[1]-radius)):
        return calc_distance(indexxy)
    else:
        return 0
       
for indexxy, item in np.ndenumerate(grid1_pad):
    grid1_distance[indexxy]=check_frame(indexxy)

grid1_distance=grid1_distance[radius:-radius, radius:-radius]
# grid1_distance[index0]=-999.25
# index0 = np.where(grid1 == img.nodata)
# index0 = np.where(grid1 == 0.000123)
grid1_distance=grid1_distance.astype(np.float32)
grid1_distance[index0]=-999.25


print("#### Processing file done \n")


if write_switch:
    print("#### Saving to output files")
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid1_distance)

    print("... Saving to npy file done")
    
    out_meta = img_target.meta.copy()
    # epsg_code = int(img.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "dtype" : 'float32',
                     "nodata" : -999.25,
                     "crs": img.crs})
    with rasterio.open(base_out_folder+city_string_in+"/"+city_string_out+out_file+".tif", "w", **out_meta) as dest:
        dest.write(grid1_distance[np.newaxis,:,:])
        
    print("... Saving to tiff file done")
    
if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    retted=show(img, ax=ax1, vmin=0, vmax=0.75*np.max(grid1))
    im = retted.get_images()[0]
    fig.colorbar(im, ax=ax1)
    
    if clip_switch:
        poly_gdf.boundary.plot(ax=ax1, color="red")
    # ax1.set_title("All Unclipped World Data", fontsize=20)
    # ax2.set_title("All Unclipped Capital Data", fontsize=20)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    plt.imshow(np.squeeze(grid1_distance),cmap="jet")
    plt.clim(-0.75*np.max(grid1_distance), 0.75*np.max(grid1_distance))
    plt.colorbar()
    plt.savefig(base_out_folder_pic+"/"+city_string_out+out_file+".png")
    plt.show()

print("#### Plotting file done \n")




