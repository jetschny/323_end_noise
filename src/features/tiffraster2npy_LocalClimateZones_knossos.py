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
import os
# from numpy import newaxis
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
        print(sys.argv[i],"\t", end = " ")
    plot_switch=str2bool(sys.argv[3])
    write_switch=str2bool(sys.argv[4])
else:
    plot_switch=True
    write_switch=True

plot_switch=False
# write_switch=False
clip_switch=True
interp_switch=True

#"Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga" "Bordeaux" "Grenoble" "Innsbruck" "Salzburg" "Kaunas" "Limassol"
#"VIE" #"PIL" #"CLF" #"RIG" "BOR" "GRE" "INN" "SAL" "KAU" "LIM" 
# city_string_in="Clermont_Ferrand"
# city_string_out="CLF" 
# city_string_in="Ljubljana"
# city_string_out="LJU" 
# city_string_in="Thessaloniki"
# city_string_out="THE" 
city_string_in="Vienna"
city_string_out="VIE" 

# city_string_in=sys.argv[1]
# city_string_out=sys.argv[2]

print("\n######################## \n")
print("_LocalClimateZones feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder  ="Z:/NoiseML/2024/city_data_raw/"
base_out_folder ="Z:/NoiseML/2024/city_data_features/"
base_out_folder_pic ="Z:/NoiseML/2024/city_data_pics/"

in_file  = '_LocalClimateZones.tif'
in_file_target='_MRoadsLden.tif'
out_file = "_feat_localclimatezones"


img = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file, 'r') 
img_target = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r') 
  
  
print("#### Loading file done\n")


print("#### Cropping file")

# code12220 =  gdf[gdf['code_2018']=='12220']
# code12220.explore("area", legend=False)

if clip_switch:
    # Create a custom polygon from bbox extend of target data
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
    # grid1 = grid1[img_target.shape]s
    # target grid is bigger by 1 GP each or more
    if (city_string_out=="PIL") or (city_string_out=="MAD"):
        grid1_0 = np.zeros(img_target.shape)-999.25
        grid1_0[0:grid1.shape[0],0:(grid1.shape[1]-1)]=grid1[0:grid1.shape[0],0:(grid1.shape[1]-1)]
        grid1=grid1_0
    # feature grid is bigger by 1 GP in Y-dim
    if (city_string_out=="SAL"):
        grid1 = grid1[:,1:1+img_target.shape[1]]
    # feature grid is bigger by 1 GP in X-dim
    if (city_string_out=="KAU") or (city_string_out=="LIM") or (city_string_out=="RIG"):
        grid1 = grid1[1:1+img_target.shape[0],:]
    # feature grid is bigger by 1 GP each
    if (city_string_out=="OSL") or (city_string_out=="CLF") or (city_string_out=="NIC") or (city_string_out=="VIE"):
        grid1 = grid1[1:1+img_target.shape[0],1:1+img_target.shape[1]]
    else :
        # grid1 = grid1[1:1+img_target.shape[0],1:1+img_target.shape[1]]
    # grid1 = resize(grid1, img_target.shape)
        45

index0 = np.where(grid1 == img.nodata)
grid1[index0]=0

nodata_value = img.nodata if img.nodata is not None else -999.25
grid1[grid1 == nodata_value] = 0

print("#### Cropping file done \n")

if  np.squeeze(grid1).shape != img_target.shape:
    print("#####################################################")
    print("#### Warning : target and feature array size mismtach")
    print("##################################################### \n´")
    
    
print("#### Processing file")

grid1_distance=grid1.astype(float)
grid1_distance[index0]=-999.25
grid1_distance=grid1_distance.astype(np.float32)
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
    # with rasterio.open(base_out_folder+city_string_in+"/"+city_string_out+out_file+".tif", "w", **out_meta) as dest:
        # dest.write(grid1_distance[np.newaxis,:,:])
        
    output_tif_path = os.path.join(base_out_folder, city_string_in, city_string_out + out_file + ".tif")

    with rasterio.open(output_tif_path, "w", **out_meta) as dest:
        dest.write(grid1_distance[np.newaxis, :, :])
    
    print("... Saving to tiff file done")
        
    
if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
  
    retted=show(img, ax=ax1, vmin=0, vmax=0.75*np.max(grid1_distance))
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
    plt.clim(0, 0.75*np.max(grid1_distance))
    plt.colorbar()
    plt.savefig(base_out_folder_pic+"/"+city_string_out+out_file+".png")
    plt.show()

    print("#### Plotting file done \n")
