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
# import geopandas as gpd
# from geocube.api.core import make_geocube
# from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import rasterio
# from rasterio.plot import show
# from rasterio.mask import mask
# from skimage.transform import resize
import sys
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

# plot_switch=True
# write_switch=True
# clip_switch=False
# interp_switch=False

print("#### Loading file")

# city_string_in="Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga"
# city_string_out="VIE" #"PIL" #"CLF" #"RIG"
# city_string_in="Athens"
# city_string_out="ATH"
city_string_in=sys.argv[1]
city_string_out=sys.argv[2]

base_in_folder_tif  ="Z:/NoiseML/2024/city_data_raw/"
base_in_folder_npy  ="Z:/NoiseML/2024/city_data_MLpredictions/"
base_out_folder     ="Z:/NoiseML/2024/city_data_MLpredictions/"
base_out_folder_pic ="Z:/NoiseML/2024/city_data_pics/"

in_file_tif  = '_Absorption.tif'
in_file_npy = "_Lden_lczpred.npy"
out_file_tif  = '_full_dnn_map.tif'
out_file_png  = '_full_dnn_map.jpg'

img = rasterio.open(base_in_folder_tif+city_string_in+"/" + city_string_in + in_file_tif, 'r') 
grid1=np.load(base_in_folder_npy+city_string_in+"/" +city_string_in+in_file_npy)
  
print("#### Loading file done\n")


print("#### Cropping file")


img_clipped =img.read()
# img_clipped=np.array(img)
out_meta = img.meta
out_meta.update({"driver": "GTiff",
             "height": img_clipped.shape[1],
             "width": img_clipped.shape[2],
             "nodata" : 0})

# if interp_switch:
#     img_clipped = np.resize(np.squeeze(img_clipped),(1400,1300))

# noise_classes_old=sorted(np.unique(grid1))
# noise_classes_new=sorted(np.unique(img_clipped))

# counter=0
# for a in noise_classes_old:
#     indexxy = np.where(grid1 ==a)
#     grid1[indexxy]=noise_classes_new[counter]
#     counter=counter+1
    

# grid1=grid1.astype(np.float32)
index0 = np.where(grid1 == img.nodata)
grid1[index0]=0

print("#### Cropping file done \n")

if plot_switch:
    print("#### Plotting file")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # show(img, ax=ax1)
       
    # if clip_switch:
    #     poly_gdf.boundary.plot(ax=ax1, color="red")
    ax1.set_title("TIFF input", fontsize=14)
    ax2.set_title("NPY predictions", fontsize=14)
    ax3.set_title("Difference", fontsize=14)
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # show(img_clipped, ax=ax2)
    im1=ax1.imshow(np.squeeze(img_clipped),cmap="jet")
    # im1.set_clim(50, 76)
    fig.colorbar(im1, orientation='vertical', ax=ax1)
    # plt.colorbar()
    # plt.savefig(base_in_folder+out_grid_file+"_clip.png")
    im2=ax2.imshow(np.squeeze(grid1),cmap="jet")
    # plt.clim(0, 76)
    im2.set_clim(50, 76)
    fig.colorbar(im2, orientation='vertical', ax=ax2)
    
    grid2=np.squeeze(img_clipped)
    grid2[index0]=0
    
    # im3=ax3.imshow(grid2-grid1,cmap="RdBu_r")
    # im3.set_clim(-15, 15)
    # fig.colorbar(im3, orientation='vertical', ax=ax3)
    
    plt.savefig(base_out_folder_pic +city_string_out+out_file_png)
    plt.show()

print("#### Plotting file done \n")


if write_switch:
    print("#### Saving to tif file")
    profile = img.profile

    with rasterio.open(base_in_folder_npy+city_string_in+"/" +city_string_in+out_file_tif, 'w', **out_meta) as dst:
        dst.write(grid1.astype(rasterio.uint8), 1)
    print("#### Saving to tif file done")
    

