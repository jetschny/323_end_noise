#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:18:23 2022

@author: sjet
"""
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import sys
import numpy as np
import matplotlib.pyplot as plt
# from numpy import newaxis
import rasterio

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
# city_string_in="Madrid"
# city_string_out="MAD" 

city_string_in=sys.argv[1]
city_string_out=sys.argv[2]

print("\n######################## \n")
print("Distance to road feature creation \n")
print("#### Loading file data from city ",city_string_in," (",city_string_out,")")
print("#### Plotting of figures is ",plot_switch," and writing of output files is ",write_switch)

# base_in_folder="/home/sjet/data/323_end_noise/"
# base_out_folder="/home/sjet/data/323_end_noise/"
base_in_folder:  str ="P:/NoiseML/2024/city_data_raw/"
base_out_folder: str ="P:/NoiseML/2024/city_data_features/"
base_out_folder_pic: str ="P:/NoiseML/2024/city_data_pics/"

print("#### Loading npy file")

in_file1="_raw_osm_roads_streetclass.npy"
in_file_target='_MRoadsLden.tif'
out_file="_feat_dist2road"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_file1)
img_target = rasterio.open(base_in_folder+city_string_in +"/" + city_string_in+in_file_target, 'r') 

print("#### Loading npy file done")

print("#### Processing file")

side_length=40
radius_road=divmod(side_length,2)[0]

grid1=grid1.astype(np.float32)
# making all data with "no data value" 0
index0 = np.where(grid1 == np.min(grid1))
grid1[index0]=0

# only needed when OSM inpout is used, not merged with UA data
indexxy = np.where(grid1 > 0)
grid1[indexxy]=1

grid1=np.pad(grid1, [radius_road,radius_road], "symmetric")
dim_grid1=grid1.shape
grid1_dist2road=np.zeros(dim_grid1)
distance_matrix=np.zeros([side_length,side_length])

for indexxy, item in np.ndenumerate(distance_matrix):
    distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius_road)**2 + (indexxy[1]-radius_road)**2))
    

def distance2road(indexxy):
    return np.sum(np.multiply(grid1[indexxy[0]-radius_road:indexxy[0]+radius_road,
                                         indexxy[1]-radius_road:indexxy[1]+radius_road],distance_matrix))

def checkframe(indexxy):
    if (min(indexxy)>=radius_road)  and (indexxy[0]<(dim_grid1[0]-radius_road)) and (indexxy[1]<(dim_grid1[1]-radius_road)):
        return distance2road(indexxy)
    else:
        return 0

for indexxy, item in np.ndenumerate(grid1):
    grid1_dist2road[indexxy]=checkframe(indexxy)

# grid1_dist2road=grid1
# removing padding area
grid1_dist2road=grid1_dist2road[radius_road:-radius_road, radius_road:-radius_road]
# cropping to initial range of input data
# re-inserting no data value
# grid1_dist2road[index0]=-999.25
index0 = np.where(grid1_dist2road <= 0)
grid1_dist2road[index0]=-999.25
grid1_dist2road=grid1_dist2road.astype(np.float32)


print("#### Processing file done")

print("#### Potting file")

x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)

if write_switch:
    print("#### Saving to output files")
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid1_dist2road)

    print("... Saving to npy file done")
    
    out_meta = img_target.meta.copy()
    # epsg_code = int(img.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "dtype" : 'float32',
                     "nodata" : -999.25,
                     "crs": img_target.crs})
    with rasterio.open(base_out_folder+city_string_in+"/"+city_string_out+out_file+".tif", "w", **out_meta) as dest:
        dest.write(grid1_dist2road[np.newaxis,:,:])
        
    print("... Saving to tiff file done")
    
if plot_switch:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
    # con2=ax2.contourf(grid1,[5000, 15000, 20000], cmap='RdGy')
    im1=ax1.imshow(grid1)
    # im2=ax2.imshow(grid1_road)
    im2=ax2.imshow(grid1_dist2road)
    
    
     # plt.axis('off')
    # plt.contourf(grid1)
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    # plt.colorbar(im3, ax=ax3)
    
    # ax1.set_xlim(600,800)
    # # ax2.set_xlim(600+radius_road,800+radius_road)
    # ax2.set_xlim(600,800)
    # ax1.set_ylim(1200,1400)
    # # ax2.set_ylim(1200+radius_road,1400+radius_road)
    # ax2.set_ylim(1200,1400)
    
    im2.set_clim(0,2)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    # ax3.set_aspect('equal', 'box')

    # plt.colorbar(con2, ax=ax2)
    

    plt.savefig(base_out_folder_pic+"/"+city_string_out+out_file+".png")
    plt.show()
    print("#### Potting file done")


