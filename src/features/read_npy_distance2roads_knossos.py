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


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

write_switch=True
plot_switch=True

city_string_in="Riga"
city_string_out="RIG"

base_in_folder="/home/sjet/data/323_end_noise/"
base_out_folder="/home/sjet/data/323_end_noise/"

print("#### Loading npy file")

in_file1="_raw_osm_roads_streetclass.npy"
out_file="_feat_dist2road"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_file1)

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
    if (min(indexxy)>radius_road)  and (indexxy[0]<(dim_grid1[0]-radius_road)) and (indexxy[1]<(dim_grid1[1]-radius_road)):
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
    print("#### Saving to npy file")
    np.save(base_out_folder+city_string_in+"/"+city_string_out+out_file+".npy",grid1_dist2road)
    print("#### Saving to npy file done")
    
    
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
    
    ax1.set_xlim(600,800)
    # ax2.set_xlim(600+radius_road,800+radius_road)
    ax2.set_xlim(600,800)
    ax1.set_ylim(1200,1400)
    # ax2.set_ylim(1200+radius_road,1400+radius_road)
    ax2.set_ylim(1200,1400)
    
    im2.set_clim(0,2)
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    # ax3.set_aspect('equal', 'box')

    # plt.colorbar(con2, ax=ax2)
    plt.show()

    plt.savefig(base_out_folder+city_string_in+"/"+city_string_out+out_file+".png")
    print("#### Potting file done")


