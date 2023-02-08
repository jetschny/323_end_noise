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

base_folder="HAN_data"
print("#### Loading npy file")
in_grid_file1="han_road_urbanatlas_osm_merge.npy"
grid1=np.load(base_folder+"/"+in_grid_file1)

write_switch=True
out_grid_file="han_dist2road_urbanatlas_osm_merge"

print("#### Loading npy file done")

print("#### Processing file")

# grid1_road=grid1*0
# indexxy = np.where(grid1 == 12210)
# grid1_road[indexxy]=1
# indexxy = np.where(grid1 == 12220)
# grid1_road[indexxy]=1
grid1_road=grid1

side_length=40
radius_road=divmod(side_length,2)[0]

grid1_road=np.pad(grid1_road, [radius_road,radius_road], "symmetric")
dim_grid1_road=grid1_road.shape
grid1_dist2road=np.zeros(dim_grid1_road)
distance_matrix=np.zeros([side_length,side_length])

for indexxy, item in np.ndenumerate(distance_matrix):
    distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius_road)**2 + (indexxy[1]-radius_road)**2))
    

def distance2road(indexxy):
    return np.sum(np.multiply(grid1_road[indexxy[0]-radius_road:indexxy[0]+radius_road,
                                         indexxy[1]-radius_road:indexxy[1]+radius_road],distance_matrix))

def checkframe(indexxy):
    if (min(indexxy)>radius_road)  and (indexxy[0]<(dim_grid1_road[0]-radius_road)) and (indexxy[1]<(dim_grid1_road[1]-radius_road)):
        return distance2road(indexxy)
    else:
        return 0

for indexxy, item in np.ndenumerate(grid1_road):
    grid1_dist2road[indexxy]=checkframe(indexxy)

grid1_dist2road=grid1_dist2road[radius_road:-radius_road, radius_road:-radius_road]
print("#### Processing file done")

print("#### Potting file")

x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
# con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con2=ax2.contourf(grid1,[5000, 15000, 20000], cmap='RdGy')
im1=ax1.imshow(grid1)
im2=ax2.imshow(grid1_road)
im3=ax3.imshow(grid1_dist2road)


 # plt.axis('off')
# plt.contourf(grid1)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.colorbar(im3, ax=ax3)

ax1.set_xlim(600,800)
ax2.set_xlim(600+radius_road,800+radius_road)
ax3.set_xlim(600,800)
ax1.set_ylim(1200,1400)
ax2.set_ylim(1200+radius_road,1400+radius_road)
ax3.set_ylim(1200,1400)

im3.set_clim(0,2)

ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

# plt.colorbar(con2, ax=ax2)
plt.show()

print("#### Potting file done")

if write_switch:
    print("#### Saving to npy file")
    out_grid_file=out_grid_file+".npy"
    np.save(base_folder+"/"+out_grid_file,grid1_dist2road)
    print("#### Saving to npy file done")
    
