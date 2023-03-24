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

base_in_folder="/home/sjet/data/323_end_noise/HAN_data/"
base_out_folder="/home/sjet/data/323_end_noise/HAN_data/"
in_file1="DE013L1_HANNOVER_UA2018_v013_clip.npy"
in_file2="OSM_roads_han_streetclass_clip.npy"

out_file="han_road_urbanatlas_osm_merge"

print("#### Loading npy file")

grid1=np.load(base_in_folder+in_file1)
grid=np.asarray(grid1,dtype=np.int32)

grid2=np.load(base_in_folder+in_file2)
grid2=np.asarray(grid2,dtype=np.int32)

write_switch=True


print("#### Loading npy file done")

print("#### Processing file")

grid1_road=grid1*0
grid2_road=grid2*0
indexxy = np.where(grid1 == 12210)
grid1_road[indexxy]=1
indexxy = np.where(grid1 == 12220)
grid1_road[indexxy]=1

indexxy = np.where(grid2 > 0)
grid2_road[indexxy]=1

# side_length=40
# radius_road=divmod(side_length,2)[0]

# grid1_road=np.pad(grid1_road, [radius_road,radius_road], "symmetric")
# dim_grid1_road=grid1_road.shape
# grid1_dist2road=np.zeros(dim_grid1_road)
# distance_matrix=np.zeros([side_length,side_length])

# for indexxy, item in np.ndenumerate(distance_matrix):
#     distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius_road)**2 + (indexxy[1]-radius_road)**2))
    

# def distance2road(indexxy):
#     return np.sum(np.multiply(grid1_road[indexxy[0]-radius_road:indexxy[0]+radius_road,
#                                          indexxy[1]-radius_road:indexxy[1]+radius_road],distance_matrix))

# def checkframe(indexxy):
#     if (min(indexxy)>radius_road)  and (indexxy[0]<(dim_grid1_road[0]-radius_road)) and (indexxy[1]<(dim_grid1_road[1]-radius_road)):
#         return distance2road(indexxy)
#     else:
#         return 0

# for indexxy, item in np.ndenumerate(grid1_road):
#     grid1_dist2road[indexxy]=checkframe(indexxy)

# grid1_dist2road=grid1_dist2road[radius_road:-radius_road, radius_road:-radius_road]
print("#### Processing file done")

print("#### Potting file")

x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)

grid3_road=grid1_road+grid2_road
grid3_road[np.where(grid3_road >1)]=1
                   
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 8))
# con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con2=ax2.contourf(grid1,[5000, 15000, 20000], cmap='RdGy')
im1=ax1.imshow(grid1_road)
im2=ax2.imshow(grid2_road)
im3=ax3.imshow(grid1_road-grid2_road)
im4=ax4.imshow(grid3_road)

# im2=ax2.imshow(grid1_road)
# im3=ax3.imshow(grid1_dist2road)


 # plt.axis('off')
# plt.contourf(grid1)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.colorbar(im3, ax=ax3)
plt.colorbar(im4, ax=ax4)

ax1.set_xlim(150,300)
ax2.set_xlim(150,300)
ax3.set_xlim(150,300)
ax4.set_xlim(150,300)
# ax2.set_xlim(600+radius_road,800+radius_road)
ax1.set_ylim(700,900)
ax2.set_ylim(700,900)
ax3.set_ylim(700,900)
ax4.set_ylim(700,900)
# ax1.set_ylim(1200,1400)
# ax2.set_ylim(1200+radius_road,1400+radius_road)
# ax3.set_ylim(1200,1400)

# im1.set_clim(0.5,1.5)
# im2.set_clim(0.5,1.5)
# im3.set_clim(-0.5,0.5)

ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')
ax4.set_aspect('equal', 'box')

# plt.colorbar(con2, ax=ax2)
plt.show()

print("#### Potting file done")

if write_switch:
    print("#### Saving to npy file")
    out_grid_filename=base_out_folder+out_file+".npy"
    np.save(out_grid_filename,grid3_road)
    print("#### Saving to npy file done")
    
