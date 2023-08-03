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

############# missing : define not a value number!!!!
write_switch=False



base_in_folder="/home/sjet/data/323_end_noise/Riga/"
base_out_folder="/home/sjet/data/323_end_noise/Riga/"

print("#### Loading npy file")
in_grid_file1="RIG_raw_ua_road_clip.npy"
grid1=np.load(base_in_folder+in_grid_file1)
# grid1=np.asarray(grid1,dtype=np.int32)

in_grid_file2="RIG_raw_osm_roads_streetclass_clip.npy"
grid2=np.load(base_in_folder+in_grid_file2)
# grid2=np.asarray(grid2,dtype=np.int32)

out_grid_file="RIG_raw_merged_road_urbanatlas_osm"

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

# ax1.set_xlim(150,300)
# ax2.set_xlim(150,300)
# ax3.set_xlim(150,300)
# ax4.set_xlim(150,300)
# ax2.set_xlim(600+radius_road,800+radius_road)
# ax1.set_ylim(700,900)
# ax2.set_ylim(700,900)
# ax3.set_ylim(700,900)
# ax4.set_ylim(700,900)
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

############# missing : define not a value number!!!!

print("#### Potting file done")

if write_switch:
    print("#### Saving to npy file")
    out_grid_file=out_grid_file+".npy"
    np.save(base_out_folder+"/"+out_grid_file,grid3_road)
    print("#### Saving to npy file done")
    
