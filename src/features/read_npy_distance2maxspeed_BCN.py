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

print("#### Loading npy file")

base_in_folder="/home/sjet/data/323_end_noise/BCN_data/"
base_out_folder="/home/sjet/data/323_end_noise/BCN_data/"

in_file="OSM_roads_bcn_maxspeed_clipfill.npy"
out_file="OSM_roads_bcn_maxspeed_clipfilldist"

grid1=np.load(base_in_folder+in_file)

write_switch=True

# grid1_trees=grid1*0
# indexxy = np.where(grid1 == 12210)
# grid1_trees[indexxy]=2
# indexxy = np.where(grid1 == 12220)
# grid1_trees[indexxy]=1
print("#### Loading npy file done")

print("#### Processing file")


side_length=5
radius=divmod(side_length,2)[0]

grid1_pad=np.pad(grid1, [radius,radius], "symmetric")
dim_grid1_pad=grid1_pad.shape
grid1_distance=np.zeros(dim_grid1_pad)
distance_matrix=np.zeros([radius,radius])

# for indexxy, item in np.ndenumerate(distance_matrix):
#     distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius)**2 + (indexxy[1]-radius)**2))
    

def calc_distance(indexxy):
    return np.mean(grid1_pad[indexxy[0]-radius:indexxy[0]+radius,indexxy[1]-radius:indexxy[1]+radius])  
  
def check_frame(indexxy):
    if (min(indexxy)>radius) and (indexxy[0]<(dim_grid1_pad[0]-radius)) and (indexxy[1]<(dim_grid1_pad[1]-radius)):
        return calc_distance(indexxy)
    else:
        return 0
       
for indexxy, item in np.ndenumerate(grid1_pad):
    grid1_distance[indexxy]=check_frame(indexxy)

grid1_distance=grid1_distance[radius:-radius, radius:-radius]
# grid1_distance_scaled=grid1_distance*(np.median(grid1[np.where(grid1 >0)])/np.median(grid1_distance[np.where(grid1_distance >0)]))
print("#### Processing file done")

print("#### Potting file")

x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con2=ax2.contourf(grid1,[5000, 15000, 20000], cmap='RdGy')
# im1=ax1.imshow(grid1)
im1=ax1.imshow(grid1)
im2=ax2.imshow(grid1_distance)
im1.set_clim(0, np.max(grid1)*2/3)
im2.set_clim(0, np.max(grid1_distance)*2/3)

y_slice=800 
plot1=ax1.plot([0,1300],[y_slice, y_slice],'-r')
plot1=ax2.plot([0,1300],[y_slice, y_slice],'-r')

# im3.set_clim(0, 100)

 # plt.axis('off')
# plt.contourf(grid1)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
# plt.colorbar(im3, ax=ax3)

ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
# ax3.set_aspect('equal', 'box')

# ax1.set_xlim(600,800)
# ax2.set_xlim(600,800)

# ax1.set_ylim(950,1150)
# ax2.set_ylim(950,1150)

# im2.set_clim(0,2)

# plt.colorbar(con2, ax=ax2)
plt.show()

fig, (axs1)  = plt.subplots(1, 1, figsize=(20, 8))
  

# plot4=axs1.plot(grid4_plot[y_slice,:],"-m")
plot1=axs1.plot(grid1[y_slice,:],"-k")
plot2=axs1.plot(grid1_distance[y_slice,:],"-c")
# plot7=axs1.plot(grid7_plot[y_slice,:],"-y")


plt.legend(['OSM Speet Limit', 'OSM Speet Limit proc'], loc="upper left")


axs1.set_ylabel("OSM Max Speed",fontsize=14)
axs1.set_ylim([0, 60])


plt.show()
 
print("#### Potting file done")


if write_switch:
    print("#### Saving to npy file")
    out_grid_filename=base_out_folder+out_file+".npy"
    np.save(out_grid_filename,grid1_distance)
    print("#### Saving to npy file done")
    
