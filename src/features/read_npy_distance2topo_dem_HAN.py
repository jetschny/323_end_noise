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

base_in_folder="/home/sjet/data/323_end_noise/HAN_data/"
base_out_folder="/home/sjet/data/323_end_noise/HAN_data/"
in_file = 'Hannover_eu_dem_v11_E40N30_clip.npy'
out_file ='han_distance2topo_dem'

grid1=np.load(base_in_folder+in_file)

write_switch=True


print("#### Loading npy file done")

print("#### Processing file")


side_length=25
radius=divmod(side_length,2)[0]

indexxy = np.where(grid1 <0)
grid1[indexxy]=0


grid1_pad=np.pad(grid1, [radius,radius], "symmetric")
dim_grid1_pad=grid1_pad.shape
grid1_distance=np.zeros(dim_grid1_pad)
distance_matrix=np.zeros([radius,radius])

# for indexxy, item in np.ndenumerate(distance_matrix):
#     distance_matrix[indexxy]=1/np.exp(np.sqrt((indexxy[0]-radius)**2 + (indexxy[1]-radius)**2))
    

def calc_distance(indexxy):
    return grid1_pad[indexxy]-np.mean(grid1_pad[indexxy[0]-radius:indexxy[0]+radius,indexxy[1]-radius:indexxy[1]+radius])  
  
def check_frame(indexxy):
    if (min(indexxy)>radius) and (indexxy[0]<(dim_grid1_pad[0]-radius)) and (indexxy[1]<(dim_grid1_pad[1]-radius)):
        return calc_distance(indexxy)
    else:
        return 0
       
for indexxy, item in np.ndenumerate(grid1_pad):
    grid1_distance[indexxy]=check_frame(indexxy)

grid1_distance=grid1_distance[radius:-radius, radius:-radius]
print("#### Processing file done")

print("#### Potting file")

x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

im1=ax1.imshow(grid1)
im2=ax2.imshow(grid1_distance)
# im1.set_clim(0, 100)
# im2.set_clim(0, 100)


 # plt.axis('off')
# plt.contourf(grid1)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
# plt.colorbar(im3, ax=ax3)

ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
# ax3.set_aspect('equal', 'box')



# ax1.set_xlim(700,900)
# ax2.set_xlim(700,900)

# ax1.set_ylim(1000,1200)
# ax2.set_ylim(1000,1200)


im1.set_clim(0,300)
im2.set_clim(-15,15)

plt.show()

print("#### Potting file done")

if write_switch:
    print("#### Saving to npy file")
    
    np.save(base_out_folder+out_file+".npy",grid1_distance)
    print("#### Saving to npy file done")
    
