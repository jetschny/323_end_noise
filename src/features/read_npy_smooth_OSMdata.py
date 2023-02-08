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
# from scipy.interpolate import griddata
import scipy.stats as st
from sklearn.neighbors import KernelDensity

plt.rc('image', cmap='jet')

plt.close('all')

print("#### Loading npy file")

# define in and output names
# in_grid_file1="OSM_roads_bcn_nlanes_clip.npy"
# in_grid_file1="OSM_roads_bcn_streetclass_clip.npy"
in_grid_file1="OSM_roads_bcn_maxspeed_clip.npy"

# out_grid_file1="OSM_roads_bcn_nlanes_clip_smooth"
# out_grid_file1="OSM_roads_bcn_streetclass_clip_smooth"
out_grid_file1="OSM_roads_bcn_maxspeed_clip_smooth"

#decide whether or not to write output data to file
write_switch=True


# load numpy array
grid1=np.load(in_grid_file1)


print("#### Loading npy file done")

print("#### Processing file")

# define square size in grid units for windows processing
side_length=25
radius=divmod(side_length,2)[0]

# indices of input array ==0 (-> to be filled)
indexxy1 = np.where(grid1 ==0)
# indices of input array >0 (-> stays at it is)
indexxy2 = np.where(grid1 >0)

# pading of array to allow for corner point processing
grid1_pad=np.pad(grid1, [radius,radius], "symmetric")

#creating new array og the same size of the padded array
dim_grid1=grid1.shape
dim_grid1_pad=grid1_pad.shape
grid1_distance=np.zeros(dim_grid1_pad)


# # apply a process to a chosen window
# def calc_distance(indexxy):
#     return np.mean(grid1_pad[indexxy[0]-radius:indexxy[0]+radius,indexxy[1]-radius:indexxy[1]+radius])  
  
# # bounary check, nothing happens if within padded area
# def check_frame(indexxy):
#     if (min(indexxy)>radius) and (indexxy[0]<(dim_grid1_pad[0]-radius)) and (indexxy[1]<(dim_grid1_pad[1]-radius)):
#         return calc_distance(indexxy)
#     else:
#         return 0


# for indexxy, item in np.ndenumerate(grid1_pad):
#     grid1_distance[indexxy]=check_frame(indexxy)


# grid1_distance=grid1_distance[radius:-radius, radius:-radius]
# grid1_distance[indexxy2]=grid1[indexxy2]

x=indexxy2[0]
y=indexxy2[1]
z=grid1[indexxy2]

x_new=np.linspace(0,dim_grid1[1]-1,dim_grid1[1])
y_new=np.linspace(0,dim_grid1[0]-1,dim_grid1[0])
x_grid, y_grid = np.meshgrid(x_new, y_new)

xy = np.vstack([x,y])
d = xy.shape[0]
n = xy.shape[1]
bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
#bw = n**(-1./(d+4)) # scott
print('bw: {}'.format(bw))

kde = KernelDensity(bandwidth=bw, metric='euclidean', kernel='gaussian', algorithm='ball_tree')
kde.fit(xy.T)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()

X, Y = np.mgrid[xmin:xmax:1400j, ymin:ymax:1300j]
positions = np.vstack([X.ravel(), Y.ravel()])

grid1_kde = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)
   
# grid1_interp = griddata((y,x), z, (x_grid, y_grid), method='linear')

# Peform the kernel density estimate
# xx, yy = np.mgrid[x_new, y_new]
# positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
# values = np.vstack([x, y])
# kernel = st.gaussian_kde(values)
# grid1_kde = np.reshape(kernel(positions).T, x_grid.shape)



print("#### Processing file done")

print("#### Potting file")

# x = np.linspace(1, grid1.shape[1], grid1.shape[1])
# y = np.linspace(1, grid1.shape[0], grid1.shape[0])
# X, Y = np.meshgrid(x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

im1=ax1.imshow(grid1)
im2=ax2.imshow(grid1_kde)

# ax2.scatter(x, y, c='k', s=5)
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


im1.set_clim(0,50)
# im2.set_clim(0,50)

plt.show()

print("#### Potting file done")

if write_switch:
    print("#### Saving to npy file")
    out_grid_file=out_grid_file1+".npy"
    np.save(out_grid_file,grid1_distance)
    print("#### Saving to npy file done")
    
