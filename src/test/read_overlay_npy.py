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

in_grid_file1="IMD_2018_010m_E36N20_03035_v020_clip.npy"
in_grid_file2="ES002L2_BARCELONA_UA2018_v013_clip.npy"
in_grid_file3="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"

grid1=np.load(in_grid_file1)
# grid1=grid1.flatten()
grid2=np.load(in_grid_file2)
grid3=np.load(in_grid_file3)

# grid2=np.pad(grid2, [(0, 1grid2.shape[0]-grid2.shape[0]), (0,0)], mode=constant)
# grid3=np.pad(grid3, [(0, grid1.shape[0]-grid3.shape[0]), (0,0)], mode="constant")



x = np.linspace(1, grid1.shape[1], grid1.shape[1])
y = np.linspace(1, grid1.shape[0], grid1.shape[0])
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
# con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
im1=ax1.imshow(grid1)
im2=ax2.imshow(grid2)
im3=ax3.imshow(grid3)

 # plt.axis('off')
# plt.contourf(grid1)
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.colorbar(im3, ax=ax3)
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')
# plt.colorbar(con2, ax=ax2)
plt.show()


fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
# im1=ax1.imshow(grid2)
# con1=ax2.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy', origin="upper")
# con2=ax1.contourf(grid2,[0, 25000, 50000], cmap='RdGy', origin="upper")
con3=ax1.contourf(grid3,[0, 5, 10], cmap='RdGy', origin="upper")


# im2=ax3.imshow(grid2)
# con2=ax3.contourf(X,Y,grid1,[20, 50, 70], cmap='RdGy')
#  # plt.axis('off')
# # plt.contourf(grid1)
# # plt.colorbar(con1, ax=ax1)

ax1.set_aspect('equal', 'box')
# ax2.set_aspect('equal', 'box')
# ax3.set_aspect('equal', 'box')

plt.colorbar(con3, ax=ax1)
# plt.colorbar(con2, ax=ax3)

plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
# im1=ax1.imshow(grid2)
# # con1=ax2.contourf(grid1,[20, 50, 70], cmap='RdGy')
# con1=ax2.contourf(grid1,[20, 50, 70], cmap='RdGy', origin="upper")


# im2=ax3.imshow(grid2)
# con2=ax3.contourf(X,Y,grid1,[20, 50, 70], cmap='RdGy')
#  # plt.axis('off')
# # plt.contourf(grid1)
# # plt.colorbar(con1, ax=ax1)

# ax1.set_aspect('equal', 'box')
# ax2.set_aspect('equal', 'box')
# ax3.set_aspect('equal', 'box')

# # plt.colorbar(con1, ax=ax2)
# # plt.colorbar(con2, ax=ax3)

# plt.show()