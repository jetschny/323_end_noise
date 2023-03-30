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
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


plt.close('all')
plot_switch=True

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

base_folder="/home/sjet/data/323_end_noise/HAN_data/"

in_grid_file1="han_dist2road_urbanatlas_osm_merge.npy"
in_grid_file2="han_distance2topo_dem.npy"
in_grid_file3="han_distance2trees_tcd.npy"
in_grid_file4="OSM_roads_han_streetclass_clip.npy"
in_grid_file5="OSM_roads_han_nlanes_clipfill.npy"
# in_grid_file5="OSM_roads_han_nlanes_clipfill_kde15.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file6="OSM_roads_han_maxspeed_clipfill_kde15.npy"
in_grid_file7="han_distance2buildings_bcd.npy"
# in_grid_file7="bcn_road_focalstats50_clip.npy"

# in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"
in_grid_target="han_noise_Aggroad_Lden_clip.npy"

grid1=np.load(base_folder+in_grid_file1)
grid2=np.load(base_folder+in_grid_file2)
grid3=np.load(base_folder+in_grid_file3)
grid4=np.load(base_folder+in_grid_file4)
grid5=np.load(base_folder+in_grid_file5)
grid6=np.load(base_folder+in_grid_file6)
grid7=np.load(base_folder+in_grid_file7)

grid_target=np.load(base_folder+in_grid_target)
grid_target= grid_target.astype(float)

# x = np.linspace(1, grid1.shape[1], grid1.shape[1])
# y = np.linspace(1, grid1.shape[0], grid1.shape[0])
# X, Y = np.meshgrid(x, y)


# values of 0 in grid_target represent not-existing/missing data
# all values in the features at the same indices will be blanked out
# indexxy = np.where(grid_target == 0)
# # grid_target[indexxy]=0
# grid1[indexxy]=0
# grid2[indexxy]=0
# grid3[indexxy]=0
# grid4[indexxy]=0
# grid5[indexxy]=0
# grid6[indexxy]=0

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

if plot_switch:
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    
    # levels = np.arange(0, 3.5, 0.5)
    
    # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
    # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
    im1=axs[0,0].imshow(grid1, cmap=colMap, vmin = 0.2)
    im2=axs[0,1].imshow(grid2, cmap="seismic")
    im3=axs[0,2].imshow(grid3, cmap=colMap, vmin = 0.2)
    im4=axs[0,3].imshow(grid4, cmap=colMap, vmin = 0.2)
    
    im5=axs[1,0].imshow(grid5, cmap=colMap, vmin = 0.2)
    im6=axs[1,1].imshow(grid6, cmap=colMap, vmin = 0.2)
    im7=axs[1,2].imshow(grid7, cmap=colMap, vmin = 0.2)
    im8=axs[1,3].imshow(grid_target, cmap=colMap, vmin = 0.2)
    
      # plt.axis('off')
    # plt.contourf(grid1)
    plt.colorbar(im1, ax=axs[0,0])
    plt.colorbar(im2, ax=axs[0,1])
    plt.colorbar(im3, ax=axs[0,2])
    plt.colorbar(im4, ax=axs[0,3])
    plt.colorbar(im5, ax=axs[1,0])
    plt.colorbar(im6, ax=axs[1,1])
    plt.colorbar(im7, ax=axs[1,2])
    plt.colorbar(im8, ax=axs[1,3])
    
    axs[0,0].set_aspect('equal', 'box')
    axs[0,1].set_aspect('equal', 'box')
    axs[0,2].set_aspect('equal', 'box')
    axs[0,3].set_aspect('equal', 'box')
    axs[1,0].set_aspect('equal', 'box')
    axs[1,1].set_aspect('equal', 'box')
    axs[1,2].set_aspect('equal', 'box')
    axs[1,3].set_aspect('equal', 'box')
    # im1.set_clim(0.1,8)
    im2.set_clim(-10,10)
    im3.set_clim(0.1,15)
    # im7.set_clim(5,400)
    
    axs[0,0].set_title('Distance to Road')
    axs[0,1].set_title('Divergence from Topo')
    axs[0,2].set_title('Mean Tree Density')
    axs[0,3].set_title('OSM Street Class')
    axs[1,0].set_title('OSM Number of Lanes')
    axs[1,1].set_title('OSM Speet Limit')
    axs[1,2].set_title('Mean Building Height density')
    axs[1,3].set_title('Modeled Noise')
    
    # plt.colorbar(con2, ax=ax2)
    plt.show()
    
    fig, (axs1, axs2, axs3)  = plt.subplots(1, 3, figsize=(12, 8))
    
    # levels = np.arange(0, 3.5, 0.5)
    
    # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
    # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
    im1=axs1.imshow(grid4, cmap=colMap, vmin = 0.2)
    
    im2=axs2.imshow(grid_target, cmap=colMap, vmin = 0.2)
    grid_target_crop=grid_target[0:1500,0:1900]
    grid4_crop=grid4
    grid_qc=np.where(grid4_crop>0, 2,0)+np.where(grid_target_crop>60, 1,0)
    im3=axs3.imshow(grid_qc, cmap=colMap, vmin = 0.2)
      # plt.axis('off')
    # plt.contourf(grid1)
    plt.colorbar(im1, ax=axs1)
   
    plt.colorbar(im2, ax=axs2)
    
    axs1.set_aspect('equal', 'box')
    axs2.set_aspect('equal', 'box')
    axs3.set_aspect('equal', 'box')
    # im1.set_clim(0.1,8)
    # im2.set_clim(-10,10)
    # im3.set_clim(0.1,15)
    # im7.set_clim(5,400)
    
    
    
    axs1.set_title('Distance to Road')
  
    axs2.set_title('Modeled Noise')
    
    # plt.colorbar(con2, ax=ax2)
    plt.show()
    
    