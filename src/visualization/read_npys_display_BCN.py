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

base_folder="BCN_data"

in_grid_file1="bcn_dist2road_urbanatlas_osm_merge.npy"
in_grid_file2="bcn_distance2topo_dem.npy"
in_grid_file3="bcn_distance2trees_tcd.npy"
in_grid_file4="OSM_roads_bcn_streetclass_clip.npy"
# in_grid_file5="OSM_roads_bcn_nlanes_clip_smooth.npy"
in_grid_file5="OSM_roads_bcn_nlanes_clipfill_kde15.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file6="OSM_roads_bcn_maxspeed_clipfill_kde15.npy"
in_grid_file7="bcn_road_focalstats50_clip.npy"

in_grid_target="2017_isofones_total_dia_mapa_estrategic_soroll_bcn_clip.npy"

grid1=np.load(base_folder+"/"+in_grid_file1)
grid2=np.load(base_folder+"/"+in_grid_file2)
grid3=np.load(base_folder+"/"+in_grid_file3)
grid4=np.load(base_folder+"/"+in_grid_file4)
grid5=np.load(base_folder+"/"+in_grid_file5)
grid6=np.load(base_folder+"/"+in_grid_file6)
grid7=np.load(base_folder+"/"+in_grid_file7)

grid_target=np.load(base_folder+"/"+in_grid_target)
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
    
    ###########################################
    #############plot all grids, panel of all
    
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))
    plt.rcParams['axes.grid'] = False
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
    im1.set_clim(0.1,8)
    im2.set_clim(-10,10)
    im3.set_clim(0.1,15)
    im7.set_clim(5,400)
    
    axs[0,0].set_title('Distance to Road')
    axs[0,1].set_title('Divergence from Topo')
    axs[0,2].set_title('Mean Tree Density')
    axs[0,3].set_title('OSM Street Class')
    axs[1,0].set_title('OSM Number of Lanes')
    axs[1,1].set_title('OSM Speet Limit')
    axs[1,2].set_title('OSM Street FocalStats')
    axs[1,3].set_title('Modeled Noise')
    
    x_window=[400, 600]
    y_window=[1200, 1400]
    axs[0,0].set_xlim(x_window)
    axs[0,1].set_xlim(x_window)
    axs[0,2].set_xlim(x_window)
    axs[0,3].set_xlim(x_window)
    axs[1,0].set_xlim(x_window)
    axs[1,1].set_xlim(x_window)
    axs[1,2].set_xlim(x_window)
    axs[1,3].set_xlim(x_window)
    
    axs[0,0].set_ylim(y_window)
    axs[0,1].set_ylim(y_window)
    axs[0,2].set_ylim(y_window)
    axs[0,3].set_ylim(y_window)
    axs[1,0].set_ylim(y_window)
    axs[1,1].set_ylim(y_window)
    axs[1,2].set_ylim(y_window)
    axs[1,3].set_ylim(y_window)
    
    
    # plt.colorbar(con2, ax=ax2)
    plt.show()


    ###########################################
    #############plot 3 grids, panel of some, zoom
    
    fig, (axs1, axs2, axs3)  = plt.subplots(1, 3, figsize=(12, 8))
    
    # levels = np.arange(0, 3.5, 0.5)
    
    # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
    # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
    im1=axs1.imshow(grid4[600:660,420:480], cmap=colMap, vmin = 0.2)
    
    im2=axs2.imshow(grid_target[600:660,420:480], cmap=colMap, vmin = 0.2)
    grid_target_crop=grid_target#[14:1300,0:1487]
    grid4_crop=grid4#[0:1286,13:1500]
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
    
    ###########################################
    #############plot one grid
    
    fig, (axs1)  = plt.subplots(1, 1, figsize=(12, 8))
    
    im1=axs1.imshow(grid_target, cmap=colMap, vmin = 0.2)
    plot1=axs1.plot([0,1300],[800, 800],'-r')
   
    plt.colorbar(im1, ax=axs1)
 
    axs1.set_aspect('equal', 'box')
    axs1.set_title('Modeled Noise')
    

    plt.show()
    
    
    fig, (axs1)  = plt.subplots(1, 1, figsize=(12, 8))
    
    indexxy = np.where(grid_target >0)
    grid1_plot=np.zeros(grid_target.shape)
    grid2_plot=np.zeros(grid_target.shape)
    grid3_plot=np.zeros(grid_target.shape)
    grid4_plot=np.zeros(grid_target.shape)
    grid5_plot=np.zeros(grid_target.shape)
    grid6_plot=np.zeros(grid_target.shape)
    grid1_plot[indexxy]=grid1[indexxy]
    grid2_plot[indexxy]=grid2[indexxy]
    grid3_plot[indexxy]=grid3[indexxy]
    grid4_plot[indexxy]=grid4[indexxy]
    grid5_plot[indexxy]=grid5[indexxy]
    grid6_plot[indexxy]=grid6[indexxy]
    
    plot_target=axs1.plot(grid_target[800,:]/np.max(grid_target[800,:]),"-r")
    plot1=axs1.plot(grid1_plot[800,:]/np.max(grid1_plot[800,:]),"-b")
    plot2=axs1.plot(grid2_plot[800,:]/np.max(grid2_plot[800,:]),"-g")
    plot3=axs1.plot(grid3_plot[800,:]/np.max(grid3_plot[800,:]),"-y")
    plot4=axs1.plot(grid4_plot[800,:]/np.max(grid4_plot[800,:]),"-m")
    plot5=axs1.plot(grid5_plot[800,:]/np.max(grid5_plot[800,:]),"-k")
    plot6=axs1.plot(grid6_plot[800,:]/np.max(grid6_plot[800,:]),"-c")
    plt.legend(["Noise","Distance to Road", 'Divergence from Topo', 'Mean Tree Density',
                'OSM Street Class', 'OSM Number of Lanes', 'OSM Speet Limit'])
    # axs[1,2].set_title('OSM Street FocalStats')
    # axs[1,3].set_title('Modeled Noise')
    
    
    plt.show()
    


