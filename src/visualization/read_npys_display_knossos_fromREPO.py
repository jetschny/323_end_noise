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

city_string_in="riga"
city_string_out="RIG"

base_in_folder="/home/sjet/repos/323_end_noise/data/processed/"
base_out_folder="/home/sjet/repos/323_end_noise/data/processed/"

# feature data
# OSM street class, distance to clipped street class
in_grid_file1="_feat_dist2road.npy"
# EU DEM, divergernce from mean topography
in_grid_file2="_feat_eu_dem_v11.npy"
# Copernicus, TreeCoverDensity, distance to TCD
in_grid_file3="_feat_dist2tree.npy"
# OSM maximum speed and number of lanes, merged and smoothed, 0-1
in_grid_file4="_feat_osmmaxspeed_nolanes_smooth.npy"
# Urban atlas land use land cover, reclassified to represent absortpion, 0-10
in_grid_file5="_feat_absoprtion.npy"
# Urban atlas building height 0-x00 m
in_grid_file6="_feat_UA2012_bheight.npy"

#output figure file 
out_file = "_panel_features"

# target noise data
in_grid_target="_target_noise_Aggroad_Lden.npy"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file1)
grid2=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file2)
grid3=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file3)
grid4=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file4)
grid5=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file5)
grid6=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file6)

grid_target=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_target)
# grid_target= grid_target.astype(float)

# x = np.linspace(1, grid1.shape[1], grid1.shape[1])
# y = np.linspace(1, grid1.shape[0], grid1.shape[0])
# X, Y = np.meshgrid(x, y)

if np.isnan(grid1).any():
    print("NAN detetcted in grid1")
if np.isnan(grid2).any():
    print("NAN detetcted in grid2")
if np.isnan(grid3).any():
    print("NAN detetcted in grid3")
if np.isnan(grid4).any():
    print("NAN detetcted in grid4")
if np.isnan(grid5).any():
    print("NAN detetcted in grid5")
if np.isnan(grid6).any():
    print("NAN detetcted in grid6")
if np.isnan(grid_target).any():
    print("NAN detetcted in grid_target")
    
if np.isinf(grid1).any():
    print("INF detetcted in grid1")
if np.isinf(grid2).any():
    print("INF detetcted in grid2")    
if np.isinf(grid3).any():
    print("INF detetcted in grid3")
if np.isinf(grid4).any():
    print("INF detetcted in grid4")
if np.isinf(grid5).any():
    print("INF detetcted in grid5")
if np.isinf(grid6).any():
    print("INF detetcted in grid6") 
if np.isinf(grid_target).any():
    print("INF detetcted in grid_target") 

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
    print("#### Plotting file")
    
    fig, axs = plt.subplots(2, 4, figsize=(25, 15))
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
    im7=axs[1,2].imshow(grid6, cmap=colMap, vmin = 0.2)
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
    im1.set_clim(0.1,2)
    im2.set_clim(-10,10)
    # im3.set_clim(0.1,15)
    im4.set_clim(0.0,0.5)
    im6.set_clim(0,10)
    im7.set_clim(0,10)
    im8.set_clim(30,80)
    
    axs[0,0].set_title('Distance to Road')
    axs[0,1].set_title('Divergence from Topo')
    axs[0,2].set_title('Mean Tree Density')
    axs[0,3].set_title('OSM Street Information')
    axs[1,0].set_title('Absoprtion')
    axs[1,1].set_title('Building Height Density')
    axs[1,2].set_title('Building Height Density')
    axs[1,3].set_title('Target Noise')
    
    
    # x_window=[400, 600]
    # y_window=[1200, 1400]
    x_window=[00, 2512]
    y_window=[00, 2283]
    
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
    plt.savefig(base_out_folder+city_string_in+"/"+city_string_out+out_file+".png")

    ###########################################
    #############plot 3 grids, panel of some, zoom
    
    fig, (axs1, axs2, axs3)  = plt.subplots(1, 3, figsize=(12, 8))
    
    # levels = np.arange(0, 3.5, 0.5)
    
    # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
    # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
    im1=axs1.imshow(grid4[1200:1400,1200:1400], cmap=colMap, vmin = 0.02)
    
    im2=axs2.imshow(grid_target[1200:1400,1200:1400], cmap=colMap, vmin = 0.02)
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
    y_line_slice=1500
    plot1=axs1.plot([0,2512],[y_line_slice, y_line_slice],'-r')
   
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
    
    plot_target=axs1.plot(grid_target[y_line_slice,:]/np.max(grid_target[y_line_slice,:]),"-r")
    plot1=axs1.plot(grid1_plot[y_line_slice,:]/np.max(grid1_plot[y_line_slice,:]),"-b")
    plot2=axs1.plot(grid2_plot[y_line_slice,:]/np.max(grid2_plot[y_line_slice,:]),"-g")
    plot3=axs1.plot(grid3_plot[y_line_slice,:]/np.max(grid3_plot[y_line_slice,:]),"-y")
    plot4=axs1.plot(grid4_plot[y_line_slice,:]/np.max(grid4_plot[y_line_slice,:]),"-m")
    plot5=axs1.plot(grid5_plot[y_line_slice,:]/np.max(grid5_plot[y_line_slice,:]),"-k")
    plot6=axs1.plot(grid6_plot[y_line_slice,:]/np.max(grid6_plot[y_line_slice,:]),"-c")
    plt.legend(["Noise","Distance to Road", 'Divergence from Topo', 'Mean Tree Density',
                'OSM Street Class', 'OSM Number of Lanes', 'OSM Speet Limit'])
    # axs[1,2].set_title('OSM Street FocalStats')
    # axs[1,3].set_title('Modeled Noise')
    
    
    plt.show()
    print("#### Plotting file done \n")


