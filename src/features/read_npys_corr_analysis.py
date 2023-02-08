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


plt.close('all')
plot_switch=True

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend


base_in_folder="BCN_data/"

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

grid1=np.load(base_in_folder+in_grid_file1)
grid2=np.load(base_in_folder+in_grid_file2)
grid3=np.load(base_in_folder+in_grid_file3)
grid4=np.load(base_in_folder+in_grid_file4)
grid5=np.load(base_in_folder+in_grid_file5)
grid6=np.load(base_in_folder+in_grid_file6)
grid7=np.load(base_in_folder+in_grid_file7)

grid_target=np.load(base_in_folder+in_grid_target)
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

# Mask the bad data:
# grid1_masked = np.ma.array(grid1,mask=np.isnan(grid1))

# if plot_switch:
#     fig, axs = plt.subplots(1, 4, figsize=(12, 8))
    
#     levels = np.arange(0, 3.5, 0.5)
    
#     # con1=ax1.contourf(grid1,[20, 50, 70], cmap='RdGy')
#     # con2=ax2.contourf(grid2,[5000, 15000, 20000], cmap='RdGy')
#     im1=axs[0].imshow(grid1, cmap=colMap, vmin = 0.2)
#     im2=axs[1].imshow(grid2, cmap="seismic")
#     im3=axs[2].imshow(grid3, cmap=colMap, vmin = 0.2)
#     im4=axs[3].imshow(grid_target, cmap=colMap, vmin = 0.2)
    
#      # plt.axis('off')
#     # plt.contourf(grid1)
#     plt.colorbar(im1, ax=axs[0])
#     plt.colorbar(im2, ax=axs[1])
#     plt.colorbar(im3, ax=axs[2])
#     plt.colorbar(im4, ax=axs[3])
#     axs[0].set_aspect('equal', 'box')
#     axs[1].set_aspect('equal', 'box')
#     axs[2].set_aspect('equal', 'box')
#     axs[3].set_aspect('equal', 'box')
#     im1.set_clim(0.1,8)
#     im2.set_clim(-10,10)
#     im3.set_clim(0.1,15)
#     im4.set_clim(40,80)
    
#     axs[0].set_title('Distance to Road')
#     axs[1].set_title('Divergence from Topo')
#     axs[2].set_title('Mean Tree Density')
#     axs[3].set_title('Modeled Noise')
    
#     # plt.colorbar(con2, ax=ax2)
#     plt.show()
    
    
# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_target >0)
# indexxy = np.where(grid_target)
# grid_target_flat=grid_target[indexxy].flatten()
#conversion from dB to linear Pascal scale
# grid_target_flat_pa=10**(grid_target_flat/20)*(2e-5)

# grid_target_flat=grid_target.flatten()
# grid1_flat=grid1.flatten()
# grid2_flat=grid2.flatten()
# grid3_flat=grid3.flatten()
# grid4_flat=grid4.flatten()
# grid5_flat=grid5.flatten()
# grid6_flat=grid6.flatten()

# grid1_flat=grid1[indexxy].flatten()
# grid2_flat=grid2[indexxy].flatten()
# grid3_flat=grid3[indexxy].flatten()
# grid4_flat=grid4[indexxy].flatten()
# grid5_flat=grid5[indexxy].flatten()
# grid6_flat=grid6[indexxy].flatten()

# grid1_flat_max=np.max(grid1_flat)
# grid2_flat_max=np.max(grid2_flat)
# grid3_flat_max=np.max(grid3_flat)

# grid_target_flat_max=np.max(grid_target_flat)

# grid1_flat=grid1_flat/grid1_flat_max
# grid2_flat=grid2_flat/grid2_flat_max
# grid3_flat=grid3_flat/grid3_flat_max

# grid_target_flat=grid_target_flat/grid_target_flat_max

# grid_target_flat_pa=10**(grid_target_flat/20)*(2e-5)

df = pd.DataFrame(np.array((grid_target[indexxy].flatten(),
                            grid1[indexxy].flatten(), 
                            grid2[indexxy].flatten(),
                            grid3[indexxy].flatten(), 
                            grid4[indexxy].flatten(), 
                            grid5[indexxy].flatten(),
                            grid6[indexxy].flatten(),
                            grid7[indexxy].flatten()  )).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                           "StreetClass","NLanes","SpeedLimit","RoadFocal"])

if plot_switch:
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()
    
    # fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    # sns.distplot(grid_target[indexxy].flatten(), label="Noise in dB", ax=axs[0])
    # sns.distplot(10**(grid_target[indexxy].flatten()/20)*(2e-5), label="Noise in Pascal", ax=axs[1])
    # axs[0].legend()
    # axs[1].legend()
    # plt.show()
    
    
    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # counter=0
    # for column in ["Dist2Road","DivTopo","MeanTreeDens", "StreetClass","NLanes","SpeedLimit"]:
    #     # print("counter  ",counter)
    #     # print("counter [] ",[np.int((counter-np.mod(counter,3))/3),np.mod(counter,3)])
    #     sns.distplot(df[column], label=column, ax=axs[np.int((counter-np.mod(counter,3))/3),np.mod(counter,3)])
    #     axs[np.int((counter-np.mod(counter,3))/3), np.mod(counter,3)].legend()
    #     counter=counter+1
    # plt.show()

  
    # fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    # for column in df:
    #     sns.kdeplot(df[column], shade=True, label=column)

    # plt.legend()
    # plt.show()


# fig, axs = plt.subplots(1, 1, figsize=(12, 8))
# sns.pairplot(df,hue='Noise', height=2.5)
# plt.show()


