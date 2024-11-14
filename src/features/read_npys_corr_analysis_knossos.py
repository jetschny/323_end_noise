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


city_string_in="Clermont_Ferrand"
city_string_out="CLF"

base_in_folder:  str ="P:/NoiseML/2024/city_data_features/"
base_out_folder: str ="P:/NoiseML/2024/city_data_features/"
base_out_folder_pic: str ="P:/NoiseML/2024/city_data_pics/"

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
out_file1 = "_analysis_correlation"
out_file2 = "_analysis_distribution"

# target noise data
in_grid_target="_target_noise_Aggroad_Lden.npy"

grid1=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file1)
grid2=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file2)
grid3=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file3)
grid4=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file4)
grid5=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file5)
grid6=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_file6)


grid_target=np.load(base_in_folder+city_string_in+"/"+city_string_out+in_grid_target)

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

    
    
# all non-existing / missing data at values of grid_target==0 will be excluded
indexxy = np.where(grid_target >0)


df = pd.DataFrame(np.array((grid_target[indexxy].flatten(),
                            grid1[indexxy].flatten(), 
                            grid2[indexxy].flatten(),
                            grid3[indexxy].flatten(), 
                            grid4[indexxy].flatten(), 
                            grid5[indexxy].flatten(),
                            grid6[indexxy].flatten(),)).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                           "StreetInfo","Absorption","BuildingHeight"])

if plot_switch:
    print("#### Plotting file")
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()
    plt.savefig(base_out_folder_pic+city_string_out+out_file1+".png")
    print("#### Saving image as ", base_out_folder_pic+city_string_out+out_file1+".png")
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    sns.histplot(df.Dist2Road[df.Dist2Road != -999.25], ax=axs[0,0], binwidth=0.05, stat="percent")
    sns.histplot(df.DivTopo[df.DivTopo != -999.25], ax=axs[0,1], binwidth=0.2, stat="percent")
    sns.histplot(df.MeanTreeDens[df.MeanTreeDens != -999.25], ax=axs[0,2], binwidth=1, stat="percent")
    sns.histplot(df.StreetInfo[df.StreetInfo != -999.25], ax=axs[1,0], binwidth=0.01,stat="percent")
    sns.histplot(df.Absorption[df.Absorption != -999.25], ax=axs[1,1], binwidth=1, stat="percent")
    sns.histplot(df.BuildingHeight[df.BuildingHeight != -999.25], ax=axs[1,2], binwidth=0.1, stat="percent")
    axs[0,0].set_xlim([-0.1, 3])
    axs[0,0].set_ylim([0, 10])
    axs[0,0].set_yticks(np.arange(0, 10, 1)) 

    axs[0,1].set_xlim([-10, 10])
    axs[0,1].set_ylim([0, 10])
    axs[0,1].set_yticks(np.arange(0, 10, 1)) 

    axs[0,2].set_xlim([-1, 50])
    axs[0,2].set_ylim([0, 10])
    axs[0,2].set_yticks(np.arange(0, 10, 1)) 
    
    axs[1,0].set_xlim([-0.02, 1])
    axs[1,0].set_ylim([0, 10])
    axs[1,0].set_yticks(np.arange(0, 10, 1)) 

    axs[1,1].set_xlim([-1, 11])
    axs[1,1].set_ylim([0, 65])
    axs[1,1].set_xticks(np.arange(0, 11, 1)) 
    # axs[1,1].set_yticks(np.arange(0, 65, 5)) 

    axs[1,2].set_xlim([-0.2, 10])
    axs[1,2].set_ylim([0, 10])
    axs[1,2].set_yticks(np.arange(0, 10, 1)) 

    plt.show()
    plt.savefig(base_out_folder_pic+city_string_out+out_file2+".png")
    print("#### Saving image as ", base_out_folder_pic+city_string_out+out_file2+".png")
   
    print("#### Plotting file done \n")


