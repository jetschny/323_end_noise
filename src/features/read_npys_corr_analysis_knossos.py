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


city_string_in="Riga"
city_string_out="RIG"

base_in_folder="/home/sjet/data/323_end_noise/"
base_out_folder="/home/sjet/data/323_end_noise/"

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
out_file = "_correlation_analysis"

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
    plt.savefig(base_out_folder+city_string_in+"/"+city_string_out+out_file+".png")
    print("#### Plotting file done \n")
   

