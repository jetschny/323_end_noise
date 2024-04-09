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
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# from sklearn.ensemble import RandomForestRegressor


plt.close('all')
plot_switch=True

default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

# city_string_in="Vienna" #"Pilsen" #"Clermont_Ferrand" #"Riga"
# city_string_out="VIE" #"PIL" #"CLF" #"RIG"
city_string_in  = "Riga"
city_string_out = "RIG"

# base_in_folder="/home/sjet/repos/323_end_noise/data/processed/"
# base_out_folder="/home/sjet/repos/323_end_noise/reports/figures/"
base_in_folder:  str ="C:/Users/jetschny/Documents/repos/323_end_noise/data/processed/"
base_out_folder: str ="C:/Users/jetschny/Documents/repos/323_end_noise/reports/figures/"

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

grid_target=np.load(base_in_folder+city_string_in+"\\"+city_string_out+in_grid_target)
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

indexxy = np.where(grid_target >0)

notavlaue=-999.25
grid1[np.where(grid1 == notavlaue)]=np.NAN
grid2[np.where(grid2 == notavlaue)]=np.NAN
grid3[np.where(grid3 == notavlaue)]=np.NAN
grid4[np.where(grid4 == notavlaue)]=np.NAN
grid5[np.where(grid5 == notavlaue)]=np.NAN
grid6[np.where(grid6 == notavlaue)]=np.NAN

grid2[np.where(grid2> 20)]=np.NAN
grid2[np.where(grid2< -20)]=np.NAN

df = pd.DataFrame(np.array((grid_target[indexxy].flatten(order='C'),
                            grid1[indexxy].flatten(order='C'), 
                            grid2[indexxy].flatten(order='C'),
                            grid3[indexxy].flatten(order='C'), 
                            grid4[indexxy].flatten(order='C'), 
                            grid5[indexxy].flatten(order='C'),
                            grid6[indexxy].flatten(order='C'))).transpose(), 
                  columns=["Noise","Dist2Road","DivTopo","MeanTreeDens",
                            "StreetInfo","Absortion","BuildingHeight"])

# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

if plot_switch:
    
    # ###########################################
    # #############plot all grids, panel of all
    # print("#### Plotting file")
    
    n_bins=30
    
    fig, axs = plt.subplots(2, 3, figsize=(25, 15))
    sns.histplot(data=df, x="Dist2Road", ax=axs[0,0], kde=True, bins=n_bins)
    sns.histplot(data=df, x="DivTopo", ax=axs[0,1],kde=True, bins=60)
    sns.histplot(data=df, x="MeanTreeDens", ax=axs[0,2], kde=True, bins=n_bins)
    sns.histplot(data=df, x="StreetInfo", ax=axs[1,0], kde=True, bins=n_bins)
    sns.histplot(data=df, x="Absortion", ax=axs[1,1], kde=True, bins=n_bins)
    sns.histplot(data=df, x="BuildingHeight", ax=axs[1,2], kde=True, bins=n_bins)
    
    axs[0,0].set_xlim([-0.1, 5.1])
    axs[0,1].set_xlim([-6, 6])
    axs[0,2].set_xlim([-1, 51])
    
    axs[1,0].set_xlim([-0.01, 0.51])
    axs[1,1].set_xlim([-1, 11])
    axs[1,2].set_xlim([-1, 21])
    
    plt.show()
    plt.savefig(base_out_folder+"/"+city_string_out+out_file+"_hist.png")
    
    n_bins=30
    
    fig, axs = plt.subplots(1, 2, figsize=(25, 15))
    sns.histplot(data=df, x="Noise", ax=axs[0], kde=True, bins=n_bins)
    # sns.histplot(data=df, x="DivTopo", ax=axs[0,1],kde=True, bins=60)
    # sns.histplot(data=df, x="MeanTreeDens", ax=axs[0,2], kde=True, bins=n_bins)
    sns.histplot(data=df, x="StreetInfo", ax=axs[1], kde=True, bins=n_bins)
    # sns.histplot(data=df, x="Absortion", ax=axs[1,1], kde=True, bins=n_bins)
    # sns.histplot(data=df, x="BuildingHeight", ax=axs[1,2], kde=True, bins=n_bins)
    # 
    axs[0].set_xlim([50, 80])
    axs[1].set_xlim([-0.01, 0.51])
    
    plt.show()
    plt.savefig(base_out_folder+"/"+city_string_out+out_file+"_noise.png")

    print("#### Plotting file done \n")


