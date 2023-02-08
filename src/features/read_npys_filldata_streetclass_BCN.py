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
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


plt.close('all')
plot_switch=True
write_switch=True


default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

base_in_folder="BCN_data/"

in_grid_file1="OSM_roads_bcn_streetclass_clip.npy"
# in_grid_file5="OSM_roads_bcn_nlanes_clip_smooth.npy"
in_grid_file2="OSM_roads_bcn_nlanes_clip.npy"
# in_grid_file6="OSM_roads_bcn_maxspeed_clip_smooth.npy"
in_grid_file3="OSM_roads_bcn_maxspeed_clip.npy"

out_grid_file2="OSM_roads_bcn_nlanes_clipfill.npy"
out_grid_file3="OSM_roads_bcn_maxspeed_clipfill.npy"

grid1=np.load(base_in_folder+in_grid_file1)
grid2=np.load(base_in_folder+in_grid_file2)
grid3=np.load(base_in_folder+in_grid_file3)

indexxy = np.where(grid1 >0)
df = pd.DataFrame(np.array((grid1[indexxy].flatten(order='C'), 
                            grid2[indexxy].flatten(order='C'),
                            grid3[indexxy].flatten(order='C'), 
                             )).transpose(), columns=["StreetClass","NLanes","SpeedLimit"])


# df[df["StreetClass"]==1]["SpeedLimit"].hist()

df['SpeedLimit_orig'] = df['SpeedLimit'] 
df['NLanes_orig'] = df['NLanes'] 

for a in df["StreetClass"].unique():
    df.loc[( (df["StreetClass"]==a) & (df["SpeedLimit"]==0)), "SpeedLimit"] = df[(df["StreetClass"]==a) & (df["SpeedLimit"]>0)]["SpeedLimit"].median()
    df.loc[( (df["StreetClass"]==a) & (df["NLanes"]==0)), "NLanes"] = df[(df["StreetClass"]==a) & (df["NLanes"]>0)]["NLanes"].median()
    
# indexxy2 = np.where(grid3[indexxy]==0)
# road_classes_old=sorted(df_hway["highway"].unique())

if write_switch:
    print("#### Saving to npy file")
    grid_target_export=np.empty(grid1.shape)*0
    indexxy = np.where(grid1)
    grid_target_export[indexxy] = np.array(df["NLanes"])
    np.save(base_in_folder+out_grid_file2,grid_target_export)
    
    grid_target_export=np.empty(grid1.shape)*0
    indexxy = np.where(grid1)
    grid_target_export[indexxy] = np.array(df["SpeedLimit"])
    np.save(base_in_folder+out_grid_file3,grid_target_export)
    print("#### Saving to npy file done")