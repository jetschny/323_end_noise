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


default_font_size=11
plt.rc('font', size=default_font_size) #controls default text size
plt.rc('axes', titlesize=default_font_size) #fontsize of the title
plt.rc('axes', labelsize=default_font_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=default_font_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=default_font_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=default_font_size) #fontsize of the legend

base_in_folder1:  str ="Z:/NoiseML/2024/city_data_features/"
base_in_folder2: str ="Z:/NoiseML/2024/city_data_MLpredictions/"
base_out_folder_pic: str ="Z:/NoiseML/2024/city_data_pics/"

city1_string_in="Vienna"
city1_string_out="VIE" 

in_grid_file1="VIEmodel_VIEpredict_xgboost_drop_feat6_extr.npy"
in_grid_target="VIE_target_noise_Aggroad_Lden.npy"


out_pic_file="VIE_target_noise_Aggroad_Lden_xgboost6_drop_feat6_extr"

grid1=np.load(base_in_folder2+city1_string_in+"/"+in_grid_file1)

grid_target=np.load(base_in_folder1+city1_string_in+"/"+in_grid_target)
grid_target= grid_target.astype(float)


# grid_target = 10**(grid_target/20)*(2e-5)
# grid1=np.log10(grid1/(2e-5))*20

noise_classes_old=sorted(np.unique(grid_target))
noise_classes_new=np.array(range(0, len(noise_classes_old), 1))

counter=0
for a in noise_classes_old:
    indexxy = np.where(grid_target ==a)
    grid_target[indexxy]=noise_classes_new[counter]
    counter=counter+1



# Get the colormap and set the under and bad colors
colMap = plt.cm.get_cmap("gist_rainbow").copy()
colMap.set_under(color='white')

if plot_switch:
    x_windows=[(400, 600),( 600, 800),( 600, 800),( 600, 800),( 900, 1100)]
    y_windows=[(1200, 1400),( 800, 1000),( 1000, 1200),( 1200, 1400),( 800, 1000)]
    
    for ii in range(5):
        
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
   
        im1=axs[0].imshow(grid_target, cmap=colMap, vmin=0.1)
        im2=axs[1].imshow(grid1, cmap=colMap, vmin=0.1)
        im3=axs[2].imshow(grid_target-grid1, cmap="seismic")
        
          # plt.axis('off')
        # plt.contourf(grid1)
        plt.colorbar(im1, ax=axs[0])
        plt.colorbar(im2, ax=axs[1])
        plt.colorbar(im3, ax=axs[2])
        
        axs[0].set_aspect('equal', 'box')
        axs[1].set_aspect('equal', 'box')
        axs[2].set_aspect('equal', 'box')
     
        # im1.set_clim(0.1,80)
        # im2.set_clim(0.1,80)
        im3.set_clim(-2,2)
        
        # x_window=
        # y_window=[1200, 1400]
        axs[0].set_xlim(x_windows[ii])
        axs[1].set_xlim(x_windows[ii])
        axs[2].set_xlim(x_windows[ii])
        
        axs[0].set_ylim(y_windows[ii])
        axs[1].set_ylim(y_windows[ii])
        axs[2].set_ylim(y_windows[ii])
        
        # axs[0].set_title('Distance to Road')
        # axs[1].set_title('Divergence from Topo')
     
        
        # plt.colorbar(con2, ax=ax2)
        plt.show()
        plt.savefig(base_out_folder_pic+out_pic_file+str(ii)+".png")
    
